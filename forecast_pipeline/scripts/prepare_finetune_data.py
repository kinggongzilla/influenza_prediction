#!/usr/bin/env python3
"""
Step 2b: Build training and validation samples for Chronos-2 fine-tuning.

For each qualifying country in data/quality_report.csv:
  - Loads full ILI time series with zero-to-NaN cleaning and leading-artifact trimming
  - Adds weather, hemisphere, week_sin/cos, and neighbor covariates
  - Creates sliding-window training samples (context=260wk, prediction=4wk, stride=4wk)
  - Applies peak oversampling (3× for target windows in top-30% of values)
  - Applies recency weighting via sample repetition (exp decay, λ=0.3)
  - Splits into train (all except last 104 weeks) and validation (last 104 weeks)

Outputs:
  data/finetune_train.pkl   - list of dicts for pipeline.fit(inputs=...)
  data/finetune_val.pkl     - list of dicts for pipeline.fit(validation_inputs=...)

Usage:
    cd forecast_pipeline
    python scripts/prepare_finetune_data.py [--dry_run]
"""

import os
import sys
import math
import pickle
import argparse
import contextlib
import io
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "scripts"))
from data_loader import load_time_series_data, load_data_type_indicator, clean_zeros_to_nan, trim_leading_artifact
from weather_fetcher import get_weather_for_country, get_country_coordinates

EXTRACTED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "extracted_data")
TRAINING_COUNTRIES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "training_countries.json")
WEATHER_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "weather_cache")
OUTPUT_TRAIN = os.path.join(os.path.dirname(__file__), "..", "data", "finetune_train.pkl")
OUTPUT_VAL = os.path.join(os.path.dirname(__file__), "..", "data", "finetune_val.pkl")

# Quality filter thresholds
MIN_WEEKS_NONNAN = 156       # 3 years of real data
MIN_YEARS_COVERAGE = 4
MAX_NAN_PCT_SPORADIC = 50.0  # for sporadic-NaN countries
MIN_MEDIAN_NONZERO = 20.0    # exclude tiny-count series (Bermuda-type)

CONTEXT_LENGTH = 260      # 5 years of weekly context per sample
PREDICTION_LENGTH = 4     # 4-week forecast horizon
STRIDE = 4                # slide window every 4 weeks
HELD_OUT_WEEKS = 104      # last 2 seasons reserved for validation
MAX_NEIGHBORS = 6         # max neighbor channels per country
PEAK_QUANTILE = 0.70      # threshold for peak oversampling
PEAK_REPEAT = 3           # how many times to repeat peak windows
RECENCY_LAMBDA = 0.3      # exponential decay (half-life ≈ 2.3 yr)

# Country name fixes for weather fetcher / coordinate lookup
COUNTRY_NAME_MAP = {
    "Côte d'Ivoire": "Ivory Coast",
    "Kosovo (in accordance with UN Security Council resolution 1244 (1999))": "Kosovo",
    "Micronesia (Federated States of)": "Micronesia",
    "Netherlands (Kingdom of the)": "Netherlands",
    "Lao People's Democratic Republic": "Laos",
    "United States of America": "United States",
    "United Kingdom, England": "United Kingdom",
    "United Kingdom, Northern Ireland": "United Kingdom",
    "United Kingdom, Scotland": "United Kingdom",
    "United Kingdom, Wales": "United Kingdom",
    "United Republic of Tanzania": "Tanzania",
    "Russian Federation": "Russia",
    "Türkiye": "Turkey",
    "Viet Nam": "Vietnam",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Iran (Islamic Republic of)": "Iran",
}


# ---------------------------------------------------------------------------
# Covariate builders
# ---------------------------------------------------------------------------

def build_hemisphere_covariate(lat: float, n: int) -> np.ndarray:
    if lat > 23.5:
        hemi = 1.0
    elif lat < -23.5:
        hemi = -1.0
    else:
        hemi = 0.0
    return np.full(n, hemi, dtype=np.float32)


def build_week_covariates(dates) -> tuple[np.ndarray, np.ndarray]:
    dates_pd = pd.to_datetime([str(d) for d in dates])
    week_num = dates_pd.isocalendar().week.astype(float).values
    sin_vals = np.sin(2 * np.pi * week_num / 52.0).astype(np.float32)
    cos_vals = np.cos(2 * np.pi * week_num / 52.0).astype(np.float32)
    return sin_vals, cos_vals


def _find_best_weather_cache(lat: float, lon: float, start_date: str, end_date: str):
    """
    Find a cache file for this lat/lon whose date range covers [start_date, end_date].
    Returns the cache file path (Path) or None.
    Prefers an exact match; falls back to any file whose range contains our range.
    """
    from pathlib import Path
    prefix = f"{lat:.4f}_{lon:.4f}_"
    cache_dir = Path(WEATHER_CACHE_DIR)

    exact = cache_dir / f"{prefix}{start_date}_{end_date}.parquet"
    if exact.exists():
        return exact

    # Look for any file with matching coordinates that spans our dates
    candidates = sorted(cache_dir.glob(f"{prefix}*.parquet"))
    for candidate in candidates:
        try:
            import re
            dates_in_name = re.findall(r'\d{4}-\d{2}-\d{2}', candidate.stem)
            if len(dates_in_name) == 2:
                cached_start, cached_end = dates_in_name[0], dates_in_name[1]
                if cached_start <= start_date and cached_end >= end_date:
                    return candidate
        except Exception:
            continue

    return None


def build_weather_covariates(country_name: str, dates, lat: float, lon: float, normalize: bool = True) -> dict:
    """
    Load weather from cache only by finding a matching parquet file and
    aggregating/normalizing directly — no API calls.
    Returns empty dict if no suitable cache file exists.
    """
    from weather_fetcher import aggregate_weather_to_weekly, normalize_weather_features

    dates_pd = pd.to_datetime([str(d) for d in dates])
    start_date = dates_pd.min().strftime("%Y-%m-%d")
    end_date = dates_pd.max().strftime("%Y-%m-%d")

    cache_file = _find_best_weather_cache(lat, lon, start_date, end_date)
    if cache_file is None:
        print(f"    ⚠  Weather not cached for {country_name}, skipping")
        return {}

    try:
        daily_df = pd.read_parquet(cache_file)
        # Trim to required date range
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        daily_df = daily_df[
            (daily_df["date"] >= start_date) & (daily_df["date"] <= end_date)
        ].reset_index(drop=True)

        # Aggregate to weekly
        with contextlib.redirect_stdout(io.StringIO()):
            weekly_df = aggregate_weather_to_weekly(daily_df, dates)

        if normalize:
            with contextlib.redirect_stdout(io.StringIO()):
                weekly_df, _ = normalize_weather_features(weekly_df)

        feature_cols = [c for c in weekly_df.columns if c != "date"]
        result = {}
        for col in feature_cols:
            vals = weekly_df[col].values.astype(np.float32)
            if len(vals) == len(dates):
                result[col] = vals
        return result
    except Exception as e:
        print(f"    ⚠  Weather load error for {country_name}: {e}")
        return {}


def find_extracted_file(country_name: str, data_type: str = None) -> str | None:
    """Find extracted data file. Prefers combined, then ILI/ARI based on data_type."""
    safe = country_name.replace(" ", "_").replace(",", "").replace("'", "").replace("(", "").replace(")", "")
    suffixes = ["_combined.csv", "_ili.csv", "_ari.csv"]
    if data_type == "ari":
        suffixes = ["_combined.csv", "_ari.csv", "_ili.csv"]

    for suffix in suffixes:
        candidates = [
            os.path.join(EXTRACTED_DATA_DIR, f"extracted_{safe}{suffix}"),
            os.path.join(EXTRACTED_DATA_DIR, f"{safe}{suffix}"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        # Search by partial name
        for fname in os.listdir(EXTRACTED_DATA_DIR):
            if fname.endswith(suffix) and safe.lower() in fname.lower():
                return os.path.join(EXTRACTED_DATA_DIR, fname)
    return None


def build_neighbor_covariates(country_name: str, target_dates) -> dict:
    """Load ILI from up to MAX_NEIGHBORS neighboring countries."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from src.scripts.country_neighbors import get_neighbors
    except ImportError:
        return {}

    neighbors = get_neighbors(country_name)[:MAX_NEIGHBORS]
    covariates = {}
    for neighbor in neighbors:
        filepath = find_extracted_file(neighbor)
        if not filepath:
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                nb_data, nb_dates, _ = load_time_series_data(
                    filepath, context_length=None, min_value=None
                )
            if nb_dates is None:
                continue
            nb_lookup = {d: v for d, v in zip(nb_dates, nb_data.tolist())}
            aligned = np.array(
                [nb_lookup.get(d, float("nan")) for d in target_dates],
                dtype=np.float32
            )
            coverage = np.sum(~np.isnan(aligned)) / len(aligned)
            if coverage >= 0.5:
                key = f"neighbor_{neighbor.replace(' ', '_')}_ili"
                covariates[key] = aligned
        except Exception:
            continue
    return covariates


# ---------------------------------------------------------------------------
# Recency weighting
# ---------------------------------------------------------------------------

def recency_repeat_count(window_end_date, most_recent_date, lambda_: float = RECENCY_LAMBDA) -> int:
    """Return integer repeat count based on exponential recency weighting."""
    years_ago = (most_recent_date - window_end_date).days / 365.25
    weight = math.exp(-lambda_ * years_ago)
    # Quantize into 1-3 repeats
    if weight >= 0.70:   # last ~1.2 years
        return 3
    elif weight >= 0.30:  # last ~4 years
        return 2
    else:
        return 1


# ---------------------------------------------------------------------------
# Window builder
# ---------------------------------------------------------------------------

def build_windows(
    values: np.ndarray,
    dates,
    all_covariates: dict,
    train_end_idx: int,
    peak_threshold: float,
    most_recent_date,
) -> tuple[list[dict], list[dict]]:
    """
    Build training and validation sample dicts.

    Training: windows where target ends at or before train_end_idx.
    Validation: single window where context is up to train_end_idx, target is next PREDICTION_LENGTH.
    """
    n = len(values)
    dates_pd = pd.to_datetime([str(d) for d in dates])

    train_samples = []
    val_samples = []

    # ---- Training windows ----
    # Stride through the training portion, building context→target pairs
    for end_ctx in range(CONTEXT_LENGTH, train_end_idx, STRIDE):
        end_tgt = end_ctx + PREDICTION_LENGTH
        if end_tgt > train_end_idx:
            break

        ctx_start = max(0, end_ctx - CONTEXT_LENGTH)
        target = values[end_ctx:end_tgt]

        # Skip windows with all-NaN targets
        if np.all(np.isnan(target)):
            continue

        context = values[ctx_start:end_ctx]

        # Build covariate slices
        sample = {"target": context}
        if all_covariates:
            past_covs = {}
            for name, arr in all_covariates.items():
                past_covs[name] = arr[ctx_start:end_ctx]
            sample["past_covariates"] = past_covs

        # Recency weighting
        window_end_date = dates_pd[end_ctx - 1]
        repeats = recency_repeat_count(window_end_date, most_recent_date)

        # Peak oversampling: check if any target value exceeds threshold
        valid_target = target[~np.isnan(target)]
        is_peak = len(valid_target) > 0 and float(np.max(valid_target)) > peak_threshold
        if is_peak:
            repeats *= PEAK_REPEAT

        for _ in range(repeats):
            train_samples.append(sample)

    # ---- Validation windows ----
    # Slide over the held-out portion (train_end_idx to end of series)
    # Build windows where context is from training portion
    for end_ctx in range(train_end_idx, n, STRIDE):
        end_tgt = end_ctx + PREDICTION_LENGTH
        if end_tgt > n:
            break

        ctx_start = max(0, end_ctx - CONTEXT_LENGTH)
        target = values[end_ctx:end_tgt]

        if np.all(np.isnan(target)):
            continue

        context = values[ctx_start:end_ctx]

        val_sample = {"target": context}
        if all_covariates:
            past_covs = {}
            for name, arr in all_covariates.items():
                past_covs[name] = arr[ctx_start:end_ctx]
            val_sample["past_covariates"] = past_covs

        val_samples.append(val_sample)

    return train_samples, val_samples


# ---------------------------------------------------------------------------
# Per-country processing
# ---------------------------------------------------------------------------

AVAILABLE_COVARIATES = ["data_type", "hemisphere", "week_of_year", "weather", "neighbors"]


def process_country(country_name: str, filepath: str, covariates: list[str] | None = None, held_out_weeks: int = HELD_OUT_WEEKS) -> tuple[list, list]:
    print(f"\n  [{country_name}]")

    # Load full series
    with contextlib.redirect_stdout(io.StringIO()):
        data, dates, _ = load_time_series_data(filepath, context_length=None, min_value=None)

    values = data.numpy().copy()
    if dates is None or len(dates) == 0:
        print("    ✗ No dates, skipping")
        return [], []

    # Clean
    values = clean_zeros_to_nan(values, max_consecutive_zeros=3)
    values, dates = trim_leading_artifact(values, dates)
    n = len(values)

    if n < CONTEXT_LENGTH + PREDICTION_LENGTH + held_out_weeks:
        print(f"    ✗ Series too short after trimming ({n} weeks), skipping")
        return [], []

    # Train/val split index
    train_end_idx = n - held_out_weeks

    # Dates
    dates_pd = pd.to_datetime([str(d) for d in dates])
    most_recent_date = dates_pd[train_end_idx - 1]

    # Peak threshold (based on training portion only)
    train_vals = values[:train_end_idx]
    nonzero_train = train_vals[(train_vals > 0) & ~np.isnan(train_vals)]
    if len(nonzero_train) > 0:
        peak_threshold = float(np.quantile(nonzero_train, PEAK_QUANTILE))
    else:
        peak_threshold = float("inf")  # no peaks → no oversampling

    # ---- Covariates ----
    all_covariates = {}
    covs = covariates or []

    if covs:
        print(f"    covariates: {covs}")
    else:
        print(f"    covariates: none (target only)")

    # Coordinates needed for hemisphere and weather
    need_coords = bool({"hemisphere", "weather"} & set(covs))
    lat, lon = 0.0, 0.0
    if need_coords:
        mapped_name = COUNTRY_NAME_MAP.get(country_name, country_name)
        try:
            lat, lon = get_country_coordinates(mapped_name)
        except Exception:
            pass

    if "hemisphere" in covs:
        all_covariates["hemisphere"] = build_hemisphere_covariate(lat, n)

    if "week_of_year" in covs:
        sin_woy, cos_woy = build_week_covariates(dates)
        all_covariates["week_sin"] = sin_woy
        all_covariates["week_cos"] = cos_woy

    if "weather" in covs:
        weather_covs = build_weather_covariates(country_name, dates, lat=lat, lon=lon, normalize=True)
        all_covariates.update(weather_covs)
        if weather_covs:
            print(f"    weather: {list(weather_covs.keys())}")
        else:
            print(f"    weather: none (will train without)")

    if "neighbors" in covs:
        neighbor_covs = build_neighbor_covariates(country_name, dates)
        all_covariates.update(neighbor_covs)
        if neighbor_covs:
            print(f"    neighbors: {list(neighbor_covs.keys())}")

    if "data_type" in covs:
        dt_indicator = load_data_type_indicator(filepath, dates)
        if dt_indicator is not None:
            all_covariates["data_type_indicator"] = dt_indicator
            n_ari = int((dt_indicator > 0.5).sum())
            if n_ari > 0:
                print(f"    data_type: {n - n_ari} ILI + {n_ari} ARI weeks")
        else:
            # Legacy file — infer from filename
            if "_ari" in filepath.lower():
                all_covariates["data_type_indicator"] = np.ones(n, dtype=np.float32)
                print(f"    data_type: all ARI (legacy file)")
            else:
                all_covariates["data_type_indicator"] = np.zeros(n, dtype=np.float32)
                print(f"    data_type: all ILI (legacy file)")

    # Build windows
    train_samples, val_samples = build_windows(
        values=values,
        dates=dates,
        all_covariates=all_covariates,
        train_end_idx=train_end_idx,
        peak_threshold=peak_threshold,
        most_recent_date=most_recent_date,
    )

    n_cov = len(all_covariates)
    print(f"    → {len(train_samples)} train windows, {len(val_samples)} val windows "
          f"| {n_cov} covariates | peak_thresh={peak_threshold:.0f}")

    return train_samples, val_samples


# ---------------------------------------------------------------------------
# Quality assessment
# ---------------------------------------------------------------------------

def classify_nan_pattern(values: np.ndarray, dates) -> str:
    """Returns 'seasonal' if NaN weeks cluster in the same months every year, else 'sporadic'."""
    df = pd.DataFrame({"v": values, "date": pd.to_datetime([str(d) for d in dates])})
    df["month"] = df["date"].dt.month
    monthly_nan = df.groupby("month")["v"].apply(lambda x: x.isna().mean())
    low_nan_months = int((monthly_nan < 0.2).sum())
    high_nan_months = int((monthly_nan > 0.8).sum())
    if low_nan_months >= 8 and high_nan_months >= 2:
        return "seasonal"
    return "sporadic"


def assess_country(filepath: str, country_name: str, data_type: str = "combined") -> dict:
    """Assess data quality for a single country. Returns dict with stats and include decision."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            data, dates, _ = load_time_series_data(filepath, context_length=None, min_value=None)
    except Exception as e:
        return {"country": country_name, "data_type": data_type, "include": False,
                "exclude_reason": str(e)}

    values = data.numpy().copy()
    values = clean_zeros_to_nan(values)
    values, dates = trim_leading_artifact(values, dates)

    n_total = len(values)
    n_nonnan = int(np.sum(~np.isnan(values)))
    nan_pct = round(100.0 * (1 - n_nonnan / n_total), 1) if n_total > 0 else 100.0

    if dates is not None and len(dates) > 0:
        dates_pd = pd.to_datetime([str(d) for d in dates])
        years_coverage = round((dates_pd[-1] - dates_pd[0]).days / 365.25, 1)
    else:
        years_coverage = 0.0

    nonzero_vals = values[(values > 0) & ~np.isnan(values)]
    scale = round(float(np.median(nonzero_vals)), 2) if len(nonzero_vals) > 0 else 0.0
    nan_pattern = classify_nan_pattern(values, dates) if dates is not None else "sporadic"

    reasons = []
    if n_nonnan < MIN_WEEKS_NONNAN:
        reasons.append(f"too_few_nonnan({n_nonnan}<{MIN_WEEKS_NONNAN})")
    if years_coverage < MIN_YEARS_COVERAGE:
        reasons.append(f"too_short({years_coverage:.1f}yr<{MIN_YEARS_COVERAGE})")
    if nan_pattern == "sporadic" and nan_pct > MAX_NAN_PCT_SPORADIC:
        reasons.append(f"high_nan_sporadic({nan_pct}%>50%)")
    if scale < MIN_MEDIAN_NONZERO:
        reasons.append(f"too_small_scale({scale}<{MIN_MEDIAN_NONZERO})")

    return {
        "country": country_name,
        "data_type": data_type,
        "include": len(reasons) == 0,
        "exclude_reason": "; ".join(reasons) if reasons else "",
        "n_weeks_nonnan": n_nonnan,
        "nan_pct": nan_pct,
        "years_coverage": years_coverage,
        "nan_pattern": nan_pattern,
        "scale": scale,
    }


def scan_extracted_data() -> list[dict]:
    """Scan all extracted CSVs, assess quality, return list of per-country dicts."""
    combined_files = {f.replace("_combined.csv", ""): (f, "combined")
                      for f in os.listdir(EXTRACTED_DATA_DIR) if f.endswith("_combined.csv")}
    ili_files = {f.replace("_ili.csv", ""): (f, "ili")
                 for f in os.listdir(EXTRACTED_DATA_DIR) if f.endswith("_ili.csv")}
    ari_files = {f.replace("_ari.csv", ""): (f, "ari")
                 for f in os.listdir(EXTRACTED_DATA_DIR) if f.endswith("_ari.csv")}

    # Priority: combined > ILI > ARI
    country_files = {}
    for files in [ari_files, ili_files, combined_files]:  # later entries overwrite
        country_files.update(files)

    results = []
    for key in sorted(country_files):
        fname, data_type = country_files[key]
        country = key.replace("_", " ")
        filepath = os.path.join(EXTRACTED_DATA_DIR, fname)
        result = assess_country(filepath, country, data_type=data_type)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import json

    parser = argparse.ArgumentParser(
        description="Assess data quality, update training_countries.json, and build fine-tuning samples"
    )
    parser.add_argument("--dry_run", action="store_true",
                        help="Print stats without saving output files")
    parser.add_argument("--covariates", nargs="*", default=None,
                        metavar="COV",
                        help=f"Covariates to include. Available: {AVAILABLE_COVARIATES}. "
                             "Omit flag for all, pass with no args for none (target only).")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix for output filenames (e.g. '_nocov')")
    parser.add_argument("--exclude_countries", type=str, default="",
                        help="Comma-separated list of countries to exclude (e.g. 'Thailand')")
    parser.add_argument("--held_out_weeks", type=int, default=HELD_OUT_WEEKS,
                        help=f"Weeks to hold out for validation (default: {HELD_OUT_WEEKS})")
    parser.add_argument("--assess_only", action="store_true",
                        help="Only run quality assessment and update training_countries.json (skip training data)")
    args = parser.parse_args()

    # Resolve covariates: None (flag omitted) → all, [] (flag with no args) → none
    if args.covariates is None:
        covariates = list(AVAILABLE_COVARIATES)
    else:
        covariates = args.covariates
        for c in covariates:
            if c not in AVAILABLE_COVARIATES:
                parser.error(f"Unknown covariate '{c}'. Available: {AVAILABLE_COVARIATES}")

    if not covariates and not args.output_suffix:
        args.output_suffix = "_nocov"

    output_train = OUTPUT_TRAIN.replace(".pkl", f"{args.output_suffix}.pkl") if args.output_suffix else OUTPUT_TRAIN
    output_val = OUTPUT_VAL.replace(".pkl", f"{args.output_suffix}.pkl") if args.output_suffix else OUTPUT_VAL

    # --- Step 1: Scan and assess all extracted data ---
    print("Scanning extracted data for quality assessment...")
    assessments = scan_extracted_data()

    exclude_set = set()
    if args.exclude_countries:
        exclude_set = {c.strip() for c in args.exclude_countries.split(",")}

    passed = []
    failed = []
    for a in assessments:
        if a["country"] in exclude_set:
            a["include"] = False
            a["exclude_reason"] = "manually excluded"
        if a["include"]:
            passed.append(a)
            print(f"  PASS {a['country']} ({a['data_type']}, {a['nan_pattern']}, "
                  f"{a['n_weeks_nonnan']}wk, scale={a['scale']:.0f})")
        else:
            failed.append(a)

    print(f"\n{'='*60}")
    print(f"  PASS: {len(passed)} countries  |  FAIL: {len(failed)} countries")
    print(f"{'='*60}")

    if failed:
        print(f"\nExcluded ({len(failed)}):")
        for a in failed:
            print(f"  {a['country']}: {a['exclude_reason']}")

    # --- Step 2: Save training_countries.json ---
    training_countries = sorted([a["country"] for a in passed])
    country_data_types = {a["country"]: a["data_type"] for a in passed}

    training_data = {
        "excluded_from_training": sorted(list(exclude_set)),
        "training_countries": training_countries,
        "country_data_types": {c: country_data_types[c] for c in training_countries},
    }

    if not args.dry_run:
        with open(TRAINING_COUNTRIES_PATH, "w") as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {TRAINING_COUNTRIES_PATH} ({len(training_countries)} countries)")
    else:
        print(f"\nDry run — would save {len(training_countries)} countries to {TRAINING_COUNTRIES_PATH}")

    if args.assess_only:
        return

    # --- Step 3: Build training samples ---
    print(f"\nBuilding training data for {len(passed)} countries")
    print(f"Context={CONTEXT_LENGTH}wk | Prediction={PREDICTION_LENGTH}wk | Stride={STRIDE}wk | Held-out={args.held_out_weeks}wk")
    print(f"Covariates: {covariates if covariates else 'none (target only)'}")
    print(f"Peak oversampling {PEAK_REPEAT}× above {PEAK_QUANTILE:.0%} quantile | Recency λ={RECENCY_LAMBDA}")

    all_train = []
    all_val = []

    for a in passed:
        country = a["country"]
        filepath = find_extracted_file(country, data_type=a["data_type"])
        if filepath is None:
            print(f"\n  {country}: data file not found, skipping")
            continue

        try:
            train_s, val_s = process_country(country, filepath, covariates=covariates, held_out_weeks=args.held_out_weeks)
            all_train.extend(train_s)
            all_val.extend(val_s)
        except Exception as e:
            print(f"\n  {country}: error — {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Total training samples : {len(all_train):,}")
    print(f"Total validation samples: {len(all_val):,}")

    if args.dry_run:
        print("Dry run — not saving.")
        return

    os.makedirs(os.path.dirname(output_train), exist_ok=True)

    with open(output_train, "wb") as f:
        pickle.dump(all_train, f)
    print(f"Saved: {output_train}")

    with open(output_val, "wb") as f:
        pickle.dump(all_val, f)
    print(f"Saved: {output_val}")


if __name__ == "__main__":
    main()
