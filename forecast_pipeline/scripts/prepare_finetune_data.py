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
from data_loader import load_time_series_data
from weather_fetcher import get_weather_for_country, get_country_coordinates

EXTRACTED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "extracted_data")
QUALITY_REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "quality_report.csv")
WEATHER_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "weather_cache")
OUTPUT_TRAIN = os.path.join(os.path.dirname(__file__), "..", "data", "finetune_train.pkl")
OUTPUT_VAL = os.path.join(os.path.dirname(__file__), "..", "data", "finetune_val.pkl")

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
# Data cleaning helpers (matches assess_data_quality.py)
# ---------------------------------------------------------------------------

def clean_zeros_to_nan(values: np.ndarray, max_consecutive_zeros: int = 3) -> np.ndarray:
    series = pd.Series(values.copy(), dtype=float)
    is_zero = series == 0
    run_id = (is_zero != is_zero.shift()).cumsum()
    run_lengths = is_zero.groupby(run_id).transform("sum")
    series[is_zero & (run_lengths > max_consecutive_zeros)] = np.nan
    return series.values


def trim_leading_artifact(values: np.ndarray, dates, threshold_pct: float = 0.10):
    nonzero = values[(values > 0) & ~np.isnan(values)]
    if len(nonzero) == 0:
        return values, dates
    overall_median = float(np.median(nonzero))
    threshold = overall_median * threshold_pct
    for i in range(len(values) - 8):
        window = values[i:i + 8]
        valid = window[~np.isnan(window)]
        if len(valid) > 0 and float(np.mean(valid)) > threshold:
            return values[i:], dates[i:]
    return values, dates


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


def find_extracted_file(country_name: str) -> str | None:
    safe = country_name.replace(" ", "_").replace(",", "").replace("'", "").replace("(", "").replace(")", "")
    candidates = [
        os.path.join(EXTRACTED_DATA_DIR, f"extracted_{safe}_ili.csv"),
        os.path.join(EXTRACTED_DATA_DIR, f"{safe}_ili.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Search by partial name
    for fname in os.listdir(EXTRACTED_DATA_DIR):
        if fname.endswith("_ili.csv") and safe.lower() in fname.lower():
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

def process_country(country_name: str, filepath: str, no_covariates: bool = False, held_out_weeks: int = HELD_OUT_WEEKS) -> tuple[list, list]:
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

    if not no_covariates:
        mapped_name = COUNTRY_NAME_MAP.get(country_name, country_name)
        try:
            lat, lon = get_country_coordinates(mapped_name)
        except Exception:
            lat, lon = 0.0, 0.0

        # Hemisphere (constant)
        all_covariates["hemisphere"] = build_hemisphere_covariate(lat, n)

        # Week of year (sin/cos)
        sin_woy, cos_woy = build_week_covariates(dates)
        all_covariates["week_sin"] = sin_woy
        all_covariates["week_cos"] = cos_woy

        # Weather
        weather_covs = build_weather_covariates(country_name, dates, lat=lat, lon=lon, normalize=True)
        all_covariates.update(weather_covs)
        if weather_covs:
            print(f"    weather: {list(weather_covs.keys())}")
        else:
            print(f"    weather: none (will train without)")

        # Neighbor ILI
        neighbor_covs = build_neighbor_covariates(country_name, dates)
        all_covariates.update(neighbor_covs)
        if neighbor_covs:
            print(f"    neighbors: {list(neighbor_covs.keys())}")
    else:
        print(f"    covariates: none (target only)")

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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Print stats without saving output files")
    parser.add_argument("--no_covariates", action="store_true",
                        help="Build samples without any covariates (target only)")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix for output filenames (e.g. '_nocov')")
    parser.add_argument("--exclude_countries", type=str, default="",
                        help="Comma-separated list of countries to exclude (e.g. 'Italy,Germany')")
    parser.add_argument("--held_out_weeks", type=int, default=HELD_OUT_WEEKS,
                        help=f"Weeks to hold out for validation (default: {HELD_OUT_WEEKS})")
    args = parser.parse_args()

    if args.no_covariates and not args.output_suffix:
        args.output_suffix = "_nocov"

    output_train = OUTPUT_TRAIN.replace(".pkl", f"{args.output_suffix}.pkl") if args.output_suffix else OUTPUT_TRAIN
    output_val = OUTPUT_VAL.replace(".pkl", f"{args.output_suffix}.pkl") if args.output_suffix else OUTPUT_VAL

    df = pd.read_csv(QUALITY_REPORT_PATH)
    included = df[df["include"]].reset_index(drop=True)

    if args.exclude_countries:
        exclude_set = {c.strip() for c in args.exclude_countries.split(",")}
        before = len(included)
        included = included[~included["country"].isin(exclude_set)].reset_index(drop=True)
        print(f"Excluded {before - len(included)} countries: {exclude_set}")

    print(f"Processing {len(included)} qualifying countries")
    print(f"Context={CONTEXT_LENGTH}wk | Prediction={PREDICTION_LENGTH}wk | Stride={STRIDE}wk | Held-out={args.held_out_weeks}wk")
    print(f"Peak oversampling {PEAK_REPEAT}× above {PEAK_QUANTILE:.0%} quantile | Recency λ={RECENCY_LAMBDA}")

    all_train = []
    all_val = []

    for i, row in included.iterrows():
        country = row["country"]
        # Find the ILI file
        safe = country.replace(" ", "_").replace(",", "").replace("'", "").replace("(", "").replace(")", "")
        # Try multiple filename patterns
        filepath = None
        for fname in os.listdir(EXTRACTED_DATA_DIR):
            if fname.endswith("_ili.csv") and safe.lower().replace("ô", "o") in fname.lower():
                filepath = os.path.join(EXTRACTED_DATA_DIR, fname)
                break
        if filepath is None:
            print(f"\n  ✗ {country}: ILI file not found, skipping")
            continue

        try:
            train_s, val_s = process_country(country, filepath, no_covariates=args.no_covariates, held_out_weeks=args.held_out_weeks)
            all_train.extend(train_s)
            all_val.extend(val_s)
        except Exception as e:
            print(f"\n  ✗ {country}: error — {e}")
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
