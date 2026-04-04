#!/usr/bin/env python3
"""
Head-to-head comparison with Influcast models using pairwise relative WIS.

Downloads all Influcast model forecasts for the 2023/24 season,
generates Chronos-2 forecasts in the same format, and computes
pairwise relative WIS (Cramer et al., 2022) as used in the paper.
"""

import argparse
import os
import sys
import time
import urllib.request
from io import StringIO

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "scripts"))
from chronos.chronos2 import Chronos2Pipeline

WEATHER_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "weather_cache")

# Italy coordinates (centroid)
ITALY_LAT, ITALY_LON = 41.8719, 12.5674

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INFLUCAST_QUANTILE_LEVELS = [
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99,
]

INFLUCAST_INTERVALS = [
    (0.01,  0.99,  0.02),
    (0.025, 0.975, 0.05),
    (0.05,  0.95,  0.10),
    (0.10,  0.90,  0.20),
    (0.15,  0.85,  0.30),
    (0.20,  0.80,  0.40),
    (0.25,  0.75,  0.50),
    (0.30,  0.70,  0.60),
    (0.35,  0.65,  0.70),
    (0.40,  0.60,  0.80),
    (0.45,  0.55,  0.90),
]

BASE_URL = "https://raw.githubusercontent.com/Predizioni-Epidemiologiche-Italia/Influcast/main"

INFLUCAST_MODELS = [
    "ISI-IPSICast",
    "ISI-GLEAM",
    "EpiQMUL-ARIMA_QMUL",
    "EpiQMUL-SEIR_QMUL",
    "comunipd-mobnetSI2R",
    "UNIPD_NEIDE-SEEIIRS_MCMC",
    "ISI-FluABCaster",
    "ISI-FluBcast",
    "CSL_PoliTo-metaFlu",
    "ev_and_modelers-DeepRE",
    "FBK_HE-REST_HE",
    "C2S2_Trento-SIR_INN",
    "Influcast-quantileBaseline",
    "Influcast-Ensemble",
]

SEASONS = {
    "2023-24": {"start_year": 2023, "start_week": 46, "end_year": 2024, "end_week": 13},
    "2024-25": {"start_year": 2024, "start_week": 45, "end_year": 2025, "end_week": 16},
}


def build_round_weeks(season: str) -> list[str]:
    s = SEASONS[season]
    weeks = []
    for w in range(s["start_week"], 53):
        weeks.append(f"{s['start_year']}_{w:02d}")
    for w in range(1, s["end_week"] + 1):
        weeks.append(f"{s['end_year']}_{w:02d}")
    return weeks


# Default to 2023-24 for backwards compat
ROUND_WEEKS = build_round_weeks("2023-24")

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "influcast_italy_national.csv")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "influcast_cache")


def load_ground_truth(season: str) -> dict[tuple[int, int], float]:
    """Load ground truth from Influcast repo (authoritative source for their eval).
    Falls back to influnet data if download fails."""
    truth = {}

    # Try Influcast repo first — this is what the models were evaluated against
    s = SEASONS[season]
    # Build list of weeks that need ground truth (forecast targets = round weeks + up to 4 ahead)
    all_weeks = []
    for w in range(s["start_week"], 53):
        all_weeks.append((s["start_year"], w))
    for w in range(1, s["end_week"] + 5):  # +4 for forecast horizon
        all_weeks.append((s["end_year"], w))

    season_str = f"{s['start_year']}-{s['end_year']}"
    cache_dir = os.path.join(CACHE_DIR, "ground_truth")
    os.makedirs(cache_dir, exist_ok=True)

    # Try to download from Influcast repo
    for year, week in all_weeks:
        if year == s["start_year"]:
            gt_season = season_str
        else:
            gt_season = season_str
        cache_file = os.path.join(cache_dir, f"{gt_season}_{year}_{week:02d}.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file)
                for _, row in df.iterrows():
                    truth[(int(row["anno"]), int(row["settimana"]))] = float(row["incidenza"])
                continue
            except Exception:
                pass

        url = f"{BASE_URL}/sorveglianza/ILI/{gt_season}/italia-{year}_{week:02d}-ILI.csv"
        try:
            resp = urllib.request.urlopen(url, timeout=10)
            text = resp.read().decode("utf-8")
            with open(cache_file, "w") as f:
                f.write(text)
            df = pd.read_csv(StringIO(text))
            for _, row in df.iterrows():
                truth[(int(row["anno"]), int(row["settimana"]))] = float(row["incidenza"])
        except Exception:
            pass

    # Fall back to influnet data for any missing weeks
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        for _, row in df.iterrows():
            yw = row["year_week"]
            year, week = int(yw.split("-")[0]), int(yw.split("-")[1])
            if (year, week) not in truth:
                truth[(year, week)] = float(row["incidence"])

    return truth


def download_forecasts(model_name: str, round_weeks: list[str] = None) -> dict[str, pd.DataFrame]:
    """Download all round forecasts for a model. Returns {round_week: DataFrame}."""
    if round_weeks is None:
        round_weeks = ROUND_WEEKS
    forecasts = {}
    for rw in round_weeks:
        cache_file = os.path.join(CACHE_DIR, model_name, f"{rw}.csv")
        if os.path.exists(cache_file):
            try:
                forecasts[rw] = pd.read_csv(cache_file)
                continue
            except Exception:
                pass

        url = f"{BASE_URL}/previsioni/{model_name}/{rw}.csv"
        try:
            resp = urllib.request.urlopen(url, timeout=10)
            text = resp.read().decode("utf-8")
            os.makedirs(os.path.join(CACHE_DIR, model_name), exist_ok=True)
            with open(cache_file, "w") as f:
                f.write(text)
            forecasts[rw] = pd.read_csv(StringIO(text))
        except Exception:
            pass  # model didn't submit this round
    return forecasts


def extract_national_quantiles(df: pd.DataFrame, horizon: int) -> np.ndarray | None:
    """Extract quantile forecast for Italy national, given horizon (1-4).
    Returns array of shape (len(INFLUCAST_QUANTILE_LEVELS),) or None."""
    # Filter to national level, ILI target, and quantile type
    mask = (
        (df["luogo"] == "IT") &
        (df["tipo_valore"] == "quantile") &
        (df["orizzonte"] == horizon)
    )
    if "target" in df.columns:
        mask = mask & (df["target"] == "ILI")
    sub = df[mask].copy()
    if len(sub) == 0:
        return None

    sub["id_valore"] = sub["id_valore"].astype(float)
    sub = sub.sort_values("id_valore")

    # Map to our quantile levels
    available = dict(zip(sub["id_valore"], sub["valore"]))
    result = []
    for q in INFLUCAST_QUANTILE_LEVELS:
        q_round = round(q, 3)
        if q_round in available:
            result.append(float(available[q_round]))
        else:
            # Try close match
            closest = min(available.keys(), key=lambda x: abs(x - q_round))
            if abs(closest - q_round) < 0.005:
                result.append(float(available[closest]))
            else:
                return None  # missing quantile
    return np.array(result)


def compute_wis(actual: float, quantiles: np.ndarray) -> float:
    """WIS with 11 Influcast intervals."""
    q_idx = {round(q, 3): i for i, q in enumerate(INFLUCAST_QUANTILE_LEVELS)}
    median = quantiles[q_idx[0.5]]
    K = len(INFLUCAST_INTERVALS)
    wis = 0.5 * abs(actual - median)
    for lower_q, upper_q, alpha in INFLUCAST_INTERVALS:
        l = quantiles[q_idx[round(lower_q, 3)]]
        u = quantiles[q_idx[round(upper_q, 3)]]
        is_score = (u - l) + (2.0 / alpha) * max(l - actual, 0.0) + (2.0 / alpha) * max(actual - u, 0.0)
        wis += (alpha / 2.0) * is_score
    return wis / (K + 0.5)


def generate_chronos_forecasts(model_path: str, use_covariates: bool, round_weeks: list[str] = None, use_weather: bool = False) -> dict[str, dict[int, np.ndarray]]:
    """Generate Chronos-2 forecasts for all rounds. Returns {round_week: {horizon: quantiles}}."""
    # Load Italian ILI data
    df = pd.read_csv(DATA_FILE)
    df = df.sort_values("year_week").reset_index(drop=True)
    values = df["incidence"].values.astype(np.float64)
    weeks = df["year_week"].tolist()
    # Convert weeks to (year, week) for lookup
    week_to_idx = {}
    for i, w in enumerate(weeks):
        year, wk = w.split("-")
        week_to_idx[(int(year), int(wk))] = i

    pipeline = Chronos2Pipeline.from_pretrained(model_path)
    data_tensor = torch.tensor(values, dtype=torch.float32)

    # Load weather covariates if requested
    weather_weekly = None
    if use_weather:
        try:
            from weather_fetcher import get_country_coordinates, aggregate_weather_to_weekly, normalize_weather_features
            import re as _re
            from pathlib import Path as _Path
            import contextlib as _ctx, io as _io

            lat, lon = get_country_coordinates("Italy")
            prefix = f"{lat:.4f}_{lon:.4f}_"
            cache_dir = _Path(WEATHER_CACHE_DIR)
            cache_file = None
            for candidate in sorted(cache_dir.glob(f"{prefix}*.parquet")):
                dates_in_name = _re.findall(r'\d{4}-\d{2}-\d{2}', candidate.stem)
                if len(dates_in_name) == 2:
                    cache_file = candidate
                    break  # take first match
            if cache_file:
                daily_df = pd.read_parquet(cache_file)
                daily_df["date"] = pd.to_datetime(daily_df["date"])
                # Convert year_weeks to dates for aggregation
                week_dates = [pd.Timestamp.fromisocalendar(int(w.split("-")[0]), int(w.split("-")[1]), 1) for w in weeks]
                with _ctx.redirect_stdout(_io.StringIO()):
                    weekly_df = aggregate_weather_to_weekly(daily_df, week_dates)
                with _ctx.redirect_stdout(_io.StringIO()):
                    weekly_df, _ = normalize_weather_features(weekly_df)
                feature_cols = [c for c in weekly_df.columns if c != "date"]
                weather_weekly = {}
                for col in feature_cols:
                    vals = weekly_df[col].values.astype(np.float32)
                    if len(vals) == len(weeks):
                        weather_weekly[col] = vals
                print(f"  Weather loaded: {len(weather_weekly)} features, {len(weeks)} weeks")
            else:
                print("  Warning: no weather cache found for Italy")
        except Exception as e:
            print(f"  Warning: weather loading failed: {e}")

    if round_weeks is None:
        round_weeks = ROUND_WEEKS
    results = {}
    for rw in round_weeks:
        year, week = int(rw.split("_")[0]), int(rw.split("_")[1])
        if (year, week) not in week_to_idx:
            continue
        cutoff_idx = week_to_idx[(year, week)]

        context = data_tensor[:cutoff_idx]

        has_covs = use_covariates or (use_weather and weather_weekly)
        if has_covs:
            past_covs = {}
            future_covs = {}

            if use_covariates:
                week_nums = []
                for w in weeks[:cutoff_idx]:
                    wn = int(w.split("-")[1])
                    week_nums.append(wn)
                week_nums_arr = np.array(week_nums, dtype=np.float64)
                past_covs["week_sin"] = np.sin(2 * np.pi * week_nums_arr / 52.0).astype(np.float32)
                past_covs["week_cos"] = np.cos(2 * np.pi * week_nums_arr / 52.0).astype(np.float32)
                past_covs["hemisphere"] = np.full(cutoff_idx, 1.0, dtype=np.float32)
                # Future covariates
                future_week_nums = []
                for fh in range(1, 5):
                    fidx = cutoff_idx + fh
                    if fidx < len(weeks):
                        future_week_nums.append(int(weeks[fidx].split("-")[1]))
                    else:
                        future_week_nums.append((week + fh - 1) % 52 + 1)
                future_week_nums = np.array(future_week_nums, dtype=np.float64)
                future_covs["week_sin"] = np.sin(2 * np.pi * future_week_nums / 52.0).astype(np.float32)
                future_covs["week_cos"] = np.cos(2 * np.pi * future_week_nums / 52.0).astype(np.float32)
                future_covs["hemisphere"] = np.full(4, 1.0, dtype=np.float32)

            if use_weather and weather_weekly:
                for col, vals in weather_weekly.items():
                    past_covs[col] = vals[:cutoff_idx]

            input_dict = {"target": context.numpy(), "past_covariates": past_covs}
            if future_covs:
                input_dict["future_covariates"] = future_covs
            inputs = [input_dict]
        else:
            inputs = context.unsqueeze(0).unsqueeze(0)

        try:
            qf, mf = pipeline.predict_quantiles(
                inputs=inputs,
                prediction_length=4,
                quantile_levels=INFLUCAST_QUANTILE_LEVELS,
            )
            qf = qf[0][0].cpu().numpy()  # (4, n_quantiles)
            results[rw] = {h: qf[h - 1] for h in range(1, 5)}
        except Exception as e:
            print(f"  Chronos failed for {rw}: {e}")

    return results


def parse_model_specs(specs: list[str]) -> list[dict]:
    """Parse model specs like 'path' or 'path:covs' or 'path:weather' into dicts."""
    results = []
    for spec in specs:
        parts = spec.split(":")
        path = parts[0]
        flags = parts[1] if len(parts) > 1 else ""
        use_covariates = "covs" in flags
        use_weather = "weather" in flags
        tag = os.path.basename(path)
        name = f"Chronos-2-{tag}"
        if use_covariates:
            name += "+covs"
        if use_weather:
            name += "+weather"
        results.append({"path": path, "use_covariates": use_covariates,
                        "use_weather": use_weather, "name": name})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None,
                        help="Single model path (legacy). Use --models for multiple.")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Multiple model specs: 'path' or 'path:covs' or 'path:weather'")
    parser.add_argument("--use_covariates", action="store_true")
    parser.add_argument("--use_weather", action="store_true", help="Add cached weather as past covariates")
    parser.add_argument("--season", type=str, default="2023-24", choices=list(SEASONS.keys()))
    args = parser.parse_args()

    # Build list of Chronos model specs
    if args.models:
        model_specs = parse_model_specs(args.models)
    else:
        path = args.model_path or "amazon/chronos-2"
        tag = os.path.basename(path)
        name = f"Chronos-2-{tag}"
        if args.use_covariates:
            name += "+covs"
        if args.use_weather:
            name += "+weather"
        model_specs = [{"path": path, "use_covariates": args.use_covariates,
                        "use_weather": args.use_weather, "name": name}]

    round_weeks = build_round_weeks(args.season)

    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1. Load ground truth
    print(f"Season: {args.season} ({len(round_weeks)} rounds)")
    print("Loading ground truth...")
    truth = load_ground_truth(args.season)
    print(f"  → {len(truth)} week observations")

    # 2. Download all Influcast model forecasts
    all_models = {}
    for model_name in INFLUCAST_MODELS:
        print(f"Downloading {model_name}...")
        all_models[model_name] = download_forecasts(model_name, round_weeks)
        n_rounds = len(all_models[model_name])
        print(f"  → {n_rounds} rounds")

    # 3. Generate Chronos forecasts for each model spec
    chronos_names = []
    for spec in model_specs:
        print(f"\nGenerating {spec['name']} forecasts...")
        forecasts = generate_chronos_forecasts(spec["path"], spec["use_covariates"], round_weeks, spec["use_weather"])
        print(f"  → {len(forecasts)} rounds")
        all_models[spec["name"]] = forecasts
        chronos_names.append(spec["name"])

    # 4. Build WIS scores and coverage per (model, round, horizon)
    # Structure: {model_name: {(round_week, horizon): wis_score}}
    print("\nComputing WIS scores and coverage...")
    all_wis = {}
    all_cov50 = {}   # {model_name: [covered_bool, ...]}
    all_cov90 = {}

    q_idx = {round(q, 3): i for i, q in enumerate(INFLUCAST_QUANTILE_LEVELS)}

    for model_name, forecasts in all_models.items():
        is_chronos = model_name in chronos_names
        model_wis = {}
        cov50_list = []
        cov90_list = []
        for rw, round_data in forecasts.items():
            year, week = int(rw.split("_")[0]), int(rw.split("_")[1])
            for h in range(1, 5):
                target_week = week + h
                target_year = year
                if target_week > 52:
                    target_week -= 52
                    target_year += 1
                actual = truth.get((target_year, target_week))
                if actual is None:
                    continue

                if is_chronos:
                    quantiles = round_data.get(h)
                else:
                    quantiles = extract_national_quantiles(round_data, h)
                if quantiles is None:
                    continue

                wis = compute_wis(actual, quantiles)
                model_wis[(rw, h)] = wis
                cov50_list.append(quantiles[q_idx[0.25]] <= actual <= quantiles[q_idx[0.75]])
                cov90_list.append(quantiles[q_idx[0.05]] <= actual <= quantiles[q_idx[0.95]])
        all_wis[model_name] = model_wis
        all_cov50[model_name] = cov50_list
        all_cov90[model_name] = cov90_list

    model_names = list(all_wis.keys())
    n_models = len(model_names)

    # 5. Print simple WIS/baseline ratio first
    print("\n" + "=" * 90)
    print("SIMPLE RELATIVE WIS (model mean WIS / baseline mean WIS)")
    print("=" * 90)
    baseline_name = "Influcast-quantileBaseline"
    baseline_wis = all_wis.get(baseline_name, {})

    simple_ratios = {}
    for model_name in model_names:
        model_scores = all_wis[model_name]
        # Only use rounds where both model and baseline have scores
        common_keys = set(model_scores.keys()) & set(baseline_wis.keys())
        if not common_keys:
            continue
        m_mean = np.mean([model_scores[k] for k in common_keys])
        b_mean = np.mean([baseline_wis[k] for k in common_keys])
        simple_ratios[model_name] = m_mean / b_mean

    print(f"  {'rWIS':>6}  {'Cov50%':>7}  {'Cov90%':>7}  {'Model'}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*40}")
    for model_name, ratio in sorted(simple_ratios.items(), key=lambda x: x[1]):
        marker = " ← US" if "Chronos" in model_name else ""
        c50 = np.mean(all_cov50[model_name]) * 100 if all_cov50.get(model_name) else float("nan")
        c90 = np.mean(all_cov90[model_name]) * 100 if all_cov90.get(model_name) else float("nan")
        n = len(set(all_wis[model_name].keys()) & set(baseline_wis.keys()))
        print(f"  {ratio:.3f}   {c50:5.1f}%   {c90:5.1f}%  {model_name} ({n} obs){marker}")

    # 6. Pairwise relative WIS (Cramer et al., 2022)
    print("\n" + "=" * 90)
    print("PAIRWISE RELATIVE WIS (Cramer et al., 2022)")
    print("=" * 90)

    # For each pair, compute mean WIS ratio on common observations
    theta = {}  # {model: geometric mean of pairwise ratios}
    for m in model_names:
        pairwise_ratios = []
        for mp in model_names:
            if m == mp:
                continue
            common = set(all_wis[m].keys()) & set(all_wis[mp].keys())
            if len(common) < 4:  # need enough common obs
                continue
            m_mean = np.mean([all_wis[m][k] for k in common])
            mp_mean = np.mean([all_wis[mp][k] for k in common])
            if mp_mean > 0:
                pairwise_ratios.append(m_mean / mp_mean)
        if pairwise_ratios:
            theta[m] = np.exp(np.mean(np.log(pairwise_ratios)))  # geometric mean

    # Normalize by baseline
    theta_baseline = theta.get(baseline_name, 1.0)
    pairwise_rwis = {m: v / theta_baseline for m, v in theta.items()}

    print(f"  {'rWIS':>6}  {'Cov50%':>7}  {'Cov90%':>7}  {'Model'}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*40}")
    for model_name, rwis in sorted(pairwise_rwis.items(), key=lambda x: x[1]):
        marker = " ← US" if "Chronos" in model_name else ""
        c50 = np.mean(all_cov50[model_name]) * 100 if all_cov50.get(model_name) else float("nan")
        c90 = np.mean(all_cov90[model_name]) * 100 if all_cov90.get(model_name) else float("nan")
        n_rounds = len(all_wis[model_name])
        print(f"  {rwis:.3f}   {c50:5.1f}%   {c90:5.1f}%  {model_name} ({n_rounds} obs){marker}")

    # 7. Per-horizon breakdown for Chronos vs key models
    print("\n" + "=" * 90)
    print("PER-HORIZON SIMPLE rWIS (model / baseline on common obs)")
    print("=" * 90)
    key_models = list(chronos_names) + ["Influcast-Ensemble", baseline_name]
    # Also find best individual Influcast model
    exclude_set = set(chronos_names) | {baseline_name, "Influcast-Ensemble"}
    best_individual = min(
        ((m, r) for m, r in simple_ratios.items()
         if m not in exclude_set),
        key=lambda x: x[1],
        default=(None, None)
    )
    if best_individual[0]:
        key_models.insert(1, best_individual[0])

    for h in range(1, 5):
        print(f"\n  Horizon {h}wk:")
        for model_name in key_models:
            if model_name not in all_wis:
                continue
            model_scores = {k: v for k, v in all_wis[model_name].items() if k[1] == h}
            bl_scores = {k: v for k, v in baseline_wis.items() if k[1] == h}
            common = set(model_scores.keys()) & set(bl_scores.keys())
            if not common:
                continue
            m_mean = np.mean([model_scores[k] for k in common])
            b_mean = np.mean([bl_scores[k] for k in common])
            ratio = m_mean / b_mean if b_mean > 0 else float("nan")
            marker = " ← US" if "Chronos" in model_name else ""
            print(f"    {ratio:.3f}  {model_name}{marker}")


if __name__ == "__main__":
    main()
