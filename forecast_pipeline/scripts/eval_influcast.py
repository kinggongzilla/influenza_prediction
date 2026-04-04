#!/usr/bin/env python3
"""
Evaluate Chronos-2 on Italian ILI data matching the Influcast paper setup.

Influcast paper (Fiandrino et al., 2025):
  - 2023/24 season: week 2023-42 through 2024-13 (20 forecasting rounds)
  - Each round: predict 1-4 weeks ahead
  - Metric: WIS with 11 symmetric intervals (α = 0.02,0.05,0.10,...,0.90)
  - 23 quantiles submitted per horizon
  - Baseline: random walk with symmetrised increments, 10k trajectories
  - Relative WIS: model WIS / baseline WIS

Best Influcast results (Table 1):
  - Mechanistic-1: rWIS = 0.54
  - Ensemble:      rWIS = 0.57
  - Baseline:      rWIS = 1.00
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "scripts"))
from chronos.chronos2 import Chronos2Pipeline

# ---------------------------------------------------------------------------
# Influcast quantile levels: 23 quantiles matching the paper
# ---------------------------------------------------------------------------
INFLUCAST_QUANTILE_LEVELS = [
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99,
]

# 11 symmetric intervals from the paper (Section 4.3)
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

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "influcast_italy_national.csv")

# Influcast season: 20 forecasting rounds
# Paper says week 2023-46 (mid-November) to 2024-13 (late March)
# The first round used data up to week 2023-42 and predicted weeks 43-46
# We'll replicate: for each round t, we use all data up to week t and predict t+1..t+4
SEASON_START_WEEK = "2023-42"  # first data cutoff
SEASON_END_WEEK = "2024-13"    # last forecasting round


def year_week_to_date(yw: str) -> pd.Timestamp:
    """Convert 'YYYY-WW' to a Monday date."""
    year, week = yw.split("-")
    return pd.Timestamp.fromisocalendar(int(year), int(week), 1)


def load_italy_data() -> tuple[np.ndarray, list[str]]:
    """Load Italian national ILI incidence. Returns (values, year_weeks)."""
    df = pd.read_csv(DATA_FILE)
    # Sort chronologically
    df = df.sort_values("year_week").reset_index(drop=True)
    values = df["incidence"].values.astype(np.float64)
    weeks = df["year_week"].tolist()
    return values, weeks


def compute_wis(actual: float, quantiles: np.ndarray, quantile_levels: list[float]) -> float:
    """WIS with K intervals, matching Influcast paper formula."""
    q_idx = {q: i for i, q in enumerate(quantile_levels)}
    median = quantiles[q_idx[0.5]]
    K = len(INFLUCAST_INTERVALS)
    wis = 0.5 * abs(actual - median)
    for lower_q, upper_q, alpha in INFLUCAST_INTERVALS:
        l = quantiles[q_idx[lower_q]]
        u = quantiles[q_idx[upper_q]]
        is_score = (u - l) + (2.0 / alpha) * max(l - actual, 0.0) + (2.0 / alpha) * max(actual - u, 0.0)
        wis += (alpha / 2.0) * is_score
    return wis / (K + 0.5)


def compute_baseline_quantiles(
    values: np.ndarray, cutoff: int, horizon: int,
    quantile_levels: list[float], n_trajectories: int = 10000,
    rng: np.random.Generator | None = None,
) -> np.ndarray | None:
    """Influcast random walk baseline."""
    last_value = values[cutoff - 1]
    if np.isnan(last_value):
        return None
    # Historical 1-step diffs (use all available history, paper uses calibration period)
    diffs = []
    for t in range(1, cutoff):
        v0, v1 = values[t - 1], values[t]
        if not np.isnan(v0) and not np.isnan(v1):
            diffs.append(v1 - v0)
    if len(diffs) < 4:
        return None
    diffs = np.array(diffs, dtype=np.float64)
    diffs_sym = np.concatenate([diffs, -diffs])
    if rng is None:
        rng = np.random.default_rng(42)
    # Cumulative random walk to horizon h
    samples = rng.choice(diffs_sym, size=(n_trajectories, horizon), replace=True)
    paths = last_value + np.cumsum(samples, axis=1)  # shape: (n_traj, horizon)
    paths = np.maximum(paths, 0.0)
    # Quantiles at each horizon step
    return np.quantile(paths[:, horizon - 1], quantile_levels)


def compute_coverage(actual: float, quantiles: np.ndarray, quantile_levels: list[float],
                     lower_q: float, upper_q: float) -> bool:
    q_idx = {q: i for i, q in enumerate(quantile_levels)}
    return quantiles[q_idx[lower_q]] <= actual <= quantiles[q_idx[upper_q]]


def main():
    parser = argparse.ArgumentParser(description="Influcast-style evaluation on Italian ILI data")
    parser.add_argument("--model_path", type=str, default="amazon/chronos-2")
    parser.add_argument("--use_week_of_year", action="store_true")
    parser.add_argument("--use_hemisphere", action="store_true")
    args = parser.parse_args()

    model_tag = os.path.basename(args.model_path) if "/" in args.model_path else args.model_path
    print(f"Model: {model_tag}")
    print(f"Covariates: week_of_year={'yes' if args.use_week_of_year else 'no'}, hemisphere={'yes' if args.use_hemisphere else 'no'}")

    # Load data
    values, weeks = load_italy_data()
    print(f"Loaded {len(values)} weeks of Italian ILI data ({weeks[0]} to {weeks[-1]})")

    # Find indices for the evaluation season
    week_to_idx = {w: i for i, w in enumerate(weeks)}
    if SEASON_START_WEEK not in week_to_idx:
        print(f"ERROR: {SEASON_START_WEEK} not found in data")
        return
    if SEASON_END_WEEK not in week_to_idx:
        print(f"ERROR: {SEASON_END_WEEK} not found in data")
        return

    start_idx = week_to_idx[SEASON_START_WEEK]
    end_idx = week_to_idx[SEASON_END_WEEK]

    print(f"Evaluation: {SEASON_START_WEEK} to {SEASON_END_WEEK} ({end_idx - start_idx + 1} rounds)")
    print()

    # Load model
    print(f"Loading model...")
    pipeline = Chronos2Pipeline.from_pretrained(args.model_path)
    print("Model loaded.\n")

    data_tensor = torch.tensor(values, dtype=torch.float32)
    rng = np.random.default_rng(42)

    # Per-horizon results
    results_by_horizon = {h: {"wis": [], "baseline_wis": [], "ae": [], "cov50": [], "cov90": []}
                          for h in range(1, 5)}

    total_start = time.time()

    for round_idx, cutoff_idx in enumerate(range(start_idx, end_idx + 1)):
        cutoff_week = weeks[cutoff_idx]
        context = data_tensor[:cutoff_idx]

        # Build covariates if requested
        covariates = {}
        if args.use_week_of_year:
            # Convert year_weeks to week numbers
            week_nums = []
            for w in weeks[:cutoff_idx]:
                wn = int(w.split("-")[1])
                week_nums.append(wn)
            week_nums = np.array(week_nums, dtype=np.float64)
            covariates["week_sin"] = torch.tensor(np.sin(2 * np.pi * week_nums / 52.0), dtype=torch.float32)
            covariates["week_cos"] = torch.tensor(np.cos(2 * np.pi * week_nums / 52.0), dtype=torch.float32)

        if args.use_hemisphere:
            # Italy: ~42°N, clearly northern hemisphere
            covariates["hemisphere"] = torch.full((cutoff_idx,), 1.0, dtype=torch.float32)

        # Build input for Chronos
        if covariates:
            # Convert to numpy for past_covariates
            past_covs = {k: v.numpy() for k, v in covariates.items()}
            input_dict = {"target": context.numpy(), "past_covariates": past_covs}
            # Future covariates for week_sin/cos (we know future weeks)
            future_covs = {}
            if args.use_week_of_year:
                future_weeks = []
                for fh in range(1, 5):
                    fw = weeks[cutoff_idx + fh] if cutoff_idx + fh < len(weeks) else weeks[-1]
                    future_weeks.append(int(fw.split("-")[1]))
                future_weeks = np.array(future_weeks, dtype=np.float64)
                future_covs["week_sin"] = np.sin(2 * np.pi * future_weeks / 52.0).astype(np.float32)
                future_covs["week_cos"] = np.cos(2 * np.pi * future_weeks / 52.0).astype(np.float32)
            if args.use_hemisphere:
                future_covs["hemisphere"] = np.full(4, 1.0, dtype=np.float32)
            if future_covs:
                input_dict["future_covariates"] = future_covs
            inputs = [input_dict]
        else:
            inputs = context.unsqueeze(0).unsqueeze(0)

        # Predict quantiles at each horizon
        try:
            quantile_forecasts, mean_forecasts = pipeline.predict_quantiles(
                inputs=inputs,
                prediction_length=4,
                quantile_levels=INFLUCAST_QUANTILE_LEVELS,
            )
        except Exception as e:
            print(f"  Round {round_idx+1} ({cutoff_week}): prediction failed ({e})")
            continue

        # quantile_forecasts shape: (1, 1, 4, n_quantiles)
        # mean_forecasts shape: (1, 1, 4)
        qf = quantile_forecasts[0][0].cpu().numpy()  # (4, n_quantiles)
        mf = mean_forecasts[0][0].cpu().numpy()       # (4,)

        # Evaluate each horizon
        for h in range(1, 5):
            target_idx = cutoff_idx + h
            if target_idx >= len(values):
                continue
            actual = values[target_idx]
            if np.isnan(actual):
                continue

            # Model quantiles at horizon h (already computed)
            model_q = qf[h - 1]  # shape: (n_quantiles,)

            # Baseline quantiles
            baseline_q = compute_baseline_quantiles(
                values, cutoff_idx, h, INFLUCAST_QUANTILE_LEVELS, rng=rng
            )

            # WIS
            model_wis = compute_wis(actual, model_q, INFLUCAST_QUANTILE_LEVELS)
            results_by_horizon[h]["wis"].append(model_wis)

            if baseline_q is not None:
                bl_wis = compute_wis(actual, baseline_q, INFLUCAST_QUANTILE_LEVELS)
                results_by_horizon[h]["baseline_wis"].append(bl_wis)

            # Absolute error (median)
            q_idx = {q: i for i, q in enumerate(INFLUCAST_QUANTILE_LEVELS)}
            median_pred = model_q[q_idx[0.5]]
            results_by_horizon[h]["ae"].append(abs(actual - median_pred))

            # Coverage
            results_by_horizon[h]["cov50"].append(
                compute_coverage(actual, model_q, INFLUCAST_QUANTILE_LEVELS, 0.25, 0.75))
            results_by_horizon[h]["cov90"].append(
                compute_coverage(actual, model_q, INFLUCAST_QUANTILE_LEVELS, 0.05, 0.95))

        # Print round summary
        h1_actual = values[cutoff_idx + 1] if cutoff_idx + 1 < len(values) else float("nan")
        h1_median = mf[0]
        print(f"  Round {round_idx+1:2d} ({cutoff_week}): actual(+1wk)={h1_actual:.2f} pred={h1_median:.2f}")

    elapsed = time.time() - total_start
    print(f"\nDone in {elapsed:.1f}s\n")

    # Summary by horizon
    print("=" * 80)
    print(f"INFLUCAST-STYLE EVALUATION — {model_tag}")
    print("=" * 80)
    print(f"{'Horizon':>8} {'N':>4} {'Mean WIS':>10} {'Mean BL WIS':>12} {'rWIS':>8} {'MAE':>8} {'Cov50%':>8} {'Cov90%':>8}")
    print("-" * 80)

    all_wis = []
    all_bl_wis = []
    all_ae = []
    all_cov50 = []
    all_cov90 = []

    for h in range(1, 5):
        r = results_by_horizon[h]
        n = len(r["wis"])
        mean_wis = np.mean(r["wis"]) if r["wis"] else float("nan")
        mean_bl = np.mean(r["baseline_wis"]) if r["baseline_wis"] else float("nan")
        rwis = mean_wis / mean_bl if mean_bl > 0 else float("nan")
        mae = np.mean(r["ae"]) if r["ae"] else float("nan")
        c50 = np.mean(r["cov50"]) * 100 if r["cov50"] else float("nan")
        c90 = np.mean(r["cov90"]) * 100 if r["cov90"] else float("nan")
        print(f"  {h}wk    {n:4d}   {mean_wis:10.3f} {mean_bl:12.3f} {rwis:8.3f} {mae:8.3f} {c50:7.1f}% {c90:7.1f}%")
        all_wis.extend(r["wis"])
        all_bl_wis.extend(r["baseline_wis"])
        all_ae.extend(r["ae"])
        all_cov50.extend(r["cov50"])
        all_cov90.extend(r["cov90"])

    print("-" * 80)
    overall_rwis = np.mean(all_wis) / np.mean(all_bl_wis) if all_bl_wis else float("nan")
    print(f"  ALL    {len(all_wis):4d}   {np.mean(all_wis):10.3f} {np.mean(all_bl_wis):12.3f} {overall_rwis:8.3f} {np.mean(all_ae):8.3f} {np.mean(all_cov50)*100:7.1f}% {np.mean(all_cov90)*100:7.1f}%")
    print()
    print("Influcast paper reference (Table 1):")
    print("  Mechanistic-1: rWIS=0.54, Cov50=75%, Cov90=89%")
    print("  Ensemble:      rWIS=0.57, Cov50=51%, Cov90=80%")
    print("  Baseline:      rWIS=1.00")


if __name__ == "__main__":
    main()
