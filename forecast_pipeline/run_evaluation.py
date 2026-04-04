#!/usr/bin/env python3
"""
Rolling evaluation script for Chronos-2 forecasts.

Metrics (FluSight-comparable):
  - WIS (Weighted Interval Score): primary probabilistic metric
  - Relative WIS: WIS / naive-baseline WIS (< 1 = better than baseline)
  - Coverage 50% / 95%: fraction of actuals inside prediction intervals
  - MAE: median absolute error of the point forecast
  - MAPE: mean absolute percentage error (retained for continuity)

Improvements over naive MAPE:
  1. Per-country off-season filter (20th pct of non-zero values) so near-zero weeks
     don't inflate MAPE via tiny denominators.
  2. Phase labels per eval point (growth / peak_window / decay / off_season) so we
     can report where the model actually fails.
  3. Peak timing error — per prediction window, compare argmax(forecast) vs
     argmax(actuals) to measure phase-shift errors in weeks.

Experiments:
  --context_length N       Experiment E: limit context to last N weeks
  --weather_mode MODE      Experiment C: full / past_only / none
  --use_neighbors          Experiment D: neighboring countries ILI as past covariates
  --use_capital_coords     Experiment A: capital city coords for weather
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "scripts"))

from chronos.chronos2 import Chronos2Pipeline
from data_loader import load_time_series_data, load_time_series_with_weather
from country_neighbors import get_neighbors

EXTRACTED_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "extracted_data")
COUNTRY_SUMMARY_FILE = os.path.join(os.path.dirname(__file__), "country_data_summary.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "evaluation")

DEFAULT_COUNTRIES = "Argentina,Austria,Bulgaria,Chile,Italy,Germany,Belgium"

# Quantile levels used for WIS computation (FluSight-style symmetric pairs + median)
# Pairs: (0.025, 0.975) → 95% interval, (0.1, 0.9) → 80% interval, (0.25, 0.75) → 50% interval
WIS_QUANTILE_LEVELS = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
# Symmetric interval definitions: (lower_q, upper_q, alpha)
# alpha = 1 - coverage, e.g. 50% interval has alpha=0.5
WIS_INTERVALS = [
    (0.025, 0.975, 0.05),   # 95% interval
    (0.1,   0.9,   0.2),    # 80% interval
    (0.25,  0.75,  0.5),    # 50% interval
]


def log(msg: str):
    print(msg, flush=True)


def find_extracted_file(country_name: str) -> str | None:
    safe_name = country_name.replace(" ", "_")
    for suffix in ("_ili.csv", "_ari.csv"):
        path = os.path.join(EXTRACTED_DATA_DIR, safe_name + suffix)
        if os.path.exists(path):
            return path
    return None


def suppress_stdout():
    return contextlib.redirect_stdout(io.StringIO())


# ---------------------------------------------------------------------------
# WIS (Weighted Interval Score) — FluSight-style probabilistic metric
# ---------------------------------------------------------------------------

def compute_wis(actual: float, quantiles: np.ndarray, quantile_levels: list[float]) -> float:
    """
    Weighted Interval Score (WIS) for a single observation.

    WIS = (1 / (K + 0.5)) * [0.5 * |y - q50| + Σ_k (α_k/2 * IS(l_k, u_k, y))]

    where IS(l, u, y) = (u - l)  +  (2/α) * max(l - y, 0)  +  (2/α) * max(y - u, 0)

    A lower WIS is better. WIS rewards both sharpness and calibration.
    """
    q_idx = {q: i for i, q in enumerate(quantile_levels)}
    median = quantiles[q_idx[0.5]]

    K = len(WIS_INTERVALS)
    wis = 0.5 * abs(actual - median)  # absolute error of median (dispersion-free term)

    for lower_q, upper_q, alpha in WIS_INTERVALS:
        l = quantiles[q_idx[lower_q]]
        u = quantiles[q_idx[upper_q]]
        # Interval score: width + penalty for observations outside the interval
        is_score = (u - l) + (2.0 / alpha) * max(l - actual, 0.0) + (2.0 / alpha) * max(actual - u, 0.0)
        wis += (alpha / 2.0) * is_score

    return wis / (K + 0.5)


def compute_naive_baseline_quantiles(
    data: torch.Tensor,
    cutoff: int,
    horizon: int,
    quantile_levels: list[float],
    history_weeks: int = 104,
    n_trajectories: int = 10000,
    _rng: np.random.Generator | None = None,
) -> np.ndarray | None:
    """
    Influcast-paper baseline (Fiandrino et al., 2025):
      - Median = last known value (vₜ)
      - 1-step differences from history: δ = {d₂, ..., dₜ}
      - Symmetrised increment pool: δ' = δ ∪ (−δ)
      - For each of n_trajectories random walks:
          fₕ = vₜ + Σⁿ⁼¹ʰ sample(δ')   (cumulative random walk)
      - Quantiles taken across trajectories at step h=horizon

    Returns array of shape (len(quantile_levels),) or None if insufficient history.
    """
    values = data.numpy()
    last_value = values[cutoff - 1]
    if np.isnan(last_value):
        return None

    # Collect historical 1-step differences (non-NaN consecutive pairs)
    hist_start = max(1, cutoff - history_weeks)
    diffs = []
    for t in range(hist_start, cutoff):
        v0 = values[t - 1]
        v1 = values[t]
        if not np.isnan(v0) and not np.isnan(v1):
            diffs.append(v1 - v0)

    if len(diffs) < 4:
        return None

    diffs = np.array(diffs, dtype=np.float64)
    diffs_sym = np.concatenate([diffs, -diffs])   # symmetrised pool

    rng = _rng if _rng is not None else np.random.default_rng()
    # Shape: (n_trajectories, horizon) — cumulative sum gives path at each step
    samples = rng.choice(diffs_sym, size=(n_trajectories, horizon), replace=True)
    endpoint = last_value + np.sum(samples, axis=1)   # value at horizon h
    endpoint = np.maximum(endpoint, 0.0)              # no negative incidence

    baseline_quantiles = np.quantile(endpoint, quantile_levels)
    return baseline_quantiles


# ---------------------------------------------------------------------------
# Phase analysis helpers
# ---------------------------------------------------------------------------

def compute_season_threshold(data: torch.Tensor, percentile: float = 20.0) -> float:
    """20th percentile of non-zero, non-NaN values — used as off-season filter."""
    values = data.numpy()
    nonzero = values[(values > 0) & ~np.isnan(values)]
    if len(nonzero) == 0:
        return 1.0
    return float(np.percentile(nonzero, percentile))


def compute_phase_labels(data: torch.Tensor, threshold: float,
                          peak_window_weeks: int = 2) -> list[str]:
    """
    Assign a phase label to every timestep:
      off_season  — below threshold (too few cases to matter clinically)
      growth      — above threshold, rising toward a seasonal peak
      peak_window — within peak_window_weeks of a seasonal peak
      decay       — above threshold, falling after a seasonal peak
    """
    values = data.numpy().copy()
    n = len(values)
    labels = ["off_season"] * n

    valid = np.where(~np.isnan(values), values, 0.0)

    # Find seasonal peaks: at least 16 weeks apart, above threshold with some prominence
    peaks, _ = find_peaks(
        valid,
        height=threshold,
        distance=16,
        prominence=threshold * 0.3,
    )

    if len(peaks) == 0:
        for t in range(n):
            if not np.isnan(values[t]) and values[t] >= threshold:
                labels[t] = "growth"
        return labels

    # Mark peak windows first
    for p in peaks:
        for offset in range(-peak_window_weeks, peak_window_weeks + 1):
            idx = p + offset
            if 0 <= idx < n and not np.isnan(values[idx]) and values[idx] >= threshold:
                labels[idx] = "peak_window"

    # Growth vs decay for all remaining above-threshold points
    for t in range(n):
        if labels[t] != "off_season":
            continue
        if np.isnan(values[t]) or values[t] < threshold:
            continue

        before = peaks[peaks < t]
        after = peaks[peaks > t]

        if len(after) > 0 and (len(before) == 0 or (after[0] - t) <= (t - before[-1])):
            labels[t] = "growth"
        elif len(before) > 0:
            labels[t] = "decay"
        else:
            labels[t] = "growth"

    return labels


# ---------------------------------------------------------------------------
# Week-of-year covariates (sin/cos encoded — always known past AND future)
# ---------------------------------------------------------------------------

def compute_week_of_year_covariates(dates: list) -> dict[str, torch.Tensor]:
    """
    Returns sin and cos encoding of week-of-year for every date.
    Using sin/cos so the transition week52→week1 is smooth.
    Since week-of-year is deterministic, these can be used as both
    past_covariates and future_covariates.
    Names match prepare_finetune_data.py: week_sin, week_cos.
    """
    weeks = []
    for d in dates:
        try:
            if hasattr(d, "isocalendar"):
                w = d.isocalendar()[1]
            else:
                w = pd.to_datetime(str(d)).isocalendar()[1]
        except Exception:
            w = 1
        weeks.append(w)

    weeks_arr = np.array(weeks, dtype=np.float32)
    sin_enc = torch.tensor(np.sin(2 * np.pi * weeks_arr / 52.0), dtype=torch.float32)
    cos_enc = torch.tensor(np.cos(2 * np.pi * weeks_arr / 52.0), dtype=torch.float32)
    return {"week_sin": sin_enc, "week_cos": cos_enc}


WEATHER_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "weather_cache")


def _find_best_weather_cache(lat: float, lon: float, start_date: str, end_date: str):
    """Find a cache parquet whose date range covers [start_date, end_date]."""
    import re
    prefix = f"{lat:.4f}_{lon:.4f}_"
    cache_dir = Path(WEATHER_CACHE_DIR)
    if not cache_dir.exists():
        return None
    exact = cache_dir / f"{prefix}{start_date}_{end_date}.parquet"
    if exact.exists():
        return exact
    for candidate in sorted(cache_dir.glob(f"{prefix}*.parquet")):
        try:
            dates_in_name = re.findall(r'\d{4}-\d{2}-\d{2}', candidate.stem)
            if len(dates_in_name) == 2:
                if dates_in_name[0] <= start_date and dates_in_name[1] >= end_date:
                    return candidate
        except Exception:
            continue
    return None


def load_weather_from_cache(country_name: str, dates: list, normalize: bool = True) -> dict[str, torch.Tensor]:
    """Load weather covariates from cache only (no API calls). Returns dict of tensors or empty dict."""
    from weather_fetcher import get_country_coordinates, aggregate_weather_to_weekly, normalize_weather_features
    name_map = {
        "Côte d'Ivoire": "Ivory Coast",
        "Kosovo (in accordance with UN Security Council resolution 1244 (1999))": "Kosovo",
        "Micronesia (Federated States of)": "Micronesia",
        "Netherlands (Kingdom of the)": "Netherlands",
        "Lao People's Democratic Republic": "Laos",
        "United States of America": "United States",
        "Russian Federation": "Russia",
        "Türkiye": "Turkey",
        "Viet Nam": "Vietnam",
    }
    mapped = name_map.get(country_name, country_name)
    try:
        lat, lon = get_country_coordinates(mapped)
    except Exception:
        return {}

    dates_pd = pd.to_datetime([str(d) for d in dates])
    start_date = dates_pd.min().strftime("%Y-%m-%d")
    end_date = dates_pd.max().strftime("%Y-%m-%d")

    cache_file = _find_best_weather_cache(lat, lon, start_date, end_date)
    if cache_file is None:
        return {}

    try:
        daily_df = pd.read_parquet(cache_file)
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        daily_df = daily_df[
            (daily_df["date"] >= start_date) & (daily_df["date"] <= end_date)
        ].reset_index(drop=True)

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
                result[col] = torch.tensor(vals, dtype=torch.float32)
        return result
    except Exception:
        return {}


def compute_hemisphere_covariate(country_name: str, dates: list) -> dict[str, torch.Tensor]:
    """
    Returns a constant hemisphere channel (+1 N, -1 S, 0 tropical).
    Matches the hemisphere covariate used in prepare_finetune_data.py.
    """
    try:
        from weather_fetcher import get_country_coordinates
        # Apply same name mapping as prepare_finetune_data.py
        name_map = {
            "Côte d'Ivoire": "Ivory Coast",
            "Kosovo (in accordance with UN Security Council resolution 1244 (1999))": "Kosovo",
            "Micronesia (Federated States of)": "Micronesia",
            "Netherlands (Kingdom of the)": "Netherlands",
            "Lao People's Democratic Republic": "Laos",
            "United States of America": "United States",
            "Russian Federation": "Russia",
            "Türkiye": "Turkey",
            "Viet Nam": "Vietnam",
        }
        mapped = name_map.get(country_name, country_name)
        lat, _ = get_country_coordinates(mapped)
    except Exception:
        lat = 0.0
    if lat > 23.5:
        hemi = 1.0
    elif lat < -23.5:
        hemi = -1.0
    else:
        hemi = 0.0
    vals = torch.full((len(dates),), hemi, dtype=torch.float32)
    return {"hemisphere": vals}


# ---------------------------------------------------------------------------
# Neighbor covariates
# ---------------------------------------------------------------------------

def load_neighbor_covariates(country_name: str, target_dates: list) -> dict[str, torch.Tensor]:
    neighbors = get_neighbors(country_name)
    covariates = {}
    for neighbor in neighbors:
        extracted = find_extracted_file(neighbor)
        if not extracted:
            continue
        try:
            with suppress_stdout():
                nb_data, nb_dates, _ = load_time_series_data(
                    extracted, context_length=None, min_value=1e-6
                )
            if nb_dates is None:
                continue
            nb_lookup = {d: v for d, v in zip(nb_dates, nb_data.tolist())}
            aligned = torch.tensor(
                [nb_lookup.get(d, float("nan")) for d in target_dates],
                dtype=torch.float32,
            )
            coverage = (~torch.isnan(aligned)).float().mean().item()
            if coverage >= 0.5:
                covariates[f"neighbor_{neighbor.replace(' ', '_')}_ili"] = aligned
        except Exception:
            continue
    return covariates


# ---------------------------------------------------------------------------
# Core rolling evaluation
# ---------------------------------------------------------------------------

def evaluate_country(
    model: Chronos2Pipeline,
    data: torch.Tensor,
    dates: list,
    eval_window: int,
    prediction_horizon: int,
    weather_covariates: dict | None = None,
    neighbor_covariates: dict | None = None,
    week_of_year_covariates: dict | None = None,
    weather_mode: str = "full",
    context_length: int | None = None,
    min_context: int = 10,
    season_threshold: float | None = None,
    eval_start_date: str | None = None,
    eval_end_date: str | None = None,
) -> dict:
    """
    Rolling evaluation with phase-labelled errors and peak timing analysis.

    season_threshold: off-season filter level; if None, auto-computed as the
                      20th percentile of non-zero values for this country.
    """
    n = len(data)

    # ---- 1. Per-country off-season threshold --------------------------------
    if season_threshold is None:
        season_threshold = compute_season_threshold(data)

    # ---- 2. Phase labels for every timestep ---------------------------------
    phase_labels = compute_phase_labels(data, season_threshold)

    eval_start = max(min_context + prediction_horizon, n - eval_window)
    has_weather = weather_covariates is not None and weather_mode != "none"

    # Parse date filters (applied to target date, i.e. the point being predicted)
    _eval_start_date = pd.Timestamp(eval_start_date) if eval_start_date else None
    _eval_end_date   = pd.Timestamp(eval_end_date)   if eval_end_date   else None
    has_neighbors = neighbor_covariates is not None and len(neighbor_covariates) > 0
    has_woy = week_of_year_covariates is not None and len(week_of_year_covariates) > 0

    eval_dates, actuals, errors, phases = [], [], [], []
    predictions: list[float] = []
    pred_q10: list[float] = []
    pred_q90: list[float] = []
    peak_timing_errors: list[int] = []   # signed: positive = predicted peak too late

    # FluSight-style probabilistic metrics
    wis_scores: list[float] = []
    wis_baseline_scores: list[float] = []
    mae_scores: list[float] = []
    covered_50: list[bool] = []    # 50% interval (q25-q75)
    covered_95: list[bool] = []    # 95% interval (q025-q975)
    _rng = np.random.default_rng(42)  # shared RNG for baseline trajectories

    for t in range(eval_start, n):
        cutoff = t - prediction_horizon
        if cutoff < min_context:
            continue

        # Date-range filter on the target date
        if (_eval_start_date is not None or _eval_end_date is not None) and dates:
            target_date = pd.Timestamp(str(dates[t]))
            if _eval_start_date is not None and target_date < _eval_start_date:
                continue
            if _eval_end_date is not None and target_date > _eval_end_date:
                continue

        actual = data[t].item()
        if actual < season_threshold or np.isnan(actual):
            continue
        if torch.isnan(data[cutoff - 1]):
            continue

        phase = phase_labels[t]

        ctx_start = max(0, cutoff - context_length) if context_length is not None else 0
        context = data[ctx_start:cutoff]

        with torch.no_grad():
            all_past_covs: dict = {}
            all_future_covs: dict = {}

            if has_weather:
                for name, tensor in weather_covariates.items():
                    all_past_covs[name] = tensor[ctx_start:cutoff].numpy()
                    if weather_mode == "full":
                        all_future_covs[name] = tensor[cutoff:cutoff + prediction_horizon].numpy()

            if has_neighbors:
                for name, tensor in neighbor_covariates.items():
                    past_slice = tensor[ctx_start:cutoff].numpy()
                    if not np.isnan(past_slice[-1]):
                        all_past_covs[name] = past_slice

            if has_woy:
                for name, tensor in week_of_year_covariates.items():
                    all_past_covs[name] = tensor[ctx_start:cutoff].numpy()
                    # week-of-year is always known, so always add as future cov
                    all_future_covs[name] = tensor[cutoff:cutoff + prediction_horizon].numpy()

            if all_past_covs:
                use_future = all_future_covs and all(
                    len(v) >= prediction_horizon and not np.any(np.isnan(v))
                    for v in all_future_covs.values()
                )
                input_dict = {"target": context.numpy(), "past_covariates": all_past_covs}
                if use_future:
                    input_dict["future_covariates"] = all_future_covs
                quantile_forecasts, mean_forecasts = model.predict_quantiles(
                    inputs=[input_dict],
                    prediction_length=prediction_horizon,
                    quantile_levels=WIS_QUANTILE_LEVELS,
                )
            else:
                quantile_forecasts, mean_forecasts = model.predict_quantiles(
                    inputs=context.unsqueeze(0).unsqueeze(0),
                    prediction_length=prediction_horizon,
                    quantile_levels=WIS_QUANTILE_LEVELS,
                )

        # Full forecast for peak timing; final step for MAPE
        full_forecast = mean_forecasts[0][0, :].cpu().numpy()  # shape: (prediction_horizon,)
        predicted = full_forecast[-1]
        if np.isnan(predicted):
            continue

        # All quantiles at the final prediction step — shape: (len(WIS_QUANTILE_LEVELS),)
        q_step = quantile_forecasts[0][0, -1, :].cpu().numpy()

        errors.append(abs(predicted - actual) / actual)
        predictions.append(float(predicted))
        # q10 = WIS_QUANTILE_LEVELS index 1 (0.1), q90 = index 5 (0.9)
        pred_q10.append(float(q_step[1]))
        pred_q90.append(float(q_step[5]))
        actuals.append(actual)
        eval_dates.append(dates[t] if dates else t)
        phases.append(phase)

        # ---- FluSight probabilistic metrics ------------------------------------
        wis = compute_wis(actual, q_step, WIS_QUANTILE_LEVELS)
        wis_scores.append(wis)
        mae_scores.append(abs(predicted - actual))

        # 50% coverage: q0.25 (idx 2) to q0.75 (idx 4)
        covered_50.append(bool(q_step[2] <= actual <= q_step[4]))
        # 95% coverage: q0.025 (idx 0) to q0.975 (idx 6)
        covered_95.append(bool(q_step[0] <= actual <= q_step[6]))

        # Naive baseline WIS (Influcast-style random walk)
        baseline_q = compute_naive_baseline_quantiles(data, cutoff, prediction_horizon, WIS_QUANTILE_LEVELS, _rng=_rng)
        if baseline_q is not None:
            wis_baseline_scores.append(compute_wis(actual, baseline_q, WIS_QUANTILE_LEVELS))

        # ---- 3. Peak timing error -------------------------------------------
        # Does the actual prediction window contain a meaningful peak?
        actual_window = data[cutoff:cutoff + prediction_horizon].numpy()
        if len(actual_window) == prediction_horizon and not np.any(np.isnan(actual_window)):
            actual_peak_idx = int(np.argmax(actual_window))
            predicted_peak_idx = int(np.argmax(full_forecast))
            # Only count if the actual peak in the window is significant
            if actual_window[actual_peak_idx] >= season_threshold * 1.5:
                peak_timing_errors.append(predicted_peak_idx - actual_peak_idx)

    # ---- Aggregate results --------------------------------------------------
    if not errors:
        return {
            "mape_pct": None,
            "median_error_pct": None,
            "mae": None,
            "wis": None,
            "relative_wis": None,
            "coverage_50": None,
            "coverage_95": None,
            "n_eval_points": 0,
            "season_threshold": round(season_threshold, 2),
            "used_weather": has_weather,
            "used_neighbors": has_neighbors,
            "phase_mape": {},
            "peak_timing_mae_weeks": None,
            "peak_timing_bias_weeks": None,
            "n_peak_timing_points": 0,
            "eval_dates": [],
            "actuals": [],
            "predictions": [],
            "pred_q10": [],
            "pred_q90": [],
            "errors_pct": [],
            "phases": [],
        }

    phase_mape = {}
    for ph in ("growth", "peak_window", "decay"):
        ph_errs = [e for e, p in zip(errors, phases) if p == ph]
        phase_mape[ph] = {
            "mape_pct": round(float(np.mean(ph_errs) * 100), 2) if ph_errs else None,
            "n_points": len(ph_errs),
        }

    mean_wis = float(np.mean(wis_scores)) if wis_scores else None
    mean_baseline_wis = float(np.mean(wis_baseline_scores)) if wis_baseline_scores else None
    relative_wis = round(mean_wis / mean_baseline_wis, 3) if (mean_wis and mean_baseline_wis) else None

    return {
        "mape_pct": round(float(np.mean(errors) * 100), 2),
        "median_error_pct": round(float(np.median(errors) * 100), 2),
        "mae": round(float(np.mean(mae_scores)), 4) if mae_scores else None,
        "wis": round(mean_wis, 4) if mean_wis is not None else None,
        "relative_wis": relative_wis,
        "coverage_50": round(float(np.mean(covered_50)) * 100, 1) if covered_50 else None,
        "coverage_95": round(float(np.mean(covered_95)) * 100, 1) if covered_95 else None,
        "n_eval_points": len(errors),
        "season_threshold": round(season_threshold, 2),
        "used_weather": has_weather,
        "used_neighbors": has_neighbors,
        "phase_mape": phase_mape,
        "peak_timing_mae_weeks": round(float(np.mean(np.abs(peak_timing_errors))), 2) if peak_timing_errors else None,
        "peak_timing_bias_weeks": round(float(np.mean(peak_timing_errors)), 2) if peak_timing_errors else None,
        "n_peak_timing_points": len(peak_timing_errors),
        "eval_dates": eval_dates,
        "actuals": actuals,
        "predictions": predictions,
        "pred_q10": pred_q10,
        "pred_q90": pred_q90,
        "errors_pct": [round(e * 100, 2) for e in errors],
        "phases": phases,
    }


# ---------------------------------------------------------------------------
# Rolling eval plot — same style as production plots, stitched rolling preds
# ---------------------------------------------------------------------------

def plot_rolling_eval(
    country_name: str,
    data: torch.Tensor,
    dates: list,
    eval_result: dict,
    eval_window: int,
    prediction_horizon: int,
    season_threshold: float,
    output_dir: str,
) -> str | None:
    """
    Same visual style as the production forecast plot (blue context, red dashed
    mean, orange actuals, green quantile band), but instead of a single
    N-week shot the red line is stitched from individual prediction_horizon-
    week-ahead predictions made at every timestep in the eval window.
    """
    import matplotlib.pyplot as plt

    if not eval_result.get("eval_dates"):
        return None

    n = len(data)
    values = data.numpy()
    all_dates = dates if dates else list(range(n))
    split_idx = n - eval_window

    eval_dates = eval_result["eval_dates"]
    preds      = eval_result["predictions"]
    q10s       = eval_result["pred_q10"]
    q90s       = eval_result["pred_q90"]

    plt.figure(figsize=(12, 6))

    # Context (blue)
    plt.plot(all_dates[:split_idx], values[:split_idx],
             color="blue", lw=1.2, label="Context")

    # Actual test data (orange)
    plt.plot(all_dates[split_idx:], values[split_idx:],
             color="orange", lw=1.5, label="Actual Test Data")

    # Rolling mean prediction (red dashed)
    plt.plot(eval_dates, preds,
             color="red", lw=1.5, ls="--", label="Mean Forecast (rolling)")

    # Quantile band (green)
    plt.plot(eval_dates, q10s, color="green", lw=0.8, ls="--", label="Quantile 0.1")
    plt.plot(eval_dates, q90s, color="green", lw=0.8, ls="--", label="Quantile 0.9")
    plt.fill_between(eval_dates, q10s, q90s, color="green", alpha=0.15,
                     label="10%-90% Quantile Range")

    # X-axis ticks
    try:
        n_ticks = 12
        tick_step = max(1, len(all_dates) // n_ticks)
        tick_positions = all_dates[::tick_step]
        plt.xticks(tick_positions, [str(d)[:10] for d in tick_positions],
                   rotation=45, ha="right", fontsize=8)
    except Exception:
        pass

    safe_name = country_name.replace(" ", "_").replace(",", "")
    plt.title(
        f"Chronos-2 — {prediction_horizon}-week-ahead rolling evaluation "
        f"({eval_window}-week window) | data/extracted_data/{safe_name}_ili.csv"
    )
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{safe_name}_chronos-2_forecast_results_eval.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rolling evaluation of Chronos-2 forecasts")
    parser.add_argument("--eval_window", type=int, default=52)
    parser.add_argument("--prediction_horizon", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--country", type=str, default=None)
    parser.add_argument("--countries", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    # Experiment flags
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--weather_mode", choices=["full", "past_only", "none"], default="none")
    parser.add_argument("--use_neighbors", action="store_true")
    parser.add_argument("--use_capital_coords", action="store_true")
    parser.add_argument("--use_week_of_year", action="store_true",
                        help="Add sin/cos week-of-year as past+future covariates (matches training)")
    parser.add_argument("--use_hemisphere", action="store_true",
                        help="Add constant hemisphere covariate (+1 N/-1 S/0 tropical, matches training)")
    parser.add_argument("--no_weather", action="store_true", help="Alias for --weather_mode none")
    parser.add_argument("--weather_cache_only", action="store_true",
                        help="Load weather from cache only (no API calls)")
    # Date range filter (applied to target dates, i.e. the weeks being predicted)
    parser.add_argument("--eval_start_date", type=str, default=None,
                        help="Only evaluate targets on or after this date (YYYY-MM-DD)")
    parser.add_argument("--eval_end_date", type=str, default=None,
                        help="Only evaluate targets on or before this date (YYYY-MM-DD)")
    # Phase filter
    parser.add_argument("--season_threshold_pct", type=float, default=20.0,
                        help="Percentile of non-zero values used as off-season filter (default: 20)")
    parser.add_argument("--model_path", type=str, default="amazon/chronos-2",
                        help="HuggingFace model ID or path to fine-tuned checkpoint directory")
    args = parser.parse_args()

    if args.no_weather:
        args.weather_mode = "none"

    summary = pd.read_csv(COUNTRY_SUMMARY_FILE)
    all_countries = summary[
        (summary["total_ili_cases"] > 0) | (summary["total_ari_cases"] > 0)
    ]["country"].tolist()

    if args.country:
        countries = [args.country]
    elif args.countries:
        countries = [c.strip() for c in args.countries.split(",")]
        missing = [c for c in countries if c not in all_countries]
        if missing:
            log(f"Warning: countries not found in summary: {missing}")
            countries = [c for c in countries if c in all_countries]
    else:
        countries = [c.strip() for c in DEFAULT_COUNTRIES.split(",")]

    if args.limit:
        countries = countries[: args.limit]

    date_range_str = ""
    if args.eval_start_date or args.eval_end_date:
        date_range_str = f" | dates={args.eval_start_date or '?'}→{args.eval_end_date or '?'}"
    model_tag = Path(args.model_path).name if args.model_path != "amazon/chronos-2" else "chronos-2"
    log(
        f"Evaluating {len(countries)} countries | "
        f"window={args.eval_window}wk | horizon={args.prediction_horizon}wk | "
        f"weather={args.weather_mode} | "
        f"neighbors={'yes' if args.use_neighbors else 'no'} | "
        f"capital_coords={'yes' if args.use_capital_coords else 'no'} | "
        f"week_of_year={'yes' if args.use_week_of_year else 'no'} | "
        f"hemisphere={'yes' if args.use_hemisphere else 'no'} | "
        f"context={'all' if args.context_length is None else f'{args.context_length}wk'} | "
        f"threshold=p{args.season_threshold_pct:.0f}"
        f"{date_range_str} | model={model_tag}"
    )

    model_label = args.model_path if args.model_path != "amazon/chronos-2" else "Chronos-2"
    log(f"Loading model: {model_label}")
    model = Chronos2Pipeline.from_pretrained(args.model_path)
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        model.model = model.model.to("cuda")
        log(f"Model on CUDA: {torch.cuda.get_device_name()}")
    else:
        log("Model on CPU")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plots_dir = os.path.join(RESULTS_DIR, "plots")

    results = {
        "eval_config": {
            "eval_window": args.eval_window,
            "prediction_horizon": args.prediction_horizon,
            "weather_mode": args.weather_mode,
            "use_neighbors": args.use_neighbors,
            "use_capital_coords": args.use_capital_coords,
            "use_week_of_year": args.use_week_of_year,
            "context_length": args.context_length,
            "season_threshold_pct": args.season_threshold_pct,
            "countries": countries,
        },
        "countries": {},
    }

    total_start = time.time()
    weather_success = 0
    weather_fallback = 0

    for i, country_name in enumerate(countries, 1):
        extracted_file = find_extracted_file(country_name)
        if not extracted_file:
            log(f"[{i}/{len(countries)}] {country_name}: no extracted data, skipping")
            continue

        data = None
        weather_covariates = None
        dates = None

        if args.weather_mode != "none":
            if args.weather_cache_only:
                # Load ILI data first, then weather from cache separately
                try:
                    with suppress_stdout():
                        data, dates, _ = load_time_series_data(
                            extracted_file, context_length=None, min_value=1e-6
                        )
                    if dates is not None:
                        weather_covariates = load_weather_from_cache(country_name, dates)
                    if weather_covariates:
                        weather_success += 1
                    else:
                        weather_fallback += 1
                except Exception:
                    weather_fallback += 1
                    weather_covariates = None
            else:
                try:
                    with suppress_stdout():
                        data, dates, _, weather_covariates = load_time_series_with_weather(
                            file_path=extracted_file,
                            country_name=country_name,
                            context_length=None,
                            min_value=1e-6,
                            normalize_weather=True,
                            use_capital_coords=args.use_capital_coords,
                        )
                    if weather_covariates:
                        weather_success += 1
                    else:
                        weather_fallback += 1
                except Exception:
                    weather_fallback += 1
                    weather_covariates = None

        if data is None:
            try:
                with suppress_stdout():
                    data, dates, _ = load_time_series_data(
                        extracted_file, context_length=None, min_value=1e-6
                    )
            except Exception as e:
                log(f"[{i}/{len(countries)}] {country_name}: failed to load data ({e})")
                continue

        if len(data) < args.prediction_horizon + 10:
            log(f"[{i}/{len(countries)}] {country_name}: too few data points, skipping")
            continue

        neighbor_covariates = None
        if args.use_neighbors and dates is not None:
            neighbor_covariates = load_neighbor_covariates(country_name, dates)

        week_of_year_covariates = None
        if args.use_week_of_year and dates is not None:
            week_of_year_covariates = compute_week_of_year_covariates(dates)
        if args.use_hemisphere and dates is not None:
            hemi_covs = compute_hemisphere_covariate(country_name, dates)
            if week_of_year_covariates is not None:
                week_of_year_covariates.update(hemi_covs)
            else:
                week_of_year_covariates = hemi_covs

        # Per-country season threshold
        season_threshold = compute_season_threshold(data, args.season_threshold_pct)

        start = time.time()
        result = evaluate_country(
            model=model,
            data=data,
            dates=dates,
            eval_window=args.eval_window,
            prediction_horizon=args.prediction_horizon,
            weather_covariates=weather_covariates,
            neighbor_covariates=neighbor_covariates,
            week_of_year_covariates=week_of_year_covariates,
            weather_mode=args.weather_mode,
            context_length=args.context_length,
            season_threshold=season_threshold,
            eval_start_date=args.eval_start_date,
            eval_end_date=args.eval_end_date,
        )
        elapsed = time.time() - start

        if args.plot and result["n_eval_points"] > 0:
            plot_path = plot_rolling_eval(
                country_name=country_name,
                data=data,
                dates=dates,
                eval_result=result,
                eval_window=args.eval_window,
                prediction_horizon=args.prediction_horizon,
                season_threshold=season_threshold,
                output_dir=plots_dir,
            )
            if plot_path:
                log(f"  Plot: {plot_path}")

        # Store compact summary (not per-point data) in JSON
        results["countries"][country_name] = {
            "mape_pct": result["mape_pct"],
            "median_error_pct": result["median_error_pct"],
            "mae": result["mae"],
            "wis": result["wis"],
            "relative_wis": result["relative_wis"],
            "coverage_50": result["coverage_50"],
            "coverage_95": result["coverage_95"],
            "n_eval_points": result["n_eval_points"],
            "season_threshold": result["season_threshold"],
            "used_weather": result["used_weather"],
            "used_neighbors": result["used_neighbors"],
            "phase_mape": result["phase_mape"],
            "peak_timing_mae_weeks": result["peak_timing_mae_weeks"],
            "peak_timing_bias_weeks": result["peak_timing_bias_weeks"],
            "n_peak_timing_points": result["n_peak_timing_points"],
        }

        if result["mape_pct"] is not None:
            tags = []
            if result["used_weather"]:   tags.append("W")
            if result["used_neighbors"]: tags.append("N")
            tag = "[" + ",".join(tags) + "]" if tags else "[-]"

            pm = result["phase_mape"]
            def pf(ph):
                v = pm.get(ph, {})
                mape = v.get("mape_pct")
                n = v.get("n_points", 0)
                return f"{mape:.0f}%({n})" if mape is not None else f"—({n})"

            pt = result["peak_timing_mae_weeks"]
            pb = result["peak_timing_bias_weeks"]
            pt_str = f"±{pt:.1f}wk bias={pb:+.1f}" if pt is not None else "—"

            wis_str = f"WIS={result['wis']:.1f}" if result["wis"] is not None else "WIS=—"
            rwis = result["relative_wis"]
            rwis_str = f"rWIS={rwis:.3f}" if rwis is not None else "rWIS=—"
            cov_str = (
                f"cov50={result['coverage_50']:.0f}% cov95={result['coverage_95']:.0f}%"
                if result["coverage_50"] is not None else ""
            )

            log(
                f"[{i}/{len(countries)}] {country_name}: "
                f"{tag} MAPE={result['mape_pct']:.1f}% | {wis_str} {rwis_str} | {cov_str} | "
                f"({result['n_eval_points']}pts, {elapsed:.1f}s) | "
                f"growth={pf('growth')} peak={pf('peak_window')} decay={pf('decay')} | "
                f"peak_timing={pt_str}"
            )
        else:
            log(f"[{i}/{len(countries)}] {country_name}: no valid eval points (threshold={season_threshold:.1f})")

    total_elapsed = time.time() - total_start

    output_file = args.output or "eval_results.json"
    output_path = os.path.join(RESULTS_DIR, output_file)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    log(f"\nDone in {total_elapsed:.0f}s → {output_path}")

    mapes = [v["mape_pct"] for v in results["countries"].values() if v["mape_pct"] is not None]
    wis_vals = [v["wis"] for v in results["countries"].values() if v.get("wis") is not None]
    rwis_vals = [v["relative_wis"] for v in results["countries"].values() if v.get("relative_wis") is not None]
    cov50_vals = [v["coverage_50"] for v in results["countries"].values() if v.get("coverage_50") is not None]
    cov95_vals = [v["coverage_95"] for v in results["countries"].values() if v.get("coverage_95") is not None]

    if mapes:
        log(f"\nOverall: {len(mapes)} countries | Mean MAPE={np.mean(mapes):.1f}% | Median MAPE={np.median(mapes):.1f}%")
    if wis_vals:
        log(f"  WIS          : mean={np.mean(wis_vals):.2f} median={np.median(wis_vals):.2f} across {len(wis_vals)} countries")
    if rwis_vals:
        rw_arr = np.array(rwis_vals)
        n_better = int(np.sum(rw_arr < 1.0))
        log(f"  Relative WIS : mean={np.mean(rw_arr):.3f} (< 1.0 = better than naive baseline; {n_better}/{len(rwis_vals)} countries beat baseline)")
    if cov50_vals:
        log(f"  Coverage 50% : mean={np.mean(cov50_vals):.1f}% (target: 50%)")
    if cov95_vals:
        log(f"  Coverage 95% : mean={np.mean(cov95_vals):.1f}% (target: 95%)")

    # Phase summary across all countries
    for ph in ("growth", "peak_window", "decay"):
        ph_mapes = [
            v["phase_mape"].get(ph, {}).get("mape_pct")
            for v in results["countries"].values()
            if v.get("phase_mape", {}).get(ph, {}).get("mape_pct") is not None
        ]
        if ph_mapes:
            log(f"  {ph:12s}: mean MAPE={np.mean(ph_mapes):.1f}% across {len(ph_mapes)} countries")

    pt_maes = [
        v["peak_timing_mae_weeks"]
        for v in results["countries"].values()
        if v.get("peak_timing_mae_weeks") is not None
    ]
    if pt_maes:
        log(f"  peak_timing  : mean MAE={np.mean(pt_maes):.2f}wk across {len(pt_maes)} countries")

    if args.weather_mode != "none":
        log(f"Weather: {weather_success} with weather, {weather_fallback} fallback")


if __name__ == "__main__":
    main()
