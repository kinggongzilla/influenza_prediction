#!/usr/bin/env python3
"""
Step 1: Assess data quality for all country ILI CSVs.

Outputs data/quality_report.csv with per-country stats and inclusion decision.

Usage:
    cd forecast_pipeline
    python scripts/assess_data_quality.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "scripts"))
from data_loader import load_time_series_data

EXTRACTED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "extracted_data")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "quality_report.csv")

MIN_WEEKS_NONNAN = 156       # 3 years of real data
MIN_YEARS_COVERAGE = 4
MAX_NAN_PCT_SPORADIC = 50.0  # for sporadic-NaN countries
MIN_MEDIAN_NONZERO = 20.0    # exclude tiny-count series (Bermuda-type)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_zeros_to_nan(values: np.ndarray, max_consecutive_zeros: int = 3) -> np.ndarray:
    """
    Convert runs of zeros longer than max_consecutive_zeros to NaN.
    Single isolated zeros within an active season are kept (could be real).
    """
    series = pd.Series(values.copy(), dtype=float)
    is_zero = series == 0
    # Label each run of zeros with a group id
    run_id = (is_zero != is_zero.shift()).cumsum()
    run_lengths = is_zero.groupby(run_id).transform("sum")
    series[is_zero & (run_lengths > max_consecutive_zeros)] = np.nan
    return series.values


def trim_leading_artifact(values: np.ndarray, dates, threshold_pct: float = 0.10):
    """
    Trim leading zeros/NaN before the series 'activates'.
    Returns (trimmed_values, trimmed_dates).
    """
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


def classify_nan_pattern(values: np.ndarray, dates) -> str:
    """
    Returns 'seasonal' if NaN weeks cluster in the same months every year
    (i.e. country only surveys during flu season), else 'sporadic'.
    """
    df = pd.DataFrame({"v": values, "date": pd.to_datetime([str(d) for d in dates])})
    df["month"] = df["date"].dt.month
    monthly_nan = df.groupby("month")["v"].apply(lambda x: x.isna().mean())
    low_nan_months = int((monthly_nan < 0.2).sum())
    high_nan_months = int((monthly_nan > 0.8).sum())
    if low_nan_months >= 8 and high_nan_months >= 2:
        return "seasonal"
    return "sporadic"


def detect_long_gaps(values: np.ndarray, gap_weeks: int = 52) -> bool:
    """Returns True if there is a run of NaN longer than gap_weeks mid-series."""
    is_nan = np.isnan(values)
    count = 0
    max_run = 0
    for v in is_nan:
        count = count + 1 if v else 0
        max_run = max(max_run, count)
    return max_run > gap_weeks


def count_leading_artifact_weeks(values: np.ndarray, dates, threshold_pct: float = 0.10) -> int:
    trimmed, _ = trim_leading_artifact(values, dates, threshold_pct)
    return len(values) - len(trimmed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def assess_country(filepath: str, country_name: str) -> dict:
    import contextlib, io
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            data, dates, _ = load_time_series_data(filepath, context_length=None, min_value=None)
    except Exception as e:
        return {"country": country_name, "error": str(e), "include": False}

    values = data.numpy().copy()

    # --- Zero cleaning ---
    values_cleaned = clean_zeros_to_nan(values)

    # --- Basic stats ---
    n_total = len(values_cleaned)
    n_nonnan = int(np.sum(~np.isnan(values_cleaned)))
    nan_pct = round(100.0 * (1 - n_nonnan / n_total), 1) if n_total > 0 else 100.0

    if dates is not None and len(dates) > 0:
        dates_pd = pd.to_datetime([str(d) for d in dates])
        date_start = str(dates_pd[0].date())
        date_end = str(dates_pd[-1].date())
        years_coverage = round((dates_pd[-1] - dates_pd[0]).days / 365.25, 1)
    else:
        date_start = date_end = "unknown"
        years_coverage = 0.0

    nonzero_vals = values_cleaned[(values_cleaned > 0) & ~np.isnan(values_cleaned)]
    median_nonzero = round(float(np.median(nonzero_vals)), 2) if len(nonzero_vals) > 0 else 0.0
    scale = median_nonzero

    # --- Pattern classification ---
    nan_pattern = classify_nan_pattern(values_cleaned, dates) if dates is not None else "sporadic"
    has_long_gap = detect_long_gaps(values_cleaned)
    leading_artifact_weeks = count_leading_artifact_weeks(values_cleaned, dates) if dates is not None else 0

    # --- Inclusion decision ---
    reasons = []

    if n_nonnan < MIN_WEEKS_NONNAN:
        reasons.append(f"too_few_nonnan({n_nonnan}<{MIN_WEEKS_NONNAN})")
    if years_coverage < MIN_YEARS_COVERAGE:
        reasons.append(f"too_short({years_coverage:.1f}yr<{MIN_YEARS_COVERAGE})")
    if nan_pattern == "sporadic" and nan_pct > MAX_NAN_PCT_SPORADIC:
        reasons.append(f"high_nan_sporadic({nan_pct}%>50%)")
    if scale < MIN_MEDIAN_NONZERO:
        reasons.append(f"too_small_scale({scale}<{MIN_MEDIAN_NONZERO})")

    include = len(reasons) == 0

    return {
        "country": country_name,
        "include": include,
        "exclude_reason": "; ".join(reasons) if reasons else "",
        "nan_pattern": nan_pattern,
        "n_weeks_total": n_total,
        "n_weeks_nonnan": n_nonnan,
        "nan_pct": nan_pct,
        "date_start": date_start,
        "date_end": date_end,
        "years_coverage": years_coverage,
        "leading_artifact_weeks": leading_artifact_weeks,
        "has_long_gap": has_long_gap,
        "median_nonzero_value": scale,
    }


def main():
    files = sorted(f for f in os.listdir(EXTRACTED_DATA_DIR) if f.endswith("_ili.csv"))
    print(f"Found {len(files)} ILI country files")

    rows = []
    for fname in files:
        country = fname.replace("_ili.csv", "").replace("_", " ")
        filepath = os.path.join(EXTRACTED_DATA_DIR, fname)
        row = assess_country(filepath, country)
        status = "✓" if row.get("include") else "✗"
        reason = f"  [{row.get('exclude_reason', '')}]" if not row.get("include") else ""
        print(f"  {status} {country}: {row.get('n_weeks_nonnan', '?')} nonnan weeks, "
              f"{row.get('nan_pct', '?')}% NaN, "
              f"scale={row.get('median_nonzero_value', '?')}, "
              f"pattern={row.get('nan_pattern', '?')}"
              f"{reason}")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)

    n_included = int(df["include"].sum())
    print(f"\nTotal: {n_included}/{len(rows)} countries pass quality filters")
    print(f"Report saved to: {OUTPUT_PATH}")

    print("\n--- Included countries ---")
    for _, row in df[df["include"]].sort_values("country").iterrows():
        print(f"  {row['country']} ({row['nan_pattern']}, {row['n_weeks_nonnan']}wk, "
              f"scale={row['median_nonzero_value']:.0f})")


if __name__ == "__main__":
    main()
