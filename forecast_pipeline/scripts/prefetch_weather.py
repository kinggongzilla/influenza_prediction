#!/usr/bin/env python3
"""
Step 2a: Prefetch weather data for all qualifying countries.

Iterates the quality_report.csv, fetches historical weather for each included
country, and waits when rate-limited. Must complete before prepare_finetune_data.py.

Usage:
    cd forecast_pipeline
    python scripts/prefetch_weather.py
"""

import argparse
import glob
import os
import re
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "scripts"))
from weather_fetcher import get_weather_for_country, get_country_coordinates

QUALITY_REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "quality_report.csv")
WEATHER_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "weather_cache")
POLITE_DELAY_SEC = 10
RATE_LIMIT_WAIT_SEC = 90
# Cache is considered stale if its end date is more than this many days before the required end date
STALE_THRESHOLD_DAYS = 14

# Map long/unusual country names to shorter forms the weather fetcher can resolve
COUNTRY_NAME_MAP = {
    "Côte d'Ivoire": "Ivory Coast",
    "Kosovo (in accordance with UN Security Council resolution 1244 (1999))": "Kosovo",
    "Micronesia (Federated States of)": "Micronesia",
    "Netherlands (Kingdom of the)": "Netherlands",
    "Lao People's Democratic Republic": "Laos",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Iran (Islamic Republic of)": "Iran",
    "Republic of Moldova": "Moldova",
    "Democratic People's Republic of Korea": "North Korea",
    "Democratic Republic of the Congo": "Democratic Republic of the Congo",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "United Kingdom, England": "United Kingdom",
    "United Kingdom, Northern Ireland": "United Kingdom",
    "United Kingdom, Scotland": "United Kingdom",
    "United Kingdom, Wales": "United Kingdom",
    "United Republic of Tanzania": "Tanzania",
    "United States of America": "United States",
    "Viet Nam": "Vietnam",
    "Russian Federation": "Russia",
    "Syrian Arab Republic": "Syria",
    "occupied Palestinian territory, including east Jerusalem": "Palestine",
    "Türkiye": "Turkey",
}


def check_cache_freshness(country_name: str, required_end: str) -> tuple[bool, str | None]:
    """Check if weather cache exists and covers up to required_end.
    Returns (is_fresh, cached_end_date or None)."""
    mapped = COUNTRY_NAME_MAP.get(country_name, country_name)
    try:
        lat, lon = get_country_coordinates(mapped)
    except Exception:
        return False, None

    prefix = f"{lat:.4f}_{lon:.4f}_"
    candidates = glob.glob(os.path.join(WEATHER_CACHE_DIR, f"{prefix}*.parquet"))
    if not candidates:
        return False, None

    best_end = None
    for f in candidates:
        dates_in_name = re.findall(r'\d{4}-\d{2}-\d{2}', os.path.basename(f))
        if len(dates_in_name) == 2:
            cached_end = dates_in_name[1]
            if best_end is None or cached_end > best_end:
                best_end = cached_end

    if best_end is None:
        return False, None

    # Check if cache covers the required range (within threshold)
    from datetime import datetime, timedelta
    required_dt = datetime.strptime(required_end, "%Y-%m-%d")
    cached_dt = datetime.strptime(best_end, "%Y-%m-%d")
    gap_days = (required_dt - cached_dt).days

    if gap_days <= STALE_THRESHOLD_DAYS:
        return True, best_end
    else:
        return False, best_end


def prefetch_country(country_name: str, date_start: str, date_end: str) -> bool:
    country_name = COUNTRY_NAME_MAP.get(country_name, country_name)
    """
    Fetch and cache weather for one country. Returns True on success.
    Raises RuntimeError on non-rate-limit failures.
    Raises RateLimitError (RuntimeError with 'rate' in message) on rate limits.
    """
    # Build a dummy weekly_dates list spanning the full range at weekly frequency
    weekly_dates = pd.date_range(start=date_start, end=date_end, freq="W-MON").tolist()
    if len(weekly_dates) == 0:
        print(f"  ⚠  No weekly dates for {country_name} ({date_start} – {date_end}), skipping")
        return False

    get_weather_for_country(
        country_name=country_name,
        start_date=date_start,
        end_date=date_end,
        weekly_dates=weekly_dates,
        normalize=False,          # raw data; normalisation happens in prepare_finetune_data.py
        cache_dir=WEATHER_CACHE_DIR,
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Prefetch weather data for qualifying countries")
    parser.add_argument("--refresh_stale", action="store_true",
                        help="Re-fetch weather for countries with stale cache (end date too old)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only report status, don't fetch anything")
    parser.add_argument("--target_end_date", type=str, default=None,
                        help="Override end date for all countries (YYYY-MM-DD). Default: use each country's date_end from quality report")
    args = parser.parse_args()

    df = pd.read_csv(QUALITY_REPORT_PATH)
    included = df[df["include"]].reset_index(drop=True)
    print(f"Qualifying countries: {len(included)}")
    print(f"Cache dir: {os.path.abspath(WEATHER_CACHE_DIR)}")
    print(f"Mode: {'dry run' if args.dry_run else 'fetch'} | refresh_stale={'yes' if args.refresh_stale else 'no'}\n")

    # First pass: check freshness
    fresh = []
    stale = []
    missing = []

    for _, row in included.iterrows():
        country = row["country"]
        required_end = args.target_end_date or row["date_end"]
        is_fresh, cached_end = check_cache_freshness(country, required_end)
        if cached_end is None:
            missing.append((country, row["date_start"], required_end))
        elif is_fresh:
            fresh.append((country, cached_end))
        else:
            stale.append((country, row["date_start"], required_end, cached_end))

    print(f"Fresh:   {len(fresh)}")
    print(f"Stale:   {len(stale)}")
    print(f"Missing: {len(missing)}")

    if stale:
        print(f"\nStale countries (cache end → required end):")
        for country, _, required_end, cached_end in stale:
            print(f"  {country}: {cached_end} → {required_end}")

    if missing:
        print(f"\nMissing countries:")
        for country, _, _ in missing:
            print(f"  {country}")

    if args.dry_run:
        print("\nDry run — no fetching.")
        return

    # Build fetch list
    to_fetch = list(missing)  # always fetch missing
    if args.refresh_stale:
        to_fetch.extend([(c, s, e) for c, s, e, _ in stale])

    if not to_fetch:
        print("\nNothing to fetch — all caches are fresh!")
        return

    print(f"\nFetching weather for {len(to_fetch)} countries...\n")

    ok = 0
    skipped = 0
    failed = []

    for i, (country, date_start, date_end) in enumerate(to_fetch):
        print(f"[{i+1}/{len(to_fetch)}] {country} ({date_start} – {date_end})")

        while True:
            try:
                prefetch_country(country, date_start, date_end)
                print(f"  ✓ done")
                ok += 1
                time.sleep(POLITE_DELAY_SEC)
                break
            except RuntimeError as e:
                msg = str(e).lower()
                if "429" in msg or "rate" in msg or "too many" in msg or "max retries" in msg:
                    print(f"  ⏳ Rate limited. Waiting {RATE_LIMIT_WAIT_SEC}s before retry…")
                    time.sleep(RATE_LIMIT_WAIT_SEC)
                    # loop back and retry
                else:
                    print(f"  ✗ Failed: {e} — skipping")
                    failed.append(country)
                    skipped += 1
                    time.sleep(POLITE_DELAY_SEC)
                    break
            except Exception as e:
                print(f"  ✗ Unexpected error: {e} — skipping")
                failed.append(country)
                skipped += 1
                time.sleep(POLITE_DELAY_SEC)
                break

    print(f"\n{'='*60}")
    print(f"Done: {ok} fetched, {skipped} failed")
    if failed:
        print(f"Failed countries: {failed}")
    else:
        print("All countries fetched successfully — ready for prepare_finetune_data.py")


if __name__ == "__main__":
    main()
