#!/usr/bin/env python3
"""
Prefetch weather data for all training countries.

Reads training_countries.json, checks cache freshness for each country,
and fetches missing/stale weather data from Open-Meteo. Stale caches are
extended incrementally (only the missing tail is fetched) to minimize API calls.

Usage:
    cd forecast_pipeline
    python scripts/prefetch_weather.py --dry_run
    python scripts/prefetch_weather.py --refresh_stale
    python scripts/prefetch_weather.py --refresh_stale --target_end_date 2026-04-01
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "scripts"))
from weather_fetcher import get_weather_for_country, get_country_coordinates

TRAINING_COUNTRIES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "training_countries.json")
EXTRACTED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "extracted_data")
WEATHER_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "weather_cache")
POLITE_DELAY_SEC = 10
RATE_LIMIT_WAIT_SEC = 90
STALE_THRESHOLD_DAYS = 14

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


def find_extracted_file(country_name: str) -> str | None:
    """Find extracted CSV for a country. Prefers combined > ILI > ARI."""
    safe = country_name.replace(" ", "_").replace(",", "").replace("'", "").replace("(", "").replace(")", "")
    for suffix in ["_combined.csv", "_ili.csv", "_ari.csv"]:
        candidates = [
            os.path.join(EXTRACTED_DATA_DIR, f"extracted_{safe}{suffix}"),
            os.path.join(EXTRACTED_DATA_DIR, f"{safe}{suffix}"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        for fname in os.listdir(EXTRACTED_DATA_DIR):
            if fname.endswith(suffix) and safe.lower() in fname.lower():
                return os.path.join(EXTRACTED_DATA_DIR, fname)
    return None


def get_country_date_range(country_name: str) -> tuple[str, str] | None:
    """Get date range from a country's extracted CSV."""
    filepath = find_extracted_file(country_name)
    if not filepath:
        return None
    try:
        df = pd.read_csv(filepath, usecols=["Time"])
        df["Time"] = pd.to_datetime(df["Time"])
        return df["Time"].min().strftime("%Y-%m-%d"), df["Time"].max().strftime("%Y-%m-%d")
    except Exception:
        return None


def check_cache_freshness(country_name: str, required_end: str) -> tuple[bool, str | None, str | None]:
    """Check if weather cache exists and covers up to required_end.
    Returns (is_fresh, cached_end_date, cache_file_path)."""
    mapped = COUNTRY_NAME_MAP.get(country_name, country_name)
    try:
        lat, lon = get_country_coordinates(mapped)
    except Exception:
        return False, None, None

    prefix = f"{lat:.4f}_{lon:.4f}_"
    candidates = glob.glob(os.path.join(WEATHER_CACHE_DIR, f"{prefix}*.parquet"))
    if not candidates:
        return False, None, None

    best_end = None
    best_file = None
    for f in candidates:
        dates_in_name = re.findall(r'\d{4}-\d{2}-\d{2}', os.path.basename(f))
        if len(dates_in_name) == 2:
            cached_end = dates_in_name[1]
            if best_end is None or cached_end > best_end:
                best_end = cached_end
                best_file = f

    if best_end is None:
        return False, None, None

    required_dt = datetime.strptime(required_end, "%Y-%m-%d")
    cached_dt = datetime.strptime(best_end, "%Y-%m-%d")
    gap_days = (required_dt - cached_dt).days

    if gap_days <= STALE_THRESHOLD_DAYS:
        return True, best_end, best_file
    else:
        return False, best_end, best_file


def extend_cache(country_name: str, cached_file: str, cached_end: str, required_end: str) -> bool:
    """Fetch only the missing tail and merge with existing cache. Returns True on success."""
    mapped = COUNTRY_NAME_MAP.get(country_name, country_name)

    # Fetch from 7 days before cached_end to required_end (small overlap for safety)
    overlap_start = (datetime.strptime(cached_end, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
    weekly_dates = pd.date_range(start=overlap_start, end=required_end, freq="W-MON").tolist()
    if len(weekly_dates) == 0:
        return False

    # Fetch the tail
    get_weather_for_country(
        country_name=mapped,
        start_date=overlap_start,
        end_date=required_end,
        weekly_dates=weekly_dates,
        normalize=False,
        cache_dir=WEATHER_CACHE_DIR,
    )

    # Find the newly created cache file for the tail
    lat, lon = get_country_coordinates(mapped)
    prefix = f"{lat:.4f}_{lon:.4f}_"
    tail_file = os.path.join(WEATHER_CACHE_DIR, f"{prefix}{overlap_start}_{required_end}.parquet")

    if not os.path.exists(tail_file):
        # Weather fetcher may have used a different filename; find it
        new_candidates = glob.glob(os.path.join(WEATHER_CACHE_DIR, f"{prefix}*{required_end}.parquet"))
        tail_file = new_candidates[0] if new_candidates else None

    if tail_file and tail_file != cached_file and os.path.exists(tail_file):
        # Merge old + new
        old_df = pd.read_parquet(cached_file)
        new_df = pd.read_parquet(tail_file)
        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged["date"] = pd.to_datetime(merged["date"])
        merged = merged.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

        # Determine full date range for merged filename
        full_start = merged["date"].min().strftime("%Y-%m-%d")
        full_end = merged["date"].max().strftime("%Y-%m-%d")
        merged_file = os.path.join(WEATHER_CACHE_DIR, f"{prefix}{full_start}_{full_end}.parquet")

        merged.to_parquet(merged_file, index=False)

        # Clean up old files (only if they differ from merged)
        for f in [cached_file, tail_file]:
            if f != merged_file and os.path.exists(f):
                os.remove(f)

        print(f"    merged: {os.path.basename(cached_file)} + tail → {os.path.basename(merged_file)}")

    return True


def prefetch_country(country_name: str, date_start: str, date_end: str) -> bool:
    """Fetch and cache weather for one country (full range). Returns True on success."""
    mapped = COUNTRY_NAME_MAP.get(country_name, country_name)
    weekly_dates = pd.date_range(start=date_start, end=date_end, freq="W-MON").tolist()
    if len(weekly_dates) == 0:
        print(f"    no weekly dates for {date_start} – {date_end}, skipping")
        return False

    get_weather_for_country(
        country_name=mapped,
        start_date=date_start,
        end_date=date_end,
        weekly_dates=weekly_dates,
        normalize=False,
        cache_dir=WEATHER_CACHE_DIR,
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Prefetch weather data for training countries")
    parser.add_argument("--refresh_stale", action="store_true",
                        help="Re-fetch weather for countries with stale cache (incremental)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only report status, don't fetch anything")
    parser.add_argument("--target_end_date", type=str, default=None,
                        help="Override end date for all countries (YYYY-MM-DD). Default: today")
    args = parser.parse_args()

    with open(TRAINING_COUNTRIES_PATH) as f:
        training_data = json.load(f)
    countries = training_data["training_countries"]

    default_end = args.target_end_date or datetime.now().strftime("%Y-%m-%d")

    print(f"Training countries: {len(countries)}")
    print(f"Cache dir: {os.path.abspath(WEATHER_CACHE_DIR)}")
    print(f"Target end date: {default_end}")
    print(f"Mode: {'dry run' if args.dry_run else 'fetch'} | refresh_stale={'yes' if args.refresh_stale else 'no'}\n")

    fresh = []
    stale = []
    missing = []
    no_data = []

    for country in countries:
        date_range = get_country_date_range(country)
        if date_range is None:
            no_data.append(country)
            continue

        date_start, date_end_csv = date_range
        required_end = max(date_end_csv, default_end)

        is_fresh, cached_end, cached_file = check_cache_freshness(country, required_end)
        if cached_end is None:
            missing.append((country, date_start, required_end))
        elif is_fresh:
            fresh.append((country, cached_end))
        else:
            stale.append((country, date_start, required_end, cached_end, cached_file))

    print(f"Fresh:   {len(fresh)}")
    print(f"Stale:   {len(stale)}")
    print(f"Missing: {len(missing)}")
    if no_data:
        print(f"No CSV:  {len(no_data)}")

    if stale:
        print(f"\nStale countries (cache end → required end):")
        for country, _, required_end, cached_end, _ in stale:
            print(f"  {country}: {cached_end} → {required_end}")

    if missing:
        print(f"\nMissing countries:")
        for country, _, _ in missing:
            print(f"  {country}")

    if args.dry_run:
        print("\nDry run — no fetching.")
        return

    # Build fetch list: always fetch missing, optionally extend stale
    to_fetch_full = list(missing)  # (country, date_start, date_end) — full fetch
    to_extend = []
    if args.refresh_stale:
        to_extend = list(stale)  # (country, date_start, required_end, cached_end, cached_file)

    total = len(to_fetch_full) + len(to_extend)
    if total == 0:
        print("\nNothing to fetch — all caches are fresh!")
        return

    print(f"\nFetching: {len(to_fetch_full)} full + {len(to_extend)} incremental\n")

    ok = 0
    failed = []
    idx = 0

    # Full fetches (missing countries)
    for country, date_start, date_end in to_fetch_full:
        idx += 1
        print(f"[{idx}/{total}] {country} (full: {date_start} → {date_end})")
        while True:
            try:
                prefetch_country(country, date_start, date_end)
                print(f"    done")
                ok += 1
                time.sleep(POLITE_DELAY_SEC)
                break
            except RuntimeError as e:
                msg = str(e).lower()
                if "429" in msg or "rate" in msg or "too many" in msg or "max retries" in msg:
                    print(f"    rate limited, waiting {RATE_LIMIT_WAIT_SEC}s...")
                    time.sleep(RATE_LIMIT_WAIT_SEC)
                else:
                    print(f"    failed: {e}")
                    failed.append(country)
                    time.sleep(POLITE_DELAY_SEC)
                    break
            except Exception as e:
                print(f"    failed: {e}")
                failed.append(country)
                time.sleep(POLITE_DELAY_SEC)
                break

    # Incremental extends (stale countries)
    for country, date_start, required_end, cached_end, cached_file in to_extend:
        idx += 1
        print(f"[{idx}/{total}] {country} (extend: {cached_end} → {required_end})")
        while True:
            try:
                extend_cache(country, cached_file, cached_end, required_end)
                print(f"    done")
                ok += 1
                time.sleep(POLITE_DELAY_SEC)
                break
            except RuntimeError as e:
                msg = str(e).lower()
                if "429" in msg or "rate" in msg or "too many" in msg or "max retries" in msg:
                    print(f"    rate limited, waiting {RATE_LIMIT_WAIT_SEC}s...")
                    time.sleep(RATE_LIMIT_WAIT_SEC)
                else:
                    print(f"    failed: {e}")
                    failed.append(country)
                    time.sleep(POLITE_DELAY_SEC)
                    break
            except Exception as e:
                print(f"    failed: {e}")
                failed.append(country)
                time.sleep(POLITE_DELAY_SEC)
                break

    print(f"\n{'='*60}")
    print(f"Done: {ok} fetched, {len(failed)} failed")
    if failed:
        print(f"Failed: {failed}")
    else:
        print("All caches up to date.")


if __name__ == "__main__":
    main()
