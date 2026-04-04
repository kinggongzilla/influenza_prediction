#!/usr/bin/env python3
"""
WHO fluID Data Updater

Downloads the latest WHO fluID epidemiological surveillance data and regenerates
the country data summary used by the forecasting pipeline.

Data source: https://xmart-api-public.who.int/FLUMART/VIW_FID_EPI?$format=csv
(linked from https://www.who.int/teams/global-influenza-programme/surveillance-and-monitoring/influenza-surveillance-outputs)

Usage:
    python update_who_data.py [--backup] [--run_inference] [--limit N] [--skip_weather_on_error]
"""

import os
import sys
import shutil
import argparse
import subprocess
import logging
from pathlib import Path

import requests
import pandas as pd

# Paths relative to this script's directory (forecast_pipeline/)
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
WHO_DATA_FILE = DATA_DIR / "who_flu_data.csv"
COUNTRY_SUMMARY_FILE = SCRIPT_DIR / "country_data_summary.csv"

WHO_FLUID_URL = "https://xmart-api-public.who.int/FLUMART/VIW_FID_EPI?$format=csv"
EXPECTED_COLUMNS = {
    "COUNTRY_AREA_TERRITORY",
    "ISO_WEEKSTARTDATE",
    "AGEGROUP_CODE",
    "ILI_CASE",
    "ARI_CASE",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download_fluid_data(output_path: Path, backup: bool) -> None:
    """Stream-download the WHO fluID CSV and save to output_path."""
    if backup and output_path.exists():
        backup_path = output_path.with_suffix(".csv.bak")
        shutil.copy2(output_path, backup_path)
        logger.info(f"Backed up existing data to {backup_path}")

    logger.info(f"Downloading fluID data from {WHO_FLUID_URL} ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(WHO_FLUID_URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 1024 * 256  # 256 KB

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = 100 * downloaded / total
                            print(
                                f"\r  {downloaded / 1_000_000:.1f} MB / {total / 1_000_000:.1f} MB ({pct:.0f}%)",
                                end="",
                                flush=True,
                            )
                        else:
                            print(
                                f"\r  {downloaded / 1_000_000:.1f} MB downloaded",
                                end="",
                                flush=True,
                            )
            print()  # newline after progress

    except requests.RequestException as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)

    logger.info(f"Saved to {output_path} ({output_path.stat().st_size / 1_000_000:.1f} MB)")


def validate_data(path: Path) -> pd.DataFrame:
    """Load and validate the downloaded CSV; return the DataFrame."""
    logger.info("Validating downloaded data ...")
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        logger.error(f"Could not read CSV: {e}")
        sys.exit(1)

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        logger.error(f"Downloaded CSV is missing expected columns: {missing}")
        sys.exit(1)

    row_count = len(df)
    if row_count < 10_000:
        logger.error(f"Downloaded CSV only has {row_count} rows — expected >10,000. Aborting.")
        sys.exit(1)

    latest_date = df["ISO_WEEKSTARTDATE"].dropna().max()
    logger.info(f"  Rows: {row_count:,}")
    logger.info(f"  Countries: {df['COUNTRY_AREA_TERRITORY'].nunique()}")
    logger.info(f"  Latest week: {latest_date}")
    return df


def regenerate_country_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Regenerate country_data_summary.csv from the WHO data."""
    logger.info("Regenerating country data summary ...")

    summary = (
        df.groupby("COUNTRY_AREA_TERRITORY", as_index=False)
        .agg(total_ili_cases=("ILI_CASE", "sum"), total_ari_cases=("ARI_CASE", "sum"))
        .rename(columns={"COUNTRY_AREA_TERRITORY": "country"})
    )

    def data_types(row):
        types = []
        if row["total_ili_cases"] > 0:
            types.append("ILI")
        if row["total_ari_cases"] > 0:
            types.append("ARI")
        return ", ".join(types) if types else "None"

    summary["data_types"] = summary.apply(data_types, axis=1)

    summary.to_csv(output_path, index=False)
    logger.info(f"Saved country summary: {len(summary)} countries → {output_path}")


def run_inference(limit: int | None, start_from: int, skip_weather_on_error: bool) -> None:
    """Invoke run_all_country_inference.py as a subprocess."""
    script = SCRIPT_DIR / "run_all_country_inference.py"
    cmd = [sys.executable, str(script)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if start_from:
        cmd += ["--start_from", str(start_from)]
    if skip_weather_on_error:
        cmd += ["--skip_weather_on_error"]

    logger.info(f"Running inference pipeline: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    if result.returncode != 0:
        logger.error("Inference pipeline exited with errors.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Download latest WHO fluID data and preprocess for the forecasting pipeline"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup existing who_flu_data.csv before overwriting",
    )
    parser.add_argument(
        "--run_inference",
        action="store_true",
        help="After preprocessing, run full country inference pipeline",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit inference to N countries (passed to run_all_country_inference.py)",
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=0,
        help="Start inference from country index N (for resuming)",
    )
    parser.add_argument(
        "--skip_weather_on_error",
        action="store_true",
        help="Fall back to non-weather inference if weather fetch fails",
    )
    args = parser.parse_args()

    # Step 1: Download
    download_fluid_data(WHO_DATA_FILE, backup=args.backup)

    # Step 2: Validate + regenerate summary
    df = validate_data(WHO_DATA_FILE)
    regenerate_country_summary(df, COUNTRY_SUMMARY_FILE)

    # Step 3: Optionally run inference
    if args.run_inference:
        run_inference(
            limit=args.limit,
            start_from=args.start_from,
            skip_weather_on_error=args.skip_weather_on_error,
        )
    else:
        logger.info(
            "Data updated. Run with --run_inference to trigger country forecasting."
        )


if __name__ == "__main__":
    main()
