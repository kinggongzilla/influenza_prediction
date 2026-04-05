#!/usr/bin/env python3
"""
Batch Processing Script for Country-by-Country Inference

Loads the Chronos model once and runs inference for all countries in-process
(no subprocess overhead per country).

Features:
- Single model load, batch country inference on GPU
- Automatic data extraction with ILI/ARI fallback
- Progress tracking and error handling
- Results organization and reporting
"""

import pandas as pd
import numpy as np
import os
import json
import subprocess
import sys
import time
import argparse
import torch
from typing import List, Optional
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from chronos.chronos2 import Chronos2Pipeline
from data_loader import load_time_series_data, load_data_type_indicator, plot_forecast_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('country_inference_batch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

PREDICTION_LENGTH = 54
MODEL_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load_country_list(country_summary_file: str, countries_file: str = None) -> pd.DataFrame:
    """Load country summary, optionally filtered by a countries JSON file."""
    df = pd.read_csv(country_summary_file)
    logger.info(f"Loaded country summary with {len(df)} countries")

    country_data_types = {}
    if countries_file:
        with open(countries_file, 'r') as f:
            countries_data = json.load(f)
        allowed = set(countries_data.get('training_countries', []))
        country_data_types = countries_data.get('country_data_types', {})
        df = df[df['country'].isin(allowed)]
        logger.info(f"Filtered to {len(df)} countries from {countries_file}")

    # Add data_type column from training_countries.json (defaults to ili)
    df = df.copy()
    df['data_type'] = df['country'].map(lambda c: country_data_types.get(c, 'ili'))

    # Skip countries with no data
    df = df[(df['total_ili_cases'] > 0) | (df['total_ari_cases'] > 0)]
    logger.info(f"{len(df)} countries have data ({(df['data_type'] == 'ari').sum()} ARI)")
    return df


def extract_country_data(country_name: str, who_data_file: str, extracted_data_dir: str,
                         data_type: str = "ili") -> Optional[str]:
    """Run the extraction script for a single country, return path to extracted file."""
    safe_name = country_name.replace(' ', '_')

    # Check if extracted file already exists (prefer combined)
    for suffix in [f"_combined.csv", f"_{data_type}.csv"]:
        for pattern in [f"{safe_name}{suffix}", f"extracted_{safe_name}{suffix}"]:
            path = os.path.join(extracted_data_dir, pattern)
            if os.path.exists(path):
                return path

    cmd = [
        sys.executable,
        "enhanced_extract_country_data.py",
        "--country", country_name,
        "--mode", "combined",
        "--input", who_data_file,
        "--output_dir", extracted_data_dir
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
    if result.returncode != 0:
        logger.error(f"Extraction failed for {country_name}: {result.stderr[:200]}")
        return None

    # Parse output for file path
    for line in result.stdout.split('\n'):
        if 'Data saved to:' in line:
            return line.split('Data saved to:')[1].strip()

    # Fallback: look for combined first, then data_type
    for suffix in ["_combined.csv", f"_{data_type}.csv"]:
        for f in os.listdir(extracted_data_dir):
            if safe_name.lower() in f.lower() and f.endswith(suffix):
                return os.path.join(extracted_data_dir, f)

    return None


def run_inference_for_country(
    model: Chronos2Pipeline,
    extracted_file: str,
    country_name: str,
    results_dir: str,
    prediction_length: int = PREDICTION_LENGTH,
    plot: bool = True,
) -> bool:
    """Run inference for a single country using the already-loaded model."""
    clean_name = country_name.replace(' ', '_').replace('-', '_')
    country_results_dir = os.path.join(results_dir, clean_name)
    os.makedirs(country_results_dir, exist_ok=True)

    # Load data
    data_file_path = extracted_file if extracted_file.startswith('data/') else f"data/{extracted_file}"
    data, dates, _ = load_time_series_data(
        data_file_path,
        context_length=None,
        min_value=1e-6,
    )

    # Load data type indicator (ILI=0, ARI=1) if available
    dt_indicator = load_data_type_indicator(data_file_path, dates)

    if dt_indicator is not None:
        # Dict-based input with data_type covariate
        input_dict = {
            "target": data.numpy(),
            "past_covariates": {"data_type_indicator": dt_indicator},
            "future_covariates": {"data_type_indicator": np.full(prediction_length, dt_indicator[-1])},
        }
        quantile_forecasts, mean_forecasts = model.predict_quantiles(
            inputs=[input_dict],
            prediction_length=prediction_length,
        )
    else:
        # Univariate inference (no covariates)
        data_batched = data.unsqueeze(0)
        data_3d = data_batched.unsqueeze(-1).transpose(1, 2)
        quantile_forecasts, mean_forecasts = model.predict_quantiles(
            inputs=data_3d,
            prediction_length=prediction_length,
        )

    forecast_mean = mean_forecasts[0]
    forecast_quantiles = quantile_forecasts[0]

    # Plot
    if plot:
        plot_country_name = extracted_file.replace('_ili_extracted.csv', '').replace('data/', '')
        plot_country_name = plot_country_name.replace('_ari_extracted.csv', '')
        plot_country_name = plot_country_name.replace('_combined.csv', '').replace('_ili.csv', '').replace('_ari.csv', '')
        plot_country_name = plot_country_name.replace('/', '_').replace(' ', '_')

        plot_forecast_results(
            context=data,
            forecast_mean=forecast_mean.squeeze(),
            forecast_quantiles=forecast_quantiles,
            prediction_length=prediction_length,
            model_quantiles=MODEL_QUANTILES,
            test_data=None,
            test_dates=None,
            data_file=extracted_file,
            dates=dates,
            country_name=plot_country_name,
            is_eval=False,
            model_name="Chronos-2",
            results_dir=country_results_dir
        )

    # Save CSV
    context_length = len(data)
    total_length = context_length + prediction_length

    results = {
        'time': list(range(total_length)),
        'context': data.tolist() + [None] * prediction_length,
        'mean_forecast': [None] * context_length + forecast_mean.squeeze().tolist()
    }
    for i, q in enumerate(MODEL_QUANTILES):
        results[f'quantile_{q}'] = [None] * context_length + forecast_quantiles[0, :, i].squeeze().tolist()

    results_df = pd.DataFrame(results)

    csv_name = extracted_file.replace('_ili_extracted.csv', '').replace('data/', '')
    csv_name = csv_name.replace('_ari_extracted.csv', '')
    csv_name = csv_name.replace('_combined.csv', '').replace('_ili.csv', '').replace('_ari.csv', '')
    csv_name = csv_name.replace('/', '_').replace(' ', '_')
    csv_filename = f'{csv_name}_forecast_results.csv'
    results_df.to_csv(os.path.join(country_results_dir, csv_filename), index=False)

    return True


def main():
    parser = argparse.ArgumentParser(description="Batch country inference (single model load)")
    parser.add_argument("--model_path", type=str, default="amazon/chronos-2",
                        help="Model checkpoint or HuggingFace model ID")
    parser.add_argument("--countries_file", type=str, default=None,
                        help="JSON file with training_countries list")
    parser.add_argument("--country_summary_file", type=str, default="country_data_summary.csv")
    parser.add_argument("--who_data_file", type=str, default="data/who_flu_data.csv")
    parser.add_argument("--extracted_data_dir", type=str, default="data/extracted_data")
    parser.add_argument("--results_dir", type=str, default="results/country_predictions")
    parser.add_argument("--prediction_length", type=int, default=PREDICTION_LENGTH)
    parser.add_argument("--no_plot", action="store_true", help="Skip plot generation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of countries")
    parser.add_argument("--start_from", type=int, default=0, help="Start from country index N")
    parser.add_argument("--test", action="store_true", help="Quick test with 3 countries")
    args = parser.parse_args()

    if args.test:
        args.limit = 3

    os.makedirs(args.extracted_data_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Load country list
    countries = load_country_list(args.country_summary_file, args.countries_file)

    # Load model ONCE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model from {args.model_path} on {device}...")
    model = Chronos2Pipeline.from_pretrained(args.model_path, device_map=device)
    logger.info(f"Model loaded. Device: {model.model.device}")

    # Process countries
    total = min(args.limit or len(countries), len(countries))
    logger.info(f"Processing {total} countries...")

    success_count = 0
    fail_count = 0
    skip_count = 0
    start_time = time.time()

    for i, (_, row) in enumerate(countries.iterrows(), 1):
        if args.limit and i > args.limit:
            break
        if i <= args.start_from:
            continue

        country_name = row['country']
        data_type = row.get('data_type', 'ili')

        # Extract data
        extracted_file = extract_country_data(country_name, args.who_data_file, args.extracted_data_dir,
                                              data_type=data_type)
        if extracted_file is None:
            logger.warning(f"[{i}/{total}] SKIP {country_name} - extraction failed")
            skip_count += 1
            continue

        # Run inference
        try:
            t0 = time.time()
            run_inference_for_country(
                model=model,
                extracted_file=extracted_file,
                country_name=country_name,
                results_dir=args.results_dir,
                prediction_length=args.prediction_length,
                plot=not args.no_plot,
            )
            elapsed = time.time() - t0
            logger.info(f"[{i}/{total}] OK {country_name} ({elapsed:.1f}s)")
            success_count += 1
        except Exception as e:
            logger.error(f"[{i}/{total}] FAIL {country_name}: {e}")
            fail_count += 1

        if i % 10 == 0:
            elapsed_total = time.time() - start_time
            logger.info(f"Progress: {i}/{total} | OK={success_count} FAIL={fail_count} SKIP={skip_count} | {elapsed_total:.0f}s elapsed")

    total_time = time.time() - start_time
    logger.info(f"\nDone. {success_count} OK, {fail_count} failed, {skip_count} skipped in {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"Results in: {args.results_dir}")


if __name__ == "__main__":
    main()
