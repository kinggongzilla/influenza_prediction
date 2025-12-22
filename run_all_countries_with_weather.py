#!/usr/bin/env python3
"""
Batch processing script to run Chronos2 inference with weather covariates for all countries.

This script processes all country ILI files in the data/ directory and generates
forecasts with weather covariates.
"""

import os
import subprocess
import sys
from pathlib import Path

# Country file to country name mapping
COUNTRY_MAPPING = {
    'argentina_ili_extracted.csv': 'Argentina',
    'austria_ili_extracted.csv': 'Austria',
    'canada_ili_extracted.csv': 'Canada',
    # 'china_ili_extracted.csv': 'China',  # Excluded per user request
    'france_ili_extracted.csv': 'France',
    # 'hongkong_ili_extracted.csv': 'Hong Kong',  # Deleted - not in bounding boxes
    'italy_ili_extracted.csv': 'Italy',
    'mexico_ili_extracted.csv': 'Mexico',
    'moldova_ili_extracted.csv': 'Moldova',
    'mongolia_ili_extracted.csv': 'Mongolia',
    'portugal_ili_extracted.csv': 'Portugal',
    'usa_ili_extracted.csv': 'United States',
}

def run_inference_for_country(data_file, country_name, prediction_length=54, test_split=0.1):
    """
    Run Chronos2 inference with weather covariates for a single country.

    Args:
        data_file: CSV filename (e.g., 'argentina_ili_extracted.csv')
        country_name: Full country name (e.g., 'Argentina')
        prediction_length: Forecast horizon
        test_split: Fraction for test set

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "=" * 80)
    print(f"Processing: {country_name}")
    print("=" * 80)

    cmd = [
        'python', 'src/scripts/chronos_inference.py',
        '--data_file', data_file,
        '--country_name', country_name,
        '--use_weather',
        '--prediction_length', str(prediction_length),
        '--test_split', str(test_split),
        '--plot'
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✅ SUCCESS: {country_name}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: {country_name}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {country_name}")
        print(f"Exception: {e}")
        return False


def main():
    """Run inference for all countries."""
    print("=" * 80)
    print("Batch Processing: Chronos2 Inference with Weather Covariates")
    print("=" * 80)
    print(f"\nTotal countries to process: {len(COUNTRY_MAPPING)}")
    print("\nCountries:")
    for i, (file, name) in enumerate(COUNTRY_MAPPING.items(), 1):
        print(f"  {i}. {name} ({file})")
    print()

    # Track results
    results = {}

    # Process each country
    for data_file, country_name in COUNTRY_MAPPING.items():
        success = run_inference_for_country(
            data_file=data_file,
            country_name=country_name,
            prediction_length=54,
            test_split=0.1
        )
        results[country_name] = success

    # Print summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)

    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]

    print(f"\n✅ Successful: {len(successful)}/{len(results)}")
    for name in successful:
        print(f"   - {name}")

    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(results)}")
        for name in failed:
            print(f"   - {name}")

    print("\n" + "=" * 80)
    print(f"Processing complete! Success rate: {len(successful)}/{len(results)}")
    print("=" * 80)

    # Exit with error code if any failed
    if failed:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
