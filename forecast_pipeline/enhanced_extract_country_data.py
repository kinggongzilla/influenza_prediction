#!/usr/bin/env python3
"""
Enhanced Country Data Extractor with ILI/ARI Fallback Logic

This script extracts time series data for countries from the WHO FLUMART dataset,
automatically choosing ILI data if available, or falling back to ARI data if ILI cases are zero.

Parallel-reporting guard: some countries report ILI and ARI *simultaneously*, with the two
series on very different scales (e.g. a small sentinel-style ILI network alongside national
ARI totals). Mixing them in one series is unusable: weeks where the smaller series reports
zero get filled with values from the larger one, splicing a different scale into the series
(Russia: tens of ILI cases per week next to 100k+ ARI weeks). When both indicators are
reported in parallel for a sustained period and their medians differ by more than an order
of magnitude, keep only the larger-scale (usually national) indicator, provided it is a
substantive series on its own (same floors as the quality screen, so a short pandemic-era
series cannot replace a long one).

The script is designed to support batch processing for running inference on all available countries.
"""

import pandas as pd
import argparse
import os
import re
from typing import Optional, Tuple
from datetime import datetime

# Parallel-reporting guard thresholds (see extract_combined_data).
# The first two mirror the quality-screen floors in scripts/prepare_finetune_data.py.
PARALLEL_MIN_BOTH_WEEKS = 26    # ~half a year of weeks where both indicators are reported
SCALE_RATIO_THRESHOLD = 10.0    # median-scale difference that makes mixing unusable
MIN_WINNER_WEEKS = 156          # winner must have at least this many positive weeks
MIN_WINNER_YEARS = 4.0          # winner must span at least this many years


def find_country_code_and_name(country_name: str, df: pd.DataFrame) -> Tuple[str, str]:
    """
    Find country code and full name given a country name or partial match.
    
    Args:
        country_name: Country name or partial name to search for
        df: DataFrame containing the WHO flu data
        
    Returns:
        Tuple of (country_code, country_name) or (None, None) if not found
    """
    # Try exact code match first
    exact_code_match = df[df['COUNTRY_CODE'] == country_name.upper()]
    if len(exact_code_match) > 0:
        return exact_code_match['COUNTRY_CODE'].iloc[0], exact_code_match['COUNTRY_AREA_TERRITORY'].iloc[0]
    
    # Try exact name match (case-insensitive)
    exact_name_match = df[df['COUNTRY_AREA_TERRITORY'].str.lower() == country_name.lower()]
    if len(exact_name_match) > 0:
        return exact_name_match['COUNTRY_CODE'].iloc[0], exact_name_match['COUNTRY_AREA_TERRITORY'].iloc[0]
    
    # Try word boundary regex match
    try:
        pattern = rf'\b{re.escape(country_name)}\b'
        regex_matches = df[df['COUNTRY_AREA_TERRITORY'].str.contains(pattern, case=False, na=False, regex=True)]
        if len(regex_matches) > 0:
            return regex_matches['COUNTRY_CODE'].iloc[0], regex_matches['COUNTRY_AREA_TERRITORY'].iloc[0]
    except Exception:
        pass
    
    # Try contains match for longer names
    if len(country_name) >= 3:
        contains_matches = df[df['COUNTRY_AREA_TERRITORY'].str.contains(country_name, case=False, na=False)]
        if len(contains_matches) == 1:
            return contains_matches['COUNTRY_CODE'].iloc[0], contains_matches['COUNTRY_AREA_TERRITORY'].iloc[0]
        elif len(contains_matches) > 1:
            # Multiple matches - return the first one
            return contains_matches['COUNTRY_CODE'].iloc[0], contains_matches['COUNTRY_AREA_TERRITORY'].iloc[0]
    
    return None, None


def extract_ili_data(country_code: str, country_name: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Extract ILI data for a specific country.
    
    Args:
        country_code: Country ISO code
        country_name: Country name
        df: DataFrame containing the WHO flu data
        
    Returns:
        DataFrame with Time and ILI_Cases columns, or None if no data
    """
    # Filter for the specific country
    country_df = df[df['COUNTRY_CODE'] == country_code].copy()
    
    if len(country_df) == 0:
        print(f"  ⚠️  No data found for {country_name} ({country_code})")
        return None
    
    # Get "All" age group data (case-insensitive)
    country_all_ages = country_df[country_df['AGEGROUP_CODE'].str.upper() == 'ALL'].copy()
    
    if len(country_all_ages) == 0:
        print(f"  ⚠️  No 'All' age group data found for {country_name}")
        return None
    
    # Select and rename columns
    country_simple = country_all_ages[['ISO_WEEKSTARTDATE', 'ILI_CASE']].copy()
    country_simple.columns = ['Time', 'ILI_Cases']
    
    # Convert to datetime and sort
    country_simple['Time'] = pd.to_datetime(country_simple['Time'])
    country_simple = country_simple.sort_values('Time').reset_index(drop=True)
    
    # Check if we have any non-zero ILI cases
    non_zero_ili = country_simple['ILI_Cases'].fillna(0).astype(float) > 0
    if not non_zero_ili.any():
        print(f"  ⚠️  No non-zero ILI cases found for {country_name}")
        return None
    
    return country_simple


def extract_ari_data(country_code: str, country_name: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Extract ARI data for a specific country.
    
    Args:
        country_code: Country ISO code
        country_name: Country name
        df: DataFrame containing the WHO flu data
        
    Returns:
        DataFrame with Time and ARI_Cases columns, or None if no data
    """
    # Filter for the specific country
    country_df = df[df['COUNTRY_CODE'] == country_code].copy()
    
    if len(country_df) == 0:
        print(f"  ⚠️  No data found for {country_name} ({country_code})")
        return None
    
    # Get "All" age group data (case-insensitive)
    country_all_ages = country_df[country_df['AGEGROUP_CODE'].str.upper() == 'ALL'].copy()
    
    if len(country_all_ages) == 0:
        print(f"  ⚠️  No 'All' age group data found for {country_name}")
        return None
    
    # Select and rename columns
    country_simple = country_all_ages[['ISO_WEEKSTARTDATE', 'ARI_CASE']].copy()
    country_simple.columns = ['Time', 'ARI_Cases']
    
    # Convert to datetime and sort
    country_simple['Time'] = pd.to_datetime(country_simple['Time'])
    country_simple = country_simple.sort_values('Time').reset_index(drop=True)
    
    # Check if we have any non-zero ARI cases
    non_zero_ari = country_simple['ARI_Cases'].fillna(0).astype(float) > 0
    if not non_zero_ari.any():
        print(f"  ⚠️  No non-zero ARI cases found for {country_name}")
        return None
    
    return country_simple


def extract_combined_data(country_code: str, country_name: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Extract combined ILI+ARI data for a country, preferring ILI where available.

    Returns DataFrame with Time, Cases, DataType columns.
    DataType: 0 = ILI, 1 = ARI.
    """
    ili_data = extract_ili_data(country_code, country_name, df)
    ari_data = extract_ari_data(country_code, country_name, df)

    if ili_data is None and ari_data is None:
        return None

    if ili_data is not None and ari_data is None:
        result = ili_data.rename(columns={'ILI_Cases': 'Cases'})
        result['DataType'] = 0
        return result

    if ili_data is None and ari_data is not None:
        result = ari_data.rename(columns={'ARI_Cases': 'Cases'})
        result['DataType'] = 1
        return result

    # Both available — merge and prefer ILI
    merged = ili_data.merge(ari_data, on='Time', how='outer').sort_values('Time').reset_index(drop=True)
    # Raw WHO columns are object dtype (numeric strings, sometimes junk):
    # coerce once so numeric comparisons below are safe (junk -> NaN).
    merged['ILI_Cases'] = pd.to_numeric(merged['ILI_Cases'], errors='coerce')
    merged['ARI_Cases'] = pd.to_numeric(merged['ARI_Cases'], errors='coerce')

    # Parallel-reporting guard (see module docstring): if both indicators are
    # reported simultaneously at very different scales, use the larger-scale
    # one only instead of interleaving the two.
    if int((merged['ILI_Cases'].notna() & merged['ARI_Cases'].notna()).sum()) >= PARALLEL_MIN_BOTH_WEEKS:
        med_ili = merged.loc[merged['ILI_Cases'] > 0, 'ILI_Cases'].median()
        med_ari = merged.loc[merged['ARI_Cases'] > 0, 'ARI_Cases'].median()
        if (pd.notna(med_ili) and pd.notna(med_ari) and med_ili > 0 and med_ari > 0
                and max(med_ili, med_ari) / min(med_ili, med_ari) > SCALE_RATIO_THRESHOLD):
            winner_col = 'ARI_Cases' if med_ari > med_ili else 'ILI_Cases'
            winner = merged[merged[winner_col] > 0]
            if len(winner) >= MIN_WINNER_WEEKS:
                dates = pd.to_datetime(winner['Time'])
                span_years = (dates.max() - dates.min()).days / 365.25
                if span_years >= MIN_WINNER_YEARS:
                    label = 'ARI' if winner_col == 'ARI_Cases' else 'ILI'
                    print(f"  [parallel-reporting guard] {country_name}: keeping {label} only "
                          f"(medians ILI={med_ili:.0f}, ARI={med_ari:.0f}; {len(winner)} weeks, {span_years:.1f} yr span)")
                    result = winner[['Time', winner_col]].rename(columns={winner_col: 'Cases'}).copy()
                    result['DataType'] = 1 if label == 'ARI' else 0
                    return result.reset_index(drop=True)

    merged['Cases'] = merged['ILI_Cases']
    merged['DataType'] = 0

    # Use ARI where ILI is missing or zero
    ari_mask = merged['ILI_Cases'].isna() | (merged['ILI_Cases'] == 0)
    ari_has_data = merged['ARI_Cases'].notna() & (merged['ARI_Cases'] > 0)
    use_ari = ari_mask & ari_has_data
    merged.loc[use_ari, 'Cases'] = merged.loc[use_ari, 'ARI_Cases']
    merged.loc[use_ari, 'DataType'] = 1

    # Drop rows where neither has data
    merged = merged[merged['Cases'].notna() & (merged['Cases'] > 0)]

    return merged[['Time', 'Cases', 'DataType']].reset_index(drop=True)


def extract_country_data_combined(country_name: str, input_file: str = "data/who_flu_data.csv",
                                  output_dir: str = "data/extracted_data") -> Optional[str]:
    """Extract combined ILI+ARI data for a country. Returns path to CSV or None."""
    print(f"Processing {country_name} (combined mode)...")

    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        print(f"  Error reading input file: {e}")
        return None

    country_code, full_country_name = find_country_code_and_name(country_name, df)
    if country_code is None:
        print(f"  Could not find country: {country_name}")
        return None

    print(f"  Found: {full_country_name} ({country_code})")

    combined = extract_combined_data(country_code, full_country_name, df)
    if combined is None:
        print(f"  No usable data found for {full_country_name}")
        return None

    output_file = os.path.join(output_dir, f"{full_country_name.replace(' ', '_')}_combined.csv")
    combined.to_csv(output_file, index=False)

    n_ili = int((combined['DataType'] == 0).sum())
    n_ari = int((combined['DataType'] == 1).sum())
    print(f"  Extracted combined data: {len(combined)} records ({n_ili} ILI, {n_ari} ARI)")
    print(f"     Date range: {combined['Time'].min()} to {combined['Time'].max()}")
    return output_file


def extract_country_data_adaptive(country_name: str, input_file: str = "data/who_flu_data.csv",
                                 output_dir: str = "data/extracted_data") -> Optional[str]:
    """
    Extract data for a country, using ILI if available, otherwise ARI.
    
    Args:
        country_name: Country name to extract data for
        input_file: Path to WHO flu data CSV
        output_dir: Directory to save extracted files
        
    Returns:
        Path to extracted CSV file, or None if extraction failed
    """
    print(f"🔍 Processing {country_name}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the full dataset
    try:
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        print(f"  ❌ Error reading input file: {e}")
        return None
    
    # Find country code and full name
    country_code, full_country_name = find_country_code_and_name(country_name, df)
    
    if country_code is None:
        print(f"  ❌ Could not find country: {country_name}")
        return None
    
    print(f"  📍 Found: {full_country_name} ({country_code})")
    
    # Try ILI data first
    ili_data = extract_ili_data(country_code, full_country_name, df)
    
    if ili_data is not None:
        # Save ILI data
        output_file = os.path.join(output_dir, f"{full_country_name.replace(' ', '_')}_ili.csv")
        ili_data.to_csv(output_file, index=False)
        print(f"  ✅ Extracted ILI data: {len(ili_data)} records")
        print(f"     Date range: {ili_data['Time'].min().date()} to {ili_data['Time'].max().date()}")
        return output_file
    
    # Fall back to ARI data
    print(f"  🔄 Falling back to ARI data...")
    ari_data = extract_ari_data(country_code, full_country_name, df)
    
    if ari_data is not None:
        # Save ARI data
        output_file = os.path.join(output_dir, f"{full_country_name.replace(' ', '_')}_ari.csv")
        ari_data.to_csv(output_file, index=False)
        print(f"  ✅ Extracted ARI data: {len(ari_data)} records")
        print(f"     Date range: {ari_data['Time'].min().date()} to {ari_data['Time'].max().date()}")
        return output_file
    
    print(f"  ❌ No usable data found for {full_country_name}")
    return None


def main():
    """
    Main function for command-line usage.
    """
    parser = argparse.ArgumentParser(
        description="Extract country data with ILI/ARI fallback from WHO FLUMART dataset"
    )
    parser.add_argument(
        "--country",
        default=None,
        help="Country name to extract data for"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract all countries from training_countries.json"
    )
    parser.add_argument(
        "--countries_file",
        default="data/training_countries.json",
        help="JSON file with training_countries list (used with --all)"
    )
    parser.add_argument(
        "--input",
        default="data/who_flu_data.csv",
        help="Path to WHO FLUMART CSV file (default: data/who_flu_data.csv)"
    )
    parser.add_argument(
        "--output_dir",
        default="data/extracted_data",
        help="Directory to save extracted files (default: data/extracted_data)"
    )
    parser.add_argument(
        "--mode",
        choices=["combined", "adaptive"],
        default="combined",
        help="Extraction mode: 'combined' merges ILI+ARI with DataType indicator (default), "
             "'adaptive' picks ILI or ARI only"
    )

    args = parser.parse_args()

    if not args.country and not args.all:
        parser.error("Either --country or --all is required")

    if args.all:
        import json
        with open(args.countries_file, 'r') as f:
            countries = json.load(f)['training_countries']

        # Load WHO data once
        print(f"Loading WHO data from {args.input}...")
        df = pd.read_csv(args.input, low_memory=False)
        os.makedirs(args.output_dir, exist_ok=True)

        total = len(countries)
        ok, fail = 0, 0
        for i, country_name in enumerate(countries, 1):
            country_code, full_name = find_country_code_and_name(country_name, df)
            if country_code is None:
                print(f"  [{i}/{total}] SKIP {country_name} (not found)")
                fail += 1
                continue

            if args.mode == "combined":
                combined = extract_combined_data(country_code, full_name, df)
                if combined is not None:
                    output_file = os.path.join(args.output_dir, f"{full_name.replace(' ', '_')}_combined.csv")
                    combined.to_csv(output_file, index=False)
                    n_ili = int((combined['DataType'] == 0).sum())
                    n_ari = int((combined['DataType'] == 1).sum())
                    print(f"  [{i}/{total}] {full_name}: {len(combined)} records ({n_ili} ILI, {n_ari} ARI)")
                    ok += 1
                else:
                    print(f"  [{i}/{total}] FAIL {full_name} (no data)")
                    fail += 1
            else:
                result = extract_country_data_adaptive(country_name, args.input, args.output_dir)
                if result:
                    ok += 1
                else:
                    fail += 1

        print(f"\nDone: {ok}/{total} extracted, {fail} failed")
    else:
        if args.mode == "combined":
            result = extract_country_data_combined(args.country, args.input, args.output_dir)
        else:
            result = extract_country_data_adaptive(args.country, args.input, args.output_dir)

        if result:
            print(f"\nSuccess! Data saved to: {result}")
        else:
            print(f"\nFailed to extract data for {args.country}")
            exit(1)


if __name__ == "__main__":
    main()