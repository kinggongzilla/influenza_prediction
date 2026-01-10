#!/usr/bin/env python3
"""
Enhanced Country Data Extractor with ILI/ARI Fallback Logic

This script extracts time series data for countries from the WHO FLUMART dataset,
automatically choosing ILI data if available, or falling back to ARI data if ILI cases are zero.

The script is designed to support batch processing for running inference on all available countries.
"""

import pandas as pd
import argparse
import os
import re
from typing import Optional, Tuple
from datetime import datetime


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
        required=True,
        help="Country name to extract data for"
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
    
    args = parser.parse_args()
    
    # Extract data for the specified country
    result = extract_country_data_adaptive(args.country, args.input, args.output_dir)
    
    if result:
        print(f"\n🎉 Success! Data saved to: {result}")
    else:
        print(f"\n💥 Failed to extract data for {args.country}")
        exit(1)


if __name__ == "__main__":
    main()