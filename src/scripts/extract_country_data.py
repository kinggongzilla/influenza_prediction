#!/usr/bin/env python3
"""
Extract ILI data for a given country from the WHO FLUMART dataset
and prepare it for use with TiRex time series forecasting.

Usage:
    python extract_country_data.py --country <COUNTRY_NAME_OR_CODE> --input <INPUT_CSV> --output <OUTPUT_CSV>

Example:
    python extract_country_data.py --country Austria --input who_flu_data.csv --output austria_ili.csv
    python extract_country_data.py --country AUT --input who_flu_data.csv --output austria_ili.csv
"""
import pandas as pd
import argparse

def extract_country_data(country, input_file, output_file):
    """
    Extract ILI data for a specific country from WHO FLUMART dataset.
    
    Args:
        country: Country name or ISO code (e.g., 'Austria' or 'AUT')
        input_file: Path to the WHO FLUMART CSV file
        output_file: Path to save the extracted data
    """
    # Read the full dataset
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    
    # Filter for the specified country (can be name or code)
    country_df = df[df['COUNTRY_CODE'] == country.upper()].copy()
    
    if len(country_df) == 0:
        # Try exact name match first (case-insensitive)
        country_df = df[df['COUNTRY_AREA_TERRITORY'].str.lower() == country.lower()].copy()
        
        if len(country_df) == 0:
            # Try substring match as fallback, but be more careful
            # Use word boundary matching to avoid partial matches
            import re
            try:
                pattern = rf'\\b{re.escape(country)}\\b'
                name_matches = df[df['COUNTRY_AREA_TERRITORY'].str.contains(pattern, case=False, na=False, regex=True)].copy()
                
                if len(name_matches) > 0:
                    country_df = name_matches.copy()
                elif len(country) >= 3:  # Only try fuzzy matching for longer names
                    # Try contains match but prioritize exact word matches
                    contains_matches = df[df['COUNTRY_AREA_TERRITORY'].str.contains(country, case=False, na=False)].copy()
                    if len(contains_matches) == 1:
                        country_df = contains_matches.copy()
                    elif len(contains_matches) > 1:
                        print(f"Warning: Multiple countries match '{country}':")
                        for idx, row in contains_matches.iterrows():
                            print(f"  - {row['COUNTRY_AREA_TERRITORY']} ({row['COUNTRY_CODE']})")
                        # Use the first match
                        country_df = contains_matches.iloc[[0]].copy()
            except Exception as e:
                print(f"Warning: Regex matching failed: {e}")
        
        if len(country_df) == 0:
            print(f"Error: No data found for country '{country}'")
            print("Available countries (first 50):")
            print(df['COUNTRY_AREA_TERRITORY'].unique()[:50])
            
            # Suggest similar country names
            similar = df[df['COUNTRY_AREA_TERRITORY'].str.contains(country[:3], case=False, na=False)]['COUNTRY_AREA_TERRITORY'].unique()[:5]
            if len(similar) > 0:
                print(f"\\nDid you mean one of these?")
                for name in similar:
                    print(f"  - {name}")
            return False
    
    # We want the "All" age group data (AGEGROUP_CODE == 'ALL' or 'All' or other variants)
    # Use case-insensitive matching to catch all variations
    country_all_ages = country_df[country_df['AGEGROUP_CODE'].str.upper() == 'ALL'].copy()
    
    # Select only the columns we need
    # ISO_WEEKSTARTDATE is the date, ILI_CASE is the ILI cases
    country_simple = country_all_ages[['ISO_WEEKSTARTDATE', 'ILI_CASE']].copy()
    
    # Rename columns to match the format expected by tirex_inference.py
    country_simple.columns = ['Time', 'ILI_Cases']
    
    # Sort by date
    country_simple = country_simple.sort_values('Time')
    
    # Reset index
    country_simple = country_simple.reset_index(drop=True)
    
    # Save to CSV
    country_simple.to_csv(output_file, index=False)
    
    country_name = country_df['COUNTRY_AREA_TERRITORY'].iloc[0]
    print(f"✓ Successfully extracted {country_name} ILI data")
    print(f"  Records: {len(country_simple)}")
    print(f"  Date range: {country_simple['Time'].min()} to {country_simple['Time'].max()}")
    print(f"  Saved to: {output_file}")
    
    return True

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Extract ILI data for a country from WHO FLUMART dataset"
    )
    parser.add_argument(
        "--country",
        required=True,
        help="Country name or ISO code (e.g., 'Austria' or 'AUT')"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the WHO FLUMART CSV file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the extracted data"
    )
    
    args = parser.parse_args()
    
    # Extract the data
    success = extract_country_data(args.country, args.input, args.output)
    
    if not success:
        exit(1)
