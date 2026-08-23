import json
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pycountry

# Surveillance data at least this old is considered stale: the model forecasts
# 4 weeks ahead, so a prediction anchored to data that is 4+ weeks old no longer
# covers "today" and must not be shown.
STALE_AFTER_DAYS = 4 * 7

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'country_predictions')
EXTRACTED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'extracted_data')
BOUNDING_BOXES_FILE = os.path.join(BASE_DIR, 'data', 'country_bounding_boxes.json')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'frontend', 'public', 'data', 'details')

def load_mappings():
    with open(BOUNDING_BOXES_FILE, 'r') as f:
        bboxes = json.load(f)
    
    name_to_code = {}
    for code, data in bboxes.items():
        name = data[0]
        name_to_code[name.lower()] = code
        
    # Manual overrides (copied from generate_map_data.py)
    overrides = {
        'bolivia_(plurinational_state_of)': 'BO',
        'iran_(islamic_republic_of)': 'IR',
        'russian_federation': 'RU',
        'united_states_of_america': 'US',
        'venezuela_(bolivarian_republic_of)': 'VE',
        'viet_nam': 'VN',
        'korea_(republic_of)': 'KR',
        'democratic_people\'s_republic_of_korea': 'KP',
        'türkiye': 'TR',
        'united_kingdom_of_great_britain_and_northern_ireland': 'GB',
        'syrian_arab_republic': 'SY',
        'republic_of_moldova': 'MD',
        'lao_people\'s_democratic_republic': 'LA',
        'micronesia_(federated_states_of)': 'FM',
        'united_republic_of_tanzania': 'TZ',
        # Countries that passed quality screening but have no region in the
        # 110m world-map geography (small-island territories, UK sub-regions,
        # non-standard names). They get their own codes so every trained
        # country has a detail page, even though it can't be colored on the map.
        'american_samoa': 'AS',
        'czechia': 'CZ',
        'côte_d’ivoire': 'CI',
        'french_polynesia': 'PF',
        'kiribati': 'KI',
        'kosovo_(in_accordance_with_un_security_council_resolution_1244_(1999))': 'XK',
        'malta': 'MT',
        'netherlands_(kingdom_of_the)': 'NL',
        'northern_mariana_islands': 'MP',
        'north_macedonia': 'MK',
        'samoa': 'WS',
        'singapore': 'SG',
        'tonga': 'TO',
        'united_kingdom,_england': 'ENG',
        'united_kingdom,_northern_ireland': 'NIR',
        'united_kingdom,_scotland': 'SCT',
        'wallis_and_futuna': 'WF'
    }
    name_to_code.update(overrides)
    return name_to_code

def find_extracted_file(country_folder_name):
    """Find extracted data file. Returns (path, data_type) tuple."""
    # Try combined first, then ILI, then ARI
    for suffix, dtype in [("_combined.csv", "combined"), ("_ili.csv", "ILI"), ("_ari.csv", "ARI")]:
        for name_variant in [country_folder_name, country_folder_name.replace('_', ' ')]:
            path = os.path.join(EXTRACTED_DATA_DIR, f"{name_variant}{suffix}")
            if os.path.exists(path):
                if dtype == "combined":
                    # Detect actual type from last rows' DataType column
                    try:
                        import pandas as pd
                        df = pd.read_csv(path)
                        if 'DataType' in df.columns:
                            last_type = int(df['DataType'].iloc[-1])
                            dtype = "ARI" if last_type == 1 else "ILI"
                    except Exception:
                        dtype = "ILI"
                return path, dtype

    # Try globbing
    files = glob.glob(os.path.join(EXTRACTED_DATA_DIR, f"*{country_folder_name}*.csv"))
    if files:
        dtype = "ARI" if "_ari" in files[0] else "ILI"
        return files[0], dtype

    return None, None

def process_countries():
    name_to_code = load_mappings()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    country_dirs = glob.glob(os.path.join(RESULTS_DIR, '*'))
    print(f"Processing {len(country_dirs)} directories...")
    
    count = 0
    for country_dir in country_dirs:
        if not os.path.isdir(country_dir):
            continue
            
        country_folder_name = os.path.basename(country_dir)
        normalized_name = country_folder_name.lower()
        
        # Get ISO Code
        code = name_to_code.get(normalized_name)
        if not code:
            code = name_to_code.get(normalized_name.replace('_', ' '))
        
        if not code:
             if 'congo' in normalized_name:
                 if 'democratic' in normalized_name:
                     code = 'CD'
                 else:
                     code = 'CG'
        
        if not code:
            print(f"Skipping {country_folder_name}: No code found")
            continue

        # Find Results CSV
        csv_pattern = os.path.join(country_dir, '*_forecast_results.csv')
        files = glob.glob(csv_pattern)
        if not files:
            if os.path.exists(os.path.join(country_dir, 'forecast_results.csv')):
                files = [os.path.join(country_dir, 'forecast_results.csv')]
            else:
                print(f"Skipping {country_folder_name}: No result CSV")
                continue
        results_csv_file = files[0]

        # Find Extracted Data CSV (for dates)
        extracted_file, data_type = find_extracted_file(country_folder_name)
        start_date = None
        last_date = None
        if extracted_file:
            try:
                extracted_df = pd.read_csv(extracted_file)
                if 'Time' in extracted_df.columns:
                    parsed_times = pd.to_datetime(extracted_df['Time'], errors='coerce')
                    start_date = parsed_times.dropna().min()
                    last_date = parsed_times.dropna().max()
            except Exception as e:
                print(f"Error reading extracted file for {country_folder_name}: {e}")

        if not start_date:
            # Fallback: Assume recent data if we can't find file, but better to skip or use dummy
            # Let's use a dummy start date or try to infer. 
            # For now, let's skip if we can't align dates to avoid misleading charts
            print(f"Skipping {country_folder_name}: Could not determine start date")
            continue

        # Stale data (e.g. stopped reporting, or seasonal gap > 4 weeks) -> no forecast is shown
        today = datetime.now()
        stale = last_date is None or (today - last_date).days >= STALE_AFTER_DAYS
        last_update = last_date.strftime('%Y-%m-%d') if last_date is not None else None

        try:
            df = pd.read_csv(results_csv_file)
            
            # Prepare data for JSON
            output_data = {
                "country": country_folder_name.replace('_', ' '),
                "id": code,
                "data_type": data_type or "ILI",
                "stale": bool(stale),
                "last_update": last_update,
                "points": []
            }
            
            # Iterate through rows
            # 'context' is history, 'mean_forecast' is prediction
            # They might overlap in the CSV structure depending on how it was saved,
            # usually context is filled then forecast starts.
            
            current_date = start_date
            
            for index, row in df.iterrows():
                point = {
                    "date": current_date.strftime('%Y-%m-%d'),
                    "historical": None,
                    "forecast": None,
                    "lower": None,      # q0.1 (90% interval)
                    "upper": None,      # q0.9 (90% interval)
                    "lower_50": None,   # q0.25 (50% interval)
                    "upper_50": None    # q0.75 (50% interval)
                }

                # Context (History)
                if pd.notna(row.get('context')):
                    point["historical"] = float(row['context'])

                # Forecast
                if pd.notna(row.get('mean_forecast')):
                    point["forecast"] = float(row['mean_forecast'])

                    # 90% interval (q0.1 - q0.9)
                    if pd.notna(row.get('quantile_0.1')):
                        point["lower"] = float(row['quantile_0.1'])
                    if pd.notna(row.get('quantile_0.9')):
                        point["upper"] = float(row['quantile_0.9'])

                    # 50% interval (interpolated q0.25 and q0.75)
                    if pd.notna(row.get('quantile_0.2')) and pd.notna(row.get('quantile_0.3')):
                        point["lower_50"] = (float(row['quantile_0.2']) + float(row['quantile_0.3'])) / 2
                    if pd.notna(row.get('quantile_0.7')) and pd.notna(row.get('quantile_0.8')):
                        point["upper_50"] = (float(row['quantile_0.7']) + float(row['quantile_0.8'])) / 2

                output_data["points"].append(point)

                # Increment date (Weekly data)
                current_date += timedelta(weeks=1)

            # Drop forecasts for stale data: a prediction anchored to old
            # observations is not a current prediction and must not be shown.
            if stale:
                for point in output_data["points"]:
                    point["forecast"] = None
                    point["lower"] = None
                    point["upper"] = None
                    point["lower_50"] = None
                    point["upper_50"] = None

            # Save JSON
            with open(os.path.join(OUTPUT_DIR, f"{code}.json"), 'w') as f:
                json.dump(output_data, f, indent=2)
            
            count += 1
            
        except Exception as e:
            print(f"Error processing {country_folder_name}: {e}")
            
    print(f"Successfully generated details for {count} countries.")

if __name__ == "__main__":
    process_countries()
