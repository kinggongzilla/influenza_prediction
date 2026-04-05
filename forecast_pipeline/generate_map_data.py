import json
import os
import glob
import pandas as pd
import numpy as np
import pycountry

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'country_predictions')
BOUNDING_BOXES_FILE = os.path.join(BASE_DIR, 'data', 'country_bounding_boxes.json')
OUTPUT_FILE = os.path.join(BASE_DIR, '..', 'frontend', 'public', 'data', 'influenza_status.json')

def load_mappings():
    with open(BOUNDING_BOXES_FILE, 'r') as f:
        bboxes = json.load(f)
    
    name_to_code = {}
    for code, data in bboxes.items():
        name = data[0]
        name_to_code[name.lower()] = code
        
    # Manual overrides
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
        'united_republic_of_tanzania': 'TZ'
    }
    name_to_code.update(overrides)
    return name_to_code

def process_countries():
    name_to_code = load_mappings()
    map_data = []
    
    country_dirs = glob.glob(os.path.join(RESULTS_DIR, '*'))
    print(f"Processing {len(country_dirs)} directories...")
    
    for country_dir in country_dirs:
        if not os.path.isdir(country_dir):
            continue
            
        country_folder_name = os.path.basename(country_dir)
        normalized_name = country_folder_name.lower()
        
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
            continue
            
        csv_pattern = os.path.join(country_dir, '*_forecast_results.csv')
        files = glob.glob(csv_pattern)
        if not files:
            if os.path.exists(os.path.join(country_dir, 'forecast_results.csv')):
                files = [os.path.join(country_dir, 'forecast_results.csv')]
            else:
                continue
            
        csv_file = files[0]
        
        try:
            df = pd.read_csv(csv_file)
            context_vals = df['context'].dropna()
            forecast_vals = df['mean_forecast'].dropna()
            
            if context_vals.empty or forecast_vals.empty:
                continue
                
            current_pred = forecast_vals.iloc[0]
            threshold = context_vals.quantile(0.85)
            status = "high" if current_pred > threshold else "low"
            
            # Calculate percentile rank (0 to 1)
            # This represents how extreme the current prediction is compared to history
            score = (context_vals < current_pred).mean()

            # Get Numeric Code for Map Matching
            country_obj = pycountry.countries.get(alpha_2=code)
            numeric_code = country_obj.numeric if country_obj else None
            
            # Remove leading zeros if any, as some topojsons behave differently
            # Usually topojson uses string "840" or "004".
            
            # Detect data type from extracted data file
            data_type = "ILI"
            extracted_dir = os.path.join(BASE_DIR, 'data', 'extracted_data')
            for ext_suffix in ['_combined.csv', '_ari.csv']:
                ext_path = os.path.join(extracted_dir, f"{country_folder_name}{ext_suffix}")
                if os.path.exists(ext_path):
                    if ext_suffix == '_combined.csv':
                        try:
                            ext_df = pd.read_csv(ext_path)
                            if 'DataType' in ext_df.columns and int(ext_df['DataType'].iloc[-1]) == 1:
                                data_type = "ARI"
                        except Exception:
                            pass
                    elif ext_suffix == '_ari.csv':
                        data_type = "ARI"
                    break

            map_data.append({
                "id": code,
                "numeric": numeric_code,
                "name": country_folder_name.replace('_', ' '),
                "data_type": data_type,
                "value": float(current_pred),
                "threshold": float(threshold),
                "score": float(score),
                "status": status
            })
            
        except Exception as e:
            print(f"Error reading {country_folder_name}: {e}")
            
    return map_data

if __name__ == "__main__":
    data = process_countries()
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Wrote data for {len(data)} countries to {OUTPUT_FILE}")