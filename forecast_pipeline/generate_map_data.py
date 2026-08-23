import json
import os
import glob
import pandas as pd
import numpy as np
import pycountry
from datetime import datetime, timedelta

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'country_predictions')
DETAILS_DIR = os.path.join(BASE_DIR, '..', 'frontend', 'public', 'data', 'details')
EXTRACTED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'extracted_data')
BOUNDING_BOXES_FILE = os.path.join(BASE_DIR, 'data', 'country_bounding_boxes.json')
OUTPUT_FILE = os.path.join(BASE_DIR, '..', 'frontend', 'public', 'data', 'influenza_status.json')

# Surveillance data at least this old is not shown as a current prediction:
# the model forecasts 4 weeks ahead, so a forecast anchored to data that is
# 4+ weeks old no longer covers "today".
STALE_AFTER_DAYS = 4 * 7

# UK sub-regions: the 110m world map only has a single GB polygon, so the
# frontend layers dedicated polygons over it (public/data/uk_subregions.json,
# built by make_uk_subregions.py from OSM admin-4 boundaries). Pseudo ISO
# numeric codes — real 3166-1 numerics are 3 digits, so these can't collide.
UK_SUBREGIONS = {
    'ENG': ('82601', 'England (UK)'),
    'SCT': ('82602', 'Scotland (UK)'),
    'NIR': ('82603', 'Northern Ireland (UK)'),
}


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
        'united_republic_of_tanzania': 'TZ',
        # UK sub-regions: WHO reports them individually; the map layers real
        # polygons for them over the single GB base polygon (see UK_SUBREGIONS).
        'united_kingdom,_england': 'ENG',
        'united_kingdom,_scotland': 'SCT',
        'united_kingdom,_northern_ireland': 'NIR'
    }
    name_to_code.update(overrides)
    return name_to_code


def find_extracted_file(country_folder_name):
    """Locate the extracted CSV for a results folder name (combined > ILI > ARI)."""
    for suffix in ('_combined.csv', '_ili.csv', '_ari.csv'):
        for variant in (country_folder_name, country_folder_name.replace('_', ' ')):
            path = os.path.join(EXTRACTED_DATA_DIR, f"{variant}{suffix}")
            if os.path.exists(path):
                return path
    return None


def get_last_observation_date(extracted_path):
    """Last date in the extracted CSV, or None if unavailable."""
    try:
        df = pd.read_csv(extracted_path)
        for col in ('Time', 'Date'):
            if col in df.columns:
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notna().any():
                    return parsed.max()
    except Exception:
        pass
    return None


def get_value_for_today(detail, today):
    """From a country detail dict, return the best current value:
    the most recent historical observation, or if the forecast covers today,
    the forecast value closest to today."""

    points = detail['points']
    today_str = today.strftime('%Y-%m-%d')

    # Find the most recent historical observation
    last_hist_val = None
    last_hist_date = None
    for p in reversed(points):
        if p['historical'] is not None:
            last_hist_val = p['historical']
            last_hist_date = p['date']
            break

    # Find the forecast closest to today
    best_fc_val = None
    best_fc_dist = float('inf')
    for p in points:
        if p['forecast'] is not None:
            dist = abs((datetime.strptime(p['date'], '%Y-%m-%d') - today).days)
            if dist < best_fc_dist:
                best_fc_dist = dist
                best_fc_val = p['forecast']

    # Prefer forecast at today if available (within 2 weeks), otherwise
    # fall back to the most recent historical observation
    if best_fc_val is not None and best_fc_dist <= 14:
        return best_fc_val

    if last_hist_val is not None:
        return last_hist_val

    # Fallback: forecast even if far from today
    return best_fc_val


def process_countries():
    name_to_code = load_mappings()
    map_data = []
    today = datetime.now()

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

            if context_vals.empty:
                continue

            # Get current value from detail JSON (historical if recent, else forecast @ today)
            detail_path = os.path.join(DETAILS_DIR, f"{code}.json")
            if not os.path.exists(detail_path):
                continue

            # Staleness check: predictions anchored to old data (e.g. a country that
            # stopped reporting years ago) must not be displayed as current activity.
            extracted_path = find_extracted_file(country_folder_name)
            last_obs = get_last_observation_date(extracted_path) if extracted_path else None
            last_obs_str = last_obs.strftime('%Y-%m-%d') if last_obs is not None else None
            stale = last_obs is None or (today - last_obs).days >= STALE_AFTER_DAYS

            # Get Numeric Code for Map Matching
            if code in UK_SUBREGIONS:
                numeric_code = UK_SUBREGIONS[code][0]
                display_name = UK_SUBREGIONS[code][1]
            else:
                country_obj = pycountry.countries.get(alpha_2=code)
                numeric_code = country_obj.numeric if country_obj else None
                display_name = country_folder_name.replace('_', ' ')

            # Detect data type from extracted data file
            data_type = "ILI"
            if extracted_path:
                for ext_suffix in ['_combined.csv', '_ari.csv']:
                    if extracted_path.endswith(ext_suffix):
                        if ext_suffix == '_combined.csv':
                            try:
                                ext_df = pd.read_csv(extracted_path)
                                if 'DataType' in ext_df.columns and int(ext_df['DataType'].iloc[-1]) == 1:
                                    data_type = "ARI"
                            except Exception:
                                pass
                        else:
                            data_type = "ARI"
                        break

            if stale:
                map_data.append({
                    "id": code,
                    "numeric": numeric_code,
                    "name": display_name,
                    "data_type": data_type,
                    "value": None,
                    "zscore": None,
                    "score": None,
                    "status": "stale",
                    "stale": True,
                    "last_update": last_obs_str,
                    "forecast_weeks": [],
                })
                continue

            with open(detail_path, 'r') as f:
                detail = json.load(f)

            current_value = get_value_for_today(detail, today)
            if current_value is None:
                continue

            hist_mean = context_vals.mean()
            hist_std = context_vals.std()

            # Z-score: how many SDs above/below the historical mean
            if hist_std > 0:
                zscore = (current_value - hist_mean) / hist_std
            else:
                zscore = 0.0

            # Map z-score to 0-1 color score:
            #   <= 0 SD  -> 0.0 (green, below/at average)
            #   ~1 SD    -> 0.5 (yellow, moderately above)
            #   >= 2 SD  -> 1.0 (red, unusually high)
            score = max(0.0, min(1.0, zscore / 2.0))

            status = "high" if zscore >= 1.0 else "low"

            # Per-week forecast values so the map can be switched between
            # prediction weeks (same z-score scale as the current value).
            # Only the display window (default week + next 3) is kept in the
            # final JSON; trimming happens in __main__.
            forecast_weeks = []
            for p in detail['points']:
                if p['forecast'] is None:
                    continue
                v = float(p['forecast'])
                wz = (v - hist_mean) / hist_std if hist_std > 0 else 0.0
                forecast_weeks.append({
                    "date": p['date'],
                    "value": round(v, 1),
                    "zscore": round(float(wz), 2),
                    "score": float(max(0.0, min(1.0, wz / 2.0))),
                    "status": "high" if wz >= 1.0 else "low",
                })

            map_data.append({
                "id": code,
                "numeric": numeric_code,
                "name": display_name,
                "data_type": data_type,
                "value": round(float(current_value), 1),
                "zscore": round(float(zscore), 2),
                "score": float(score),
                "status": status,
                "stale": False,
                "last_update": last_obs_str,
                "forecast_weeks": forecast_weeks,
            })

        except Exception as e:
            print(f"Error reading {country_folder_name}: {e}")

    return map_data

if __name__ == "__main__":
    data = process_countries()
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # The map only offers the current prediction week (closest to today) and
    # the next 3 weeks — the 4-week forecast horizon. Everything else is
    # trimmed from the JSON; the full horizon stays on the detail pages.
    all_weeks = sorted({w['date'] for c in data for w in c.get('forecast_weeks', [])})
    today = datetime.now()
    if all_weeks:
        idx = min(
            range(len(all_weeks)),
            key=lambda i: abs((datetime.strptime(all_weeks[i], '%Y-%m-%d') - today).days),
        )
        default_week = all_weeks[idx]
        display_weeks = all_weeks[idx:idx + 4]
        display_set = set(display_weeks)
        for c in data:
            c['forecast_weeks'] = [w for w in c.get('forecast_weeks', []) if w['date'] in display_set]
    else:
        default_week = None
        display_weeks = []

    out = {
        "generated_at": today.strftime('%Y-%m-%d %H:%M'),
        "default_week": default_week,
        "weeks": display_weeks,
        "countries": data,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote data for {len(data)} countries (display weeks: {display_weeks}) to {OUTPUT_FILE}")

    # --- Country list for the methodology page ------------------------------
    # One row per country that has a detail JSON (all trained countries),
    # so the website can link to every forecast — including the ones that
    # can't be colored on the map.
    details_dir = os.path.join(BASE_DIR, '..', 'frontend', 'public', 'data', 'details')
    display_names = {
        'Kosovo (in accordance with UN Security Council resolution 1244 (1999))': 'Kosovo',
        'Netherlands (Kingdom of the)': 'Netherlands',
        'United Kingdom, England': 'England (UK)',
        'United Kingdom, Scotland': 'Scotland (UK)',
        'United Kingdom, Northern Ireland': 'Northern Ireland (UK)',
    }
    map_ids = {c['id'] for c in data}
    rows = []
    for fn in sorted(os.listdir(details_dir)):
        if not fn.endswith('.json'):
            continue
        try:
            d = json.load(open(os.path.join(details_dir, fn)))
        except Exception as e:
            print(f"country_list: skipping {fn}: {e}")
            continue
        since = next((p['date'] for p in d['points'] if p['historical'] is not None), None)
        raw_name = d.get('country') or fn[:-5]
        rows.append({
            'name': display_names.get(raw_name, raw_name),
            'code': d['id'],
            'data_type': d.get('data_type', 'ILI'),
            'since': since,
            'last_update': d.get('last_update'),
            'stale': bool(d.get('stale', False)),
            'on_map': d['id'] in map_ids,
        })
    rows.sort(key=lambda r: r['name'].lower())
    try:
        tc = json.load(open(os.path.join(BASE_DIR, 'data', 'training_countries.json')))
        excluded = tc.get('excluded_from_training', [])
    except Exception:
        excluded = []
    list_out = {
        'generated_at': today.strftime('%Y-%m-%d %H:%M'),
        'total': len(rows),
        'on_map': sum(1 for r in rows if r['on_map']),
        'countries': rows,
        'excluded_from_training': excluded,
    }
    list_file = os.path.join(os.path.dirname(OUTPUT_FILE), 'country_list.json')
    with open(list_file, 'w') as f:
        json.dump(list_out, f, indent=2)
    print(f"Wrote country list ({list_out['total']} countries, {list_out['on_map']} on map) to {list_file}")
