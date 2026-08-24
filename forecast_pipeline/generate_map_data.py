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

# The WHO feed reports the UK's devolved administrations as separate series
# (and the 110m map only has a single GB polygon), so the site shows the UK
# as one country: the map entry below is built by summing the component
# series (see combine_uk_details in generate_country_details.py, which
# writes details/GB.json).
UK_PART_NAMES = {
    'united_kingdom,_england',
    'united_kingdom,_scotland',
    'united_kingdom,_northern_ireland',
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
        'united_republic_of_tanzania': 'TZ'
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


def get_value_for_week(detail, week_str):
    """Value for an exact weekly-grid date from a country detail dict:
    the reported (historical) value if one exists for that week, otherwise
    the model's forecast for that week.

    Returns (value, source) with source 'actual' or 'forecast', or
    (None, None) if the series has neither for that week (e.g. the model's
    4-week horizon does not reach it).
    """
    for p in detail['points']:
        if p['date'] == week_str:
            if p['historical'] is not None:
                return float(p['historical']), 'actual'
            if p['forecast'] is not None:
                # Clamp at zero: the model mean can dip slightly negative
                # on low-activity series (display artifact, not a real
                # forecast). Actuals are never clamped.
                return max(0.0, float(p['forecast'])), 'forecast'
    return None, None


def process_countries(current_week_str, future_weeks):
    """Build the map entries.

    current_week_str: the nowcast week — the calendar's current ISO week
        (see __main__). Each country shows its ACTUAL value for that week
        if one has been reported, otherwise its FORECAST for it (only
        exists within the 4-week horizon).
    future_weeks: the 4 Mondays after that — the selectable forecast
        weeks in the map dropdown (always forecasts).
    """
    name_to_code = load_mappings()
    map_data = []
    today = datetime.now()
    future_set = set(future_weeks)
    uk_part_dirs = []

    country_dirs = glob.glob(os.path.join(RESULTS_DIR, '*'))
    print(f"Processing {len(country_dirs)} directories...")

    for country_dir in country_dirs:
        if not os.path.isdir(country_dir):
            continue

        country_folder_name = os.path.basename(country_dir)
        normalized_name = country_folder_name.lower()

        # UK sub-regions are combined into a single United Kingdom entry
        # (built below); skip them in the per-country loop.
        if normalized_name in UK_PART_NAMES:
            uk_part_dirs.append(country_dir)
            continue

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
                    "value_source": None,
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

            # Default view (nowcast): this week's actual if reported,
            # otherwise the forecast for this week.
            current_value, current_source = get_value_for_week(detail, current_week_str)
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

            # Per-week forecast values for the selectable future weeks
            # (same z-score scale as the current value). Forecasts are only
            # offered up to 4 weeks (STALE_AFTER_DAYS) after the country's
            # last data point — the model's evaluated horizon; beyond that
            # the week isn't offered for that country (gray).
            forecast_weeks = []
            for p in detail['points']:
                if p['date'] not in future_set or p['forecast'] is None:
                    continue
                if (datetime.strptime(p['date'], '%Y-%m-%d') - last_obs).days > STALE_AFTER_DAYS:
                    continue
                # Clamp at zero: the model mean can go slightly negative on
                # low-activity series (display artifact, not a real forecast).
                v = max(0.0, float(p['forecast']))
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
                "value_source": current_source,
                "zscore": round(float(zscore), 2),
                "score": float(score),
                "status": status,
                "stale": False,
                "last_update": last_obs_str,
                "forecast_weeks": forecast_weeks,
            })

        except Exception as e:
            print(f"Error reading {country_folder_name}: {e}")

    # --- United Kingdom (combined from England + Scotland + N. Ireland) ----
    gb_detail_path = os.path.join(DETAILS_DIR, 'GB.json')
    if os.path.exists(gb_detail_path):
        with open(gb_detail_path, 'r') as f:
            gb_detail = json.load(f)

        # Staleness: the combined series is as fresh as its freshest part.
        last_obs = None
        for part_dir in uk_part_dirs:
            part_path = find_extracted_file(os.path.basename(part_dir))
            part_obs = get_last_observation_date(part_path) if part_path else None
            if part_obs is not None:
                last_obs = part_obs if last_obs is None else max(last_obs, part_obs)
        last_obs_str = last_obs.strftime('%Y-%m-%d') if last_obs is not None else None
        stale = last_obs is None or (today - last_obs).days >= STALE_AFTER_DAYS

        if stale:
            map_data.append({
                "id": "GB",
                "numeric": 826,
                "name": gb_detail.get('country', 'United Kingdom'),
                "data_type": gb_detail.get('data_type', 'ILI'),
                "value": None,
                "value_source": None,
                "zscore": None,
                "score": None,
                "status": "stale",
                "stale": True,
                "last_update": last_obs_str,
                "forecast_weeks": [],
            })
        else:
            current_value, current_source = get_value_for_week(gb_detail, current_week_str)
            if current_value is not None:
                # Z-score baseline: the combined series' full historical
                # context (same spirit as per-country, which uses each
                # component's full context window).
                hist_vals = [p['historical'] for p in gb_detail['points'] if p['historical'] is not None]
                hist_mean = float(np.mean(hist_vals)) if hist_vals else 0.0
                hist_std = float(np.std(hist_vals)) if hist_vals else 0.0

                if hist_std > 0:
                    zscore = (current_value - hist_mean) / hist_std
                else:
                    zscore = 0.0
                score = max(0.0, min(1.0, zscore / 2.0))
                status = "high" if zscore >= 1.0 else "low"

                forecast_weeks = []
                for p in gb_detail['points']:
                    if p['date'] not in future_set or p['forecast'] is None:
                        continue
                    if (datetime.strptime(p['date'], '%Y-%m-%d') - last_obs).days > STALE_AFTER_DAYS:
                        continue
                    v = max(0.0, float(p['forecast']))
                    wz = (v - hist_mean) / hist_std if hist_std > 0 else 0.0
                    forecast_weeks.append({
                        "date": p['date'],
                        "value": round(v, 1),
                        "zscore": round(float(wz), 2),
                        "score": float(max(0.0, min(1.0, wz / 2.0))),
                        "status": "high" if wz >= 1.0 else "low",
                    })

                map_data.append({
                    "id": "GB",
                    "numeric": 826,
                    "name": gb_detail.get('country', 'United Kingdom'),
                    "data_type": gb_detail.get('data_type', 'ILI'),
                    "value": round(float(current_value), 1),
                    "value_source": current_source,
                    "zscore": round(float(zscore), 2),
                    "score": float(score),
                    "status": status,
                    "stale": False,
                    "last_update": last_obs_str,
                    "forecast_weeks": forecast_weeks,
                })
            else:
                print("UK: no current value available, skipping")

    return map_data

if __name__ == "__main__":
    # The dashboard's default view is the NOWCAST: the calendar's current
    # ISO week (labeled by its Monday). Each country shows its actual
    # value for that week if one has been reported yet — the WHO feed
    # fills in during the week as countries report — otherwise the
    # model's forecast for it. The selectable future weeks are always
    # forecasts (up to 4 weeks ahead, the model's horizon).
    today = datetime.now()
    current_week = (today - timedelta(days=today.weekday())).strftime('%Y-%m-%d')
    future_weeks = [(datetime.strptime(current_week, '%Y-%m-%d') + timedelta(days=7 * i)).strftime('%Y-%m-%d') for i in range(1, 5)]

    data = process_countries(current_week, future_weeks)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    display_weeks = [current_week] + future_weeks

    out = {
        "generated_at": today.strftime('%Y-%m-%d %H:%M'),
        "default_week": current_week,
        "weeks": display_weeks,
        "countries": data,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote data for {len(data)} countries (nowcast week: {current_week}, "
          f"future weeks: {future_weeks}) to {OUTPUT_FILE}")

    # --- Country list for the methodology page ------------------------------
    # One row per country that has a detail JSON (all trained countries),
    # so the website can link to every forecast — including the ones that
    # can't be colored on the map.
    details_dir = os.path.join(BASE_DIR, '..', 'frontend', 'public', 'data', 'details')
    display_names = {
        'Kosovo (in accordance with UN Security Council resolution 1244 (1999))': 'Kosovo',
        'Netherlands (Kingdom of the)': 'Netherlands',
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
