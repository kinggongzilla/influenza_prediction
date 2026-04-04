"""
Weather Data Fetcher for Time Series Forecasting

This module fetches and processes weather data from the Open-Meteo Historical Weather API
to be used as covariates in time series forecasting models like Chronos2.

Features:
- Fetch historical weather data by geographic coordinates
- Cache weather data locally to minimize API calls
- Convert country names to coordinates using bounding boxes
- Aggregate daily weather data to weekly frequency
- Normalize weather features for model input

API: https://open-meteo.com/en/docs/historical-weather-api
"""

import os
import json
import time
import logging
from pathlib import Path
from functools import wraps
from typing import Optional, Tuple, Dict
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np
import pycountry
from sklearn.preprocessing import StandardScaler
import random


# Rate limiting decorator
def rate_limit(calls_per_second: float = 1.0):
    """
    Decorator to rate limit function calls.

    Args:
        calls_per_second: Maximum number of calls per second
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

# Retry decorator with exponential backoff
def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, 
                      rate_limit_calls: float = 1.0):
    """
    Decorator to retry function calls with exponential backoff and rate limiting.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        rate_limit_calls: Rate limit in calls per second
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Rate limiting
            min_interval = 1.0 / rate_limit_calls
            last_called = [0.0]
            
            for attempt in range(max_retries + 1):
                # Apply rate limiting
                elapsed = time.time() - last_called[0]
                left_to_wait = min_interval - elapsed
                if left_to_wait > 0:
                    time.sleep(left_to_wait)
                
                try:
                    result = func(*args, **kwargs)
                    last_called[0] = time.time()
                    return result
                except Exception as e:
                    if attempt < max_retries:
                        # Exponential backoff with jitter
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                        logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        logging.error(f"Max retries ({max_retries}) exceeded for {func.__name__}: {e}")
                        raise
        return wrapper
    return decorator


def load_country_bounding_boxes() -> dict:
    """
    Load country bounding box coordinates from GitHub repository.

    Downloads and caches the bounding boxes JSON file locally.

    Returns:
        dict: Country code -> [name, [min_lon, min_lat, max_lon, max_lat]]

    Raises:
        RuntimeError: If download fails and no cached file exists
    """
    cache_path = Path("data/country_bounding_boxes.json")

    # Check if cached file exists
    if cache_path.exists():
        print(f"Loading country bounding boxes from cache: {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)

    # Download from GitHub
    url = "https://raw.githubusercontent.com/sandstrom/country-bounding-boxes/master/bounding-boxes.json"
    print(f"Downloading country bounding boxes from {url}")

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        bounding_boxes = response.json()

        # Cache the file
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(bounding_boxes, f, indent=2)

        print(f"Cached bounding boxes to {cache_path}")
        return bounding_boxes

    except Exception as e:
        raise RuntimeError(f"Failed to download country bounding boxes: {e}")


def get_country_coordinates(country_name: str, use_capital: bool = False) -> Tuple[float, float]:
    """
    Convert country name to coordinates.

    Args:
        country_name: Country name (e.g., "Argentina", "Austria")
        use_capital: If True, use capital city coordinates instead of bbox centroid.

    Returns:
        tuple: (latitude, longitude)

    Raises:
        ValueError: If country cannot be found or matched
    """
    # Try capital city coordinates first if requested
    if use_capital:
        capital_path = Path("data/capital_coords.json")
        if capital_path.exists():
            with open(capital_path, "r") as f:
                capitals = json.load(f)
            if country_name in capitals:
                lat, lon = capitals[country_name]
                print(f"Country: {country_name} -> Capital coordinates: ({lat:.4f}, {lon:.4f})")
                return lat, lon

    # Load bounding boxes
    bounding_boxes = load_country_bounding_boxes()

    # Normalize country name to ISO alpha-2 code
    country_code = _normalize_country_name(country_name)

    # Look up bounding box
    if country_code not in bounding_boxes:
        # Try to find similar country names or use fallback coordinates
        similar_countries = [name for name in bounding_boxes.keys() if country_name.lower() in name.lower()]
        if similar_countries:
            # Use the first similar country as fallback
            fallback_code = similar_countries[0]
            print(f"⚠️  Country code '{country_code}' not found, using fallback: {fallback_code}")
            bbox_data = bounding_boxes[fallback_code]
        else:
            # Use a default fallback location (e.g., center of the world)
            print(f"⚠️  Country code '{country_code}' not found, using default coordinates")
            return 0.0, 0.0  # Default to (0,0) - will likely fail weather fetch but won't crash

    bbox_data = bounding_boxes[country_code]
    bbox = bbox_data[1]  # [min_lon, min_lat, max_lon, max_lat]

    # Calculate centroid
    lat, lon = _calculate_centroid(bbox)

    print(f"Country: {country_name} ({country_code}) -> Centroid coordinates: ({lat:.4f}, {lon:.4f})")

    return lat, lon


def _normalize_country_name(country_name: str) -> str:
    """
    Convert various country name formats to ISO alpha-2 code.

    Args:
        country_name: Country name in various formats

    Returns:
        str: ISO alpha-2 country code (e.g., "AR", "AT")

    Raises:
        ValueError: If country cannot be matched
    """
    # Manual mapping for common variations and edge cases
    manual_map = {
        'argentina': 'AR',
        'austria': 'AT',
        'australia': 'AU',
        'canada': 'CA',
        'france': 'FR',
        'germany': 'DE',
        'italy': 'IT',
        'spain': 'ES',
        'united kingdom': 'GB',
        'uk': 'GB',
        'usa': 'US',
        'united states': 'US',
        'netherlands': 'NL',
        'belgium': 'BE',
        'switzerland': 'CH',
        'sweden': 'SE',
        'norway': 'NO',
        'denmark': 'DK',
        'finland': 'FI',
        'poland': 'PL',
        'portugal': 'PT',
        'greece': 'GR',
        'czech republic': 'CZ',
        'czechia': 'CZ',
        'ireland': 'IE',
        'new zealand': 'NZ',
        'japan': 'JP',
        'south korea': 'KR',
        'korea': 'KR',
        'brazil': 'BR',
        'mexico': 'MX',
        'chile': 'CL',
        'colombia': 'CO',
        'marshall islands': 'MH',
        'american samoa': 'AS',
        'anguilla': 'AI',
        'antigua and barbuda': 'AG',
        'aruba': 'AW',
        'bahamas': 'BS',
        'bahrain': 'BH',
        'barbados': 'BB',
        'belarus': 'BY',
        'belize': 'BZ',
        'bermuda': 'BM',
        'bhutan': 'BT',
        'bosnia and herzegovina': 'BA',
        'botswana': 'BW',
        'brunei': 'BN',
        'burundi': 'BI',
        'cabo verde': 'CV',
        'cameroon': 'CM',
        'cayman islands': 'KY',
        'central african republic': 'CF',
        'chad': 'TD',
        'comoros': 'KM',
        'congo': 'CG',
        'cook islands': 'CK',
        'costa rica': 'CR',
        'cote d ivoire': 'CI',
        'curaçao': 'CW',
        'djibouti': 'DJ',
        'dominica': 'DM',
        'dominican republic': 'DO',
        'equatorial guinea': 'GQ',
        'eritrea': 'ER',
        'eswatini': 'SZ',
        'fiji': 'FJ',
        'gabon': 'GA',
        'gambia': 'GM',
        'georgia': 'GE',
        'ghana': 'GH',
        'grenada': 'GD',
        'guatemala': 'GT',
        'guinea': 'GN',
        'guinea-bissau': 'GW',
        'guyana': 'GY',
        'haiti': 'HT',
        'honduras': 'HN',
        'jamaica': 'JM',
        'kiribati': 'KI',
        'laos': 'LA',
        'lesotho': 'LS',
        'liberia': 'LR',
        'libya': 'LY',
        'madagascar': 'MG',
        'malawi': 'MW',
        'maldives': 'MV',
        'mali': 'ML',
        'malta': 'MT',
        'mauritania': 'MR',
        'mauritius': 'MU',
        'micronesia': 'FM',
        'monaco': 'MC',
        'montenegro': 'ME',
        'montserrat': 'MS',
        'myanmar': 'MM',
        'namibia': 'NA',
        'nauru': 'NR',
        'nepal': 'NP',
        'nicaragua': 'NI',
        'niger': 'NE',
        'north macedonia': 'MK',
        'oman': 'OM',
        'palau': 'PW',
        'panama': 'PA',
        'papua new guinea': 'PG',
        'paraguay': 'PY',
        'peru': 'PE',
        'philippines': 'PH',
        'rwanda': 'RW',
        'samoa': 'WS',
        'san marino': 'SM',
        'sao tome and principe': 'ST',
        'saudi arabia': 'SA',
        'senegal': 'SN',
        'seychelles': 'SC',
        'sierra leone': 'SL',
        'singapore': 'SG',
        'solomon islands': 'SB',
        'somalia': 'SO',
        'south africa': 'ZA',
        'south sudan': 'SS',
        'sri lanka': 'LK',
        'sudan': 'SD',
        'suriname': 'SR',
        'tanzania': 'TZ',
        'thailand': 'TH',
        'timor-leste': 'TL',
        'togo': 'TG',
        'tonga': 'TO',
        'trinidad and tobago': 'TT',
        'tunisia': 'TN',
        'turkmenistan': 'TM',
        'tuvalu': 'TV',
        'uganda': 'UG',
        'united arab emirates': 'AE',
        'uruguay': 'UY',
        'uzbekistan': 'UZ',
        'vanuatu': 'VU',
        'venezuela': 'VE',
        'vietnam': 'VN',
        'yemen': 'YE',
        'zambia': 'ZM',
        'zimbabwe': 'ZW',
    }

    country_lower = country_name.lower().strip()

    # Check manual map first
    if country_lower in manual_map:
        return manual_map[country_lower]

    # Try exact match with pycountry
    try:
        country = pycountry.countries.get(name=country_name)
        if country:
            return country.alpha_2
    except (KeyError, AttributeError):
        pass

    # Try fuzzy search with pycountry
    try:
        results = pycountry.countries.search_fuzzy(country_name)
        if results:
            return results[0].alpha_2
    except (LookupError, AttributeError):
        pass

    raise ValueError(
        f"Could not match country name '{country_name}' to ISO code. "
        f"Please use a standard country name or add to manual mapping."
    )


def _calculate_centroid(bbox: list) -> Tuple[float, float]:
    """
    Calculate centroid from bounding box.

    Args:
        bbox: [min_lon, min_lat, max_lon, max_lat]

    Returns:
        tuple: (latitude, longitude)
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    lat = (min_lat + max_lat) / 2.0
    lon = (min_lon + max_lon) / 2.0
    return lat, lon


@retry_with_backoff(max_retries=5, base_delay=3.0, rate_limit_calls=0.5)
def fetch_weather_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    cache_dir: str = "data/weather_cache"
) -> pd.DataFrame:
    """
    Fetch daily weather data from Open-Meteo Historical Weather API.

    Features fetched:
    - temperature_2m_mean: Daily mean temperature at 2m height
    - temperature_2m_max: Daily maximum temperature
    - temperature_2m_min: Daily minimum temperature
    - relative_humidity_2m_mean: Daily mean relative humidity
    - relative_humidity_2m_min: Daily minimum relative humidity

    Implements caching to avoid repeated API calls.

    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        cache_dir: Directory for caching weather data

    Returns:
        DataFrame with columns: [date, temp_mean, temp_max, temp_min,
                                  humidity_mean, humidity_min]

    Raises:
        RuntimeError: If API call fails and no cache is available
    """
    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Generate cache filename
    cache_file = cache_path / f"{latitude:.4f}_{longitude:.4f}_{start_date}_{end_date}.parquet"

    # Check cache first
    if cache_file.exists():
        print(f"Loading weather data from cache: {cache_file.name}")
        return pd.read_parquet(cache_file)

    # Fetch from API
    print(f"Fetching weather data from Open-Meteo API...")
    print(f"  Location: ({latitude:.4f}, {longitude:.4f})")
    print(f"  Date range: {start_date} to {end_date}")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date,
        'end_date': end_date,
        'daily': [
            'temperature_2m_mean',
            'temperature_2m_max',
            'temperature_2m_min',
            'relative_humidity_2m_mean',
            'relative_humidity_2m_min'
        ],
        'timezone': 'UTC'
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        # Parse response
        if 'daily' not in data:
            raise RuntimeError(f"Unexpected API response format: {data}")

        daily_data = data['daily']

        # Create DataFrame
        df = pd.DataFrame({
            'date': pd.to_datetime(daily_data['time']),
            'temp_mean': daily_data.get('temperature_2m_mean'),
            'temp_max': daily_data.get('temperature_2m_max'),
            'temp_min': daily_data.get('temperature_2m_min'),
            'humidity_mean': daily_data.get('relative_humidity_2m_mean'),
            'humidity_min': daily_data.get('relative_humidity_2m_min')
        })

        # Check for missing data
        missing_pct = (df.isna().sum() / len(df) * 100)
        if missing_pct.any() > 10:
            print(f"Warning: High percentage of missing weather data:")
            for col, pct in missing_pct.items():
                if pct > 10:
                    print(f"  {col}: {pct:.1f}%")

        # Cache the result
        df.to_parquet(cache_file, index=False)
        print(f"Cached weather data to {cache_file.name}")

        print(f"Fetched {len(df)} days of weather data")

        return df

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch weather data from API: {e}")
    except (KeyError, ValueError) as e:
        raise RuntimeError(f"Failed to parse API response: {e}")


def aggregate_weather_to_weekly(
    daily_weather_df: pd.DataFrame,
    weekly_dates: list
) -> pd.DataFrame:
    """
    Aggregate daily weather to weekly frequency aligned with ILI dates.

    Aggregation strategy:
    - temperature_mean: Weekly average of daily means
    - temperature_max: Weekly maximum of daily maxes
    - temperature_min: Weekly minimum of daily mins
    - humidity_mean: Weekly average of daily means
    - humidity_min: Weekly minimum of daily mins

    Args:
        daily_weather_df: DataFrame with daily weather data
        weekly_dates: List of weekly ILI dates (ISO format strings or datetime)

    Returns:
        DataFrame with weekly weather features aligned to weekly_dates
    """
    # Convert weekly_dates to datetime
    weekly_dates_dt = pd.to_datetime(weekly_dates)

    # Ensure daily weather has datetime index
    if 'date' in daily_weather_df.columns:
        daily_weather_df = daily_weather_df.set_index('date')

    daily_weather_df.index = pd.to_datetime(daily_weather_df.index)

    # Create weekly aggregation
    weekly_weather = []

    for i in range(len(weekly_dates_dt)):
        start = weekly_dates_dt[i]
        # End is 7 days after start, or until next week's start
        if i < len(weekly_dates_dt) - 1:
            end = weekly_dates_dt[i + 1]
        else:
            end = start + pd.Timedelta(days=7)

        # Filter data for this week
        week_data = daily_weather_df[
            (daily_weather_df.index >= start) &
            (daily_weather_df.index < end)
        ]

        if len(week_data) == 0:
            # No data for this week - use NaN
            weekly_weather.append({
                'date': start,
                'temp_mean': np.nan,
                'temp_max': np.nan,
                'temp_min': np.nan,
                'humidity_mean': np.nan,
                'humidity_min': np.nan,
            })
        else:
            # Aggregate based on feature type
            weekly_weather.append({
                'date': start,
                'temp_mean': week_data['temp_mean'].mean(),
                'temp_max': week_data['temp_max'].max(),
                'temp_min': week_data['temp_min'].min(),
                'humidity_mean': week_data['humidity_mean'].mean(),
                'humidity_min': week_data['humidity_min'].min(),
            })

    result_df = pd.DataFrame(weekly_weather)

    print(f"Aggregated daily weather to {len(result_df)} weekly observations")

    # Check for missing weeks
    missing_weeks = result_df['temp_mean'].isna().sum()
    if missing_weeks > 0:
        print(f"Warning: {missing_weeks} weeks have missing weather data")

    return result_df


def normalize_weather_features(
    weather_df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normalize weather features to mean=0, std=1.

    Args:
        weather_df: DataFrame with weather features
        scaler: Optional pre-fitted StandardScaler (for using train scaler on test data)

    Returns:
        tuple: (normalized_df, fitted_scaler)
    """
    # Extract feature columns (exclude 'date' if present)
    feature_cols = [col for col in weather_df.columns if col != 'date']

    # Create or use provided scaler
    if scaler is None:
        scaler = StandardScaler()
        normalized_values = scaler.fit_transform(weather_df[feature_cols])
        print(f"Normalized {len(feature_cols)} weather features")
    else:
        normalized_values = scaler.transform(weather_df[feature_cols])
        print(f"Applied existing scaler to {len(feature_cols)} weather features")

    # Create normalized DataFrame
    normalized_df = weather_df.copy()
    normalized_df[feature_cols] = normalized_values

    # Print normalization stats
    if scaler is not None:
        print("Normalization stats (mean, std):")
        for i, col in enumerate(feature_cols):
            print(f"  {col}: mean={scaler.mean_[i]:.2f}, std={scaler.scale_[i]:.2f}")

    return normalized_df, scaler


def get_weather_for_country(
    country_name: str,
    start_date: str,
    end_date: str,
    weekly_dates: list,
    normalize: bool = True,
    cache_dir: str = "data/weather_cache",
    use_capital: bool = False,
) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """
    Convenience function to fetch, aggregate, and normalize weather data for a country.

    Args:
        country_name: Country name (e.g., "Argentina")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        weekly_dates: List of weekly dates to align with
        normalize: Whether to normalize features
        cache_dir: Cache directory for weather data

    Returns:
        tuple: (weekly_weather_df, scaler or None)
    """
    # Get country coordinates
    lat, lon = get_country_coordinates(country_name, use_capital=use_capital)

    # Fetch daily weather data
    daily_weather = fetch_weather_data(
        latitude=lat,
        longitude=lon,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir
    )

    # Aggregate to weekly
    weekly_weather = aggregate_weather_to_weekly(daily_weather, weekly_dates)

    # Normalize if requested
    scaler = None
    if normalize:
        weekly_weather, scaler = normalize_weather_features(weekly_weather)

    return weekly_weather, scaler



if __name__ == "__main__":
    # Test the module
    print("Testing weather fetcher module...")

    # Test country coordinate lookup
    print("\n" + "=" * 60)
    print("Test 1: Country coordinate lookup")
    print("=" * 60)
    try:
        coords_ar = get_country_coordinates("Argentina")
        print(f"SUCCESS: Argentina coordinates: {coords_ar}")

        coords_at = get_country_coordinates("Austria")
        print(f"SUCCESS: Austria coordinates: {coords_at}")
    except Exception as e:
        print(f"ERROR: {e}")

    # Test weather data fetch
    print("\n" + "=" * 60)
    print("Test 2: Weather data fetch")
    print("=" * 60)
    try:
        lat, lon = coords_ar
        weather_df = fetch_weather_data(
            latitude=lat,
            longitude=lon,
            start_date="2012-01-01",
            end_date="2012-01-31"
        )
        print(f"SUCCESS: Fetched {len(weather_df)} days of weather data")
        print("\nFirst 5 rows:")
        print(weather_df.head())
        print("\nData info:")
        print(weather_df.info())
    except Exception as e:
        print(f"ERROR: {e}")

    # Test weekly aggregation
    print("\n" + "=" * 60)
    print("Test 3: Weekly aggregation")
    print("=" * 60)
    try:
        weekly_dates = pd.date_range("2012-01-02", periods=4, freq='W-MON')
        weekly_weather = aggregate_weather_to_weekly(weather_df, weekly_dates)
        print(f"SUCCESS: Aggregated to {len(weekly_weather)} weeks")
        print("\nWeekly data:")
        print(weekly_weather)
    except Exception as e:
        print(f"ERROR: {e}")

    # Test normalization
    print("\n" + "=" * 60)
    print("Test 4: Feature normalization")
    print("=" * 60)
    try:
        normalized_weather, scaler = normalize_weather_features(weekly_weather)
        print("SUCCESS: Normalized weather features")
        print("\nNormalized data:")
        print(normalized_weather)
    except Exception as e:
        print(f"ERROR: {e}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
