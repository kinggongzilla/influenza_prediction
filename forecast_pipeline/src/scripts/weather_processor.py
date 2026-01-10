#!/usr/bin/env python3
"""
Weather data processor for time series forecasting.

This module provides simplified weather data processing functionality
to reduce complexity in the main weather fetcher module.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from sklearn.preprocessing import StandardScaler


class WeatherProcessor:
    """
    Simplified weather data processor for time series forecasting.
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.scaler = None

    def process_weather_data(
        self,
        daily_weather: pd.DataFrame,
        weekly_dates: list,
        split_date: Optional[str] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[StandardScaler]]:
        """
        Process weather data from daily to weekly and split into past/future.

        Args:
            daily_weather: Daily weather data DataFrame
            weekly_dates: List of weekly dates to align with
            split_date: Optional date to split past/future weather

        Returns:
            tuple: (past_weather, future_weather, scaler)
        """
        # Aggregate daily to weekly
        weekly_weather = self._aggregate_to_weekly(daily_weather, weekly_dates)

        if len(weekly_weather) == 0:
            return None, None, None

        # Normalize if requested
        if self.normalize:
            weekly_weather, self.scaler = self._normalize_features(weekly_weather)

        # Split into past and future if split_date provided
        if split_date:
            return self._split_weather_data(weekly_weather, split_date)
        else:
            return weekly_weather, None, self.scaler

    def _aggregate_to_weekly(
        self,
        daily_weather: pd.DataFrame,
        weekly_dates: list
    ) -> pd.DataFrame:
        """
        Aggregate daily weather data to weekly frequency.
        """
        if len(daily_weather) == 0:
            return pd.DataFrame()

        # Convert weekly_dates to datetime
        weekly_dates_dt = pd.to_datetime(weekly_dates)

        # Ensure daily weather has datetime index
        if 'date' in daily_weather.columns:
            daily_weather = daily_weather.set_index('date')
        daily_weather.index = pd.to_datetime(daily_weather.index)

        # Create weekly aggregation
        weekly_weather = []

        for i in range(len(weekly_dates_dt)):
            start = weekly_dates_dt[i]
            end = start + pd.Timedelta(days=7)

            # Filter data for this week
            week_data = daily_weather[
                (daily_weather.index >= start) &
                (daily_weather.index < end)
            ]

            if len(week_data) == 0:
                # No data for this week
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

        return pd.DataFrame(weekly_weather)

    def _normalize_features(
        self,
        weather_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Normalize weather features to mean=0, std=1.
        """
        # Extract feature columns (exclude 'date' if present)
        feature_cols = [col for col in weather_df.columns if col != 'date']

        if len(feature_cols) == 0:
            return weather_df, None

        # Create and fit scaler
        scaler = StandardScaler()
        normalized_values = scaler.fit_transform(weather_df[feature_cols])

        # Create normalized DataFrame
        normalized_df = weather_df.copy()
        normalized_df[feature_cols] = normalized_values

        return normalized_df, scaler

    def _split_weather_data(
        self,
        weekly_weather: pd.DataFrame,
        split_date: str
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Split weather data into past and future at split_date.
        """
        # Convert to datetime for comparison
        weekly_weather['date_dt'] = pd.to_datetime(weekly_weather['date'])
        split_datetime = pd.to_datetime(split_date)

        # Past: weeks where date <= split_date
        past_mask = weekly_weather['date_dt'] <= split_datetime
        past_weather = weekly_weather[past_mask].copy().drop(columns=['date_dt'])

        # Future: weeks where date > split_date
        future_mask = weekly_weather['date_dt'] > split_datetime
        future_weather = weekly_weather[future_mask].copy().drop(columns=['date_dt'])

        # Filter future weeks to only include those with sufficient data
        if len(future_weather) > 0:
            feature_cols = ['temp_mean', 'temp_max', 'temp_min', 'humidity_mean', 'humidity_min']
            future_weather['valid_days'] = future_weather[feature_cols].notna().sum(axis=1)

            # Keep only weeks with at least 4 valid days
            sufficient_coverage_mask = future_weather['valid_days'] >= 4
            future_weather = future_weather[sufficient_coverage_mask].copy()

            if len(future_weather) == 0:
                future_weather = None
            else:
                future_weather = future_weather.drop(columns=['valid_days'])
        else:
            future_weather = None

        return past_weather, future_weather

    def apply_scaler(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply existing scaler to new weather data.
        """
        if self.scaler is None:
            return weather_df

        feature_cols = [col for col in weather_df.columns if col != 'date']
        normalized_values = self.scaler.transform(weather_df[feature_cols])

        normalized_df = weather_df.copy()
        normalized_df[feature_cols] = normalized_values

        return normalized_df