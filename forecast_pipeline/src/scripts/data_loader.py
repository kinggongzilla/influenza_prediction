#!/usr/bin/env python3
"""
Shared data loading module for time series forecasting.

Provides consistent data loading with daily grid preservation and weekly resampling
for Chronos models.
"""

import pandas as pd
import numpy as np
import torch

# Import weather fetcher for covariate support
try:
    from weather_fetcher import get_weather_for_country
    WEATHER_AVAILABLE = True
except ImportError:
    WEATHER_AVAILABLE = False
    print("Warning: weather_fetcher module not found. Weather covariates will not be available.")


def load_time_series_data(file_path, context_length=128, min_value=1e-6, max_value=None, clip_outliers=False):
    """
    Load time series data using daily grid for preservation and weekly grid for inference.
    
    Uses a two-phase approach:
    1. Load data with daily grid to preserve exact timing (NaN for missing days)
    2. Resample to weekly grid for model inference
    
    Args:
        file_path: Path to CSV file
        context_length: Number of time steps to use as context (None for all data)
        min_value: Minimum value to use instead of 0
        max_value: Maximum value for clipping outliers (None to disable)
        clip_outliers: Whether to clip outliers using IQR method
        
    Returns:
        tuple: (torch.Tensor: weekly resampled time series data, list: weekly dates, numpy.ndarray: original daily data)
    """
    try:
        # Read CSV with comment handling
        df = pd.read_csv(file_path, comment='#')
        
        # Check if we have date information
        has_dates = 'Time' in df.columns or 'Date' in df.columns or 'time' in df.columns or 'date' in df.columns
        
        if has_dates:
            # Extract date column
            date_col = None
            if 'Time' in df.columns:
                date_col = 'Time'
            elif 'Date' in df.columns:
                date_col = 'Date'
            elif 'time' in df.columns:
                date_col = 'time'
            elif 'date' in df.columns:
                date_col = 'date'
            
            # Parse dates
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Sort by date
            df = df.sort_values(date_col).reset_index(drop=True)
            
            # Extract values column name
            if len(df.columns) > 1:
                print(f"Using column: {df.columns[1]} for time series values")
                values_col_name = df.columns[1]
            else:
                print(f"Using column: {df.columns[0]} for time series values")
                values_col_name = df.columns[0]
            
            # PHASE 1: Load with daily grid to preserve exact timing
            print("=== PHASE 1: Daily Grid Loading ===")
            
            # Create complete daily date range
            daily_date_range = pd.date_range(start=df[date_col].min(), end=df[date_col].max(), freq='D')
            print(f"Created daily grid with {len(daily_date_range)} time steps")
            print(f"Daily grid covers: {daily_date_range[0]} to {daily_date_range[-1]}")
            
            # Create DataFrame with all daily dates
            daily_df = pd.DataFrame({'Time': daily_date_range})
            
            # Merge with original data to get NaN for missing days
            daily_df = daily_df.merge(df[[date_col, values_col_name]], left_on='Time', right_on=date_col, how='left')
            
            # Extract daily values with NaN for missing days
            daily_values = daily_df[values_col_name].values
            
            # Count missing values in daily data
            daily_n_missing = np.sum(np.isnan(daily_values))
            daily_n_total = len(daily_values)
            daily_missing_percent = 100 * daily_n_missing / daily_n_total
            
            print(f"Daily data: {daily_n_total} time steps")
            print(f"Missing values (NaN): {daily_n_missing} out of {daily_n_total} ({daily_missing_percent:.1f}%)")
            
            # PHASE 2: Resample to weekly grid for model inference
            print("\n=== PHASE 2: Weekly Resampling ===")
            
            # Set Time column as index for resampling
            daily_df = daily_df.set_index('Time')
            
            # Resample to weekly frequency, using mean aggregation
            weekly_df = daily_df.resample('W').mean()
            
            # Reset index to get Time column back
            weekly_df = weekly_df.reset_index()
            
            # Extract weekly values and dates
            weekly_values = weekly_df[values_col_name].values
            weekly_dates = weekly_df['Time'].dt.strftime('%Y-%m-%d').tolist()
            
            # Count missing values in weekly data
            weekly_n_missing = np.sum(np.isnan(weekly_values))
            weekly_n_total = len(weekly_values)
            weekly_missing_percent = 100 * weekly_n_missing / weekly_n_total
            
            print(f"Weekly data: {weekly_n_total} time steps")
            print(f"Missing values (NaN): {weekly_n_missing} out of {weekly_n_total} ({weekly_missing_percent:.1f}%)")
            print(f"Date range: {weekly_dates[0]} to {weekly_dates[-1]}")
            
            # Apply post-processing to weekly data
            weekly_values = np.array(weekly_values, dtype=np.float32)
            
            # Handle NaN values - keep them as NaN for the model to recognize as missing
            # Replace 0 with min_value only for non-NaN values
            mask_nonzero = ~np.isnan(weekly_values)
            if np.any(mask_nonzero):
                weekly_values[mask_nonzero & (weekly_values == 0)] = min_value
            
            # Clip outliers if requested
            if clip_outliers and max_value is not None:
                # Calculate IQR
                q1 = np.percentile(weekly_values, 25)
                q3 = np.percentile(weekly_values, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Clip values
                weekly_values = np.clip(weekly_values, lower_bound, upper_bound)
                weekly_values = np.clip(weekly_values, None, max_value)
            elif max_value is not None:
                # Just clip to max_value
                weekly_values = np.clip(weekly_values, None, max_value)
            
            # Convert to tensor
            weekly_tensor = torch.tensor(weekly_values, dtype=torch.float32)
            
            # Truncate if context_length is specified
            if context_length is not None and len(weekly_tensor) > context_length:
                weekly_tensor = weekly_tensor[-context_length:]
                weekly_dates = weekly_dates[-context_length:]
                print(f"Data truncated to last {context_length} weekly time steps")
            
            return weekly_tensor, weekly_dates, daily_values
            
        else:
            # No date column - treat as regular time series
            print(f"No date column found, treating as regularly sampled time series")
            
            # Use the second column (index 1) for time series values
            if len(df.columns) < 2:
                raise ValueError(f"CSV file must have at least 2 columns. Found {len(df.columns)} columns.")
            
            print(f"Using second column: {df.columns[1]} for time series values")
            values = df.iloc[:, 1].values
            
            # Convert to numpy array
            values = np.array(values, dtype=np.float32)
            
            # Handle NaN values - keep them as NaN
            # Replace 0 with min_value only for non-NaN values
            mask_nonzero = ~np.isnan(values)
            if np.any(mask_nonzero):
                values[mask_nonzero & (values == 0)] = min_value
            
            # Clip outliers if requested
            if clip_outliers and max_value is not None:
                # Calculate IQR
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Clip values
                values = np.clip(values, lower_bound, upper_bound)
                values = np.clip(values, None, max_value)
            elif max_value is not None:
                # Just clip to max_value
                values = np.clip(values, None, max_value)
            
            # Convert to tensor
            data = torch.tensor(values, dtype=torch.float32)
            
            # Truncate if context_length is specified
            if context_length is not None and len(data) > context_length:
                data = data[-context_length:]
                print(f"Data truncated to last {context_length} time steps")
            
            return data, None, values
            
    except Exception as e:
        raise ValueError(f"Error loading data from {file_path}: {e}")


def load_data_for_evaluation(file_path, min_value=1e-6, max_value=None, clip_outliers=False, test_split=None):
    """
    Load data specifically for evaluation purposes (with test split).
    
    Args:
        file_path: Path to CSV file
        min_value: Minimum value to use instead of 0
        max_value: Maximum value for clipping outliers
        clip_outliers: Whether to clip outliers
        test_split: Fraction of data to use for testing
        
    Returns:
        tuple: (training_data, test_data, training_dates, test_dates)
    """
    # Load full data
    data, dates, _ = load_time_series_data(
        file_path,
        context_length=None,  # Load all data
        min_value=min_value,
        max_value=max_value,
        clip_outliers=clip_outliers
    )
    
    if test_split is not None:
        if test_split <= 0 or test_split >= 1:
            raise ValueError(f"test_split must be between 0 and 1, got {test_split}")
        
        split_idx = int(len(data) * (1 - test_split))
        test_data = data[split_idx:]
        training_data = data[:split_idx]
        
        if dates:
            test_dates = dates[split_idx:]
            training_dates = dates[:split_idx]
        else:
            test_dates = None
            training_dates = None
        
        print(f"Split data: {len(training_data)} for training, {len(test_data)} for testing")
        return training_data, test_data, training_dates, test_dates
    else:
        return data, None, dates, None


def load_time_series_with_weather(
    file_path,
    country_name,
    context_length=128,
    min_value=1e-6,
    max_value=None,
    clip_outliers=False,
    normalize_weather=True
):
    """
    Load time series data with weather covariates.

    Extends load_time_series_data() by fetching and integrating weather data
    as covariates for forecasting models that support external features.

    Args:
        file_path: Path to CSV file with ILI data
        country_name: Country name for weather data fetching (e.g., 'Argentina')
        context_length: Number of time steps to use as context (None for all data)
        min_value: Minimum value to use instead of 0 for ILI data
        max_value: Maximum value for clipping outliers in ILI data
        clip_outliers: Whether to clip outliers using IQR method
        normalize_weather: Whether to normalize weather features (recommended)

    Returns:
        tuple: (
            torch.Tensor: weekly ILI data,
            list: weekly dates,
            numpy.ndarray: daily ILI values,
            dict: weather_covariates {feature_name: torch.Tensor}
        )

    Raises:
        ImportError: If weather_fetcher module is not available
        ValueError: If weather data cannot be fetched or aligned
    """
    if not WEATHER_AVAILABLE:
        raise ImportError(
            "weather_fetcher module not available. "
            "Cannot load data with weather covariates."
        )

    print("\n" + "=" * 60)
    print("Loading Time Series with Weather Covariates")
    print("=" * 60)

    # Step 1: Load ILI data using existing function
    print("\n[1/5] Loading ILI time series data...")
    weekly_tensor, weekly_dates, daily_values = load_time_series_data(
        file_path=file_path,
        context_length=context_length,
        min_value=min_value,
        max_value=max_value,
        clip_outliers=clip_outliers
    )

    # Step 2: Extract date range for weather fetching
    print("\n[2/5] Determining date range for weather data...")
    if weekly_dates is None:
        raise ValueError("Cannot fetch weather data: No date information in ILI data")

    # Convert weekly dates to datetime for range calculation
    weekly_dates_dt = pd.to_datetime(weekly_dates)
    start_date = weekly_dates_dt.min().strftime('%Y-%m-%d')
    end_date = weekly_dates_dt.max().strftime('%Y-%m-%d')

    print(f"Weather data range: {start_date} to {end_date}")
    print(f"Number of weekly observations: {len(weekly_dates)}")

    # Step 3: Fetch weather data
    print(f"\n[3/5] Fetching weather data for {country_name}...")
    try:
        weekly_weather, scaler = get_weather_for_country(
            country_name=country_name,
            start_date=start_date,
            end_date=end_date,
            weekly_dates=weekly_dates,
            normalize=normalize_weather
        )
    except Exception as e:
        raise ValueError(f"Failed to fetch weather data: {e}")

    # Step 4: Validate alignment
    print("\n[4/5] Validating data alignment...")
    if len(weekly_weather) != len(weekly_dates):
        raise ValueError(
            f"Weather data length ({len(weekly_weather)}) does not match "
            f"ILI data length ({len(weekly_dates)})"
        )

    # Check for missing weather data
    feature_cols = [col for col in weekly_weather.columns if col != 'date']
    for col in feature_cols:
        missing_count = weekly_weather[col].isna().sum()
        if missing_count > 0:
            missing_pct = 100 * missing_count / len(weekly_weather)
            print(f"Warning: {col} has {missing_count} missing values ({missing_pct:.1f}%)")

    print("Data alignment validated successfully")

    # Step 5: Convert weather data to torch tensors
    print("\n[5/5] Converting weather features to tensors...")
    weather_covariates = {}

    for col in feature_cols:
        # Convert to numpy array, preserving NaN values
        values = weekly_weather[col].values.astype(np.float32)

        # Convert to torch tensor
        tensor = torch.tensor(values, dtype=torch.float32)

        weather_covariates[col] = tensor
        print(f"  {col}: shape {tensor.shape}, "
              f"mean={tensor.nanmean():.2f}, "
              f"std={torch.std(tensor[~torch.isnan(tensor)]):.2f}")

    print("\n" + "=" * 60)
    print("Data loading completed successfully!")
    print(f"  ILI data: {weekly_tensor.shape}")
    print(f"  Weather covariates: {len(weather_covariates)} features")
    print(f"  Time range: {len(weekly_dates)} weeks")
    print("=" * 60 + "\n")

    return weekly_tensor, weekly_dates, daily_values, weather_covariates


def load_data_for_evaluation_with_weather(
    file_path,
    country_name,
    min_value=1e-6,
    max_value=None,
    clip_outliers=False,
    normalize_weather=True,
    test_split=None
):
    """
    Load data with weather covariates for evaluation purposes (with test split).

    Args:
        file_path: Path to CSV file
        country_name: Country name for weather data
        min_value: Minimum value to use instead of 0
        max_value: Maximum value for clipping outliers
        clip_outliers: Whether to clip outliers
        normalize_weather: Whether to normalize weather features
        test_split: Fraction of data to use for testing

    Returns:
        tuple: (training_data, test_data, training_dates, test_dates,
                training_weather_covs, test_weather_covs)
    """
    # Load full data with weather
    data, dates, daily_values, weather_covariates = load_time_series_with_weather(
        file_path=file_path,
        country_name=country_name,
        context_length=None,  # Load all data
        min_value=min_value,
        max_value=max_value,
        clip_outliers=clip_outliers,
        normalize_weather=normalize_weather
    )

    if test_split is not None:
        if test_split <= 0 or test_split >= 1:
            raise ValueError(f"test_split must be between 0 and 1, got {test_split}")

        split_idx = int(len(data) * (1 - test_split))

        # Split ILI data
        test_data = data[split_idx:]
        training_data = data[:split_idx]

        # Split dates
        if dates:
            test_dates = dates[split_idx:]
            training_dates = dates[:split_idx]
        else:
            test_dates = None
            training_dates = None

        # Split weather covariates
        training_weather_covs = {}
        test_weather_covs = {}
        for feature_name, feature_tensor in weather_covariates.items():
            training_weather_covs[feature_name] = feature_tensor[:split_idx]
            test_weather_covs[feature_name] = feature_tensor[split_idx:]

        print(f"Split data with weather: {len(training_data)} for training, {len(test_data)} for testing")
        return (training_data, test_data, training_dates, test_dates,
                training_weather_covs, test_weather_covs)
    else:
        return data, None, dates, None, weather_covariates, None


def plot_forecast_results(context, forecast_mean, forecast_quantiles, prediction_length, 
                         model_quantiles, test_data=None, test_dates=None, 
                         data_file=None, dates=None, country_name=None, 
                         is_eval=False, model_name="Chronos-2", results_dir="results"):
    """
    Plot the context data and forecast results.
    
    This is a consolidated plotting function that replaces duplicate plot_results
    functions in chronos_inference.py and base_inference.py.
    
    Args:
        context: Context time series data
        forecast_mean: Mean forecast
        forecast_quantiles: Quantile forecasts (shape: [batch, time, quantiles])
        prediction_length: Length of the forecast
        model_quantiles: List of quantile values
        test_data: Optional test data to plot against forecast (for evaluation)
        test_dates: Optional dates for test data
        data_file: Optional path to CSV file for title
        dates: Optional list of dates for each time step in context
        country_name: Optional country name for filename
        is_eval: Whether this is an evaluation plot
        model_name: Name of the model for title and filename
    """
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    
    plt.figure(figsize=(12, 6))
    
    # Plot context
    time_context = torch.arange(len(context))
    plt.plot(time_context, context, 'b-', label='Context', linewidth=2)
    
    # Plot forecast
    time_forecast = torch.arange(len(context), len(context) + prediction_length)
    plt.plot(time_forecast, forecast_mean, 'r--', label='Mean Forecast', linewidth=2)
    
    # Plot test data if provided
    if test_data is not None:
        # Only plot the portion of test data that corresponds to the forecast period
        # Use min() to handle cases where test data is shorter than prediction length
        test_data_to_plot = test_data[:min(prediction_length, len(test_data))]
        if len(test_data_to_plot) > 0:
            plt.plot(time_forecast[:len(test_data_to_plot)], test_data_to_plot, 'orange', label='Actual Test Data', linewidth=2)
    
    # Plot quantiles - select a few for visualization (skip 0.5 as it's same as mean)
    selected_quantiles = [0.1, 0.9]  # Removed 0.5 to avoid duplication with mean forecast
    for q in selected_quantiles:
        if q in model_quantiles:
            idx = model_quantiles.index(q)
            plt.plot(time_forecast, forecast_quantiles[0, :, idx], 
                    'g--', alpha=0.7, linewidth=1.5, label=f'Quantile {q}')
    
    # Plot uncertainty range
    if 0.1 in model_quantiles and 0.9 in model_quantiles:
        idx_low = model_quantiles.index(0.1)
        idx_high = model_quantiles.index(0.9)
        plt.fill_between(time_forecast, 
                        forecast_quantiles[0, :, idx_low], 
                        forecast_quantiles[0, :, idx_high], 
                        color='g', alpha=0.2, label='10%-90% Quantile Range')
    
    # Set x-axis labels with dates if available
    if dates:
        try:
            # Use the provided dates
            # Create a list of formatted date labels
            date_labels = [date.split('-')[1] + ' ' + date.split('-')[0][-2:] for date in dates]
            
            # Improved tick positioning for weekly data
            total_length = len(context) + prediction_length
            
            # For weekly data, try to place ticks at yearly intervals
            # Find positions that correspond to the same month/day across years
            if len(dates) > 52:  # More than a year of weekly data
                # Try to find January dates for yearly ticks
                jan_positions = []
                jan_labels = []
                
                for i, date in enumerate(dates):
                    if i < len(context):  # Only consider context dates for ticks
                        if date.endswith('-01'):  # January dates
                            jan_positions.append(i)
                            jan_labels.append(date_labels[i])
                
                # If we found enough January dates, use them
                if len(jan_positions) >= 3:
                    plt.xticks(jan_positions, jan_labels, rotation=45)
                else:
                    # Fallback to regular spacing
                    n_ticks = min(10, total_length)
                    tick_positions = np.linspace(0, total_length - 1, n_ticks, dtype=int)
                    tick_labels = [date_labels[i] if i < len(date_labels) else '' for i in tick_positions]
                    plt.xticks(tick_positions, tick_labels, rotation=45)
            else:
                # For shorter datasets, use regular spacing
                n_ticks = min(10, total_length)
                tick_positions = np.linspace(0, total_length - 1, n_ticks, dtype=int)
                tick_labels = [date_labels[i] if i < len(date_labels) else '' for i in tick_positions]
                plt.xticks(tick_positions, tick_labels, rotation=45)
        except Exception as e:
            print(f"Warning: Could not set date labels: {e}")
    elif data_file:
        try:
            # Read the CSV file to get dates
            import pandas as pd
            date_df = pd.read_csv(data_file)
            
            # Extract dates (assuming first column is 'Time')
            dates = date_df['Time'].tolist()
            
            # Create a list of formatted date labels
            date_labels = [date.split('-')[1] + ' ' + date.split('-')[0][-2:] for date in dates]
            
            # Set x-axis ticks at regular intervals
            total_length = len(context) + prediction_length
            n_ticks = min(10, total_length)  # Show up to 10 ticks
            tick_positions = np.linspace(0, total_length - 1, n_ticks, dtype=int)
            
            # Get labels for the tick positions - only show dates for context period
            tick_labels = [date_labels[i] if i < len(date_labels) and i < len(context) else '' for i in tick_positions]
            
            plt.xticks(tick_positions, tick_labels, rotation=45)
        except Exception as e:
            print(f"Warning: Could not set date labels: {e}")
    
    plt.title(f'{model_name} Forecasting Results for {data_file}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Determine country name for filename
    if country_name is None and data_file:
        country_name = data_file.replace('_ili_extracted.csv', '').replace('data/', '')
    
    if country_name is None:
        country_name = "unknown"
    
    # Ensure results directory exists
    import os
    os.makedirs(results_dir, exist_ok=True)
    
    eval_suffix = "_eval" if is_eval else ""
    plot_filename = f'{results_dir}/{country_name}_{model_name.lower()}_forecast_results{eval_suffix}.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Saved forecast plot to '{plot_filename}'")


def print_inference_configuration(args, model_name="Chronos-2", use_weather=False, country_name=None):
    """
    Standardized configuration printing for all inference scripts.
    
    This replaces duplicate configuration printing code in chronos_inference.py.
    
    Args:
        args: Parsed command line arguments
        model_name: Name of the model being used
        use_weather: Whether weather covariates are being used
        country_name: Optional country name for weather data
    """
    print("=" * 80)
    print(f"{model_name} Time Series Forecasting")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Context length: {args.context_length}")
    print(f"  - Prediction length: {args.prediction_length}")
    print(f"  - Backend: {args.backend}")
    print(f"  - Quantiles: All model quantiles")
    print(f"  - Data source: File: {args.data_file}")
    if hasattr(args, 'test_split') and args.test_split:
        print(f"  - Test split: {args.test_split}")
    else:
        print(f"  - Test split: None")
    print(f"  - Min value (for 0 replacement): {args.min_value}")
    print(f"  - Max value (for outlier clipping): {args.max_value if args.max_value else 'None'}")
    print(f"  - Clip outliers (IQR method): {args.clip_outliers}")
    if use_weather:
        print(f"  - Weather covariates: Enabled")
        if country_name:
            print(f"  - Country: {country_name}")
        if hasattr(args, 'normalize_weather'):
            print(f"  - Normalize weather: {args.normalize_weather}")
    else:
        print(f"  - Weather covariates: Disabled")
    print("=" * 80)
    print()