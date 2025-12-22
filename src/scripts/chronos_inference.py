#!/usr/bin/env python3
"""
Inference script for Chronos-2 model (amazon/chronos-2)

Chronos-2 is a pre-trained time series forecasting model that provides
zero-shot forecasting capabilities with quantile predictions.

Usage:
    python chronos_inference.py --help

Examples:
    # Basic usage with random data
    python chronos_inference.py --prediction_length 24

    # With custom data from CSV
    python chronos_inference.py --data_file data.csv --prediction_length 48

    # Using GPU with CUDA backend
    python chronos_inference.py --backend cuda --prediction_length 24
"""

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Import model classes to register them
from chronos.chronos2 import Chronos2Pipeline

# Import shared data loading module
from data_loader import load_time_series_data, load_data_for_evaluation, load_time_series_with_weather


def load_data_from_file(file_path, context_length=128, min_value=1e-6, max_value=None, clip_outliers=False):
    """
    Load time series data from a CSV file using the shared data loader.
    
    Args:
        file_path: Path to CSV file
        context_length: Number of time steps to use as context
        min_value: Minimum value to use instead of 0 (to avoid Chronos's nan_mask_value=0)
        max_value: Maximum value for clipping outliers (None to disable)
        clip_outliers: Whether to clip outliers using IQR method
        
    Returns:
        tuple: (torch.Tensor: weekly resampled time series data, list: weekly dates)
    """
    # Use the shared data loading function
    data, dates, _ = load_time_series_data(
        file_path,
        context_length=context_length,
        min_value=min_value,
        max_value=max_value,
        clip_outliers=clip_outliers
    )
    
    return data, dates

def generate_random_data(length=128):
    """
    Generate random time series data for testing.
    
    Args:
        length: Length of the time series
        
    Returns:
        torch.Tensor: Random time series data
    """
    # Generate random walk data
    data = [100.0]
    for _ in range(length - 1):
        # Random walk with some trend and seasonality
        trend = 0.1
        seasonality = 5 * np.sin(2 * np.pi * (_ / length))
        noise = np.random.normal(0, 2)
        next_val = data[-1] + trend + seasonality + noise
        data.append(max(0.1, next_val))
    
    return torch.tensor(data, dtype=torch.float32)


def plot_results(context, forecast_mean, forecast_quantiles, prediction_length, model_quantiles, test_data=None, test_dates=None, data_file=None, dates=None, country_name=None, is_eval=False):
    """
    Plot the context data and forecast results.
    
    Args:
        context: Context time series data
        forecast_mean: Mean forecast
        forecast_quantiles: Quantile forecasts (shape: [batch, time, quantiles])
        prediction_length: Length of the forecast
        model_quantiles: List of quantile values
        test_data: Optional test data to plot against forecast (for evaluation)
        test_dates: Optional dates for test data
        data_file: Optional path to CSV file to extract actual dates for x-axis labels
        dates: Optional list of dates for each time step in context
        country_name: Optional country name for filename
    """
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
    
    plt.title(f'Chronos-2 Forecasting Results for {data_file}')
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
    
    eval_suffix = "_eval" if is_eval else ""
    plot_filename = f'results/{country_name}_chronos_forecast_results{eval_suffix}.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Saved forecast plot to '{plot_filename}'")


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Chronos-2 Time Series Forecasting"
    )
    parser.add_argument(
        "--data_file",
        help="Path to CSV file containing time series data (relative to data/ directory, second column will be used)"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=128,
        help="Number of time steps to use as context (default: 128)"
    )
    parser.add_argument(
        "--prediction_length",
        type=int,
        default=24,
        help="Number of time steps to forecast (default: 24)"
    )
    parser.add_argument(
        "--backend",
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help="Backend to use (default: auto)"
    )
    parser.add_argument(
        "--test_split",
        type=float,
        help="Fraction of data to use for testing (e.g., 0.1 for 10%%)"
    )
    parser.add_argument(
        "--min_value",
        type=float,
        default=1e-6,
        help="Minimum value to use instead of 0 (default: 1e-6)"
    )
    parser.add_argument(
        "--max_value",
        type=float,
        help="Maximum value for clipping outliers (None to disable)"
    )
    parser.add_argument(
        "--clip_outliers",
        action='store_true',
        help="Clip outliers using IQR method"
    )

    # Weather covariate arguments
    parser.add_argument(
        "--use_weather",
        action='store_true',
        help="Include weather covariates in forecasting"
    )
    parser.add_argument(
        "--country_name",
        type=str,
        help="Country name for weather data (required with --use_weather)"
    )
    parser.add_argument(
        "--normalize_weather",
        action='store_true',
        default=True,
        help="Normalize weather features (default: True)"
    )
    parser.add_argument(
        "--use_future_weather",
        action='store_true',
        help="Use future weather forecasts as covariates (requires --use_weather). "
             "With --test_split: uses first ~2 weeks of test weather. "
             "Without --test_split: fetches real 14-day forecasts from API."
    )

    parser.add_argument(
        "--plot",
        action='store_true',
        help="Generate and save a plot of the results"
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Print detailed output"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 80)
    print("Chronos-2 Time Series Forecasting")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Context length: {args.context_length}")
    print(f"  - Prediction length: {args.prediction_length}")
    print(f"  - Backend: {args.backend}")
    print(f"  - Quantiles: All model quantiles")
    if args.data_file:
        print(f"  - Data source: File: {args.data_file}")
    else:
        print(f"  - Data source: Random data")
    if args.test_split:
        print(f"  - Test split: {args.test_split}")
    print(f"  - Min value (for 0 replacement): {args.min_value}")
    print(f"  - Max value (for outlier clipping): {args.max_value}")
    print(f"  - Clip outliers (IQR method): {args.clip_outliers}")
    if args.use_weather:
        print(f"  - Weather covariates: Enabled")
        print(f"  - Country: {args.country_name}")
        print(f"  - Normalize weather: {args.normalize_weather}")
    else:
        print(f"  - Weather covariates: Disabled")

    print("=" * 80)
    print()

    # Validate weather arguments
    if args.use_weather and not args.country_name:
        raise ValueError("--country_name is required when --use_weather is enabled")

    if args.use_future_weather and not args.use_weather:
        raise ValueError("--use_future_weather requires --use_weather to be enabled")

    # Load data
    weather_covariates = None
    if args.data_file:
        # Prepend 'data/' directory if not already present
        data_file_path = args.data_file if args.data_file.startswith('data/') else f"data/{args.data_file}"
        print(f"Loading data from {data_file_path}...")

        # Load data with or without weather
        if args.use_weather:
            # Load full data with weather covariates
            data, dates, _, weather_covariates = load_time_series_with_weather(
                file_path=data_file_path,
                country_name=args.country_name,
                context_length=None,  # Don't truncate here - load all data
                min_value=args.min_value,
                max_value=args.max_value,
                clip_outliers=args.clip_outliers,
                normalize_weather=args.normalize_weather
            )
        else:
            # Load full data without truncation
            data, dates = load_data_from_file(
                data_file_path,
                None,  # Don't truncate here - load all data
                args.min_value,
                args.max_value,
                args.clip_outliers
            )

        # Handle test split if requested
        if args.test_split is not None:
            split_idx = int(len(data) * (1 - args.test_split))
            test_data = data[split_idx:]
            data = data[:split_idx]

            # Split weather covariates if present
            test_weather_covariates = None
            future_weather_covariates = None

            if weather_covariates is not None:
                test_weather_covariates = {
                    name: tensor[split_idx:]
                    for name, tensor in weather_covariates.items()
                }
                weather_covariates = {
                    name: tensor[:split_idx]
                    for name, tensor in weather_covariates.items()
                }

                # Extract future covariates from test weather if enabled
                if args.use_future_weather:
                    # Use first min(2 weeks, test_length) from test weather
                    test_length = len(next(iter(test_weather_covariates.values())))
                    future_weeks_available = min(2, test_length)

                    if future_weeks_available > 0:
                        # Extract available forecast weeks (up to 2)
                        future_weather_covariates = {
                            name: tensor[:future_weeks_available]
                            for name, tensor in test_weather_covariates.items()
                        }
                        print(f"Using {future_weeks_available} weeks of test weather as future_covariates (simulates 14-day forecast)")
                        if future_weeks_available < args.prediction_length:
                            print(f"Will use two-pass inference: {future_weeks_available} weeks with future weather, then {args.prediction_length - future_weeks_available} weeks extended")
                    else:
                        print(f"Warning: Test data has no weeks available for future_covariates")

            # When test_split is used, use ALL training data as context (ignore context_length)
            print(f"Split data: {len(data)} for training, {len(test_data)} for testing")
            print(f"Using all {len(data)} training points as context for forecasting")
            if weather_covariates is not None:
                print(f"Weather covariates also split: {len(weather_covariates)} features")
        else:
            # No test split - initialize future_weather_covariates for real forecasting
            future_weather_covariates = None

            # Fetch real weather forecasts if enabled (no test split)
            if args.use_weather and args.use_future_weather and dates:
                print("\n" + "=" * 60)
                print("Fetching Real Weather Forecasts")
                print("=" * 60)

                from datetime import datetime
                from weather_fetcher import get_weather_with_forecast

                last_ili_date = dates[-1]

                try:
                    past_weather_df, future_weather_df, scaler = get_weather_with_forecast(
                        country_name=args.country_name,
                        start_date=dates[0],
                        last_ili_date=last_ili_date,
                        prediction_length_weeks=args.prediction_length,
                        normalize=args.normalize_weather
                    )

                    if future_weather_df is not None and len(future_weather_df) > 0:
                        feature_cols = [col for col in future_weather_df.columns if col != 'date']
                        future_weather_covariates = {
                            col: torch.tensor(future_weather_df[col].values, dtype=torch.float32)
                            for col in feature_cols
                        }
                        forecast_weeks = len(future_weather_df)
                        print(f"\nLoaded {forecast_weeks} weeks of future weather forecasts")

                        if forecast_weeks < args.prediction_length:
                            print(f"Future weather covers {forecast_weeks} weeks, prediction_length is {args.prediction_length} weeks")
                            print(f"Will use two-pass inference: {forecast_weeks} weeks with future weather, then {args.prediction_length - forecast_weeks} weeks extended")
                except Exception as e:
                    print(f"\nWarning: Failed to fetch forecast: {e}")
                    print("Continuing without future_covariates.")

    else:
        print(f"\nGenerating random data with {args.context_length} time steps...")
        data = generate_random_data(args.context_length)
        if args.use_weather:
            print("Warning: Cannot use weather covariates with random data. Ignoring --use_weather.")
            weather_covariates = None
    
    # Add batch dimension (required by Chronos)
    data = data.unsqueeze(0)  # Shape: [1, context_length]
    
    # Load model
    print("\nLoading Chronos-2 model...")
    model = Chronos2Pipeline.from_pretrained("amazon/chronos-2")
    
    print(f"Model loaded successfully")
    print(f"Device: {model.model.device}")
    
    # Generate forecast
    print(f"\nGenerating forecast for {args.prediction_length} time steps...")

    # Check if we need two-pass inference
    has_future_covariates = 'future_weather_covariates' in locals() and future_weather_covariates is not None

    if weather_covariates and has_future_covariates:
        # Check if future covariates are shorter than prediction_length
        first_future_feature = next(iter(future_weather_covariates.values()))
        future_length = len(first_future_feature)

        # Always use two-pass inference when future weather is available
        # Use min(2 weeks, available future weather) for Pass 1
        pass1_length = min(2, future_length)
        
        if pass1_length > 0:
            # TWO-PASS INFERENCE
            print(f"\n{'='*60}")
            print(f"Two-Pass Inference (using {pass1_length} weeks of future weather, prediction: {args.prediction_length} weeks)")
            print(f"{'='*60}")

            # PASS 1: Predict with available future covariates (up to 2 weeks)
            print(f"\n[Pass 1] Forecasting {pass1_length} steps with future weather covariates")

            past_covariates = {
                name: tensor.numpy()
                for name, tensor in weather_covariates.items()
            }

            # Use only the first pass1_length weeks of future covariates
            future_covariates_np = {
                name: tensor.numpy()[:pass1_length]
                for name, tensor in future_weather_covariates.items()
            }

            input_dict_pass1 = {
                "target": data.squeeze().numpy(),
                "past_covariates": past_covariates,
                "future_covariates": future_covariates_np
            }

            quantile_forecasts_pass1, mean_forecasts_pass1 = model.predict_quantiles(
                inputs=[input_dict_pass1],
                prediction_length=pass1_length
            )

            print(f"Pass 1 complete: Generated {pass1_length} week forecast")

            # PASS 2: Predict remaining steps using extended context
            remaining_steps = args.prediction_length - pass1_length
            print(f"\n[Pass 2] Forecasting remaining {remaining_steps} steps")
            print(f"Extending context with Pass 1 forecast")

            # Extend target with pass 1 mean forecast
            pass1_forecast = mean_forecasts_pass1[0].squeeze()  # Shape: [pass1_length]
            extended_target = np.concatenate([
                data.squeeze().numpy(),
                pass1_forecast.numpy()
            ])

            # Extend past_covariates with future_covariates from pass 1
            extended_past_covariates = {}
            for name in weather_covariates.keys():
                past_values = weather_covariates[name].numpy()
                future_values = future_weather_covariates[name].numpy()[:pass1_length]
                extended_past_covariates[name] = np.concatenate([past_values, future_values])

            input_dict_pass2 = {
                "target": extended_target,
                "past_covariates": extended_past_covariates
            }

            quantile_forecasts_pass2, mean_forecasts_pass2 = model.predict_quantiles(
                inputs=[input_dict_pass2],
                prediction_length=remaining_steps
            )

            print(f"Pass 2 complete: Generated {remaining_steps} week forecast")

            # Concatenate forecasts
            print(f"\nCombining forecasts from both passes")
            mean_forecasts = [torch.cat([
                mean_forecasts_pass1[0],
                mean_forecasts_pass2[0]
            ], dim=1)]

            quantile_forecasts = [torch.cat([
                quantile_forecasts_pass1[0],
                quantile_forecasts_pass2[0]
            ], dim=1)]

        else:
            # SINGLE-PASS INFERENCE (no future weather or pass1_length = 0)
            print(f"\nUsing {len(weather_covariates)} weather covariates (single-pass inference):")
            for name in weather_covariates.keys():
                print(f"  - {name}")

            past_covariates = {
                name: tensor.numpy()
                for name, tensor in weather_covariates.items()
            }

            input_dict = {
                "target": data.squeeze().numpy(),
                "past_covariates": past_covariates
            }

            quantile_forecasts, mean_forecasts = model.predict_quantiles(
                inputs=[input_dict],
                prediction_length=args.prediction_length
            )

    elif weather_covariates:
        # SINGLE-PASS with only past_covariates
        print(f"\nUsing {len(weather_covariates)} weather covariates:")
        for name in weather_covariates.keys():
            print(f"  - {name}")

        past_covariates = {
            name: tensor.numpy()
            for name, tensor in weather_covariates.items()
        }

        input_dict = {
            "target": data.squeeze().numpy(),
            "past_covariates": past_covariates
        }

        print(f"\nModel using only past_covariates (no future weather)")

        quantile_forecasts, mean_forecasts = model.predict_quantiles(
            inputs=[input_dict],
            prediction_length=args.prediction_length
        )
    else:
        # Traditional univariate input format
        data_3d = data.unsqueeze(-1).transpose(1, 2)

        quantile_forecasts, mean_forecasts = model.predict_quantiles(
            inputs=data_3d,
            prediction_length=args.prediction_length
        )
    
    print("\nForecast generated successfully!")
    
    # Extract mean forecast
    # mean_forecasts[0] has shape [1, prediction_length]
    forecast_mean = mean_forecasts[0]
    
    # Extract quantile forecasts
    # quantile_forecasts[0] has shape [1, prediction_length, num_quantiles]
    forecast_quantiles = quantile_forecasts[0]
    
    print(f"Mean forecast shape: {forecast_mean.shape}")
    print(f"Quantile forecasts shape: {forecast_quantiles.shape}")
    
    # Get model quantiles - use the default quantile levels from predict_quantiles
    # By default, predict_quantiles uses [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    model_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Print some statistics
    if args.verbose:
        print("\nForecast statistics:")
        print(f"  Mean forecast range: [{forecast_mean.min():.4f}, {forecast_mean.max():.4f}]")
        print(f"  Mean forecast mean: {forecast_mean.mean():.4f}")
        print(f"  Mean forecast std: {forecast_mean.std():.4f}")
        
        print(f"\nModel quantiles: {model_quantiles}")
        print("\nQuantile forecast ranges:")
        for i, q in enumerate(model_quantiles):
            q_min = forecast_quantiles[0, :, i].min().item()
            q_max = forecast_quantiles[0, :, i].max().item()
            q_mean = forecast_quantiles[0, :, i].mean().item()
            print(f"  Quantile {q}: [{q_min:.4f}, {q_max:.4f}] (mean: {q_mean:.4f})")
    
    # Plot results if requested
    if args.plot:
        print("\nGenerating plot...")
        
        # Get test data if available
        test_data = None
        test_dates = None
        if args.test_split is not None and args.data_file:
            # Reconstruct the split to get test data
            # Use the same file path handling as main loading
            data_file_path = args.data_file if args.data_file.startswith('data/') else f"data/{args.data_file}"
            original_data, original_dates = load_data_from_file(
                data_file_path, 
                None, 
                args.min_value,
                args.max_value,
                args.clip_outliers
            )
            split_idx = int(len(original_data) * (1 - args.test_split))
            test_data = original_data[split_idx:]
            if original_dates:
                test_dates = original_dates[split_idx:]
        
        # Determine country name for plot filename
        country_name = "unknown"
        if args.data_file:
            country_name = args.data_file.replace('_ili_extracted.csv', '').replace('data/', '')
        
        plot_results(
            context=data.squeeze(),
            forecast_mean=forecast_mean.squeeze(),
            forecast_quantiles=forecast_quantiles,  # Don't squeeze - keep 3D tensor
            prediction_length=args.prediction_length,
            model_quantiles=model_quantiles,
            test_data=test_data,
            test_dates=test_dates,
            data_file=args.data_file,
            dates=dates,
            country_name=country_name,
            is_eval=args.test_split is not None
        )
    
    # Save results to CSV
    print("\nSaving results to CSV...")
    
    # Create combined time array for context + forecast
    context_length = len(data.squeeze())
    prediction_length = args.prediction_length
    total_length = context_length + prediction_length
    
    results = {
        'time': list(range(total_length)),
        'context': data.squeeze().tolist() + [None] * prediction_length,
        'mean_forecast': [None] * context_length + forecast_mean.squeeze().tolist()
    }
    
    # Add forecast data - include all quantiles
    for i, q in enumerate(model_quantiles):
        quantile_data = [None] * context_length + forecast_quantiles[0, :, i].squeeze().tolist()
        results[f'quantile_{q}'] = quantile_data
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/forecast_results.csv', index=False)
    print("Saved forecast results to 'results/forecast_results.csv'")
    
    print("\n" + "=" * 80)
    print("Inference completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
