#!/usr/bin/env python3
"""
Inference script for Chronos-2 model (amazon/chronos-2)

Chronos-2 is a pre-trained time series forecasting model that provides
zero-shot forecasting capabilities with quantile predictions.

Usage:
    python chronos_inference.py --help

Examples:
    # Basic usage with CSV data
    python chronos_inference.py --data_file data.csv --prediction_length 24

    # Using GPU with CUDA backend
    python chronos_inference.py --data_file data.csv --backend cuda --prediction_length 48
"""

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import model classes to register them
from chronos.chronos2 import Chronos2Pipeline
from data_loader import load_time_series_data, load_data_for_evaluation, load_time_series_with_weather, plot_forecast_results, print_inference_configuration



def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Chronos-2 Time Series Forecasting"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="amazon/chronos-2",
        help="Path to model checkpoint or HuggingFace model ID (default: amazon/chronos-2)"
    )
    parser.add_argument(
        "--data_file",
        required=True,
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
        "--plot",
        action='store_true',
        help="Generate and save a plot of the results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)"
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Print detailed output"
    )
    
    args = parser.parse_args()
    
    # Print configuration using standardized function
    print_inference_configuration(
        args,
        model_name="Chronos-2",
        use_weather=args.use_weather,
        country_name=args.country_name if args.use_weather else None
    )

    # Validate weather arguments
    if args.use_weather and not args.country_name:
        raise ValueError("--country_name is required when --use_weather is enabled")

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
            data, dates, _ = load_time_series_data(
                data_file_path,
                context_length=None,  # Don't truncate here - load all data
                min_value=args.min_value,
                max_value=args.max_value,
                clip_outliers=args.clip_outliers
            )

        # Handle test split if requested
        if args.test_split is not None:
            split_idx = int(len(data) * (1 - args.test_split))
            test_data = data[split_idx:]
            data = data[:split_idx]

            # Split weather covariates if present
            if weather_covariates is not None:
                weather_covariates = {
                    name: tensor[:split_idx]
                    for name, tensor in weather_covariates.items()
                }

            # When test_split is used, use ALL training data as context (ignore context_length)
            print(f"Split data: {len(data)} for training, {len(test_data)} for testing")
            print(f"Using all {len(data)} training points as context for forecasting")
            if weather_covariates is not None:
                print(f"Weather covariates also split: {len(weather_covariates)} features")

    # No random data generation - data_file is now required
    
    # Add batch dimension (required by Chronos)
    data = data.unsqueeze(0)  # Shape: [1, context_length]
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading Chronos-2 model on {device}...")
    model = Chronos2Pipeline.from_pretrained(args.model_path, device_map=device)
    
    print(f"Model loaded successfully")
    print(f"Device: {model.model.device}")
    
    # Generate forecast
    print(f"\nGenerating forecast for {args.prediction_length} time steps...")

    if weather_covariates:
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
            original_data, original_dates, _ = load_time_series_data(
                data_file_path, 
                context_length=None, 
                min_value=args.min_value,
                max_value=args.max_value,
                clip_outliers=args.clip_outliers
            )
            split_idx = int(len(original_data) * (1 - args.test_split))
            test_data = original_data[split_idx:]
            if original_dates:
                test_dates = original_dates[split_idx:]
        
        # Determine country name for plot filename
        country_name = "unknown"
        if args.data_file:
            country_name = args.data_file.replace('_ili_extracted.csv', '').replace('data/', '')
            country_name = country_name.replace('_ari_extracted.csv', '')
            country_name = country_name.replace('/', '_')
            country_name = country_name.replace(' ', '_')
        
        plot_forecast_results(
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
            is_eval=args.test_split is not None,
            model_name="Chronos-2",
            results_dir=args.results_dir
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
    
    # Generate country-specific filename
    if args.data_file:
        country_name = args.data_file.replace('_ili_extracted.csv', '').replace('data/', '')
        country_name = country_name.replace('_ari_extracted.csv', '')
        country_name = country_name.replace('/', '_')
        country_name = country_name.replace(' ', '_')
        
        # Create country-specific filename
        csv_filename = f'{country_name}_forecast_results.csv'
        
        # Ensure results directory exists
        os.makedirs(args.results_dir, exist_ok=True)
        
        # Save to specified results directory
        results_df.to_csv(os.path.join(args.results_dir, csv_filename), index=False)
        print(f"Saved forecast results to '{os.path.join(args.results_dir, csv_filename)}'")
    else:
        # Fallback to original behavior if no data file specified
        os.makedirs(args.results_dir, exist_ok=True)
        results_df.to_csv(os.path.join(args.results_dir, 'forecast_results.csv'), index=False)
        print(f"Saved forecast results to '{os.path.join(args.results_dir, 'forecast_results.csv')}'")
    
    print("\n" + "=" * 80)
    print("Inference completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
