#!/usr/bin/env python3
"""
Inference script for TiRex model (NX-AI/TiRex)

TiRex is a pre-trained time series forecasting model that provides
zero-shot forecasting capabilities with quantile predictions.

Usage:
    python tirex_inference.py --help

Examples:
    # Basic usage with random data
    python tirex_inference.py --prediction_length 24

    # With custom data from CSV
    python tirex_inference.py --data_file data.csv --prediction_length 48

    # Using GPU with CUDA backend
    python tirex_inference.py --backend cuda --prediction_length 24
"""

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Import model classes to register them
from tirex.models.tirex import TiRexZero
from tirex.base import load_model

# Import shared data loading module
from data_loader import load_time_series_data, load_data_for_evaluation


def load_data_from_file(file_path, context_length=128, min_value=1e-6, max_value=None, clip_outliers=False, fill_strategy='zero', freq='W'):
    """
    Load time series data from a CSV file using the shared data loader.
    
    This wrapper maintains the TiRex interface but uses the shared data loading module.
    Note: fill_strategy and freq parameters are ignored as the shared loader uses
    a fixed daily→weekly resampling approach.
    
    Args:
        file_path: Path to CSV file
        context_length: Number of time steps to use as context
        min_value: Minimum value to use instead of 0 (to avoid TiRex's nan_mask_value=0)
        max_value: Maximum value for clipping outliers (None to disable)
        clip_outliers: Whether to clip outliers using IQR method
        fill_strategy: Strategy for handling missing dates (ignored - always uses NaN)
        freq: Frequency for resampling (ignored - always uses weekly)
        
    Returns:
        tuple: (torch.Tensor: time series data, list: dates for each time step)
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
    Generate random time series data for demonstration.
    
    Args:
        length: Length of the time series
        
    Returns:
        torch.Tensor: Random time series data
    """
    # Generate random walk data
    data = torch.randn(length)
    data = torch.cumsum(data, dim=0)
    data = (data - data.min()) / (data.max() - data.min())  # Normalize to [0, 1]
    return data


def plot_results(context, forecast_mean, forecast_quantiles, prediction_length, model_quantiles, test_data=None, data_file=None, dates=None, country_name=None, is_eval=False):
    """
    Plot the context data and forecast results.
    
    Args:
        context: Context time series data
        forecast_mean: Mean forecast
        forecast_quantiles: Quantile forecasts (shape: [batch, time, quantiles])
        prediction_length: Length of the forecast
        model_quantiles: List of quantile values
        test_data: Optional test data to plot against forecast (for evaluation)
        data_file: Optional path to CSV file to extract actual dates for x-axis labels
        dates: Optional list of date strings for x-axis labels
        country_name: Optional country name for filename
    """
    plt.figure(figsize=(12, 6))
    
    # Plot context
    if dates and len(dates) >= len(context):
        # Use actual dates for x-axis
        time_context = dates[-len(context):]
        plt.plot(range(len(context)), context, 'b-', label='Context', linewidth=2)
    else:
        time_context = torch.arange(len(context))
        plt.plot(time_context, context, 'b-', label='Context', linewidth=2)
    
    # Plot forecast
    if dates and len(dates) >= len(context):
        # Generate forecast dates by extending the last date
        last_date = pd.to_datetime(dates[-1])
        forecast_dates = [(last_date + pd.Timedelta(weeks=i+1)).strftime('%Y-%m-%d') for i in range(prediction_length)]
        time_forecast = forecast_dates
        plt.plot(range(len(context), len(context) + prediction_length), forecast_mean, 'r--', label='Mean Forecast', linewidth=2)
    else:
        time_forecast = torch.arange(len(context), len(context) + prediction_length)
        plt.plot(time_forecast, forecast_mean, 'r--', label='Mean Forecast', linewidth=2)
    
    # Plot test data if provided
    if test_data is not None:
        # Only plot the portion of test data that corresponds to the forecast period
        # Use min() to handle cases where test data is shorter than prediction length
        test_data_to_plot = test_data[:min(prediction_length, len(test_data))]
        if len(test_data_to_plot) > 0:
            if dates and len(dates) >= len(context):
                plt.plot(range(len(context), len(context) + len(test_data_to_plot)), test_data_to_plot, 'orange', label='Actual Test Data', linewidth=2)
            else:
                plt.plot(time_forecast[:len(test_data_to_plot)], test_data_to_plot, 'orange', label='Actual Test Data', linewidth=2)
    
    # Plot quantiles - select a few for visualization (skip 0.5 as it's same as mean)
    selected_quantiles = [0.1, 0.9]  # Removed 0.5 to avoid duplication with mean forecast
    for q in selected_quantiles:
        if q in model_quantiles:
            idx = model_quantiles.index(q)
            if dates and len(dates) >= len(context):
                plt.plot(range(len(context), len(context) + prediction_length), forecast_quantiles[0, :, idx], 
                        'g--', alpha=0.7, linewidth=1.5, label=f'Quantile {q}')
            else:
                plt.plot(time_forecast, forecast_quantiles[0, :, idx], 
                        'g--', alpha=0.7, linewidth=1.5, label=f'Quantile {q}')
    
    # Plot uncertainty range
    if 0.1 in model_quantiles and 0.9 in model_quantiles:
        idx_low = model_quantiles.index(0.1)
        idx_high = model_quantiles.index(0.9)
        if dates and len(dates) >= len(context):
            plt.fill_between(range(len(context), len(context) + prediction_length), 
                            forecast_quantiles[0, :, idx_low], 
                            forecast_quantiles[0, :, idx_high], 
                            color='g', alpha=0.2, label='10%-90% Quantile Range')
        else:
            plt.fill_between(time_forecast, 
                            forecast_quantiles[0, :, idx_low], 
                            forecast_quantiles[0, :, idx_high], 
                            color='g', alpha=0.2, label='10%-90% Quantile Range')
    
    # Set x-axis labels with dates if dates are provided
    if dates and len(dates) >= len(context):
        try:
            # Use the actual dates for x-axis labels
            # Show every nth date to avoid crowding
            total_length = len(context) + prediction_length
            n_ticks = min(15, total_length)  # Show up to 15 ticks
            step = max(1, len(dates) // n_ticks)
            
            # Get tick positions and labels - only show dates for context period
            tick_positions = list(range(0, len(context), step))
            tick_labels = [dates[i] if i < len(dates) else '' for i in tick_positions]
            
            # Format dates nicely
            formatted_labels = [label.split('-')[1] + '-' + label.split('-')[0][-2:] for label in tick_labels]
            
            plt.xticks(tick_positions, formatted_labels, rotation=45)
        except Exception as e:
            print(f"Warning: Could not set date labels: {e}")
    elif data_file:
        try:
            # Fallback: Read the CSV file to get dates
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
    
    plt.title(f'TiRex Forecasting Results for {data_file}')
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
    plot_filename = f'results/{country_name}_tirex_forecast_results{eval_suffix}.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Saved forecast plot to '{plot_filename}'")


def main():
    """
    Main function to run TiRex inference.
    """
    parser = argparse.ArgumentParser(
        description='TiRex Time Series Forecasting Inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_file',
        type=str,
        default=None,
        help='Path to CSV file containing time series data (relative to data/ directory). If not provided, random data will be generated.'
    )
    
    parser.add_argument(
        '--min_value',
        type=float,
        default=1e-6,
        help='Minimum value to use instead of 0 to avoid TiRex\'s nan_mask_value=0. Default: 1e-6'
    )
    
    parser.add_argument(
        '--max_value',
        type=float,
        default=None,
        help='Maximum value for clipping outliers. None to disable. Default: None'
    )
    
    parser.add_argument(
        '--clip_outliers',
        action='store_true',
        help='Clip outliers using IQR method (Q1-1.5*IQR to Q3+1.5*IQR)'
    )
    
    parser.add_argument(
        '--context_length',
        type=int,
        default=128,
        help='Number of time steps to use as context for forecasting'
    )
    
    parser.add_argument(
        '--prediction_length',
        type=int,
        required=True,
        help='Number of time steps to forecast ahead'
    )
    
    parser.add_argument(
        '--backend',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Backend to use for computation'
    )
    
    parser.add_argument(
        '--quantiles',
        nargs='+',
        default=None,
        type=float,
        help='Quantiles to select from model output. If None, uses all model quantiles.'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate and save a plot of the results'
    )
    
    parser.add_argument(
        '--test_split',
        type=float,
        default=None,
        help='Fraction of data to hold out for testing (e.g., 0.2 for 20%%). '
             'When provided, the forecast will be plotted against actual test data.'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TiRex Time Series Forecasting")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Context length: {args.context_length}")
    print(f"  - Prediction length: {args.prediction_length}")
    print(f"  - Backend: {args.backend}")
    print(f"  - Quantiles: {args.quantiles if args.quantiles else 'All model quantiles'}")
    print(f"  - Data source: {'File: ' + args.data_file if args.data_file else 'Random data'}")
    print(f"  - Test split: {args.test_split if args.test_split else 'None (no evaluation)'}")
    print(f"  - Min value (for 0 replacement): {args.min_value}")
    print(f"  - Max value (for outlier clipping): {args.max_value if args.max_value else 'None'}")
    print(f"  - Clip outliers (IQR method): {args.clip_outliers}")
    print("=" * 80)
    
    # Load data
    if args.data_file:
        print(f"\nLoading data from {args.data_file}...")
        
        # Load full data without truncation
        # Apply the same file path handling as Chronos script
        data_file_path = args.data_file if args.data_file.startswith('data/') else f"data/{args.data_file}"
        data, dates = load_data_from_file(
            data_file_path, 
            None, 
            args.min_value,
            args.max_value,
            args.clip_outliers
        )
        print(f"Loaded {len(data)} time steps")
        
        # Split data into training and test sets if test_split is provided
        if args.test_split is not None:
            if args.test_split <= 0 or args.test_split >= 1:
                raise ValueError(f"test_split must be between 0 and 1, got {args.test_split}")
            
            split_idx = int(len(data) * (1 - args.test_split))
            test_data = data[split_idx:]
            data = data[:split_idx]
            
            # When test_split is used, use ALL training data as context (ignore context_length)
            print(f"Split data: {len(data)} for training, {len(test_data)} for testing")
            print(f"Using all {len(data)} training points as context for forecasting")
            
    else:
        print(f"\nGenerating random data with {args.context_length} time steps...")
        data = generate_random_data(args.context_length)
        dates = None
    
    # Add batch dimension (required by TiRex)
    data = data.unsqueeze(0)  # Shape: [1, context_length]
    
    # Load model
    print("\nLoading TiRex model...")
    if args.backend != 'auto':
        model = load_model("NX-AI/TiRex", backend=args.backend)
    else:
        model = load_model("NX-AI/TiRex")
    
    print(f"Model loaded successfully")
    print(f"Device: {next(model.parameters()).device}")
    
    # Generate forecast
    print(f"\nGenerating forecast for {args.prediction_length} time steps...")
    forecast_quantiles, forecast_mean = model.forecast(
        context=data,
        prediction_length=args.prediction_length
    )
    
    print("\nForecast generated successfully!")
    print(f"Mean forecast shape: {forecast_mean.shape}")
    print(f"Quantile forecasts shape: {forecast_quantiles.shape}")
    
    # Get model quantiles
    model_quantiles = model.config.quantiles
    
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
        if args.test_split is not None and args.data_file:
            # Reconstruct the split to get test data
            # Apply the same file path handling as main loading
            data_file_path = args.data_file if args.data_file.startswith('data/') else f"data/{args.data_file}"
            original_data, _ = load_data_from_file(
                data_file_path, 
                None, 
                args.min_value,
                args.max_value,
                args.clip_outliers
            )
            split_idx = int(len(original_data) * (1 - args.test_split))
            test_data = original_data[split_idx:]
        
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


if __name__ == '__main__':
    main()
