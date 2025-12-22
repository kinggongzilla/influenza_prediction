#!/usr/bin/env python3
"""
Comprehensive inference script to run both TiRex and Chronos models
on all country datasets with and without test splits.

This script automates the process of running inference for each country
in the data/ directory, testing both models with different configurations.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def get_country_files(data_dir="data/"):
    """
    Get list of country CSV files from the data directory.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        List of country file paths (excluding non-country files)
    """
    country_files = []
    
    # List all CSV files in data directory
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # Filter for country files (exclude example_data.csv and who_flu_data.csv)
    for filename in all_files:
        if filename in ['example_data.csv', 'who_flu_data.csv']:
            continue
        if '_ili_extracted.csv' in filename:
            country_files.append(filename)
    
    return sorted(country_files)

def run_inference_command(model_name, country_file, prediction_length=24, test_split=None, backend='auto'):
    """
    Run inference command for a specific model and country.
    
    Args:
        model_name: 'tirex' or 'chronos'
        country_file: Country filename (e.g., 'argentina_ili_extracted.csv')
        prediction_length: Number of time steps to forecast
        test_split: Fraction for test split (None for no split)
        backend: 'auto', 'cpu', or 'cuda'
        
    Returns:
        True if successful, False otherwise
    """
    script_path = f"src/scripts/{model_name}_inference.py"
    
    # Build command
    cmd = [
        sys.executable, script_path,
        f"--data_file", f"data/{country_file}",
        f"--prediction_length", str(prediction_length),
        f"--backend", backend,
        f"--plot"
    ]
    
    # Add test_split if provided
    if test_split is not None:
        cmd.extend([f"--test_split", str(test_split)])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ {model_name.upper()} inference completed successfully for {country_file}")
        if test_split:
            print(f"  Test split: {test_split}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {model_name.upper()} inference failed for {country_file}")
        print(f"  Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error running {model_name.upper()} inference for {country_file}")
        print(f"  Error: {str(e)}")
        return False

def main():
    """
    Main function to run comprehensive inference on all countries.
    """
    print("=" * 80)
    print("COMPREHENSIVE INFERENCE SCRIPT")
    print("Running TiRex and Chronos models on all country datasets")
    print("=" * 80)
    
    # Configuration
    prediction_length = 54  # Forecast 24 weeks ahead
    backend = 'auto'  # Use auto backend (will use GPU if available)
    test_splits = [None, 0.1]  # Run without test split and with 10% test split
    
    # Get country files
    country_files = get_country_files()
    
    if not country_files:
        print("No country files found in data/ directory!")
        return
    
    print(f"\nFound {len(country_files)} country datasets:")
    for i, filename in enumerate(country_files, 1):
        country_name = filename.replace('_ili_extracted.csv', '')
        print(f"  {i}. {country_name}")
    
    print(f"\nConfiguration:")
    print(f"  - Prediction length: {prediction_length} weeks")
    print(f"  - Backend: {backend}")
    print(f"  - Test splits: {test_splits}")
    print(f"  - Models: TiRex, Chronos")
    
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    total_runs = len(country_files) * len(test_splits) * 2  # 2 models
    run_count = 0
    
    print(f"\nStarting inference runs... (total: {total_runs})")
    print("-" * 80)
    
    # Run inference for each country, test split, and model
    for country_file in country_files:
        country_name = country_file.replace('_ili_extracted.csv', '')
        print(f"\n🌍 Processing {country_name.upper()}")
        
        for test_split in test_splits:
            test_split_str = f" (test_split={test_split})" if test_split else " (no test split)"
            print(f"\n  {country_name.upper()}{test_split_str}")
            
            # Run TiRex
            print(f"    🔮 Running TiRex...")
            start_time = time.time()
            success = run_inference_command(
                'tirex', 
                country_file, 
                prediction_length, 
                test_split, 
                backend
            )
            tirex_time = time.time() - start_time
            run_count += 1
            print(f"    TiRex completed in {tirex_time:.1f} seconds")
            
            # Run Chronos
            print(f"    ⏱️  Running Chronos...")
            start_time = time.time()
            success = run_inference_command(
                'chronos', 
                country_file, 
                prediction_length, 
                test_split, 
                backend
            )
            chronos_time = time.time() - start_time
            run_count += 1
            print(f"    Chronos completed in {chronos_time:.1f} seconds")
            
            print(f"    ✓ Completed {run_count}/{total_runs} runs")
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE INFERENCE COMPLETED!")
    print(f"Processed {len(country_files)} countries × 2 models × {len(test_splits)} configurations")
    print(f"Total runs: {run_count}")
    print("=" * 80)
    
    # Summary of generated files
    print(f"\nGenerated files summary:")
    results_files = []
    for country_file in country_files:
        country_name = country_file.replace('_ili_extracted.csv', '')
        
        # Check for generated files
        tirex_no_split = f"results/{country_name}_tirex_forecast_results.png"
        tirex_with_split = f"results/{country_name}_tirex_forecast_results_eval.png"
        chronos_no_split = f"results/{country_name}_chronos_forecast_results.png"
        chronos_with_split = f"results/{country_name}_chronos_forecast_results_eval.png"
        
        results_files.extend([tirex_no_split, tirex_with_split, chronos_no_split, chronos_with_split])
    
    # Count actually generated files
    generated_count = sum(1 for f in results_files if os.path.exists(f))
    print(f"  Generated {generated_count} plot files in results/ directory")
    
    print(f"\n📊 Results saved in: {os.path.abspath(results_dir)}")

if __name__ == "__main__":
    main()