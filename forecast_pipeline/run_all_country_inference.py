#!/usr/bin/env python3
"""
Batch Processing Script for Country-by-Country Inference

This script automates the process of extracting data and running chronos inference
for all countries in the WHO flu dataset. It uses ILI data when available and
falls back to ARI data when ILI cases are zero.

Features:
- Automatic data extraction with ILI/ARI fallback
- Batch processing with progress tracking
- Error handling and logging
- Integration with chronos inference workflow
- Results organization and reporting
"""

import pandas as pd
import os
import subprocess
import sys
import time
import argparse
from typing import List, Dict, Optional
from datetime import datetime
import logging
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('country_inference_batch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CountryInferenceProcessor:
    """Main class for processing country inference in batch."""
    
    def __init__(self, 
                 country_summary_file: str = "country_data_summary.csv",
                 who_data_file: str = "data/who_flu_data.csv",
                 extracted_data_dir: str = "data/extracted_data",
                 results_dir: str = "results/country_predictions",
                 skip_weather_on_error: bool = False):
        """
        Initialize the country inference processor.
        
        Args:
            country_summary_file: Path to country data summary CSV
            who_data_file: Path to WHO flu data CSV
            extracted_data_dir: Directory for extracted data files
            results_dir: Directory for inference results
        """
        self.country_summary_file = country_summary_file
        self.who_data_file = who_data_file
        self.extracted_data_dir = extracted_data_dir
        self.results_dir = results_dir
        self.skip_weather_on_error = skip_weather_on_error
        
        # Create directories if they don't exist
        os.makedirs(self.extracted_data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load country summary data
        self.country_summary = self._load_country_summary()
        
        # Statistics tracking
        self.stats = {
            'total_countries': len(self.country_summary),
            'successful_extractions': 0,
            'failed_extractions': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'skipped_countries': 0,
            'processing_times': [],
            'countries_with_weather': [],
            'countries_without_weather': [],
            'weather_usage_by_country': {}  # Track weather usage per country
        }
    
    def _load_country_summary(self) -> pd.DataFrame:
        """Load and validate the country summary data."""
        try:
            df = pd.read_csv(self.country_summary_file)
            logger.info(f"Loaded country summary with {len(df)} countries")
            return df
        except Exception as e:
            logger.error(f"Failed to load country summary: {e}")
            raise
    
    def _run_extraction_command(self, country_name: str) -> Optional[str]:
        """Run the enhanced extraction script for a single country."""
        try:
            cmd = [
                sys.executable, 
                "enhanced_extract_country_data.py",
                "--country", country_name,
                "--input", self.who_data_file,
                "--output_dir", self.extracted_data_dir
            ]
            
            logger.info(f"Running extraction: {' '.join(cmd)}")
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            end_time = time.time()
            
            processing_time = end_time - start_time
            self.stats['processing_times'].append(processing_time)
            
            if result.returncode == 0:
                logger.info(f"✅ Extraction successful for {country_name}")
                self.stats['successful_extractions'] += 1
                
                # Extract the output file path from the extraction script output
                for line in result.stdout.split('\n'):
                    if 'Data saved to:' in line:
                        extracted_file = line.split('Data saved to:')[1].strip()
                        return extracted_file
                
                # If we can't find the file path, look for it in the directory
                country_files = [f for f in os.listdir(self.extracted_data_dir) 
                                if country_name.replace(' ', '_') in f]
                if country_files:
                    return os.path.join(self.extracted_data_dir, country_files[0])
                
                return None
            else:
                logger.error(f"❌ Extraction failed for {country_name}: {result.stderr}")
                self.stats['failed_extractions'] += 1
                return None
                
        except Exception as e:
            logger.error(f"❌ Extraction error for {country_name}: {e}")
            self.stats['failed_extractions'] += 1
            return None
    
    def _run_chronos_inference(self, extracted_file: str, country_name: str, skip_weather_on_error: bool) -> bool:
        """Run chronos inference on extracted data (no covariates)."""
        try:
            # Clean country name for results directory
            clean_country_name = country_name.replace(' ', '_').replace('-', '_')
            country_results_dir = os.path.join(self.results_dir, clean_country_name)
            os.makedirs(country_results_dir, exist_ok=True)

            cmd = [
                sys.executable,
                "src/scripts/chronos_inference.py",
                "--data_file", extracted_file,
                "--country_name", country_name,
                "--prediction_length", "54",
                "--plot",
                "--results_dir", country_results_dir
            ]

            logger.info(f"Running chronos inference for {country_name}...")

            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            end_time = time.time()

            processing_time = end_time - start_time
            self.stats['processing_times'].append(processing_time)

            if result.returncode == 0:
                logger.info(f"Inference successful for {country_name}")
                self.stats['successful_inferences'] += 1
                self.stats['countries_without_weather'].append(country_name)
                self.stats['weather_usage_by_country'][country_name] = False
                return True
            else:
                logger.error(f"Inference failed for {country_name}: {result.stderr}")
                self.stats['failed_inferences'] += 1
                return False

        except Exception as e:
            logger.error(f"Inference error for {country_name}: {e}")
            self.stats['failed_inferences'] += 1
            return False
    
    def _should_skip_country(self, row: pd.Series) -> bool:
        """Determine if a country should be skipped based on data availability."""
        # Skip countries with no data at all
        if row['total_ili_cases'] == 0 and row['total_ari_cases'] == 0:
            logger.info(f"🔮 Skipping {row['country']} - no ILI or ARI data available")
            self.stats['skipped_countries'] += 1
            return True
            
        return False
    
    def process_country(self, country_row: pd.Series) -> bool:
        """Process a single country: extraction + inference."""
        country_name = country_row['country']
        
        if self._should_skip_country(country_row):
            return False
        
        logger.info(f"\n🚀 Processing {country_name}")
        logger.info(f"   ILI cases: {country_row['total_ili_cases']:,}")
        logger.info(f"   ARI cases: {country_row['total_ari_cases']:,}")
        logger.info(f"   Data types: {country_row['data_types']}")
        
        # Step 1: Extract data
        extracted_file = self._run_extraction_command(country_name)
        
        if extracted_file is None:
            logger.warning(f"💥 Skipping inference for {country_name} - no data extracted")
            return False
        
        # Step 2: Run chronos inference
        success = self._run_chronos_inference(extracted_file, country_name, self.skip_weather_on_error)
        
        return success
    
    def process_all_countries(self, limit: Optional[int] = None, start_from: int = 0) -> None:
        """Process all countries in the summary dataset."""
        logger.info(f"🌍 Starting batch processing of countries...")
        logger.info(f"📊 Total countries to process: {min(limit or len(self.country_summary), len(self.country_summary))}")
        
        start_time = time.time()
        
        # Process countries in order (optionally with limits)
        for i, (_, row) in enumerate(self.country_summary.iterrows(), 1):
            if limit and i > limit:
                break
            if i <= start_from:
                continue
                
            try:
                self.process_country(row)
                
                # Print progress every 5 countries
                if i % 5 == 0:
                    self._print_progress(i)
                    
            except KeyboardInterrupt:
                logger.warning("🛑 Processing interrupted by user")
                break
            except Exception as e:
                logger.error(f"💥 Unexpected error processing country {row['country']}: {e}")
                continue
        
        end_time = time.time()
        total_time = end_time - start_time
        
        self._print_final_summary(total_time)
    
    def _print_progress(self, current_count: int) -> None:
        """Print processing progress."""
        avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times']) if self.stats['processing_times'] else 0
        
        logger.info(f"\n📈 PROGRESS UPDATE")
        logger.info(f"   Countries processed: {current_count}/{self.stats['total_countries']}")
        logger.info(f"   Successful extractions: {self.stats['successful_extractions']}")
        logger.info(f"   Failed extractions: {self.stats['failed_extractions']}")
        logger.info(f"   Successful inferences: {self.stats['successful_inferences']}")
        logger.info(f"   Failed inferences: {self.stats['failed_inferences']}")
        logger.info(f"   Skipped countries: {self.stats['skipped_countries']}")
        logger.info(f"   Average processing time: {avg_time:.1f}s per country")
    
    def _print_final_summary(self, total_time: float) -> None:
        """Print final processing summary."""
        logger.info(f"\n🎉 BATCH PROCESSING COMPLETE")
        logger.info("="*60)
        
        success_rate = (self.stats['successful_inferences'] / (self.stats['total_countries'] - self.stats['skipped_countries'])) * 100 if (self.stats['total_countries'] - self.stats['skipped_countries']) > 0 else 0
        
        logger.info(f"📊 SUMMARY STATISTICS:")
        logger.info(f"   Total countries in dataset: {self.stats['total_countries']}")
        logger.info(f"   Countries processed: {self.stats['successful_extractions'] + self.stats['failed_extractions']}")
        logger.info(f"   Countries skipped (no data): {self.stats['skipped_countries']}")
        logger.info(f"   Successful extractions: {self.stats['successful_extractions']}")
        logger.info(f"   Failed extractions: {self.stats['failed_extractions']}")
        logger.info(f"   Successful inferences: {self.stats['successful_inferences']}")
        logger.info(f"   Failed inferences: {self.stats['failed_inferences']}")
        logger.info(f"   Overall success rate: {success_rate:.1f}%")
        
        # Weather usage summary
        total_with_weather = len(self.stats['countries_with_weather'])
        total_without_weather = len(self.stats['countries_without_weather'])
        
        if total_with_weather + total_without_weather > 0:
            weather_percentage = (total_with_weather / (total_with_weather + total_without_weather)) * 100
            logger.info(f"\n🌦️ WEATHER USAGE SUMMARY:")
            logger.info(f"   Countries with weather: {total_with_weather} ({weather_percentage:.1f}%)")
            logger.info(f"   Countries without weather: {total_without_weather} ({100-weather_percentage:.1f}%)")
            
            # List countries without weather if any
            if total_without_weather > 0:
                logger.info(f"\n📋 COUNTRIES WITHOUT WEATHER DATA:")
                for country in sorted(self.stats['countries_without_weather']):
                    logger.info(f"   ❌ {country}")
        logger.info(f"   Total processing time: {total_time:.1f} seconds ({total_time/3600:.2f} hours)")
        
        if self.stats['processing_times']:
            avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            logger.info(f"   Average time per country: {avg_time:.1f} seconds")
            
            # Estimate time for full dataset
            remaining_countries = self.stats['total_countries'] - (self.stats['successful_extractions'] + self.stats['failed_extractions'] + self.stats['skipped_countries'])
            if remaining_countries > 0:
                estimated_remaining = remaining_countries * avg_time / 60  # minutes
                logger.info(f"   Estimated time for remaining countries: {estimated_remaining:.1f} minutes")
        
        # Save weather usage report
        self._save_weather_usage_report()
        
        logger.info("="*60)
        logger.info(f"📁 RESULTS SAVED TO: {self.results_dir}")
        logger.info(f"📁 EXTRACTED DATA IN: {self.extracted_data_dir}")
        logger.info(f"📄 DETAILED LOG: country_inference_batch.log")
        logger.info(f"📊 WEATHER REPORT: weather_usage_report.csv")
    
    def get_countries_with_ili_data(self) -> List[str]:
        """Get list of countries with ILI data."""
        return self.country_summary[self.country_summary['total_ili_cases'] > 0]['country'].tolist()
    
    def get_countries_with_ari_data(self) -> List[str]:
        """Get list of countries with ARI data."""
        return self.country_summary[self.country_summary['total_ari_cases'] > 0]['country'].tolist()
    
    def get_countries_with_both_data(self) -> List[str]:
        """Get list of countries with both ILI and ARI data."""
        return self.country_summary[
            (self.country_summary['total_ili_cases'] > 0) & 
            (self.country_summary['total_ari_cases'] > 0)
        ]['country'].tolist()
    
    def _save_weather_usage_report(self) -> None:
        """Save weather usage data to CSV file."""
        try:
            import csv
            
            # Create weather usage data
            weather_data = []
            for country, used_weather in self.stats['weather_usage_by_country'].items():
                weather_data.append({
                    'country': country,
                    'used_weather': 'YES' if used_weather else 'NO',
                    'weather_status': 'Enabled' if used_weather else 'Disabled'
                })
            
            # Add summary statistics
            total_with = len(self.stats['countries_with_weather'])
            total_without = len(self.stats['countries_without_weather'])
            total = total_with + total_without
            
            weather_data.append({
                'country': 'TOTAL',
                'used_weather': f'{total_with}/{total}',
                'weather_status': f'{(total_with/total*100):.1f}% with weather' if total > 0 else 'N/A'
            })
            
            # Save to CSV
            with open('weather_usage_report.csv', 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['country', 'used_weather', 'weather_status']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(weather_data)
            
            logger.info(f"📊 Saved weather usage report with {len(weather_data)-1} countries")
            
        except Exception as e:
            logger.error(f"❌ Failed to save weather usage report: {e}")


def main():
    """Main function for batch processing."""
    parser = argparse.ArgumentParser(
        description="Batch process country inference with ILI/ARI fallback"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of countries to process (for testing)"
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=0,
        help="Start processing from country index N (for resuming)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a quick test with 3 countries"
    )
    parser.add_argument(
        "--show_stats",
        action="store_true",
        help="Show dataset statistics and exit"
    )
    parser.add_argument(
        "--skip_weather_on_error",
        action="store_true",
        help="Skip weather data instead of failing when country not found"
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = CountryInferenceProcessor(skip_weather_on_error=args.skip_weather_on_error)
    
    if args.show_stats:
        logger.info(f"📊 DATASET STATISTICS:")
        logger.info(f"   Total countries: {processor.stats['total_countries']}")
        logger.info(f"   Countries with ILI data: {len(processor.get_countries_with_ili_data())}")
        logger.info(f"   Countries with ARI data: {len(processor.get_countries_with_ari_data())}")
        logger.info(f"   Countries with both: {len(processor.get_countries_with_both_data())}")
        return
    
    if args.test:
        logger.info("🧪 Running test with 3 sample countries...")
        args.limit = 3
    
    # Run batch processing
    processor.process_all_countries(limit=args.limit, start_from=args.start_from)


if __name__ == "__main__":
    main()