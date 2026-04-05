# Influenza Prediction Pipeline

Global influenza forecasting using [Chronos-2](https://github.com/amazon-science/chronos-forecasting) fine-tuned on WHO surveillance data (89 countries). Produces 54-week forecasts displayed on an interactive world map.

## Quick Start

All commands run from `forecast_pipeline/`.

```bash
cd forecast_pipeline
```

### 1. Fetch latest WHO data

Downloads ~73MB of WHO fluID surveillance data and regenerates the country summary.

```bash
python update_who_data.py --backup
```

### 2. Extract country time series

Extracts combined ILI+ARI data per country (prefers ILI, falls back to ARI when ILI stops). Produces `data/extracted_data/<Country>_combined.csv` with a `DataType` column (0=ILI, 1=ARI).

```bash
# All training countries
python enhanced_extract_country_data.py --all

# Single country
python enhanced_extract_country_data.py --country Italy
```

### 3. Run inference (all countries)

Loads the fine-tuned model once on GPU and runs inference for all training countries.

```bash
python run_all_country_inference.py \
  --model_path models/chronos2-ili-finetuned-nocov/finetuned-ckpt \
  --countries_file data/training_countries.json
```

Results go to `results/country_predictions/<Country>/`.

### 4. Generate frontend data

```bash
python generate_map_data.py
python generate_country_details.py
```

This writes JSON files to `frontend/public/data/` for the world map and country detail pages.

### 5. Run the frontend

```bash
cd ../frontend
npm install
npm run dev
```

Open http://localhost:3000.

---

## Evaluation

Rolling evaluation with WIS, coverage, and phase-aware MAPE metrics. Use `--covariates` to match training covariates.

```bash
# Single country with data_type covariate
python run_evaluation.py \
  --model_path models/chronos2-ili-finetuned-nocov/finetuned-ckpt \
  --country Italy \
  --covariates data_type \
  --eval_start_date 2025-10-01 \
  --prediction_horizon 4 \
  --plot

# Multiple countries, all covariates
python run_evaluation.py \
  --model_path models/chronos2-ili-finetuned-nocov/finetuned-ckpt \
  --countries "Italy,Germany,France,Spain" \
  --covariates data_type hemisphere week_of_year \
  --eval_start_date 2025-04-01 \
  --prediction_horizon 4 \
  --plot \
  --output eval_2025_2026.json

# Benchmark against Influcast models (Italy, 2025-26 season)
python scripts/eval_influcast_pairwise.py \
  --model_path models/chronos2-ili-finetuned-nocov/finetuned-ckpt \
  --covariates data_type \
  --season 2025-26
```

Plots saved to `results/evaluation/plots/`, metrics to `results/evaluation/`.

---

## Training

### Assess data quality and prepare training data

Scans all extracted CSVs, applies quality filters (min 3yr data, min scale, max NaN%), updates `data/training_countries.json`, and builds sliding-window training samples.

```bash
# Assess only (update training_countries.json without building samples)
python scripts/prepare_finetune_data.py --assess_only --exclude_countries "Thailand"

# Build training data with all covariates (default)
python scripts/prepare_finetune_data.py --exclude_countries "Thailand"

# Select specific covariates (available: data_type, hemisphere, week_of_year, weather, neighbors)
python scripts/prepare_finetune_data.py --covariates data_type week_of_year --exclude_countries "Thailand"

# Without covariates (target only)
python scripts/prepare_finetune_data.py --covariates --exclude_countries "Thailand"

# Preview without saving
python scripts/prepare_finetune_data.py --dry_run
```

Outputs: `data/training_countries.json`, `data/finetune_train.pkl`, `data/finetune_val.pkl`.

### Fine-tune Chronos-2

```bash
# Default (2000 steps, lr=1e-6, batch=32)
python scripts/finetune_chronos.py

# Custom settings
python scripts/finetune_chronos.py \
  --steps 5000 \
  --lr 1e-5 \
  --batch_size 64 \
  --output_dir models/my-finetuned-model

# Without Weights & Biases logging
python scripts/finetune_chronos.py --no_wandb
```

Checkpoint saved to `models/chronos2-ili-finetuned/finetuned-ckpt`.

### Prefetch weather data (optional)

Pre-caches weather from Open-Meteo for all training countries. Stale caches are extended incrementally (only the missing tail is fetched).

```bash
# Check cache status
python scripts/prefetch_weather.py --dry_run

# Fetch missing + extend stale caches
python scripts/prefetch_weather.py --refresh_stale
```

### Full training pipeline

```bash
python update_who_data.py --backup
python enhanced_extract_country_data.py --all
python scripts/prefetch_weather.py --refresh_stale          # optional, only if using weather covariate
python scripts/prepare_finetune_data.py --covariates data_type --exclude_countries "Thailand"
python scripts/finetune_chronos.py --output_dir models/chronos2-ili-finetuned-nocov
```

---

## Project Structure

```
forecast_pipeline/
├── update_who_data.py                 # Download WHO fluID data
├── enhanced_extract_country_data.py   # Extract per-country CSVs (ILI+ARI combined)
├── run_all_country_inference.py       # Batch inference for all countries
├── run_evaluation.py                  # Rolling evaluation with WIS metrics
├── generate_map_data.py               # Generate frontend map JSON
├── generate_country_details.py        # Generate frontend country detail JSON
├── scripts/
│   ├── prepare_finetune_data.py       # Quality assessment + training sample builder
│   ├── finetune_chronos.py            # Fine-tune Chronos-2
│   ├── prefetch_weather.py            # Pre-cache weather data from Open-Meteo
│   └── eval_influcast_pairwise.py     # Benchmark against Influcast models (Italy)
├── data/
│   ├── who_flu_data.csv               # Raw WHO surveillance data
│   ├── extracted_data/                # Per-country CSVs
│   └── training_countries.json        # Country list + data types (auto-generated)
├── models/                            # Fine-tuned model checkpoints
└── results/                           # Inference results and eval plots

frontend/                              # Next.js world map dashboard
```
