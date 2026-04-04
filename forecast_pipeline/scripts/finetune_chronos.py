#!/usr/bin/env python3
"""
Step 4: Fine-tune Chronos-2 on global ILI data.

Loads pre-built training samples from data/finetune_train.pkl, fine-tunes
Chronos-2 with W&B logging, and saves checkpoint to models/chronos2-ili-finetuned/.

Prerequisites:
    python scripts/prefetch_weather.py    # fetch/cache all country weather
    python scripts/prepare_finetune_data.py  # build training/val sample files

Usage:
    cd forecast_pipeline
    python scripts/finetune_chronos.py [--steps 2000] [--lr 1e-6] [--batch_size 32]
"""

import os
import sys
import pickle
import argparse
from pathlib import Path

DEFAULT_TRAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "finetune_train.pkl")
DEFAULT_VAL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "finetune_val.pkl")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "chronos2-ili-finetuned")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--prediction_length", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=260,
                        help="Max context length for training windows")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging (dry test mode)")
    parser.add_argument("--train_data", type=str, default=DEFAULT_TRAIN_PATH,
                        help="Path to training data pickle")
    parser.add_argument("--val_data", type=str, default=DEFAULT_VAL_PATH,
                        help="Path to validation data pickle")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for finetuned model checkpoint")
    return parser.parse_args()


def setup_wandb(args):
    """Initialize W&B. Pause and ask user to log in if not authenticated."""
    if args.no_wandb:
        print("W&B disabled via --no_wandb")
        return False

    try:
        import wandb
    except ImportError:
        print("\n⚠️  Weights & Biases not installed.")
        print("Run: pip install wandb")
        print("Then re-run this script.")
        sys.exit(1)

    try:
        wandb.init(
            project="chronos2-ili-finetune",
            config={
                "num_steps": args.steps,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "prediction_length": args.prediction_length,
                "context_length": args.context_length,
                "base_model": "amazon/chronos-2",
                "finetune_mode": "full",
            }
        )
        print(f"W&B run: {wandb.run.url}")
        return True
    except wandb.errors.AuthenticationError:
        print("\n" + "="*60)
        print("⚠️  Not logged in to Weights & Biases.")
        print()
        print("To log in, run one of:")
        print("  wandb login")
        print("  export WANDB_API_KEY=<your-key>")
        print()
        print("Then re-run this script.")
        print("="*60)
        sys.exit(1)


def main():
    args = parse_args()

    # ---- Check input files ----
    if not os.path.exists(args.train_data):
        print(f"Error: training data not found at {args.train_data}")
        print("Run: python scripts/prepare_finetune_data.py")
        sys.exit(1)

    # ---- W&B setup ----
    use_wandb = setup_wandb(args)

    # ---- Load data ----
    print(f"Loading training data from {args.train_data}...")
    with open(args.train_data, "rb") as f:
        train_samples = pickle.load(f)
    print(f"  {len(train_samples):,} training samples")

    val_samples = None
    if os.path.exists(args.val_data):
        with open(args.val_data, "rb") as f:
            val_samples = pickle.load(f)
        print(f"  {len(val_samples):,} validation samples")

    # ---- Load model ----
    print("\nLoading Chronos-2 base model...")
    from chronos import Chronos2Pipeline
    import torch
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
    )
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Model on {device}")

    # ---- Fine-tune ----
    print(f"\nStarting fine-tuning: {args.steps} steps, lr={args.lr}, batch={args.batch_size}")
    print(f"Output: {args.output_dir}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    extra_kwargs = {}
    if use_wandb:
        extra_kwargs["report_to"] = "wandb"

    finetuned = pipeline.fit(
        inputs=train_samples,
        prediction_length=args.prediction_length,
        validation_inputs=val_samples,
        finetune_mode="full",
        context_length=args.context_length,
        learning_rate=args.lr,
        num_steps=args.steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        finetuned_ckpt_name="finetuned-ckpt",
        **extra_kwargs,
    )

    print(f"\nFine-tuning complete. Checkpoint saved to: {args.output_dir}")

    if use_wandb:
        import wandb
        wandb.finish()

    return finetuned


if __name__ == "__main__":
    main()
