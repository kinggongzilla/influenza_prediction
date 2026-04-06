#!/usr/bin/env python3
"""
Covariate ablation experiments for Chronos-2 influenza forecasting.

Runs all 2^5 = 32 covariate combinations through:
  1. Training data preparation
  2. Fine-tuning
  3. Evaluation with run_evaluation.py (WIS, coverage, rWIS vs naive baseline)
  4. Evaluation against Influcast models (pairwise rWIS)

Results are collected into:
  - results/ablation/ablation_results.json  (machine-readable)
  - results/ablation/ablation_summary.md    (human-readable report)

Usage:
    cd forecast_pipeline
    python scripts/run_ablation.py                     # Run all 32 experiments
    python scripts/run_ablation.py --experiments 0,1,6 # Run specific experiments
    python scripts/run_ablation.py --eval_only         # Re-evaluate existing models
    python scripts/run_ablation.py --skip_influcast    # Skip Influcast comparison
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PIPELINE_DIR, "models", "ablation")
EVAL_RESULTS_DIR = os.path.join(PIPELINE_DIR, "results", "evaluation")
ABLATION_RESULTS_DIR = os.path.join(PIPELINE_DIR, "results", "ablation")

ALL_COVARIATES = ["data_type", "hemisphere", "week_of_year", "weather", "neighbors"]
SUFFIX_MAP = {
    "data_type": "dt",
    "hemisphere": "hemi",
    "week_of_year": "woy",
    "weather": "weather",
    "neighbors": "neighbors",
}

# Training defaults (match user's recent training run)
DEFAULT_STEPS = 4000
DEFAULT_LR = 5e-6
DEFAULT_BATCH_SIZE = 32

# Evaluation defaults
EVAL_START_DATE = "2025-10-01"
PREDICTION_HORIZON = 4
INFLUCAST_SEASON = "2024-25"


def build_experiments():
    """Generate all 2^5 covariate combinations plus a zero-shot baseline."""
    experiments = []

    # Experiment 0: zero-shot base Chronos-2 (no finetuning)
    experiments.append({
        "id": 0,
        "covariates": [],
        "suffix": "base-zeroshot",
        "model_dir": None,  # signals: use pretrained model directly
        "model_path": "amazon/chronos-2",
        "skip_training": True,
    })

    idx = 1
    for r in range(len(ALL_COVARIATES) + 1):
        for combo in itertools.combinations(ALL_COVARIATES, r):
            covs = list(combo)
            suffix = "-".join(SUFFIX_MAP[c] for c in covs) if covs else "nocov"
            model_dir = os.path.join(MODELS_DIR, suffix)
            experiments.append({
                "id": idx,
                "covariates": covs,
                "suffix": suffix,
                "model_dir": model_dir,
                "model_path": os.path.join(model_dir, "finetuned-ckpt"),
                "skip_training": False,
            })
            idx += 1
    return experiments


def run_cmd(cmd, description, cwd=None):
    """Run a subprocess command, streaming output. Returns (success, elapsed_seconds)."""
    print(f"\n{'─'*70}")
    print(f"  {description}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'─'*70}")
    start = time.time()
    result = subprocess.run(cmd, cwd=cwd or PIPELINE_DIR)
    elapsed = time.time() - start
    success = result.returncode == 0
    status = "OK" if success else f"FAILED (exit {result.returncode})"
    print(f"  → {status} ({elapsed:.0f}s)")
    return success, elapsed


def prepare_data(experiment):
    """Prepare training data for an experiment."""
    cmd = [sys.executable, "scripts/prepare_finetune_data.py",
           "--exclude_countries", "Thailand", "--covariates"]
    cmd.extend(experiment["covariates"])
    label = ", ".join(experiment["covariates"]) or "none"
    return run_cmd(cmd, f"Prepare data [{experiment['suffix']}] covariates=[{label}]")


def finetune(experiment, steps, lr, batch_size):
    """Fine-tune a model for an experiment. Skips if checkpoint exists."""
    ckpt_config = os.path.join(experiment["model_dir"], "finetuned-ckpt", "config.json")
    if os.path.exists(ckpt_config):
        print(f"\n  Checkpoint exists: {experiment['model_dir']}/finetuned-ckpt — skipping training")
        return True, 0.0

    cmd = [sys.executable, "scripts/finetune_chronos.py",
           "--output_dir", experiment["model_dir"],
           "--steps", str(steps),
           "--lr", str(lr),
           "--batch_size", str(batch_size)]
    return run_cmd(cmd, f"Fine-tune [{experiment['suffix']}] steps={steps} lr={lr}")


def evaluate_rolling(experiment):
    """Run rolling evaluation (run_evaluation.py)."""
    output_file = f"ablation_{experiment['suffix']}.json"
    model_path = experiment["model_path"]
    cmd = [sys.executable, "run_evaluation.py",
           "--model_path", model_path,
           "--eval_start_date", EVAL_START_DATE,
           "--prediction_horizon", str(PREDICTION_HORIZON),
           "--output", output_file,
           "--covariates"]
    cmd.extend(experiment["covariates"])
    return run_cmd(cmd, f"Evaluate [{experiment['suffix']}] → {output_file}")


def evaluate_influcast(experiment):
    """Run Influcast pairwise evaluation."""
    output_json = os.path.join(ABLATION_RESULTS_DIR, f"influcast_{experiment['suffix']}.json")
    model_path = experiment["model_path"]
    cmd = [sys.executable, "scripts/eval_influcast_pairwise.py",
           "--model_path", model_path,
           "--season", INFLUCAST_SEASON,
           "--output_json", output_json,
           "--covariates"]
    cmd.extend(experiment["covariates"])
    return run_cmd(cmd, f"Influcast [{experiment['suffix']}] season={INFLUCAST_SEASON}")


def collect_results(experiments):
    """Collect all evaluation results and generate summary."""
    results = []

    for exp in experiments:
        entry = {
            "id": exp["id"],
            "suffix": exp["suffix"],
            "covariates": exp["covariates"],
            "covariates_str": ", ".join(exp["covariates"]) or "(none)",
        }

        # Load rolling eval results
        eval_file = os.path.join(EVAL_RESULTS_DIR, f"ablation_{exp['suffix']}.json")
        if os.path.exists(eval_file):
            with open(eval_file) as f:
                eval_data = json.load(f)
            countries = eval_data.get("countries", {})
            if countries:
                rwis_vals = [v["relative_wis"] for v in countries.values() if v.get("relative_wis") is not None]
                wis_vals = [v["wis"] for v in countries.values() if v.get("wis") is not None]
                cov50_vals = [v["coverage_50"] for v in countries.values() if v.get("coverage_50") is not None]
                cov95_vals = [v["coverage_95"] for v in countries.values() if v.get("coverage_95") is not None]
                mae_vals = [v["mae"] for v in countries.values() if v.get("mae") is not None]

                entry["mean_rwis"] = sum(rwis_vals) / len(rwis_vals) if rwis_vals else None
                entry["mean_wis"] = sum(wis_vals) / len(wis_vals) if wis_vals else None
                entry["mean_cov50"] = sum(cov50_vals) / len(cov50_vals) if cov50_vals else None
                entry["mean_cov95"] = sum(cov95_vals) / len(cov95_vals) if cov95_vals else None
                entry["mean_mae"] = sum(mae_vals) / len(mae_vals) if mae_vals else None
                entry["per_country"] = {
                    name: {
                        "rwis": v.get("relative_wis"),
                        "wis": v.get("wis"),
                        "cov50": v.get("coverage_50"),
                        "cov95": v.get("coverage_95"),
                        "mae": v.get("mae"),
                        "per_horizon_rwis": v.get("per_horizon_rwis", {}),
                    }
                    for name, v in countries.items()
                }
                # Aggregate per-horizon rWIS across countries
                per_h = {}
                for h in ["1", "2", "3", "4"]:
                    h_vals = [v.get("per_horizon_rwis", {}).get(h) for v in countries.values()
                              if v.get("per_horizon_rwis", {}).get(h) is not None]
                    per_h[h] = float(np.mean(h_vals)) if h_vals else None
                entry["per_horizon_rwis"] = per_h
            else:
                entry["mean_rwis"] = None
        else:
            entry["mean_rwis"] = None

        # Load Influcast results
        influcast_file = os.path.join(ABLATION_RESULTS_DIR, f"influcast_{exp['suffix']}.json")
        if os.path.exists(influcast_file):
            with open(influcast_file) as f:
                infl_data = json.load(f)
            chronos_models = infl_data.get("chronos_models", {})
            if chronos_models:
                # Take the first (and usually only) Chronos model
                cm = list(chronos_models.values())[0]
                entry["influcast_simple_rwis"] = cm.get("simple_rwis")
                entry["influcast_pairwise_rwis"] = cm.get("pairwise_rwis")
                entry["influcast_cov50"] = cm.get("cov50")
                entry["influcast_cov90"] = cm.get("cov90")
                entry["influcast_per_horizon"] = cm.get("per_horizon_rwis", {})
            else:
                entry["influcast_simple_rwis"] = None
                entry["influcast_pairwise_rwis"] = None
        else:
            entry["influcast_simple_rwis"] = None
            entry["influcast_pairwise_rwis"] = None

        results.append(entry)

    return results


def write_summary(results):
    """Write markdown summary report."""
    lines = []
    lines.append("# Covariate Ablation Results")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Training: {DEFAULT_STEPS} steps, lr={DEFAULT_LR}, batch_size={DEFAULT_BATCH_SIZE}")
    lines.append(f"Evaluation: eval_start={EVAL_START_DATE}, horizon={PREDICTION_HORIZON}wk")
    lines.append(f"Influcast season: {INFLUCAST_SEASON}")
    lines.append("")

    # Overview table
    lines.append("## Overview")
    lines.append("")
    lines.append("| # | Covariates | Mean rWIS | Mean WIS | Cov50% | Cov95% | MAE | Influcast Simple rWIS | Influcast Pairwise rWIS | Influcast Cov50% | Influcast Cov90% |")
    lines.append("|---|-----------|----------|---------|--------|--------|-----|----------------------|------------------------|-----------------|-----------------|")

    # Sort by mean rWIS (best first)
    sorted_results = sorted(results, key=lambda x: x.get("mean_rwis") or 999)

    for r in sorted_results:
        def fmt(v, decimals=3):
            return f"{v:.{decimals}f}" if v is not None else "—"

        lines.append(
            f"| {r['id']} | {r['covariates_str']} "
            f"| {fmt(r.get('mean_rwis'))} "
            f"| {fmt(r.get('mean_wis'), 1)} "
            f"| {fmt(r.get('mean_cov50'), 1)} "
            f"| {fmt(r.get('mean_cov95'), 1)} "
            f"| {fmt(r.get('mean_mae'), 2)} "
            f"| {fmt(r.get('influcast_simple_rwis'))} "
            f"| {fmt(r.get('influcast_pairwise_rwis'))} "
            f"| {fmt(r.get('influcast_cov50'), 1)} "
            f"| {fmt(r.get('influcast_cov90'), 1)} |"
        )

    # Per-country breakdown
    lines.append("")
    lines.append("## Per-Country rWIS")
    lines.append("")

    # Collect all country names
    all_countries = set()
    for r in results:
        if "per_country" in r:
            all_countries.update(r["per_country"].keys())
    all_countries = sorted(all_countries)

    if all_countries:
        header = "| # | Covariates | " + " | ".join(all_countries) + " |"
        sep = "|---|-----------|" + "|".join(["------"] * len(all_countries)) + "|"
        lines.append(header)
        lines.append(sep)

        for r in sorted_results:
            pc = r.get("per_country", {})
            vals = []
            for c in all_countries:
                v = pc.get(c, {}).get("rwis")
                vals.append(f"{v:.3f}" if v is not None else "—")
            lines.append(f"| {r['id']} | {r['covariates_str']} | " + " | ".join(vals) + " |")

    # Rolling eval per-horizon breakdown
    lines.append("")
    lines.append("## Rolling Eval Per-Horizon rWIS")
    lines.append("")
    lines.append("| # | Covariates | H1 | H2 | H3 | H4 |")
    lines.append("|---|-----------|----|----|----|----|")

    for r in sorted_results:
        ph = r.get("per_horizon_rwis", {})
        h_vals = []
        for h in ["1", "2", "3", "4"]:
            v = ph.get(h)
            h_vals.append(f"{v:.3f}" if v is not None else "—")
        lines.append(f"| {r['id']} | {r['covariates_str']} | " + " | ".join(h_vals) + " |")

    # Influcast per-horizon breakdown
    lines.append("")
    lines.append("## Influcast Per-Horizon Simple rWIS")
    lines.append("")
    lines.append("| # | Covariates | H1 | H2 | H3 | H4 |")
    lines.append("|---|-----------|----|----|----|----|")

    for r in sorted_results:
        ph = r.get("influcast_per_horizon", {})
        h_vals = []
        for h in ["1", "2", "3", "4"]:
            v = ph.get(h)
            h_vals.append(f"{v:.3f}" if v is not None else "—")
        lines.append(f"| {r['id']} | {r['covariates_str']} | " + " | ".join(h_vals) + " |")

    # Best configurations
    lines.append("")
    lines.append("## Best Configurations")
    lines.append("")

    valid_rwis = [r for r in results if r.get("mean_rwis") is not None]
    valid_infl_simple = [r for r in results if r.get("influcast_simple_rwis") is not None]
    valid_infl_pairwise = [r for r in results if r.get("influcast_pairwise_rwis") is not None]
    valid_cov95 = [r for r in results if r.get("mean_cov95") is not None]

    if valid_rwis:
        best = min(valid_rwis, key=lambda x: x["mean_rwis"])
        lines.append(f"- **Best mean rWIS (vs naive):** #{best['id']} ({best['covariates_str']}) = {best['mean_rwis']:.3f}")
    if valid_infl_simple:
        best = min(valid_infl_simple, key=lambda x: x["influcast_simple_rwis"])
        lines.append(f"- **Best Influcast simple rWIS:** #{best['id']} ({best['covariates_str']}) = {best['influcast_simple_rwis']:.3f}")
    if valid_infl_pairwise:
        best = min(valid_infl_pairwise, key=lambda x: x["influcast_pairwise_rwis"])
        lines.append(f"- **Best Influcast pairwise rWIS:** #{best['id']} ({best['covariates_str']}) = {best['influcast_pairwise_rwis']:.3f}")
    if valid_cov95:
        best = max(valid_cov95, key=lambda x: x["mean_cov95"])
        lines.append(f"- **Best 95% coverage:** #{best['id']} ({best['covariates_str']}) = {best['mean_cov95']:.1f}%")

    # Individual covariate impact
    lines.append("")
    lines.append("## Individual Covariate Impact")
    lines.append("")
    lines.append("Average rWIS improvement when adding each covariate (across all combinations):")
    lines.append("")

    nocov_results = {r["suffix"]: r for r in results}
    for cov in ALL_COVARIATES:
        improvements = []
        for r in results:
            if cov in r["covariates"]:
                # Find the version without this covariate
                without = [c for c in r["covariates"] if c != cov]
                without_suffix = "-".join(SUFFIX_MAP[c] for c in without) if without else "nocov"
                without_r = nocov_results.get(without_suffix)
                if without_r and r.get("mean_rwis") is not None and without_r.get("mean_rwis") is not None:
                    improvements.append(without_r["mean_rwis"] - r["mean_rwis"])
        if improvements:
            avg_imp = sum(improvements) / len(improvements)
            direction = "better" if avg_imp > 0 else "worse"
            lines.append(f"- **{cov}**: {avg_imp:+.4f} avg rWIS change ({direction}, n={len(improvements)} pairs)")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Covariate ablation experiments")
    parser.add_argument("--experiments", type=str, default=None,
                        help="Comma-separated experiment IDs to run (default: all)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip data prep and training, only run evaluations")
    parser.add_argument("--skip_influcast", action="store_true",
                        help="Skip Influcast evaluation")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip rolling evaluation (run_evaluation.py)")
    parser.add_argument("--collect_only", action="store_true",
                        help="Only collect results from previous runs, no training or evaluation")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    all_experiments = build_experiments()

    if args.experiments:
        ids = [int(x.strip()) for x in args.experiments.split(",")]
        experiments = [e for e in all_experiments if e["id"] in ids]
        if not experiments:
            print(f"No experiments matched IDs: {ids}")
            sys.exit(1)
    else:
        experiments = all_experiments

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(ABLATION_RESULTS_DIR, exist_ok=True)
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

    print(f"{'='*70}")
    print(f"  COVARIATE ABLATION — {len(experiments)} experiments (incl. zero-shot baseline)")
    print(f"  Training: steps={args.steps}, lr={args.lr}, batch={args.batch_size}")
    print(f"  Eval: start={EVAL_START_DATE}, horizon={PREDICTION_HORIZON}wk")
    print(f"  Influcast: season={INFLUCAST_SEASON}")
    print(f"{'='*70}")

    for i, exp in enumerate(experiments):
        label = ", ".join(exp["covariates"]) or "(none)"
        print(f"  [{exp['id']:>2}] {exp['suffix']:<30} covariates=[{label}]")

    if args.collect_only:
        print("\n  Collect-only mode — skipping all training and evaluation.\n")
    else:
        total_start = time.time()
        succeeded = 0
        failed = []

        for i, exp in enumerate(experiments):
            print(f"\n{'='*70}")
            print(f"  EXPERIMENT {exp['id']}/{len(all_experiments)-1} ({i+1}/{len(experiments)}): {exp['suffix']}")
            label = ", ".join(exp["covariates"]) or "(none)"
            print(f"  Covariates: [{label}]")
            print(f"{'='*70}")

            exp_ok = True

            if exp.get("skip_training"):
                print(f"  Zero-shot model ({exp['model_path']}) — skipping training")
            elif not args.eval_only:
                # Step 1: Prepare training data
                ok, _ = prepare_data(exp)
                if not ok:
                    print(f"  FAILED: data preparation for {exp['suffix']}")
                    failed.append((exp["id"], exp["suffix"], "prepare"))
                    continue

                # Step 2: Fine-tune
                ok, _ = finetune(exp, args.steps, args.lr, args.batch_size)
                if not ok:
                    print(f"  FAILED: fine-tuning for {exp['suffix']}")
                    failed.append((exp["id"], exp["suffix"], "finetune"))
                    continue

            # Check model exists before evaluation (skip for HuggingFace models)
            if not exp.get("skip_training"):
                ckpt_path = os.path.join(exp["model_dir"], "finetuned-ckpt", "config.json")
                if not os.path.exists(ckpt_path):
                    print(f"  No checkpoint at {exp['model_dir']}/finetuned-ckpt — skipping evaluation")
                    failed.append((exp["id"], exp["suffix"], "no_checkpoint"))
                    continue

            # Step 3: Rolling evaluation
            if not args.skip_eval:
                ok, _ = evaluate_rolling(exp)
                if not ok:
                    exp_ok = False
                    failed.append((exp["id"], exp["suffix"], "eval"))

            # Step 4: Influcast evaluation
            if not args.skip_influcast:
                ok, _ = evaluate_influcast(exp)
                if not ok:
                    exp_ok = False
                    failed.append((exp["id"], exp["suffix"], "influcast"))

            if exp_ok:
                succeeded += 1

        total_elapsed = time.time() - total_start
        print(f"\n{'='*70}")
        print(f"  TRAINING & EVALUATION COMPLETE")
        print(f"  {succeeded}/{len(experiments)} succeeded, {len(failed)} failures")
        print(f"  Total time: {total_elapsed/60:.1f} min")
        if failed:
            print(f"  Failed: {failed}")
        print(f"{'='*70}")

    # Collect results (always runs, even if some experiments failed)
    print("\nCollecting results...")
    results = collect_results(all_experiments if args.collect_only else experiments)

    # Write full JSON results
    results_json_path = os.path.join(ABLATION_RESULTS_DIR, "ablation_results.json")
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  JSON → {results_json_path}")

    # Write markdown summary
    summary = write_summary(results)
    summary_path = os.path.join(ABLATION_RESULTS_DIR, "ablation_summary.md")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"  Summary → {summary_path}")

    print(f"\nDone. View results: cat {summary_path}")


if __name__ == "__main__":
    main()
