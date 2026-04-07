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
INFLUCAST_SEASON = "2025-26"


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


def evaluate_rolling(experiment, nocov_eval=False, countries_str=None, weather_cache_only=False):
    """Run rolling evaluation (run_evaluation.py)."""
    suffix = experiment['suffix']
    if nocov_eval:
        output_file = f"ablation_{suffix}_nocov-eval.json"
    else:
        output_file = f"ablation_{suffix}.json"
    model_path = experiment["model_path"]
    cmd = [sys.executable, "run_evaluation.py",
           "--model_path", model_path,
           "--eval_start_date", EVAL_START_DATE,
           "--prediction_horizon", str(PREDICTION_HORIZON),
           "--output", output_file,
           "--quiet",
           "--covariates"]
    if not nocov_eval:
        cmd.extend(experiment["covariates"])
    if countries_str:
        cmd.extend(["--countries", countries_str])
    if weather_cache_only:
        cmd.append("--weather_cache_only")
    return run_cmd(cmd, f"Evaluate [{suffix}] nocov_eval={nocov_eval} → {output_file}")


def evaluate_influcast(experiment, nocov_eval=False):
    """Run Influcast pairwise evaluation."""
    suffix = experiment['suffix']
    if nocov_eval:
        output_json = os.path.join(ABLATION_RESULTS_DIR, f"influcast_{suffix}_nocov-eval.json")
    else:
        output_json = os.path.join(ABLATION_RESULTS_DIR, f"influcast_{suffix}.json")
    model_path = experiment["model_path"]
    cmd = [sys.executable, "scripts/eval_influcast_pairwise.py",
           "--model_path", model_path,
           "--season", INFLUCAST_SEASON,
           "--output_json", output_json,
           "--covariates"]
    if not nocov_eval:
        cmd.extend(experiment["covariates"])
    return run_cmd(cmd, f"Influcast [{suffix}] nocov_eval={nocov_eval} season={INFLUCAST_SEASON}")


def collect_results(experiments, nocov_eval=False):
    """Collect all evaluation results and generate summary."""
    results = []
    file_tag = "_nocov-eval" if nocov_eval else ""

    for exp in experiments:
        entry = {
            "id": exp["id"],
            "suffix": exp["suffix"],
            "covariates": exp["covariates"],
            "covariates_str": ", ".join(exp["covariates"]) or "(none)",
        }

        # Load rolling eval results
        eval_file = os.path.join(EVAL_RESULTS_DIR, f"ablation_{exp['suffix']}{file_tag}.json")
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

                mape_vals = [v["mape_pct"] for v in countries.values() if v.get("mape_pct") is not None]

                entry["mean_rwis"] = sum(rwis_vals) / len(rwis_vals) if rwis_vals else None
                entry["mean_wis"] = sum(wis_vals) / len(wis_vals) if wis_vals else None
                entry["mean_cov50"] = sum(cov50_vals) / len(cov50_vals) if cov50_vals else None
                entry["mean_cov95"] = sum(cov95_vals) / len(cov95_vals) if cov95_vals else None
                entry["mean_mae"] = sum(mae_vals) / len(mae_vals) if mae_vals else None
                entry["mean_mape"] = sum(mape_vals) / len(mape_vals) if mape_vals else None
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
        influcast_file = os.path.join(ABLATION_RESULTS_DIR, f"influcast_{exp['suffix']}{file_tag}.json")
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
                entry["influcast_mape"] = cm.get("mape_pct")
                entry["influcast_per_horizon_mape"] = cm.get("per_horizon_mape", {})
            else:
                entry["influcast_simple_rwis"] = None
                entry["influcast_pairwise_rwis"] = None
                entry["influcast_mape"] = None
        else:
            entry["influcast_simple_rwis"] = None
            entry["influcast_pairwise_rwis"] = None
            entry["influcast_mape"] = None

        results.append(entry)

    return results


def write_summary(results, nocov_eval=False):
    """Write markdown summary report."""
    lines = []
    title = "# Covariate Ablation Results"
    if nocov_eval:
        title += " (No-Covariate Evaluation)"
    lines.append(title)
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Training: {DEFAULT_STEPS} steps, lr={DEFAULT_LR}, batch_size={DEFAULT_BATCH_SIZE}")
    lines.append(f"Evaluation: eval_start={EVAL_START_DATE}, horizon={PREDICTION_HORIZON}wk")
    lines.append(f"Influcast season: {INFLUCAST_SEASON}")
    lines.append("")

    # Overview table — split into Rolling Eval and Influcast for readability
    # Sort by mean rWIS (best first)
    sorted_results = sorted(results, key=lambda x: x.get("mean_rwis") or 999)

    def fmt(v, decimals=3):
        return f"{v:.{decimals}f}" if v is not None else "—"

    lines.append("## Rolling Evaluation (all countries)")
    lines.append("")
    lines.append("| # | Covariates | rWIS | MAPE% | WIS | Cov50% | Cov95% | MAE | Countries |")
    lines.append("|---|-----------|------|-------|-----|--------|--------|-----|-----------|")

    for r in sorted_results:
        n_countries = len(r.get("per_country", {})) if "per_country" in r else "—"
        lines.append(
            f"| {r['id']} | {r['covariates_str']} "
            f"| {fmt(r.get('mean_rwis'))} "
            f"| {fmt(r.get('mean_mape'), 1)} "
            f"| {fmt(r.get('mean_wis'), 1)} "
            f"| {fmt(r.get('mean_cov50'), 1)} "
            f"| {fmt(r.get('mean_cov95'), 1)} "
            f"| {fmt(r.get('mean_mae'), 2)} "
            f"| {n_countries} |"
        )

    lines.append("")
    lines.append("## Influcast Evaluation (Italy)")
    lines.append("")
    lines.append("| # | Covariates | Simple rWIS | MAPE% | Pairwise rWIS | Cov50% | Cov90% |")
    lines.append("|---|-----------|------------|-------|--------------|--------|--------|")

    for r in sorted_results:
        lines.append(
            f"| {r['id']} | {r['covariates_str']} "
            f"| {fmt(r.get('influcast_simple_rwis'))} "
            f"| {fmt(r.get('influcast_mape'), 1)} "
            f"| {fmt(r.get('influcast_pairwise_rwis'))} "
            f"| {fmt(r.get('influcast_cov50'), 1)} "
            f"| {fmt(r.get('influcast_cov90'), 1)} |"
        )

    # Per-country breakdown — vertical layout (one row per country) for readability
    lines.append("")
    lines.append("## Per-Country rWIS")
    lines.append("")

    # Collect all country names
    all_countries = set()
    for r in results:
        if "per_country" in r:
            all_countries.update(r["per_country"].keys())
    all_countries = sorted(all_countries)

    if all_countries and len(sorted_results) <= 10:
        # Few experiments: use wide table (experiments as columns)
        header = "| Country | " + " | ".join(f"#{r['id']}" for r in sorted_results) + " |"
        sep = "|---------|" + "|".join(["------"] * len(sorted_results)) + "|"
        lines.append(f"Columns: " + ", ".join(f"#{r['id']}={r['covariates_str']}" for r in sorted_results))
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for c in all_countries:
            vals = []
            for r in sorted_results:
                v = r.get("per_country", {}).get(c, {}).get("rwis")
                vals.append(f"{v:.3f}" if v is not None else "—")
            lines.append(f"| {c} | " + " | ".join(vals) + " |")
    elif all_countries:
        # Many experiments: top-5 table per experiment
        lines.append("Showing top-5 and bottom-5 countries per experiment (sorted by mean rWIS).")
        lines.append("")
        for r in sorted_results:
            pc = r.get("per_country", {})
            if not pc:
                continue
            scored = [(c, v.get("rwis")) for c, v in pc.items() if v.get("rwis") is not None]
            if not scored:
                continue
            scored.sort(key=lambda x: x[1])
            n_scored = len(scored)
            lines.append(f"### #{r['id']} {r['covariates_str']} (mean rWIS={r.get('mean_rwis', '—')}, n={n_scored} countries)")
            lines.append("")
            lines.append("| Rank | Country | rWIS |")
            lines.append("|------|---------|------|")
            for i, (c, v) in enumerate(scored[:5], 1):
                lines.append(f"| {i} | {c} | {v:.3f} |")
            if n_scored > 10:
                lines.append("| ... | ... | ... |")
                for i, (c, v) in enumerate(scored[-5:], n_scored - 4):
                    lines.append(f"| {i} | {c} | {v:.3f} |")
            lines.append("")

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

    # Influcast per-horizon MAPE
    lines.append("")
    lines.append("## Influcast Per-Horizon MAPE%")
    lines.append("")
    lines.append("| # | Covariates | H1 | H2 | H3 | H4 |")
    lines.append("|---|-----------|----|----|----|----|")

    for r in sorted_results:
        ph = r.get("influcast_per_horizon_mape", {})
        h_vals = []
        for h in ["1", "2", "3", "4"]:
            v = ph.get(h)
            h_vals.append(f"{v:.1f}" if v is not None else "—")
        lines.append(f"| {r['id']} | {r['covariates_str']} | " + " | ".join(h_vals) + " |")

    # Best configurations
    lines.append("")
    lines.append("## Best Configurations")
    lines.append("")

    valid_rwis = [r for r in results if r.get("mean_rwis") is not None]
    valid_mape = [r for r in results if r.get("mean_mape") is not None]
    valid_infl_simple = [r for r in results if r.get("influcast_simple_rwis") is not None]
    valid_infl_pairwise = [r for r in results if r.get("influcast_pairwise_rwis") is not None]
    valid_infl_mape = [r for r in results if r.get("influcast_mape") is not None]
    valid_cov95 = [r for r in results if r.get("mean_cov95") is not None]

    if valid_rwis:
        best = min(valid_rwis, key=lambda x: x["mean_rwis"])
        lines.append(f"- **Best rolling rWIS:** #{best['id']} ({best['covariates_str']}) = {best['mean_rwis']:.3f}")
    if valid_mape:
        best = min(valid_mape, key=lambda x: x["mean_mape"])
        lines.append(f"- **Best rolling MAPE:** #{best['id']} ({best['covariates_str']}) = {best['mean_mape']:.1f}%")
    if valid_infl_simple:
        best = min(valid_infl_simple, key=lambda x: x["influcast_simple_rwis"])
        lines.append(f"- **Best Influcast rWIS:** #{best['id']} ({best['covariates_str']}) = {best['influcast_simple_rwis']:.3f}")
    if valid_infl_mape:
        best = min(valid_infl_mape, key=lambda x: x["influcast_mape"])
        lines.append(f"- **Best Influcast MAPE:** #{best['id']} ({best['covariates_str']}) = {best['influcast_mape']:.1f}%")
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
    parser.add_argument("--nocov_eval", action="store_true",
                        help="Evaluate all models WITHOUT covariates (regardless of training covariates)")
    parser.add_argument("--all_countries", action="store_true",
                        help="Evaluate on all training countries (from training_countries.json) instead of default 7")
    parser.add_argument("--weather_cache_only", action="store_true",
                        help="Pass --weather_cache_only to rolling eval (avoids API rate limits)")
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

    # Note: --nocov_eval no longer implies --eval_only, so you can train + eval without covariates in one run

    # Load all-countries list if requested
    countries_str = None
    if args.all_countries:
        tc_path = os.path.join(PIPELINE_DIR, "data", "training_countries.json")
        with open(tc_path) as f:
            tc_data = json.load(f)
        countries_str = ",".join(tc_data["training_countries"])
        print(f"  Using all {len(tc_data['training_countries'])} training countries for evaluation")

    print(f"{'='*70}")
    print(f"  COVARIATE ABLATION — {len(experiments)} experiments (incl. zero-shot baseline)")
    if args.nocov_eval:
        print(f"  MODE: No-covariate evaluation (all models evaluated WITHOUT covariates)")
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
                ok, _ = evaluate_rolling(exp, nocov_eval=args.nocov_eval, countries_str=countries_str, weather_cache_only=args.weather_cache_only)
                if not ok:
                    exp_ok = False
                    failed.append((exp["id"], exp["suffix"], "eval"))

            # Step 4: Influcast evaluation
            if not args.skip_influcast:
                ok, _ = evaluate_influcast(exp, nocov_eval=args.nocov_eval)
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
    results = collect_results(all_experiments if args.collect_only else experiments, nocov_eval=args.nocov_eval)

    # Write full JSON results
    nocov_tag = "_nocov-eval" if args.nocov_eval else ""
    results_json_path = os.path.join(ABLATION_RESULTS_DIR, f"ablation_results{nocov_tag}.json")
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  JSON → {results_json_path}")

    # Write markdown summary
    summary = write_summary(results, nocov_eval=args.nocov_eval)
    summary_path = os.path.join(ABLATION_RESULTS_DIR, f"ablation_summary{nocov_tag}.md")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"  Summary → {summary_path}")

    print(f"\nDone. View results: cat {summary_path}")


if __name__ == "__main__":
    main()
