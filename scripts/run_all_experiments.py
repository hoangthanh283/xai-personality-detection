#!/usr/bin/env python
"""Orchestrate the full experiment matrix.

Usage:
    python scripts/run_all_experiments.py --group baselines
    python scripts/run_all_experiments.py --group rag_xpr
    python scripts/run_all_experiments.py --group ablations
    python scripts/run_all_experiments.py --all --wandb_project rag-xpr
"""
import argparse
import subprocess
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging


def run_command(cmd: list[str]) -> int:
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error(f"Command failed with exit code {result.returncode}")
    return result.returncode


def run_baselines(args) -> None:
    """Run all baseline experiments (B1-B9)."""
    wandb = ["--wandb_project", args.wandb_project] if args.wandb_project else []
    base_cmd = [sys.executable, "scripts/train_baseline.py"]

    experiments = [
        # ML baselines
        ["--model", "logistic_regression", "--dataset", "mbti", "--task", "16class"],
        ["--model", "svm", "--dataset", "mbti", "--task", "16class"],
        ["--model", "xgboost", "--dataset", "mbti", "--task", "16class"],
        ["--model", "ensemble", "--dataset", "mbti", "--task", "16class"],
        # Transformer baselines
        ["--model", "distilbert", "--dataset", "mbti", "--task", "16class"],
        ["--model", "roberta", "--dataset", "mbti", "--task", "16class"],
        ["--model", "distilbert", "--dataset", "mbti", "--task", "4dim"],
        ["--model", "distilbert", "--dataset", "essays", "--task", "ocean_binary"],
        ["--model", "distilbert", "--dataset", "pandora", "--task", "ocean_binary"],
    ]

    for exp_args in experiments:
        run_command(base_cmd + exp_args + wandb)


def run_rag_xpr(args) -> None:
    """Run RAG-XPR experiments (R1-R6)."""
    base_cmd = [sys.executable, "scripts/run_rag_xpr.py", "--config", "configs/rag_xpr_config.yaml"]

    experiments = [
        [
            "--dataset", "mbti",
            "--llm_provider", "openrouter",
            "--llm_model", "qwen/qwen3.6-plus-preview:free",
        ],
        ["--dataset", "mbti"],
        ["--dataset", "mbti", "--llm_provider", "vllm", "--llm_model", "meta-llama/Llama-3.1-8B-Instruct"],
        ["--dataset", "essays", "--framework", "ocean"],
        ["--dataset", "pandora", "--framework", "ocean"],
        ["--dataset", "personality_evd"],
    ]

    for exp_args in experiments:
        run_command(base_cmd + exp_args)


def run_ablations(args) -> None:
    """Run ablation studies (A1-A8)."""
    base_cmd = [sys.executable, "scripts/run_rag_xpr.py", "--config", "configs/rag_xpr_config.yaml",
                "--dataset", "mbti", "--sample", "500"]
    ablations = ["no_kb", "no_evidence_filter", "no_cope"]
    for ablation in ablations:
        run_command(base_cmd + ["--ablation", ablation, "--output", f"outputs/predictions/ablation_{ablation}.jsonl"])


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--group", choices=["baselines", "rag_xpr", "ablations", "personality_evd"])
    parser.add_argument("--all", action="store_true", help="Run all experiment groups")
    parser.add_argument("--wandb_project", default="rag-xpr")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds")
    parser.add_argument("--output_dir", default="outputs/")
    args = parser.parse_args()

    setup_logging()

    seeds = [int(s) for s in args.seeds.split(",")]
    logger.info(f"Running with seeds: {seeds}")

    groups_to_run = []
    if args.all:
        groups_to_run = ["baselines", "rag_xpr", "ablations"]
    elif args.group:
        groups_to_run = [args.group]

    for group in groups_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment group: {group}")
        logger.info(f"{'='*60}")
        if group == "baselines":
            run_baselines(args)
        elif group == "rag_xpr":
            run_rag_xpr(args)
        elif group == "ablations":
            run_ablations(args)


if __name__ == "__main__":
    main()
