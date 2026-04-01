#!/usr/bin/env python
"""Run full evaluation suite.

Usage:
    python scripts/evaluate.py --mode full --predictions_dir outputs/predictions/ --output outputs/reports/
    python scripts/evaluate.py --mode baseline_predictions --models_dir outputs/models/
    python scripts/evaluate.py --mode generate_human_eval --n_samples 50
"""
import argparse
import json
import sys
from pathlib import Path

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging
from src.utils.seed import set_seed


def load_predictions(path: str) -> list[dict]:
    """Load predictions from a JSONL file."""
    predictions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions


def run_full_evaluation(args, config: dict) -> None:
    """Compute full metrics suite for all prediction files."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    from src.evaluation.classification_metrics import \
        compute_classification_metrics
    from src.evaluation.statistical_tests import bootstrap_confidence_interval
    from src.evaluation.xai_metrics import evidence_grounding_score

    pred_dir = Path(args.predictions_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    pred_files = list(pred_dir.glob("*.jsonl"))
    if not pred_files:
        logger.warning(f"No prediction files found in {pred_dir}")
        return

    for pred_file in pred_files:
        logger.info(f"\nEvaluating: {pred_file.name}")
        predictions = load_predictions(str(pred_file))
        if not predictions:
            continue

        y_true = [p.get("gold_label", "") for p in predictions]
        y_pred = [p.get("predicted_label", "") for p in predictions]

        # Filter invalid labels
        valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if t and p]
        if not valid_pairs:
            logger.warning(f"No valid label pairs in {pred_file.name}")
            continue
        y_true, y_pred = zip(*valid_pairs)

        # Classification metrics
        metrics = compute_classification_metrics(list(y_true), list(y_pred))

        # XAI metrics
        metrics["evidence_grounding"] = evidence_grounding_score(predictions)

        # Bootstrap CI for accuracy
        y_true_arr = (np.array(y_true) == np.array(y_pred)).astype(int)  # correctness mask
        ci = bootstrap_confidence_interval(
            y_true_arr, y_true_arr,  # dummy — compute from correctness
            metric_fn=lambda a, b: float(a.mean()),
            n_bootstrap=config.get("statistical_tests", {}).get("n_bootstrap", 1000),
        )
        metrics["accuracy_ci"] = ci

        all_results[pred_file.stem] = metrics
        logger.info(
            f"  Accuracy: {metrics['accuracy']:.4f} | "
            f"F1-macro: {metrics['f1_macro']:.4f} | "
            f"Grounding: {metrics['evidence_grounding']:.4f}"
        )

    # Save results
    results_file = output_dir / "classification_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_file}")

    # Generate comparison table
    table_file = output_dir / "comparison_tables.md"
    with open(table_file, "w") as f:
        f.write("# Results Comparison\n\n")
        f.write("| Method | Accuracy | F1-macro | F1-weighted | Evidence Grounding |\n")
        f.write("|--------|----------|----------|-------------|-------------------|\n")
        for method, metrics in all_results.items():
            f.write(
                f"| {method} | {metrics.get('accuracy', 0):.4f} | "
                f"{metrics.get('f1_macro', 0):.4f} | "
                f"{metrics.get('f1_weighted', 0):.4f} | "
                f"{metrics.get('evidence_grounding', 0):.4f} |\n"
            )
    logger.info(f"Comparison table saved to {table_file}")


def generate_human_eval(args) -> None:
    """Generate human evaluation materials."""
    from src.evaluation.human_eval import HumanEvalGenerator

    method_predictions: dict[str, list[dict]] = {}
    methods = args.methods.split(",") if args.methods else []

    pred_dir = Path(args.predictions_dir) if args.predictions_dir else Path("outputs/predictions/")
    for method in methods:
        pred_file = pred_dir / f"{method}.jsonl"
        if pred_file.exists():
            method_predictions[method] = load_predictions(str(pred_file))
        else:
            logger.warning(f"Prediction file not found for method: {method}")

    if not method_predictions:
        logger.error("No prediction files found. Specify --methods and ensure --predictions_dir is correct.")
        return

    generator = HumanEvalGenerator(seed=42)
    output_dir = args.output or "outputs/human_eval/"
    generator.run(method_predictions, output_dir, n_samples=args.n_samples or 50)


def main():
    parser = argparse.ArgumentParser(description="Run evaluation suite")
    parser.add_argument("--mode", choices=["full", "baseline_predictions", "generate_human_eval", "statistical_tests"], required=True)
    parser.add_argument("--config", default="configs/eval_config.yaml")
    parser.add_argument("--predictions_dir", default="outputs/predictions/")
    parser.add_argument("--models_dir", default="outputs/models/")
    parser.add_argument("--output", default="outputs/reports/")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--methods", help="Comma-separated method names for human eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)

    if args.mode == "full":
        run_full_evaluation(args, config)
    elif args.mode == "generate_human_eval":
        generate_human_eval(args)
    else:
        logger.info(f"Mode '{args.mode}' not fully implemented yet")


if __name__ == "__main__":
    main()
