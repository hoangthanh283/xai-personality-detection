#!/usr/bin/env python
"""Run full evaluation suite.

Usage:
    python scripts/evaluate.py --mode full --predictions_dir outputs/predictions/ --output outputs/reports/
    python scripts/evaluate.py --mode baseline_predictions --models_dir outputs/models/
    python scripts/evaluate.py --mode generate_human_eval --n_samples 50
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
import yaml  # noqa: E402
from loguru import logger  # noqa: E402

try:
    from dotenv import load_dotenv  # noqa: E402
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on shell env vars

import wandb  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def load_predictions(path: str) -> list[dict]:
    """Load predictions from a JSONL file."""
    predictions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions


def plot_confusion_matrix(y_true, y_pred, labels, output_path: Path, method_name: str) -> None:
    """Generate and save a confusion matrix PNG."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {method_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_full_evaluation(args, config: dict) -> None:
    """Compute full metrics suite for all prediction files."""
    import numpy as np

    from src.evaluation.classification_metrics import \
        compute_classification_metrics
    from src.evaluation.statistical_tests import bootstrap_confidence_interval
    from src.evaluation.xai_metrics import evidence_grounding_score

    pred_dir = Path(args.predictions_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Any] = {}
    xai_results = {}

    pred_files = list(pred_dir.glob("*.jsonl"))
    if not pred_files:
        logger.warning(f"No prediction files found in {pred_dir}")
        return

    # Initialise W&B evaluation run
    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    wb_run = None
    compare_table = None
    if wandb_project:
        wb_run = wandb.init(
            project=wandb_project,
            name=f"eval_{Path(args.predictions_dir).stem}",
            job_type="evaluation",
            config={
                "predictions_dir": args.predictions_dir,
                "output_dir": args.output,
                "n_pred_files": len(pred_files),
                "seed": args.seed,
            },
            tags=["evaluation"],
            reinit=True,
        )
        compare_table = wandb.Table(columns=["method", "accuracy", "f1_macro", "f1_weighted", "evidence_grounding"])

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
        xai: Dict[str, Any] = {}
        xai["evidence_grounding"] = evidence_grounding_score(predictions)

        # Evidence Relevance (for Personality Evd)
        gold_evidences = [p.get("gold_evidence") for p in predictions]
        if any(gold_evidences):
            from src.evaluation.xai_metrics import evidence_relevance_f1
            pred_evs, gold_evs = [], []
            for p in predictions:
                if "gold_evidence" in p:
                    pred_ev = " ".join([e.get("evidence", "") for e in p.get("evidence_chain", [])])
                    gold_ev = " ".join(p["gold_evidence"]) if isinstance(p["gold_evidence"], list) else str(p["gold_evidence"])
                    pred_evs.append(pred_ev)
                    gold_evs.append(gold_ev)
            xai["evidence_relevance_f1"] = evidence_relevance_f1(pred_evs, gold_evs)

        # Explanation Consistency & Faithfulness
        try:
            from src.evaluation.xai_metrics import (explanation_consistency,
                                                    faithfulness_score)
            from src.rag_pipeline.llm_client import build_llm_client
            from src.rag_pipeline.pipeline import RAGXPRPipeline

            # Try to build LLM for consistency checking
            llm_config = config.get("llm", {"provider": "openrouter", "model": "qwen/qwen3.6-plus-preview:free"})
            llm_client = build_llm_client(llm_config)

            # Check consistency
            xai["explanation_consistency"] = explanation_consistency(predictions, llm_client)
            # For faithfulness, we need the pipeline instantiated.
            # We ONLY run faithfulness for RAG-XPR (which generates evidence chains).
            if "rag_xpr" in pred_file.stem and any(p.get("evidence_chain") for p in predictions):
                rag_cfg_path = "configs/rag_xpr_config.yaml"
                if Path(rag_cfg_path).exists():
                    pipeline = RAGXPRPipeline.from_config_file(rag_cfg_path)
                    xai["faithfulness"] = faithfulness_score(pipeline, predictions, n_samples=min(20, len(predictions)))
        except Exception as e:
            logger.warning(f"Could not compute LLM-based XAI metrics for {pred_file.stem}: {e}")

        # Store XAI back to metrics, and also into dedicated dict
        for k, v in xai.items():
            metrics[k] = v
        xai_results[pred_file.stem] = xai

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

        # Push per-file metrics to W&B
        if wb_run:
            method = pred_file.stem
            prefix = f"{method}/"
            scalar_metrics = {
                f"{prefix}{k}": v
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
            wandb.log(scalar_metrics)

            # Per-class metrics table
            report_str = metrics.get("classification_report", "")
            if report_str:
                lines = [line for line in report_str.strip().splitlines() if line.strip()]
                per_class_table = wandb.Table(columns=["class", "precision", "recall", "f1-score", "support"])
                for line in lines[1:]:
                    parts = line.split()
                    if len(parts) >= 5 and parts[0] not in ("accuracy", "macro", "weighted"):
                        try:
                            per_class_table.add_data(parts[0], float(parts[1]), float(parts[2]), float(parts[3]), int(parts[4]))
                        except (ValueError, IndexError):
                            pass
                if per_class_table.data:
                    wandb.log({f"{prefix}per_class_metrics": per_class_table})

            # Confusion matrix plot
            cm_dir = output_dir / "confusion_matrices"
            cm_dir.mkdir(parents=True, exist_ok=True)
            cm_path = cm_dir / f"{method}.png"
            labels = sorted(list(set(y_true) | set(y_pred)))
            plot_confusion_matrix(y_true, y_pred, labels, cm_path, method)

            if wb_run:
                wandb.log({f"{prefix}confusion_matrix": wandb.Image(str(cm_path))})

            # Accumulate into comparison table
            compare_table.add_data(
                method,
                metrics.get("accuracy", 0),
                metrics.get("f1_macro", 0),
                metrics.get("f1_weighted", 0),
                metrics.get("evidence_grounding", 0),
            )

    # Save results
    results_file = output_dir / "classification_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nClassification results saved to {results_file}")

    xai_results_file = output_dir / "xai_results.json"
    with open(xai_results_file, "w") as f:
        json.dump(xai_results, f, indent=2, default=str)
    logger.info(f"XAI metrics saved to {xai_results_file}")

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

    # Finalise W&B run
    if wb_run:
        wandb.log({"comparison_table": compare_table})
        wb_run.summary["results_file"] = str(results_file)
        wb_run.summary["n_methods_evaluated"] = len(all_results)
        # Best model summary
        if all_results:
            best = max(all_results.items(), key=lambda kv: kv[1].get("f1_macro", 0) if isinstance(kv[1], dict) else 0)
            wb_run.summary["best_method"] = best[0]
            wb_run.summary["best_f1_macro"] = best[1].get("f1_macro", 0) if isinstance(best[1], dict) else 0
        wb_run.finish()
        logger.info("Evaluation results pushed to W&B")


def run_statistical_tests(args, config: dict) -> None:
    """Run pairwise statistical tests (McNemar, paired bootstrap) across all methods."""
    import numpy as np
    from sklearn.metrics import f1_score

    from src.evaluation.statistical_tests import (mcnemar_test,
                                                  paired_bootstrap_test)

    pred_dir = Path(args.predictions_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = list(pred_dir.glob("*.jsonl"))
    if len(pred_files) < 2:
        logger.warning(f"Need at least 2 prediction files for statistical tests, found {len(pred_files)} in {pred_dir}")
        return

    # Load all method predictions
    method_data = {}
    for pred_file in pred_files:
        preds = load_predictions(str(pred_file))
        if preds:
            method_data[pred_file.stem] = {str(p.get("id", i)): p for i, p in enumerate(preds)}

    # Find common IDs across all methods
    common_ids = set.intersection(*[set(data.keys()) for data in method_data.values()])
    if not common_ids:
        logger.error("No common IDs found across prediction files to perform paired tests.")
        return

    common_ids = sorted(list(common_ids))
    logger.info(f"Running statistical tests on {len(common_ids)} common samples across {len(method_data)} methods.")

    # Assume gold labels are consistent across methods for the same ID
    first_method = list(method_data.values())[0]
    y_true = np.array([first_method[idx].get("gold_label", "") for idx in common_ids])

    results = []
    methods = sorted(list(method_data.keys()))

    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method_a = methods[i]
            method_b = methods[j]
            pred_a = np.array([method_data[method_a][idx].get("predicted_label", "") for idx in common_ids])
            pred_b = np.array([method_data[method_b][idx].get("predicted_label", "") for idx in common_ids])

            logger.info(f"\nTesting: {method_a} vs {method_b}")

            # 1. McNemar's Test on Accuracy
            mcn_res = mcnemar_test(y_true, pred_a, pred_b)
            logger.info(f"  McNemar p-value: {mcn_res['p_value']:.4f} (significant: {mcn_res['significant']})")

            # 2. Paired Bootstrap Test on F1-macro
            def f1_macro_fn(y_t, y_p):
                return f1_score(y_t, y_p, average="macro", zero_division=0)

            n_boot = config.get("statistical_tests", {}).get("n_bootstrap", 1000)
            boot_res = paired_bootstrap_test(y_true, pred_a, pred_b, metric_fn=f1_macro_fn, n_bootstrap=n_boot)
            logger.info(f"  Bootstrap p-val: {boot_res['p_value']:.4f} (significant: {boot_res['significant']}) | Δ F1: {boot_res['delta']:.4f}")

            results.append({
                "method_a": method_a,
                "method_b": method_b,
                "mcnemar": mcn_res,
                "bootstrap_f1": boot_res,
            })

    results_file = output_dir / "statistical_tests.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nStatistical test results saved to {results_file}")


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


def generate_baseline_predictions(args, config: dict) -> None:
    """Load baseline models and generate predictions on test datasets."""
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        return

    logger.info(f"Scanning for models in {models_dir}")

    # Simple heuristic to identify models, their datasets and tasks
    # Ideally, this should rely on metadata saved alongside the model
    # Here we demonstrate loading logic assuming standard naming from train_baseline.py

    for path in models_dir.glob("*"):
        if path.is_file() and path.suffix == ".pkl":
            continue  # Needs dataset info. Better to run baseline predictions using the script provided in `train_baseline.py` actually.

    logger.error("Generating predictions iteratively across unknown PKL/checkpoints is not fully self-contained.")
    logger.info("Please use 'scripts/train_baseline.py' which automatically outputs .jsonl test predictions upon completion.")


def main():
    parser = argparse.ArgumentParser(description="Run evaluation suite")
    parser.add_argument("--mode", choices=["full", "baseline_predictions", "generate_human_eval", "statistical_tests"], required=True)
    parser.add_argument("--config", default="configs/eval_config.yaml")
    parser.add_argument("--predictions_dir", default="outputs/predictions/")
    parser.add_argument("--models_dir", default="outputs/models/")
    parser.add_argument("--output", default="outputs/reports/")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--methods", help="Comma-separated method names for human eval")
    parser.add_argument("--wandb_project", help="W&B project name (overrides WANDB_PROJECT env var)")
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
    elif args.mode == "statistical_tests":
        run_statistical_tests(args, config)
    elif args.mode == "baseline_predictions":
        generate_baseline_predictions(args, config)
    else:
        logger.info(f"Mode '{args.mode}' not fully implemented yet")


if __name__ == "__main__":
    main()
