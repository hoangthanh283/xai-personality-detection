#!/usr/bin/env python
"""Train baseline models (ML and transformer).

Usage:
    # ML baselines
    python scripts/train_baseline.py --model logistic_regression --dataset mbti --task 16class
    python scripts/train_baseline.py --model all_ml --dataset mbti --task all --grid_search
    python scripts/train_baseline.py --model ensemble --dataset mbti --task 16class

    # Transformer baselines
    python scripts/train_baseline.py --model distilbert --dataset mbti --task 16class
    python scripts/train_baseline.py --model roberta --dataset mbti --task 4dim
    python scripts/train_baseline.py --model distilbert --dataset essays --task ocean_binary
"""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from loguru import logger

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on shell env vars

import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

ML_MODELS = ["logistic_regression", "svm", "naive_bayes", "xgboost", "random_forest"]
TRANSFORMER_MODELS = ["distilbert", "roberta"]


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_task_data(dataset: str, task: str, config: dict) -> tuple:
    """Load train/val/test data for the specified task."""
    from src.data.loader import DataLoader
    loader = DataLoader("data/processed")

    train_records = loader.load_split(dataset, "train")
    val_records = loader.load_split(dataset, "val")
    test_records = loader.load_split(dataset, "test")

    if task == "16class":
        train_texts, train_labels = loader.get_texts_and_labels(train_records, "mbti")
        val_texts, val_labels = loader.get_texts_and_labels(val_records, "mbti")
        test_texts, test_labels = loader.get_texts_and_labels(test_records, "mbti")
    elif task in ("IE", "SN", "TF", "JP"):
        train_texts, train_labels = loader.get_texts_and_labels(train_records, "mbti_dim", dimension=task)
        val_texts, val_labels = loader.get_texts_and_labels(val_records, "mbti_dim", dimension=task)
        test_texts, test_labels = loader.get_texts_and_labels(test_records, "mbti_dim", dimension=task)
    elif task.startswith("ocean_") or task in ("O", "C", "E", "A", "N"):
        trait = task.split("_")[-1] if "_" in task else task
        train_texts, train_labels = loader.get_texts_and_labels(train_records, "ocean", dimension=trait)
        val_texts, val_labels = loader.get_texts_and_labels(val_records, "ocean", dimension=trait)
        test_texts, test_labels = loader.get_texts_and_labels(test_records, "ocean", dimension=trait)
    else:
        raise ValueError(f"Unknown task: {task}")

    logger.info(f"Loaded: train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}")
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def save_predictions(texts: list[str], gold_labels: list[str], predicted_labels: list[str], model_name: str, dataset: str, task: str, output_dir: str = "outputs/predictions"):
    """Save predictions to JSONL for evaluation."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{model_name}_{dataset}_{task}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for i, (text, gold, pred) in enumerate(zip(texts, gold_labels, predicted_labels)):
            record = {
                "id": f"{model_name}_{i}",
                "text": text,
                "gold_label": gold,
                "predicted_label": pred
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Predictions saved to {output_file}")


def _log_ml_metrics_to_wandb(metrics: dict) -> None:
    """Log ML metrics to W&B, handling classification_report as a Table."""
    scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    wandb.log(scalar_metrics)

    # Log per-class classification report as a W&B Table
    report_str = metrics.get("classification_report", "")
    if report_str:
        lines = [line for line in report_str.strip().splitlines() if line.strip()]
        table = wandb.Table(columns=["class", "precision", "recall", "f1-score", "support"])
        for line in lines[1:]:  # skip header
            parts = line.split()
            if len(parts) >= 5 and parts[0] not in ("accuracy", "macro", "weighted"):
                label = parts[0]
                try:
                    table.add_data(label, float(parts[1]), float(parts[2]), float(parts[3]), int(parts[4]))
                except (ValueError, IndexError):
                    pass
        if table.data:
            wandb.log({"per_class_metrics": table})


def train_ml_model(model_name: str, dataset: str, task: str, config: dict, args) -> dict:
    """Train a single ML baseline model."""
    from src.baselines.ml_baselines import MLBaselineTrainer

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = get_task_data(dataset, task, config)

    # Initialize W&B for ML if project provided
    run = None
    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    if wandb_project:
        model_cfg = config.get(model_name, {})
        run = wandb.init(
            project=wandb_project,
            name=f"{model_name}_{dataset}_{task}",
            config={
                **model_cfg,
                "model": model_name,
                "dataset": dataset,
                "task": task,
                "grid_search": args.grid_search,
                "seed": args.seed,
                "train_size": len(train_texts),
                "test_size": len(test_texts),
            },
            tags=["ml", model_name, dataset, task],
            reinit=True,
        )

    trainer = MLBaselineTrainer(model_name, config)
    trainer.fit(train_texts, train_labels, use_grid_search=args.grid_search)

    logger.info("Evaluating on test set...")
    metrics = trainer.evaluate(test_texts, test_labels)

    # Log metrics to W&B
    if run:
        _log_ml_metrics_to_wandb(metrics)
        # Log summary scalars so they appear on the W&B run overview
        run.summary.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        run.finish()

    # Save predictions
    preds = trainer.predict(test_texts)
    save_predictions(test_texts, test_labels, preds.tolist(), model_name, dataset, task)

    output_path = f"outputs/models/tfidf_{model_name}_{dataset}_{task}.pkl"
    trainer.save(output_path)

    return metrics


def train_ensemble(dataset: str, task: str, config: dict, args) -> dict:
    """Train ensemble baseline."""
    from src.baselines.ml_baselines import EnsembleClassifier

    train_texts, train_labels, _, _, test_texts, test_labels = get_task_data(dataset, task, config)
    members = args.ensemble_members.split(",") if args.ensemble_members else ["logistic_regression", "xgboost", "random_forest"]

    # Initialize W&B
    run = None
    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    if wandb_project:
        run = wandb.init(
            project=wandb_project,
            name=f"ensemble_{dataset}_{task}",
            config={
                "members": members,
                "dataset": dataset,
                "task": task,
                "seed": args.seed,
                "train_size": len(train_texts),
                "test_size": len(test_texts),
            },
            tags=["ensemble", dataset, task],
            reinit=True,
        )

    ensemble = EnsembleClassifier(members=members, config=config)
    ensemble.fit(train_texts, train_labels)

    logger.info("Evaluating on test set...")
    metrics = ensemble.evaluate(test_texts, test_labels)

    if run:
        scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        wandb.log(scalar_metrics)
        run.summary.update(scalar_metrics)
        run.finish()

    # Save predictions
    preds = ensemble.predict(test_texts)
    save_predictions(test_texts, test_labels, preds.tolist(), "ensemble", dataset, task)

    return metrics


def _train_transformer_single(
    model_name: str,
    dataset: str,
    dim: str,
    model_cfg: dict,
    config: dict,
    args,
    extra_tags: list[str] | None = None,
) -> dict:
    """Train and evaluate a single transformer for one dimension/class."""
    from src.baselines.transformer_baseline import (TransformerBaseline,
                                                    TransformerConfig)

    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    output_dir = args.output_dir or f"outputs/models/{model_name}_{dataset}_{dim}"
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = get_task_data(dataset, dim, config)

    transformer_config = TransformerConfig(
        model_name=model_cfg.get("model_name", f"{model_name}-base-uncased"),
        **{k: v for k, v in model_cfg.items() if k != "model_name"},
        output_dir=output_dir,
    )

    # Initialize W&B BEFORE calling trainer.train(), so HF Trainer picks it up
    if wandb_project:
        tags = ["transformer", model_name, dataset, dim] + (extra_tags or [])
        wandb.init(
            project=wandb_project,
            name=f"{model_name}_{dataset}_{dim}",
            config={
                **{k: v for k, v in vars(transformer_config).items()},
                "dataset": dataset,
                "task": dim,
                "train_size": len(train_texts),
                "val_size": len(val_texts),
                "test_size": len(test_texts),
                "seed": args.seed,
            },
            tags=tags,
            reinit=True,
        )

    trainer = TransformerBaseline(transformer_config)
    trainer.train(train_texts, train_labels, val_texts, val_labels, output_dir, wandb_project)

    metrics = trainer.evaluate(test_texts, test_labels)
    if wandb.run is not None:
        test_metrics = {f"test_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
        wandb.log(test_metrics)
        wandb.run.summary.update(test_metrics)
        wandb.finish()

    return metrics


def train_transformer(model_name: str, dataset: str, task: str, config: dict, args) -> dict:
    """Train a transformer baseline."""
    model_cfg = config.get("transformer", {}).get(model_name, {})

    if task == "4dim":
        all_metrics = {}
        for dim in ["IE", "SN", "TF", "JP"]:
            logger.info(f"\n=== Training {model_name} on {dim} dimension ===")
            metrics = _train_transformer_single(model_name, dataset, dim, model_cfg, config, args, extra_tags=["4dim"])
            all_metrics[dim] = metrics
        return all_metrics

    elif task == "ocean_binary":
        all_metrics = {}
        for trait in ["O", "C", "E", "A", "N"]:
            logger.info(f"\n=== Training {model_name} on OCEAN trait {trait} ===")
            metrics = _train_transformer_single(model_name, dataset, trait, model_cfg, config, args, extra_tags=["ocean"])
            all_metrics[trait] = metrics
        return all_metrics

    else:
        output_dir = args.output_dir or f"outputs/models/{model_name}_{dataset}_{task}"
        metrics = _train_transformer_single(model_name, dataset, task, model_cfg, config, args)

        # Save predictions (only for single tasks, not multi-dim loops)
        from src.baselines.transformer_baseline import TransformerBaseline
        trainer = TransformerBaseline.load(output_dir)
        preds = trainer.predict(
            get_task_data(dataset, task, config)[4]  # test_texts
        )
        test_labels = get_task_data(dataset, task, config)[5]  # test_labels
        save_predictions(
            get_task_data(dataset, task, config)[4], test_labels, preds, model_name, dataset, task
        )
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--model", required=True, help="Model name or 'all_ml'")
    parser.add_argument("--dataset", required=True, choices=["mbti", "essays", "pandora", "personality_evd"])
    parser.add_argument("--task", required=True, help="16class, 4dim, ocean_binary, IE, SN, TF, JP")
    parser.add_argument("--config", default="configs/baseline_config.yaml")
    parser.add_argument("--output_dir", help="Override output directory")
    parser.add_argument("--wandb_project", help="W&B project name")
    parser.add_argument("--grid_search", action="store_true", help="Use hyperparameter grid search")
    parser.add_argument("--ensemble_members", help="Comma-separated list for ensemble")
    parser.add_argument("--resume_from", help="Resume transformer from checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)
    config = load_config(args.config)

    tasks = ["16class", "4dim", "ocean_binary"] if args.task == "all" else [args.task]
    models = ML_MODELS if args.model == "all_ml" else [args.model]

    all_results = {}
    for task in tasks:
        for model in models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model} on {args.dataset}/{task}")
            logger.info(f"{'='*60}")
            try:
                if model in TRANSFORMER_MODELS:
                    metrics = train_transformer(model, args.dataset, task, config, args)
                elif model == "ensemble":
                    metrics = train_ensemble(args.dataset, task, config, args)
                else:
                    metrics = train_ml_model(model, args.dataset, task, config, args)

                key = f"{model}_{args.dataset}_{task}"
                all_results[key] = metrics
                logger.info(f"Results: {json.dumps({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, indent=2)}")
            except Exception as e:
                logger.error(f"Training failed for {model}/{task}: {e}")
                raise

    # Save results summary
    output_path = Path("outputs/reports")
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / "baseline_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nAll results saved to {results_file}")

    # Log aggregate summary run to W&B (useful when running multiple models/tasks)
    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    if wandb_project and len(all_results) > 1:
        with wandb.init(
            project=wandb_project,
            name=f"summary_{args.model}_{args.dataset}_{args.task}",
            job_type="summary",
            tags=["summary", args.dataset, args.task],
            reinit=True,
        ) as summary_run:
            # Build a comparison table
            table = wandb.Table(columns=["experiment", "accuracy", "f1_macro", "f1_weighted"])
            for exp_key, m in all_results.items():
                if isinstance(m, dict) and "accuracy" in m:
                    table.add_data(exp_key, m.get("accuracy", 0), m.get("f1_macro", 0), m.get("f1_weighted", 0))
                elif isinstance(m, dict):
                    # Multi-dim results (4dim / ocean_binary)
                    for sub_key, sub_m in m.items():
                        if isinstance(sub_m, dict):
                            table.add_data(
                                f"{exp_key}/{sub_key}",
                                sub_m.get("accuracy", 0),
                                sub_m.get("f1_macro", 0),
                                sub_m.get("f1_weighted", 0),
                            )
            wandb.log({"results_summary": table})
            summary_run.summary["results_file"] = str(results_file)
        logger.info("Aggregate summary logged to W&B")


if __name__ == "__main__":
    main()
