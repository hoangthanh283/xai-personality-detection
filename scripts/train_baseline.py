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
import sys
from pathlib import Path

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging
from src.utils.seed import set_seed

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


def train_ml_model(model_name: str, dataset: str, task: str, config: dict, args) -> dict:
    """Train a single ML baseline model."""
    from src.baselines.ml_baselines import MLBaselineTrainer

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = get_task_data(dataset, task, config)

    trainer = MLBaselineTrainer(model_name, config)
    trainer.fit(train_texts, train_labels, use_grid_search=args.grid_search)

    logger.info("Evaluating on test set...")
    metrics = trainer.evaluate(test_texts, test_labels)

    output_path = f"outputs/models/tfidf_{model_name}_{dataset}_{task}.pkl"
    trainer.save(output_path)

    return metrics


def train_ensemble(dataset: str, task: str, config: dict, args) -> dict:
    """Train ensemble baseline."""
    from src.baselines.ml_baselines import EnsembleClassifier

    train_texts, train_labels, _, _, test_texts, test_labels = get_task_data(dataset, task, config)
    members = args.ensemble_members.split(",") if args.ensemble_members else ["logistic_regression", "xgboost", "random_forest"]

    ensemble = EnsembleClassifier(members=members, config=config)
    ensemble.fit(train_texts, train_labels)
    metrics = ensemble.evaluate(test_texts, test_labels)
    logger.info(f"Ensemble metrics: {metrics}")
    return metrics


def train_transformer(model_name: str, dataset: str, task: str, config: dict, args) -> dict:
    """Train a transformer baseline."""
    from src.baselines.transformer_baseline import (TransformerBaseline,
                                                    TransformerConfig)

    model_cfg = config.get("transformer", {}).get(model_name, {})

    if task == "4dim":
        # Train 4 separate binary classifiers
        all_metrics = {}
        for dim in ["IE", "SN", "TF", "JP"]:
            logger.info(f"\n=== Training {model_name} on {dim} dimension ===")
            output_dir = args.output_dir or f"outputs/models/{model_name}_{dataset}_{dim}"
            train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = get_task_data(dataset, dim, config)
            transformer_config = TransformerConfig(
                model_name=model_cfg.get("model_name", f"{model_name}-base-uncased"),
                **{k: v for k, v in model_cfg.items() if k != "model_name"},
                output_dir=output_dir,
            )
            trainer = TransformerBaseline(transformer_config)
            trainer.train(train_texts, train_labels, val_texts, val_labels, output_dir, args.wandb_project)
            metrics = trainer.evaluate(test_texts, test_labels)
            all_metrics[dim] = metrics
        return all_metrics
    elif task == "ocean_binary":
        all_metrics = {}
        for trait in ["O", "C", "E", "A", "N"]:
            logger.info(f"\n=== Training {model_name} on OCEAN trait {trait} ===")
            output_dir = args.output_dir or f"outputs/models/{model_name}_{dataset}_{trait}"
            train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = get_task_data(dataset, trait, config)
            transformer_config = TransformerConfig(
                model_name=model_cfg.get("model_name", f"{model_name}-base-uncased"),
                **{k: v for k, v in model_cfg.items() if k != "model_name"},
                output_dir=output_dir,
            )
            trainer = TransformerBaseline(transformer_config)
            trainer.train(train_texts, train_labels, val_texts, val_labels, output_dir, args.wandb_project)
            metrics = trainer.evaluate(test_texts, test_labels)
            all_metrics[trait] = metrics
        return all_metrics
    else:
        output_dir = args.output_dir or f"outputs/models/{model_name}_{dataset}_{task}"
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = get_task_data(dataset, task, config)
        transformer_config = TransformerConfig(
            model_name=model_cfg.get("model_name", f"{model_name}-base-uncased"),
            **{k: v for k, v in model_cfg.items() if k != "model_name"},
            output_dir=output_dir,
        )
        trainer = TransformerBaseline(transformer_config)
        trainer.train(train_texts, train_labels, val_texts, val_labels, output_dir, args.wandb_project)
        return trainer.evaluate(test_texts, test_labels)


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


if __name__ == "__main__":
    main()
