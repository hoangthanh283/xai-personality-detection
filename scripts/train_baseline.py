#!/usr/bin/env python
"""Train baseline models (ML, transformer, and LSTM).

Usage:
    # ML baselines
    python scripts/train_baseline.py --model logistic_regression --dataset mbti --task 16class
    python scripts/train_baseline.py --model all_ml --dataset mbti --task all --grid_search
    python scripts/train_baseline.py --model ensemble --dataset mbti --task 16class

    # Transformer baselines
    python scripts/train_baseline.py --model distilbert --dataset mbti --task 16class
    python scripts/train_baseline.py --model roberta --dataset mbti --task 4dim
    python scripts/train_baseline.py --model distilbert --dataset essays --task ocean_binary

    # LSTM baseline
    python scripts/train_baseline.py --model lstm --dataset mbti --task 16class
    python scripts/train_baseline.py --model lstm --dataset mbti --task 4dim
    python scripts/train_baseline.py --model lstm --dataset essays --task ocean_binary
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
TRANSFORMER_MODELS = ["distilbert", "roberta"]  # legacy end-to-end fine-tuning — kept for reproducibility
FROZEN_MODELS = ["frozen_bert_svm", "roberta_mlp"]  # new published paradigms
LSTM_MODELS = ["lstm"]
MBTI_DIMENSIONS = ["IE", "SN", "TF", "JP"]
OCEAN_TRAITS = ["O", "C", "E", "A", "N"]


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def expand_tasks(dataset: str, task: str) -> list[str]:
    """Expand task shortcuts into the valid task list for a dataset."""
    if task != "all":
        return [task]

    if dataset == "mbti":
        return ["16class", "4dim"]

    if dataset in {"essays", "pandora", "personality_evd"}:
        return ["ocean_binary"]

    raise ValueError(f"Unsupported dataset for task expansion: {dataset}")


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
    elif task == "4dim":
        raise ValueError("Task shortcut '4dim' must be expanded before calling get_task_data().")
    elif task == "ocean_binary":
        raise ValueError("Task shortcut 'ocean_binary' must be expanded before calling get_task_data().")
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


def _build_split_metrics(train_metrics: dict, eval_metrics: dict, test_metrics: dict) -> dict:
    """Flatten split metrics while preserving legacy top-level test keys."""
    combined = {}
    for split_name, split_metrics in (
        ("train", train_metrics),
        ("eval", eval_metrics),
        ("test", test_metrics),
    ):
        for key, value in split_metrics.items():
            if isinstance(value, (int, float)):
                combined[f"{split_name}_{key}"] = value

    # Keep legacy aliases so existing summaries and reports still work.
    combined.update({k: v for k, v in test_metrics.items() if isinstance(v, (int, float))})

    if "classification_report" in test_metrics:
        combined["classification_report"] = test_metrics["classification_report"]

    return combined


def _log_classification_report_table(report_str: str, key: str) -> None:
    """Log a sklearn classification report string as a W&B table."""
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
            wandb.log({key: table})


def _log_split_metrics_to_wandb(split_metrics: dict[str, dict], log_test_table: bool = False) -> dict:
    """Log prefixed split metrics to W&B and return the flattened scalar view."""
    combined_metrics = _build_split_metrics(
        split_metrics["train"],
        split_metrics["eval"],
        split_metrics["test"],
    )
    scalar_metrics = {k: v for k, v in combined_metrics.items() if isinstance(v, (int, float))}
    wandb.log(scalar_metrics)

    if log_test_table:
        _log_classification_report_table(
            split_metrics["test"].get("classification_report", ""),
            "test_per_class_metrics",
        )

    return combined_metrics


def _train_ml_single(model_name: str, dataset: str, task: str, config: dict, args) -> dict:
    """Train and evaluate one ML baseline for one concrete task."""
    from src.baselines.ml_baselines import MLBaselineTrainer

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = get_task_data(dataset, task, config)

    # Initialize W&B for ML if project provided
    run = None
    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    if wandb_project:
        model_cfg = config.get("ml_models", {}).get(model_name, {})
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
                "val_size": len(val_texts),
                "test_size": len(test_texts),
            },
            tags=["ml", model_name, dataset, task],
            reinit=True,
        )

    trainer = MLBaselineTrainer(model_name, config)
    trainer.fit(train_texts, train_labels, use_grid_search=args.grid_search)

    logger.info("Evaluating final train/eval/test metrics...")
    split_metrics = {
        "train": trainer.evaluate(train_texts, train_labels),
        "eval": trainer.evaluate(val_texts, val_labels),
        "test": trainer.evaluate(test_texts, test_labels),
    }
    metrics = _build_split_metrics(
        split_metrics["train"],
        split_metrics["eval"],
        split_metrics["test"],
    )

    # Log metrics to W&B
    if run:
        logged_metrics = _log_split_metrics_to_wandb(split_metrics, log_test_table=True)
        run.summary.update({k: v for k, v in logged_metrics.items() if isinstance(v, (int, float))})
        run.finish()

    # Save predictions
    preds = trainer.predict(test_texts)
    save_predictions(test_texts, test_labels, preds.tolist(), model_name, dataset, task)

    output_path = f"outputs/models/tfidf_{model_name}_{dataset}_{task}.pkl"
    trainer.save(output_path)

    return metrics


def train_ml_model(model_name: str, dataset: str, task: str, config: dict, args) -> dict:
    """Train ML baselines, expanding multi-task shortcuts when needed."""
    if task == "4dim":
        all_metrics = {}
        for dim in MBTI_DIMENSIONS:
            logger.info(f"\n=== Training {model_name} on {dim} dimension ===")
            all_metrics[dim] = _train_ml_single(model_name, dataset, dim, config, args)
        return all_metrics

    if task == "ocean_binary":
        all_metrics = {}
        for trait in OCEAN_TRAITS:
            logger.info(f"\n=== Training {model_name} on OCEAN trait {trait} ===")
            all_metrics[trait] = _train_ml_single(model_name, dataset, trait, config, args)
        return all_metrics

    return _train_ml_single(model_name, dataset, task, config, args)


def _train_ensemble_single(dataset: str, task: str, config: dict, args) -> dict:
    """Train and evaluate one ensemble for one concrete task."""
    from src.baselines.ml_baselines import EnsembleClassifier

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = get_task_data(dataset, task, config)
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
                "val_size": len(val_texts),
                "test_size": len(test_texts),
            },
            tags=["ensemble", dataset, task],
            reinit=True,
        )

    ensemble = EnsembleClassifier(members=members, config=config)
    ensemble.fit(train_texts, train_labels)

    logger.info("Evaluating final train/eval/test metrics...")
    split_metrics = {
        "train": ensemble.evaluate(train_texts, train_labels),
        "eval": ensemble.evaluate(val_texts, val_labels),
        "test": ensemble.evaluate(test_texts, test_labels),
    }
    metrics = _build_split_metrics(
        split_metrics["train"],
        split_metrics["eval"],
        split_metrics["test"],
    )

    if run:
        logged_metrics = _log_split_metrics_to_wandb(split_metrics, log_test_table=True)
        run.summary.update({k: v for k, v in logged_metrics.items() if isinstance(v, (int, float))})
        run.finish()

    # Save predictions
    preds = ensemble.predict(test_texts)
    save_predictions(test_texts, test_labels, preds.tolist(), "ensemble", dataset, task)

    return metrics


def train_ensemble(dataset: str, task: str, config: dict, args) -> dict:
    """Train ensemble baselines, expanding multi-task shortcuts when needed."""
    if task == "4dim":
        all_metrics = {}
        for dim in MBTI_DIMENSIONS:
            logger.info(f"\n=== Training ensemble on {dim} dimension ===")
            all_metrics[dim] = _train_ensemble_single(dataset, dim, config, args)
        return all_metrics

    if task == "ocean_binary":
        all_metrics = {}
        for trait in OCEAN_TRAITS:
            logger.info(f"\n=== Training ensemble on OCEAN trait {trait} ===")
            all_metrics[trait] = _train_ensemble_single(dataset, trait, config, args)
        return all_metrics

    return _train_ensemble_single(dataset, task, config, args)


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
    dataset_override_cfg = config.get("transformer", {}).get("dataset_overrides", {}).get(dataset, {})
    common_override_cfg = {
        k: v for k, v in dataset_override_cfg.items()
        if k not in TRANSFORMER_MODELS
    }
    model_specific_override_cfg = dataset_override_cfg.get(model_name, {})
    resolved_model_cfg = {**model_cfg, **common_override_cfg, **model_specific_override_cfg}

    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(TransformerConfig)}
    transformer_config = TransformerConfig(
        model_name=resolved_model_cfg.get("model_name", f"{model_name}-base-uncased"),
        **{
            k: v
            for k, v in resolved_model_cfg.items()
            if k != "model_name" and k in valid_fields
        },
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

    split_metrics = {
        "train": trainer.evaluate(train_texts, train_labels),
        "eval": trainer.evaluate(val_texts, val_labels),
        "test": trainer.evaluate(test_texts, test_labels),
    }
    metrics = _build_split_metrics(
        split_metrics["train"],
        split_metrics["eval"],
        split_metrics["test"],
    )
    if wandb.run is not None:
        logged_metrics = _log_split_metrics_to_wandb(split_metrics, log_test_table=True)
        wandb.run.summary.update({k: v for k, v in logged_metrics.items() if isinstance(v, (int, float))})
        wandb.finish()

    return metrics


def _train_frozen_single(
    model_name: str,
    dataset: str,
    task: str,
    config: dict,
    args,
    extra_tags: list[str] | None = None,
) -> dict:
    """Train and evaluate one frozen-encoder baseline for a single task.

    Supports ``frozen_bert_svm`` (Kazameini 2020) and ``roberta_mlp`` (Gao 2024).
    Embeddings are cached to outputs/embeddings/{encoder}/{dataset}_{split}.npy so
    encoding cost amortises over the 4 MBTI binary tasks + 16-class.
    """
    from src.baselines.frozen_transformer_baselines import (
        FrozenBertSvmBaseline, RobertaMlpBaseline)

    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    output_dir = args.output_dir or f"outputs/models/{model_name}_{dataset}_{task}"
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = get_task_data(dataset, task, config)

    base_cfg = dict(config.get(model_name, {}) or {})
    # Allow per-dataset overrides mirroring the transformer/lstm pattern
    dataset_override = (base_cfg.get("dataset_overrides", {}) or {}).get(dataset, {})
    if dataset_override:
        base_cfg = {**base_cfg, **dataset_override}
    base_cfg.pop("dataset_overrides", None)
    base_cfg["seed"] = args.seed

    # Cache keys — tied to encoder+dataset+split (task-agnostic so embeddings reuse)
    enc_name = (base_cfg.get("encoder", {}) or {}).get("model_name", "roberta-base")
    enc_slug = enc_name.replace("/", "__")
    ck_train = f"{dataset}_train"
    ck_val = f"{dataset}_val"
    ck_test = f"{dataset}_test"

    run = None
    if wandb_project:
        tags = ["frozen", model_name, dataset, task] + (extra_tags or [])
        run = wandb.init(
            project=wandb_project,
            name=f"{model_name}_{dataset}_{task}",
            config={
                **base_cfg,
                "model": model_name,
                "encoder_model": enc_name,
                "dataset": dataset,
                "task": task,
                "seed": args.seed,
                "train_size": len(train_texts),
                "val_size": len(val_texts),
                "test_size": len(test_texts),
            },
            tags=tags,
            reinit=True,
        )

    if model_name == "frozen_bert_svm":
        trainer = FrozenBertSvmBaseline(config=base_cfg, model_name=model_name)
        trainer.fit(train_texts, train_labels, cache_key_train=ck_train)
    elif model_name == "roberta_mlp":
        trainer = RobertaMlpBaseline(config=base_cfg, model_name=model_name)
        trainer.fit(
            train_texts, train_labels,
            val_texts=val_texts, val_labels=val_labels,
            cache_key_train=ck_train, cache_key_val=ck_val,
        )
    else:
        raise ValueError(f"Unknown frozen baseline: {model_name}")

    logger.info("Evaluating on train/eval/test splits...")
    split_metrics = {
        "train": trainer.evaluate(train_texts, train_labels, cache_key=ck_train),
        "eval": trainer.evaluate(val_texts, val_labels, cache_key=ck_val),
        "test": trainer.evaluate(test_texts, test_labels, cache_key=ck_test),
    }
    metrics = _build_split_metrics(
        split_metrics["train"],
        split_metrics["eval"],
        split_metrics["test"],
    )

    if run:
        logged_metrics = _log_split_metrics_to_wandb(split_metrics, log_test_table=True)
        run.summary.update({k: v for k, v in logged_metrics.items() if isinstance(v, (int, float))})
        run.finish()

    preds = trainer.predict(test_texts, cache_key=ck_test).tolist()
    save_predictions(test_texts, test_labels, preds, model_name, dataset, task)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_path = f"{output_dir}/model.pt" if model_name == "roberta_mlp" else f"{output_dir}/model.pkl"
    trainer.save(model_path)

    return metrics


def train_frozen(model_name: str, dataset: str, task: str, config: dict, args) -> dict:
    """Expand multi-task shortcuts (4dim, ocean_binary) for frozen baselines."""
    if task == "4dim":
        return {
            dim: _train_frozen_single(model_name, dataset, dim, config, args, extra_tags=["4dim"])
            for dim in MBTI_DIMENSIONS
        }
    if task == "ocean_binary":
        return {
            trait: _train_frozen_single(model_name, dataset, trait, config, args, extra_tags=["ocean"])
            for trait in OCEAN_TRAITS
        }
    return _train_frozen_single(model_name, dataset, task, config, args)


def _train_lstm_single(dataset: str, task: str, config: dict, args) -> dict:
    """Train and evaluate one LSTM model for a single concrete task/dimension."""
    import dataclasses
    from src.baselines.lstm_baseline import LSTMBaseline, LSTMConfig

    output_dir = args.output_dir or f"outputs/models/lstm_{dataset}_{task}"
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = get_task_data(
        dataset, task, config
    )

    lstm_cfg_dict = config.get("lstm", {})
    valid_fields = {f.name for f in dataclasses.fields(LSTMConfig)}
    lstm_config = LSTMConfig(
        **{k: v for k, v in lstm_cfg_dict.items() if k in valid_fields},
        output_dir=output_dir,
        seed=args.seed,
    )

    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    if wandb_project:
        wandb.init(
            project=wandb_project,
            name=f"lstm_{dataset}_{task}",
            config=vars(lstm_config),
            tags=["lstm", dataset, task],
            reinit=True,
        )

    trainer = LSTMBaseline(lstm_config)
    trainer.train(train_texts, train_labels, val_texts, val_labels, output_dir=output_dir)

    metrics = {
        "train": trainer.evaluate(train_texts, train_labels),
        "eval": trainer.evaluate(val_texts, val_labels),
        "test": trainer.evaluate(test_texts, test_labels),
    }
    flat = {f"{split}_{k}": v for split, m in metrics.items() for k, v in m.items() if isinstance(v, float)}
    logger.info(
        f"lstm/{dataset}/{task} — "
        f"test_acc={flat.get('test_accuracy', 0):.4f} "
        f"test_f1={flat.get('test_f1_macro', 0):.4f}"
    )

    if wandb_project and wandb.run:
        # Log final split metrics as a summary step for easy comparison
        final_metrics = {
            "final/train_accuracy": flat.get("train_accuracy", 0),
            "final/train_f1_macro": flat.get("train_f1_macro", 0),
            "final/eval_accuracy": flat.get("eval_accuracy", 0),
            "final/eval_f1_macro": flat.get("eval_f1_macro", 0),
            "final/test_accuracy": flat.get("test_accuracy", 0),
            "final/test_f1_macro": flat.get("test_f1_macro", 0),
            "final/test_precision_macro": flat.get("test_precision_macro", 0),
            "final/test_recall_macro": flat.get("test_recall_macro", 0),
            "final/test_f1_weighted": flat.get("test_f1_weighted", 0),
        }
        wandb.log(final_metrics)
        wandb.run.summary.update(flat)
        wandb.finish()

    return flat


def train_lstm(dataset: str, task: str, config: dict, args) -> dict:
    """Train LSTM baseline, expanding multi-task shortcuts when needed."""
    if task == "4dim":
        return {dim: _train_lstm_single(dataset, dim, config, args) for dim in MBTI_DIMENSIONS}
    if task == "ocean_binary":
        return {trait: _train_lstm_single(dataset, trait, config, args) for trait in OCEAN_TRAITS}
    return _train_lstm_single(dataset, task, config, args)


def train_transformer(model_name: str, dataset: str, task: str, config: dict, args) -> dict:
    """Train a transformer baseline."""
    model_cfg = config.get("transformer", {}).get(model_name, {})

    if task == "4dim":
        all_metrics = {}
        for dim in MBTI_DIMENSIONS:
            logger.info(f"\n=== Training {model_name} on {dim} dimension ===")
            metrics = _train_transformer_single(model_name, dataset, dim, model_cfg, config, args, extra_tags=["4dim"])
            all_metrics[dim] = metrics
        return all_metrics

    elif task == "ocean_binary":
        all_metrics = {}
        for trait in OCEAN_TRAITS:
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
    parser.add_argument(
        "--dataset", required=True, choices=["mbti", "essays", "pandora", "personality_evd"]
    )
    parser.add_argument("--task", required=True, help="16class, 4dim, ocean_binary, IE, SN, TF, JP")
    parser.add_argument("--config", default="configs/baseline_config.yaml")
    parser.add_argument("--output_dir", help="Override output directory")
    parser.add_argument("--wandb_project", help="W&B project name")
    parser.add_argument("--grid_search", action="store_true", help="Use hyperparameter grid search")
    parser.add_argument("--ensemble_members", help="Comma-separated list for ensemble")
    parser.add_argument("--resume_from", help="Resume transformer from checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--set", metavar="KEY=VALUE", action="append", dest="overrides", default=[],
        help="Override a config value using dot-path notation, e.g. --set transformer.distilbert.loss_weighting=sqrt_balanced",
    )
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)
    config = load_config(args.config)

    for override in args.overrides:
        if "=" not in override:
            raise ValueError(f"--set requires KEY=VALUE format, got: {override!r}")
        key_path, _, raw_value = override.partition("=")
        keys = key_path.split(".")
        node = config
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        try:
            import ast
            node[keys[-1]] = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            node[keys[-1]] = raw_value

    tasks = expand_tasks(args.dataset, args.task)
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
                elif model in FROZEN_MODELS:
                    metrics = train_frozen(model, args.dataset, task, config, args)
                elif model in LSTM_MODELS:
                    metrics = train_lstm(args.dataset, task, config, args)
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

    # Save results summary — merge with existing file so parallel runs accumulate
    output_path = Path("outputs/reports")
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / "baseline_results.json"
    existing = {}
    if results_file.exists():
        try:
            with open(results_file) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing.update(all_results)
    with open(results_file, "w") as f:
        json.dump(existing, f, indent=2, default=str)
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
