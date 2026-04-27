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
import ast
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

import wandb

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.classification_diagnostics import (  # noqa: E402
    build_classification_diagnostics, persist_classification_diagnostics)
from src.utils.logging_config import setup_logging  # noqa: E402
from src.utils.observability import MultiBackendLogger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

ML_MODELS = ["logistic_regression", "svm", "naive_bayes", "xgboost", "random_forest"]
TRANSFORMER_MODELS = [
    "distilbert",
    "roberta",
]  # legacy end-to-end fine-tuning — kept for reproducibility
FROZEN_MODELS = ["frozen_bert_svm", "roberta_mlp"]  # new published paradigms
LSTM_MODELS = ["lstm"]
MBTI_DIMENSIONS = ["IE", "SN", "TF", "JP"]
OCEAN_TRAITS = ["O", "C", "E", "A", "N"]
TIER_MODEL_TYPE_MAP = {
    "logistic_regression": "ml",
    "roberta": "transformer",
    "distilbert": "transformer",
    "roberta_mlp": "frozen",
    "frozen_bert_svm": "frozen",
    "lstm": "lstm",
}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def expand_tasks(dataset: str, task: str) -> list[str]:
    """Expand task shortcuts into the valid task list for a dataset."""
    if task != "all":
        return [task]

    # Strip optional `_augmented` / `_uncleaned` suffix when resolving task scope.
    base = dataset
    for suffix in ("_augmented", "_uncleaned"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    if base == "mbti":
        return ["16class", "4dim"]

    if base in {"essays", "pandora", "personality_evd"}:
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


def save_predictions(
    texts: list[str],
    gold_labels: list[str],
    predicted_labels: list[str],
    model_name: str,
    dataset: str,
    task: str,
    output_dir: str = "outputs/predictions",
):
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
                "predicted_label": pred,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Predictions saved to {output_file}")


def _as_list(values) -> list:
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def _build_split_diagnostics(
    *,
    split_truths: dict[str, list],
    split_predictions: dict[str, list],
    tier_id: str,
    model_name: str,
    dataset: str,
    trait: str,
) -> dict[str, dict]:
    diagnostics = {}
    for split, y_true in split_truths.items():
        y_pred = split_predictions.get(split)
        if y_pred is None:
            continue
        diagnostics[split] = build_classification_diagnostics(
            _as_list(y_true),
            _as_list(y_pred),
            metadata={
                "tier": tier_id,
                "model": model_name,
                "dataset": dataset,
                "trait": trait,
                "split": split,
            },
        )
    return diagnostics


def _flatten_split_diagnostic_metrics(split_diagnostics: dict[str, dict]) -> dict:
    """Flatten diagnostic scalar metrics while preserving legacy test aliases."""
    combined = {}
    for split, diagnostics in split_diagnostics.items():
        for key, value in diagnostics.get("scalar_metrics", {}).items():
            if isinstance(value, (int, float)):
                combined[f"{split}_{key}"] = value
                if split == "test":
                    combined[key] = value
    return combined


def _log_split_diagnostics(
    *,
    logger_obj: MultiBackendLogger,
    split_diagnostics: dict[str, dict],
    run_name: str,
    tier_id: str,
    model_name: str,
    dataset: str,
    trait: str,
    output_dir: str | Path = "outputs/reports/classification_diagnostics",
) -> None:
    """Persist and log all-split classification reports and confusion matrices."""
    for split, diagnostics in split_diagnostics.items():
        stem = f"{tier_id}_{model_name}_{dataset}_{trait}_{split}"
        title = f"{tier_id} | {model_name} | {dataset} | trait={trait} | split={split}"
        try:
            paths = persist_classification_diagnostics(
                diagnostics,
                Path(output_dir) / run_name,
                stem,
                title=title,
                render_png=True,
            )
        except Exception as exc:
            logger.debug(f"classification diagnostics persistence skipped for {stem}: {exc}")
            paths = {}

        logger_obj.log_table(
            f"{split}/classification_report",
            diagnostics["report_columns"],
            diagnostics["report_rows"],
        )
        logger_obj.log_table(
            f"{split}/confusion_matrix_table",
            diagnostics["confusion_matrix_columns"],
            diagnostics["confusion_matrix_rows"],
        )
        png_path = paths.get("png")
        if png_path:
            logger_obj.log_image(f"{split}/confusion_matrix_plot", png_path)


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
    """Update W&B summary with final split metrics and return the flattened view."""
    combined_metrics = _build_split_metrics(
        split_metrics["train"],
        split_metrics["eval"],
        split_metrics["test"],
    )
    scalar_metrics = {k: v for k, v in combined_metrics.items() if isinstance(v, (int, float))}
    if wandb.run is not None:
        wandb.run.summary.update(scalar_metrics)

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
    split_truths = {"train": train_labels, "eval": val_labels, "test": test_labels}
    split_predictions = {
        "train": _as_list(trainer.predict(train_texts)),
        "eval": _as_list(trainer.predict(val_texts)),
        "test": _as_list(trainer.predict(test_texts)),
    }
    split_diagnostics = _build_split_diagnostics(
        split_truths=split_truths,
        split_predictions=split_predictions,
        tier_id="legacy",
        model_name=model_name,
        dataset=dataset,
        trait=task,
    )
    metrics = _flatten_split_diagnostic_metrics(split_diagnostics)

    run_name = run.name if run else f"legacy_{model_name}_{dataset}_{task}"
    run_logger = MultiBackendLogger(wandb_run=run, tb_writer=None)
    if run:
        run_logger.update_summary(metrics)
    _log_split_diagnostics(
        logger_obj=run_logger,
        split_diagnostics=split_diagnostics,
        run_name=run_name,
        tier_id="legacy",
        model_name=model_name,
        dataset=dataset,
        trait=task,
    )
    if run:
        run.finish()

    # Save predictions
    save_predictions(test_texts, test_labels, split_predictions["test"], model_name, dataset, task)

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
    members = (
        args.ensemble_members.split(",")
        if args.ensemble_members
        else ["logistic_regression", "xgboost", "random_forest"]
    )

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
    split_truths = {"train": train_labels, "eval": val_labels, "test": test_labels}
    split_predictions = {
        "train": _as_list(ensemble.predict(train_texts)),
        "eval": _as_list(ensemble.predict(val_texts)),
        "test": _as_list(ensemble.predict(test_texts)),
    }
    split_diagnostics = _build_split_diagnostics(
        split_truths=split_truths,
        split_predictions=split_predictions,
        tier_id="legacy",
        model_name="ensemble",
        dataset=dataset,
        trait=task,
    )
    metrics = _flatten_split_diagnostic_metrics(split_diagnostics)

    run_name = run.name if run else f"legacy_ensemble_{dataset}_{task}"
    run_logger = MultiBackendLogger(wandb_run=run, tb_writer=None)
    if run:
        run_logger.update_summary(metrics)
    _log_split_diagnostics(
        logger_obj=run_logger,
        split_diagnostics=split_diagnostics,
        run_name=run_name,
        tier_id="legacy",
        model_name="ensemble",
        dataset=dataset,
        trait=task,
    )
    if run:
        run.finish()

    # Save predictions
    save_predictions(test_texts, test_labels, split_predictions["test"], "ensemble", dataset, task)

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
    common_override_cfg = {k: v for k, v in dataset_override_cfg.items() if k not in TRANSFORMER_MODELS}
    model_specific_override_cfg = dataset_override_cfg.get(model_name, {})
    resolved_model_cfg = {**model_cfg, **common_override_cfg, **model_specific_override_cfg}

    import dataclasses

    valid_fields = {f.name for f in dataclasses.fields(TransformerConfig)}
    transformer_config = TransformerConfig(
        model_name=resolved_model_cfg.get("model_name", f"{model_name}-base-uncased"),
        **{k: v for k, v in resolved_model_cfg.items() if k != "model_name" and k in valid_fields},
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

    split_truths = {"train": train_labels, "eval": val_labels, "test": test_labels}
    split_predictions = {
        "train": _as_list(trainer.predict(train_texts)),
        "eval": _as_list(trainer.predict(val_texts)),
        "test": _as_list(trainer.predict(test_texts)),
    }
    split_diagnostics = _build_split_diagnostics(
        split_truths=split_truths,
        split_predictions=split_predictions,
        tier_id="legacy",
        model_name=model_name,
        dataset=dataset,
        trait=dim,
    )
    metrics = _flatten_split_diagnostic_metrics(split_diagnostics)
    save_predictions(test_texts, test_labels, split_predictions["test"], model_name, dataset, dim)
    current_run = wandb.run
    run_name = current_run.name if current_run is not None else f"legacy_{model_name}_{dataset}_{dim}"
    run_logger = MultiBackendLogger(wandb_run=current_run, tb_writer=None)
    if current_run is not None:
        run_logger.update_summary(metrics)
    _log_split_diagnostics(
        logger_obj=run_logger,
        split_diagnostics=split_diagnostics,
        run_name=run_name,
        tier_id="legacy",
        model_name=model_name,
        dataset=dataset,
        trait=dim,
    )
    if current_run is not None:
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
            train_texts,
            train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            cache_key_train=ck_train,
            cache_key_val=ck_val,
        )
    else:
        raise ValueError(f"Unknown frozen baseline: {model_name}")

    logger.info("Evaluating on train/eval/test splits...")
    split_truths = {"train": train_labels, "eval": val_labels, "test": test_labels}
    split_predictions = {
        "train": _as_list(trainer.predict(train_texts, cache_key=ck_train)),
        "eval": _as_list(trainer.predict(val_texts, cache_key=ck_val)),
        "test": _as_list(trainer.predict(test_texts, cache_key=ck_test)),
    }
    split_diagnostics = _build_split_diagnostics(
        split_truths=split_truths,
        split_predictions=split_predictions,
        tier_id="legacy",
        model_name=model_name,
        dataset=dataset,
        trait=task,
    )
    metrics = _flatten_split_diagnostic_metrics(split_diagnostics)

    run_name = run.name if run else f"legacy_{model_name}_{dataset}_{task}"
    run_logger = MultiBackendLogger(wandb_run=run, tb_writer=None)
    if run:
        run_logger.update_summary(metrics)
    _log_split_diagnostics(
        logger_obj=run_logger,
        split_diagnostics=split_diagnostics,
        run_name=run_name,
        tier_id="legacy",
        model_name=model_name,
        dataset=dataset,
        trait=task,
    )
    if run:
        run.finish()

    save_predictions(test_texts, test_labels, split_predictions["test"], model_name, dataset, task)

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
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = get_task_data(dataset, task, config)

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

    split_truths = {"train": train_labels, "eval": val_labels, "test": test_labels}
    split_predictions = {
        "train": _as_list(trainer.predict(train_texts)),
        "eval": _as_list(trainer.predict(val_texts)),
        "test": _as_list(trainer.predict(test_texts)),
    }
    split_diagnostics = _build_split_diagnostics(
        split_truths=split_truths,
        split_predictions=split_predictions,
        tier_id="legacy",
        model_name="lstm",
        dataset=dataset,
        trait=task,
    )
    flat = _flatten_split_diagnostic_metrics(split_diagnostics)
    logger.info(
        f"lstm/{dataset}/{task} — "
        f"test_acc={flat.get('test_accuracy', 0):.4f} "
        f"test_f1={flat.get('test_f1_macro', 0):.4f}"
    )

    current_run = wandb.run if wandb_project else None
    run_name = current_run.name if current_run is not None else f"legacy_lstm_{dataset}_{task}"
    run_logger = MultiBackendLogger(wandb_run=current_run, tb_writer=None)
    if current_run is not None:
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
        run_logger.update_summary({**flat, **final_metrics})
    _log_split_diagnostics(
        logger_obj=run_logger,
        split_diagnostics=split_diagnostics,
        run_name=run_name,
        tier_id="legacy",
        model_name="lstm",
        dataset=dataset,
        trait=task,
    )
    if current_run is not None:
        wandb.finish()

    save_predictions(test_texts, test_labels, split_predictions["test"], "lstm", dataset, task)
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
            metrics = _train_transformer_single(
                model_name, dataset, trait, model_cfg, config, args, extra_tags=["ocean"]
            )
            all_metrics[trait] = metrics
        return all_metrics

    else:
        metrics = _train_transformer_single(model_name, dataset, task, model_cfg, config, args)
        return metrics


# Tier-config-driven orchestration (single W&B run per (model, dataset, task))


def _expand_traits_for_tier(dataset: str, task: str) -> list[str]:
    """Return the list of binary heads to train inside a single W&B run."""
    if task == "16class":
        return ["16class"]
    if task == "4dim":
        return MBTI_DIMENSIONS[:]
    if task == "ocean_binary":
        return OCEAN_TRAITS[:]
    if task in MBTI_DIMENSIONS or task in OCEAN_TRAITS:
        return [task]
    raise ValueError(f"Cannot expand task {task!r} for tier orchestration")


def _setup_tier_logger(
    *,
    tier_id: str,
    tier_cfg: dict,
    model_name: str,
    dataset: str,
    task: str,
    seed: int,
    wandb_project: str | None,
    extra_tag: str | None = None,
) -> tuple[MultiBackendLogger, str, str]:
    """Create the parent MultiBackendLogger for one (model, dataset, task) run."""
    setting = "default"
    name_parts = [tier_id, model_name, dataset]
    if task in {"4dim", "ocean_binary", "16class"}:
        # Multi-trait — keep run name compact at task level
        pass
    else:
        name_parts.append(task)
    if extra_tag:
        name_parts.append(extra_tag)
        setting = extra_tag
    run_name = "_".join(name_parts)
    tb_enabled = (tier_cfg.get("tensorboard", {}) or {}).get("enabled", True)
    tb_dir = f"outputs/tensorboard/{tier_id}/{run_name}" if tb_enabled else None

    wandb_block = tier_cfg.get("wandb", {}) or {}
    tags = list(wandb_block.get("tags", []))
    for t in [tier_id, model_name, dataset, setting]:
        if t and t not in tags:
            tags.append(t)
    group = wandb_block.get("group", tier_id)

    config_dump = {
        "tier": tier_id,
        "model": model_name,
        "dataset": dataset,
        "task": task,
        "seed": seed,
        "setting": setting,
        **{k: v for k, v in tier_cfg.items() if k not in {"_base"}},
    }
    parent = MultiBackendLogger.init_run(
        project=wandb_project,
        name=run_name,
        tags=tags,
        group=group,
        config=config_dump,
        tensorboard_dir=tb_dir,
    )
    return parent, run_name, tb_dir or ""


def _safe_evaluate(trainer, texts, labels, **kwargs) -> dict:
    """Wrap evaluate to a flat scalar dict (drops non-numeric report fields)."""
    raw = trainer.evaluate(texts, labels, **kwargs)
    return {k: v for k, v in raw.items() if isinstance(v, (int, float))}


def _run_tier_ml(
    *,
    tier_id: str,
    tier_cfg: dict,
    dataset: str,
    task: str,
    args,
) -> dict:
    """Tier 1 LR (and other ML) flow: TF-IDF + LogReg with learning curves.

    Single W&B run; per-trait metrics namespaced via `trait_{T}/...`; aggregate
    `mean/test_macro_f1` etc. computed across traits.
    """
    from src.baselines.ml_baselines import MLBaselineTrainer

    model_name = tier_cfg.get("model", {}).get("type", "logistic_regression")
    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    parent, run_name, _ = _setup_tier_logger(
        tier_id=tier_id,
        tier_cfg=tier_cfg,
        model_name=model_name,
        dataset=dataset,
        task=task,
        seed=args.seed,
        wandb_project=wandb_project,
    )

    # Translate tier YAML → legacy ml_baselines config keys.
    tfidf_block = tier_cfg.get("model", {}).get("tfidf", {})
    clf_block = tier_cfg.get("model", {}).get("classifier", {})
    legacy_cfg = {
        "tfidf": dict(tfidf_block),
        "tfidf_char": {
            "analyzer": "char_wb",
            "ngram_range": [3, 5],
            "max_features": 20000,
            "sublinear_tf": True,
            "min_df": 3,
            "max_df": 0.95,
        },
        "dimensionality_reduction": {"enabled": False, "n_components": 300},
        "ml_models": {model_name: dict(clf_block)},
    }
    # Apply per-dataset override
    overrides = (tier_cfg.get("dataset_overrides", {}) or {}).get(dataset, {})
    if overrides:
        if "model" in overrides and "tfidf" in overrides["model"]:
            legacy_cfg["tfidf"].update(overrides["model"]["tfidf"])
    # Apply CLI overrides (supports legacy --set tfidf.max_features=X)
    legacy_cfg = _apply_overrides(legacy_cfg, args.overrides)

    traits = _expand_traits_for_tier(dataset, task)
    lc_block = tier_cfg.get("learning_curve", {}) or {}
    lc_enabled = lc_block.get("enabled", False)
    lc_method = lc_block.get("method", "sweep")  # "sweep" | "sgd"
    lc_iters: list[int] = lc_block.get("iter_checkpoints", []) if lc_enabled else []
    lc_patience = int(lc_block.get("early_stopping_patience", 15))
    lc_min_delta = float(lc_block.get("early_stopping_min_delta", 1e-4))
    lc_monitor = str(lc_block.get("early_stopping_monitor", "eval_loss"))

    # Pre-load per-trait data (texts shared across traits; labels differ).
    trait_data: dict[str, tuple] = {}
    for trait in traits:
        trait_data[trait] = get_task_data(dataset, trait, legacy_cfg)

    # Phase 1: SGD lockstep learning curves (multi-trait, per-epoch aggregates,
    # per-trait early stopping). Only runs for method="sgd"; legacy sweep mode
    # falls back to per-trait calls below.
    if lc_enabled and lc_method == "sgd":
        logger.info(f"\n=== {tier_id} | SGD lockstep learning curves ({len(traits)} traits) ===")
        _run_lr_sgd_lockstep(
            parent=parent,
            traits=traits,
            trait_data=trait_data,
            legacy_cfg=legacy_cfg,
            num_epochs=int(lc_iters[-1]) if lc_iters else 100,
            patience=lc_patience,
            min_delta=lc_min_delta,
            monitor=lc_monitor,
        )

    per_trait_metrics: dict[str, dict[str, float]] = {}
    try:
        for trait in traits:
            logger.info(f"\n=== {tier_id} | {model_name} | {dataset} | trait={trait} ===")
            child = parent.with_prefix(f"trait_{trait}")
            train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = trait_data[trait]

            # Legacy sweep mode (cheap multi-max_iter refit) — kept for users who
            # still want it. SGD method handled above via lockstep.
            if lc_enabled and lc_method != "sgd":
                _log_lr_learning_curve(
                    child=child,
                    model_name=model_name,
                    legacy_cfg=legacy_cfg,
                    train_texts=train_texts,
                    train_labels=train_labels,
                    val_texts=val_texts,
                    val_labels=val_labels,
                    iters=lc_iters,
                    method=lc_method,
                )

            trainer = MLBaselineTrainer(model_name, legacy_cfg)
            trainer.fit(train_texts, train_labels, use_grid_search=False)

            split_truths = {"train": train_labels, "eval": val_labels, "test": test_labels}
            split_predictions = {
                "train": _as_list(trainer.predict(train_texts)),
                "eval": _as_list(trainer.predict(val_texts)),
                "test": _as_list(trainer.predict(test_texts)),
            }
            split_diagnostics = _build_split_diagnostics(
                split_truths=split_truths,
                split_predictions=split_predictions,
                tier_id=tier_id,
                model_name=model_name,
                dataset=dataset,
                trait=trait,
            )
            split_metrics = {split: diagnostics["scalar_metrics"] for split, diagnostics in split_diagnostics.items()}
            flat = {f"{split}/{k}": v for split, m in split_metrics.items() for k, v in m.items()}
            # Per-trait final scalars → summary tab only (NO log_dict). Train/eval
            # curves are already streamed during training; test/* is inherently
            # a single end-of-training scalar. Pushing these to wandb.log()
            # creates misleading single-dot chart panels.
            child.update_summary(flat)

            # XAI: top features per class for LR
            if model_name == "logistic_regression":
                top_n = (tier_cfg.get("wandb", {}) or {}).get("log_top_features", 0)
                if top_n:
                    _log_top_features(child, trainer, top_n)

            preds_list = split_predictions["test"]
            save_predictions(test_texts, test_labels, preds_list, model_name, dataset, trait)
            output_path = f"outputs/models/tfidf_{model_name}_{dataset}_{trait}.pkl"
            trainer.save(output_path)

            _log_split_diagnostics(
                logger_obj=child,
                split_diagnostics=split_diagnostics,
                run_name=run_name,
                tier_id=tier_id,
                model_name=model_name,
                dataset=dataset,
                trait=trait,
            )

            _pretty_log_split_metrics(
                tier_id=tier_id,
                model_name=model_name,
                dataset=dataset,
                trait=trait,
                split_metrics=split_metrics,
                y_true_test=list(test_labels),
                y_pred_test=preds_list,
                run_name=run_name,
            )

            per_trait_metrics[trait] = {k.replace("/", "_"): v for k, v in flat.items()}

        # Aggregate across traits → run summary scalar only (no chart panels).
        aggregates = MultiBackendLogger.aggregate_per_trait(per_trait_metrics)
        parent.update_summary(aggregates)
        _pretty_log_aggregates(aggregates=aggregates, run_name=run_name)
    finally:
        parent.finish()

    return {
        "per_trait": per_trait_metrics,
        "aggregates": aggregates if "aggregates" in locals() else {},
    }


def _log_lr_learning_curve(
    *,
    child: MultiBackendLogger,
    model_name: str,
    legacy_cfg: dict,
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    iters: list[int],
    method: str = "sweep",
) -> None:
    """Log a learning curve for the LR baseline (legacy single-trait sweep).

    Note: `method="sgd"` is now handled at the outer level via
    `_run_lr_sgd_lockstep` to enable multi-trait per-epoch aggregates and
    per-trait early stopping. This function only handles the lbfgs sweep
    fallback for users who explicitly opt out of SGD.
    """
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    from src.baselines.ml_baselines import MLBaselineTrainer

    for n_iter in iters:
        cfg_copy = json.loads(json.dumps(legacy_cfg))
        cfg_copy["ml_models"][model_name]["max_iter"] = int(n_iter)
        trainer = MLBaselineTrainer(model_name, cfg_copy)
        trainer.fit(train_texts, train_labels, use_grid_search=False)
        try:
            val_preds = trainer.predict(val_texts)
            val_proba = trainer.predict_proba(val_texts) if hasattr(trainer, "predict_proba") else None
        except Exception:
            val_preds, val_proba = None, None
        if val_preds is not None:
            macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
            acc = accuracy_score(val_labels, val_preds)
            metrics = {
                "learning_curve/iter": float(n_iter),
                "learning_curve/val_macro_f1": float(macro_f1),
                "learning_curve/val_accuracy": float(acc),
            }
            if val_proba is not None:
                try:
                    classes = sorted(set(train_labels))
                    metrics["learning_curve/val_log_loss"] = float(log_loss(val_labels, val_proba, labels=classes))
                except Exception:
                    pass
            child.log_dict(metrics)


def _run_lr_sgd_lockstep(
    *,
    parent: MultiBackendLogger,
    traits: list[str],
    trait_data: dict[str, tuple],
    legacy_cfg: dict,
    num_epochs: int = 100,
    patience: int = 15,
    min_delta: float = 1e-4,
    monitor: str = "eval_loss",
) -> None:
    """Train SGDClassifier per trait in lockstep epochs to produce real
    learning curves with per-epoch aggregates across traits.

    For each epoch step, logs:
      - per-trait scalars `trait_{T}/{train,eval}/{loss,accuracy,f1_macro,...}`
      - aggregate scalars `agg_{mean,min,max}/{train,eval}/{metric}` across
        traits still active at that epoch.

    Per-trait early stopping: each trait monitors `monitor` (default eval_loss)
    on its own validation set. When no improvement for `patience` epochs, that
    trait's classifier is restored to best-epoch coefficients and frozen
    (excluded from subsequent updates). Loop breaks early when all traits stop.

    Logs `early_stopping/{trait}/{best_epoch,best_val_loss,stopped_at_epoch}`
    as run summary scalars at the end.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                                 precision_score, recall_score)
    from sklearn.utils.class_weight import compute_class_weight

    tfidf_cfg = legacy_cfg.get("tfidf", {})
    # Texts are shared across traits within the same dataset; vectorize once.
    sample_train_texts, _, sample_val_texts, _, _, _ = next(iter(trait_data.values()))
    vec = TfidfVectorizer(
        max_features=tfidf_cfg.get("max_features", 20000),
        ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
        sublinear_tf=tfidf_cfg.get("sublinear_tf", True),
        min_df=tfidf_cfg.get("min_df", 3),
        max_df=tfidf_cfg.get("max_df", 0.9),
    )
    X_train = vec.fit_transform(sample_train_texts)
    X_val = vec.transform(sample_val_texts)
    n_train = X_train.shape[0]

    # Per-trait state
    state: dict[str, dict] = {}
    for trait in traits:
        _, train_labels, _, val_labels, _, _ = trait_data[trait]
        y_train = np.array(train_labels)
        y_val = np.array(val_labels)
        classes = np.array(sorted(set(train_labels)))
        cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        clf = SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            learning_rate="adaptive",
            eta0=0.05,
            random_state=42,
            class_weight={c: float(w) for c, w in zip(classes, cw)},
        )
        state[trait] = {
            "clf": clf,
            "y_train": y_train,
            "y_val": y_val,
            "classes": classes,
            "child": parent.with_prefix(f"trait_{trait}"),
            "best_val_loss": float("inf"),
            "best_val_f1": -float("inf"),
            "best_epoch": 0,
            "best_coef": None,
            "best_intercept": None,
            "patience_counter": 0,
            "stopped_at": None,
            "done": False,
        }

    for epoch in range(1, num_epochs + 1):
        epoch_train: dict[str, dict[str, float]] = {}
        epoch_eval: dict[str, dict[str, float]] = {}

        # Build one merged dict of per-trait logs for this epoch — single
        # log call ensures all per-trait + aggregate metrics share the same
        # W&B step axis position.
        merged: dict[str, float] = {"epoch": float(epoch)}

        for trait in traits:
            st = state[trait]
            if st["done"]:
                continue

            clf, classes = st["clf"], st["classes"]
            y_train, y_val = st["y_train"], st["y_val"]

            # Shuffle and partial_fit
            rng = np.random.default_rng(42 + epoch)
            order = rng.permutation(n_train)
            clf.partial_fit(X_train[order], y_train[order], classes=classes)

            # Train metrics
            train_proba = clf.predict_proba(X_train)
            train_preds = clf.predict(X_train)
            try:
                tr_loss = float(log_loss(y_train, train_proba, labels=classes))
            except ValueError:
                tr_loss = float("nan")
            tr_metrics = {
                "loss": tr_loss,
                "accuracy": float(accuracy_score(y_train, train_preds)),
                "f1_macro": float(f1_score(y_train, train_preds, average="macro", zero_division=0)),
                "f1_weighted": float(f1_score(y_train, train_preds, average="weighted", zero_division=0)),
                "precision_macro": float(precision_score(y_train, train_preds, average="macro", zero_division=0)),
                "precision_weighted": float(precision_score(y_train, train_preds, average="weighted", zero_division=0)),
                "recall_macro": float(recall_score(y_train, train_preds, average="macro", zero_division=0)),
                "recall_weighted": float(recall_score(y_train, train_preds, average="weighted", zero_division=0)),
            }

            # Val metrics
            val_proba = clf.predict_proba(X_val)
            val_preds = clf.predict(X_val)
            try:
                v_loss = float(log_loss(y_val, val_proba, labels=classes))
            except ValueError:
                v_loss = float("nan")
            v_metrics = {
                "loss": v_loss,
                "accuracy": float(accuracy_score(y_val, val_preds)),
                "f1_macro": float(f1_score(y_val, val_preds, average="macro", zero_division=0)),
                "f1_weighted": float(f1_score(y_val, val_preds, average="weighted", zero_division=0)),
                "precision_macro": float(precision_score(y_val, val_preds, average="macro", zero_division=0)),
                "precision_weighted": float(precision_score(y_val, val_preds, average="weighted", zero_division=0)),
                "recall_macro": float(recall_score(y_val, val_preds, average="macro", zero_division=0)),
                "recall_weighted": float(recall_score(y_val, val_preds, average="weighted", zero_division=0)),
            }

            epoch_train[trait] = tr_metrics
            epoch_eval[trait] = v_metrics

            # Merge into per-epoch log dict using full prefixed keys so a
            # single parent.log_dict call lands them at the same step.
            for k, v in tr_metrics.items():
                merged[f"trait_{trait}/train/{k}"] = v
            for k, v in v_metrics.items():
                merged[f"trait_{trait}/eval/{k}"] = v

            # Early stopping check (per-trait independent)
            if monitor == "eval_loss":
                current = v_loss
                improved = current < st["best_val_loss"] - min_delta
            elif monitor in ("eval_f1_macro", "f1_macro"):
                current = v_metrics["f1_macro"]
                improved = current > st["best_val_f1"] + min_delta
            else:
                current = v_loss
                improved = current < st["best_val_loss"] - min_delta

            if improved:
                st["best_val_loss"] = v_loss
                st["best_val_f1"] = v_metrics["f1_macro"]
                st["best_epoch"] = epoch
                st["best_coef"] = clf.coef_.copy()
                st["best_intercept"] = clf.intercept_.copy()
                st["patience_counter"] = 0
            else:
                st["patience_counter"] += 1
                if st["patience_counter"] >= patience:
                    if st["best_coef"] is not None:
                        clf.coef_ = st["best_coef"]
                        clf.intercept_ = st["best_intercept"]
                    st["done"] = True
                    st["stopped_at"] = epoch
                    logger.info(
                        f"[ES] trait={trait} stopped at epoch {epoch} "
                        f"(best epoch={st['best_epoch']}, best val_loss={st['best_val_loss']:.4f})"
                    )

        # Aggregate across traits that ran this epoch
        if epoch_train:
            for split, src in (("train", epoch_train), ("eval", epoch_eval)):
                for metric_key in (
                    "loss",
                    "accuracy",
                    "f1_macro",
                    "f1_weighted",
                    "precision_macro",
                    "precision_weighted",
                    "recall_macro",
                    "recall_weighted",
                ):
                    vals = [
                        src[t][metric_key] for t in src if not (src[t][metric_key] != src[t][metric_key])
                    ]  # filter NaN
                    if not vals:
                        continue
                    merged[f"agg_mean/{split}/{metric_key}"] = sum(vals) / len(vals)
                    merged[f"agg_min/{split}/{metric_key}"] = min(vals)
                    merged[f"agg_max/{split}/{metric_key}"] = max(vals)

        # Single log call → all per-trait + aggregate metrics share step axis
        parent.log_dict(merged, step=epoch)

        if all(state[t]["done"] for t in traits):
            logger.info(f"[ES] All traits stopped by epoch {epoch}; ending lockstep early.")
            break

    # Run summary: ES diagnostics
    es_summary: dict[str, float] = {}
    for trait in traits:
        st = state[trait]
        es_summary[f"early_stopping/{trait}/best_epoch"] = float(st["best_epoch"])
        es_summary[f"early_stopping/{trait}/best_val_loss"] = float(st["best_val_loss"])
        if st["stopped_at"] is not None:
            es_summary[f"early_stopping/{trait}/stopped_at_epoch"] = float(st["stopped_at"])
        else:
            es_summary[f"early_stopping/{trait}/stopped_at_epoch"] = float(num_epochs)
    parent.update_summary(es_summary)


_METRIC_HEADERS = ("acc", "f1_macro", "f1_weight", "p_macro", "p_weight", "r_macro", "r_weight")
_METRIC_KEYS = (
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "precision_weighted",
    "recall_macro",
    "recall_weighted",
)


def _format_metric_table(rows: list[tuple[str, dict[str, float]]]) -> str:
    """Render a fixed-column ASCII table for split metrics.

    `rows` is `[(row_label, metrics_dict), ...]`. Missing keys render as `--`.
    """
    col_widths = [max(len("split"), max(len(r[0]) for r in rows))]
    col_widths.extend(max(len(h), 8) for h in _METRIC_HEADERS)

    def _fmt_row(label: str, values: list[str]) -> str:
        cells = [label.ljust(col_widths[0])]
        cells.extend(v.rjust(col_widths[i + 1]) for i, v in enumerate(values))
        return " | ".join(cells)

    sep = "-+-".join("-" * w for w in col_widths)
    lines = [_fmt_row("split", list(_METRIC_HEADERS)), sep]
    for label, metrics in rows:
        values = []
        for k in _METRIC_KEYS:
            v = metrics.get(k)
            values.append(f"{v:.4f}" if isinstance(v, (int, float)) else "--")
        lines.append(_fmt_row(label, values))
    return "\n".join(lines)


def _format_aggregate_table(aggregates: dict[str, float], prefix: str = "summary/test_") -> str:
    """Render mean/min/max table from `aggregate_per_trait` output.

    Keys expected: `summary/test_mean/{metric}`, `summary/test_min/{metric}`,
    `summary/test_max/{metric}`. Anything outside the prefix is ignored.
    """
    metrics: dict[str, dict[str, float]] = {}
    for full_key, value in aggregates.items():
        if not full_key.startswith(prefix):
            continue
        rest = full_key[len(prefix) :]  # e.g. "mean/f1_macro"
        if "/" not in rest:
            continue
        stat, metric = rest.split("/", 1)
        metrics.setdefault(metric, {})[stat] = value

    if not metrics:
        return ""

    metric_col = max(len("metric"), max(len(m) for m in metrics))
    stat_cols = ("mean", "min", "max")
    stat_w = 8

    def _fmt(label: str, values: list[str]) -> str:
        cells = [label.ljust(metric_col)]
        cells.extend(v.rjust(stat_w) for v in values)
        return " | ".join(cells)

    sep = "-+-".join(["-" * metric_col] + ["-" * stat_w] * len(stat_cols))
    lines = [_fmt("metric", list(stat_cols)), sep]
    for metric in sorted(metrics):
        row = metrics[metric]
        values = [f"{row.get(s, float('nan')):.4f}" if s in row else "--" for s in stat_cols]
        lines.append(_fmt(metric, values))
    return "\n".join(lines)


def _pretty_log_split_metrics(
    *,
    tier_id: str,
    model_name: str,
    dataset: str,
    trait: str,
    split_metrics: dict[str, dict[str, float]],
    y_true_test: list | None = None,
    y_pred_test: list | None = None,
    output_dir: str | Path = "outputs/reports/run_summaries",
    run_name: str | None = None,
) -> str:
    """Pretty-print per-trait final metrics + sklearn classification_report.

    Renders an ASCII table to the console (loguru) and persists a markdown
    copy under `output_dir/{run_name}/trait_{trait}.md`. Returns the rendered
    table string for callers that want to embed it elsewhere.
    """
    from sklearn.metrics import classification_report

    rows = [(split, m) for split, m in split_metrics.items() if m]
    table = _format_metric_table(rows)
    header = f"=== {tier_id} | {model_name} | {dataset} | trait={trait} — final metrics ==="
    block = f"\n{header}\n{table}\n"

    report_str = ""
    if y_true_test is not None and y_pred_test is not None and len(y_true_test) > 0:
        try:
            report_str = classification_report(y_true_test, y_pred_test, zero_division=0)
            block += f"\nTest classification report:\n{report_str}\n"
        except Exception as exc:
            logger.debug(f"classification_report skipped for {trait}: {exc}")

    logger.info(block)

    if run_name:
        try:
            out_dir = Path(output_dir) / run_name
            out_dir.mkdir(parents=True, exist_ok=True)
            md_path = out_dir / f"trait_{trait}.md"
            md_path.write_text(
                f"# {header}\n\n```\n{table}\n```\n\n"
                + (f"## Test classification report\n\n```\n{report_str}\n```\n" if report_str else "")
            )
        except Exception as exc:
            logger.debug(f"run_summary write skipped for {trait}: {exc}")

    return table


def _pretty_log_aggregates(
    *,
    aggregates: dict[str, float],
    run_name: str | None = None,
    output_dir: str | Path = "outputs/reports/run_summaries",
) -> None:
    """Pretty-print test-set aggregates + persist markdown copy."""
    table = _format_aggregate_table(aggregates)
    if not table:
        # Fall back to legacy single-line dump when no `summary/test_*` keys.
        logger.info(f"Aggregates: {aggregates}")
        return
    block = f"\n=== Aggregate (test-set across traits) ===\n{table}\n"
    logger.info(block)
    if run_name:
        try:
            out_dir = Path(output_dir) / run_name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "_aggregates.md").write_text(f"# Aggregate (test-set across traits)\n\n```\n{table}\n```\n")
        except Exception as exc:
            logger.debug(f"aggregate summary write skipped: {exc}")


def _log_top_features(child: MultiBackendLogger, trainer, top_n: int) -> None:
    """Log top-N TF-IDF features per class for LR XAI angle."""
    try:
        clf = trainer.classifier
        feature_names = trainer.vectorizer.get_feature_names_out() if hasattr(trainer, "vectorizer") else None
        if feature_names is None or not hasattr(clf, "coef_"):
            return
        import numpy as np

        coefs = clf.coef_
        classes = list(getattr(clf, "classes_", []))
        rows: list[list] = []
        for i, cls_label in enumerate(classes):
            if coefs.shape[0] == 1:
                # Binary LR: positive coefs → class 1, negative → class 0
                if i == 1:
                    sign = +1
                    indices = np.argsort(-coefs[0])[:top_n]
                else:
                    sign = -1
                    indices = np.argsort(coefs[0])[:top_n]
                weights = sign * coefs[0][indices]
            else:
                indices = np.argsort(-coefs[i])[:top_n]
                weights = coefs[i][indices]
            for rank, (idx, weight) in enumerate(zip(indices, weights), start=1):
                rows.append([str(cls_label), rank, str(feature_names[idx]), float(weight)])
        child.log_table("top_features", ["class", "rank", "feature", "weight"], rows)
    except Exception as exc:
        logger.warning(f"top-features logging skipped: {exc}")


def _run_tier_transformer(
    *,
    tier_id: str,
    tier_cfg: dict,
    dataset: str,
    task: str,
    args,
) -> dict:
    """Tier 2a — fine-tune RoBERTa-base end-to-end with truncation."""
    import dataclasses

    from src.baselines.transformer_baseline import (TransformerBaseline,
                                                    TransformerConfig)
    from src.utils.wandb_callbacks import MultiBackendCallback

    model_block = tier_cfg.get("model", {})
    training_block = tier_cfg.get("training", {})
    logging_block = tier_cfg.get("logging", {})
    model_name_full = model_block.get("model_name", "roberta-base")
    # Use short model id for run names: "roberta", "distilbert"
    model_short = model_name_full.split("-")[0]
    extra_tag = "weighted" if training_block.get("loss_weighting") == "sqrt_balanced" else None

    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    parent, run_name, tb_dir = _setup_tier_logger(
        tier_id=tier_id,
        tier_cfg=tier_cfg,
        model_name=model_short,
        dataset=dataset,
        task=task,
        seed=args.seed,
        wandb_project=wandb_project,
        extra_tag=extra_tag,
    )

    traits = _expand_traits_for_tier(dataset, task)
    per_trait_metrics: dict[str, dict[str, float]] = {}

    try:
        for trait in traits:
            logger.info(f"\n=== {tier_id} | {model_short} | {dataset} | trait={trait} ===")
            train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = get_task_data(
                dataset, trait, {}
            )
            # Build TransformerConfig honoring tier YAML + dataset overrides
            ds_override = (tier_cfg.get("dataset_overrides", {}) or {}).get(dataset, {})
            merged_model = {**model_block, **(ds_override.get("model", {}) or {})}
            merged_training = {**training_block, **(ds_override.get("training", {}) or {})}
            merged_logging = {**logging_block, **(ds_override.get("logging", {}) or {})}

            output_dir = args.output_dir or f"outputs/models/{tier_id}_{model_short}_{dataset}_{trait}" + (
                "_weighted" if extra_tag else ""
            )
            valid_fields = {f.name for f in dataclasses.fields(TransformerConfig)}
            init_kwargs = {
                "model_name": merged_model.get("model_name", "roberta-base"),
                "max_length": merged_model.get("max_length", 512),
                "dropout": merged_model.get("dropout"),
                "use_pretrained": merged_model.get("use_pretrained", True),
                **{k: v for k, v in merged_training.items() if k in valid_fields},
                **{k: v for k, v in merged_logging.items() if k in valid_fields},
                "tensorboard_dir": tb_dir or None,
                "output_dir": output_dir,
                "seed": args.seed,
            }
            init_kwargs = {k: v for k, v in init_kwargs.items() if k in valid_fields}
            transformer_config = TransformerConfig(**init_kwargs)

            trainer = TransformerBaseline(transformer_config)
            trait_logger = parent.with_prefix(f"trait_{trait}")
            cb = MultiBackendCallback(logger=trait_logger)
            trainer.train(
                train_texts,
                train_labels,
                val_texts,
                val_labels,
                output_dir=output_dir,
                wandb_project=wandb_project,  # so HF Trainer routes to existing W&B run
                external_callbacks=[cb],
            )

            split_truths = {"train": train_labels, "eval": val_labels, "test": test_labels}
            split_predictions = {
                "train": _as_list(trainer.predict(train_texts)),
                "eval": _as_list(trainer.predict(val_texts)),
                "test": _as_list(trainer.predict(test_texts)),
            }
            model_label = f"{model_short}{'_weighted' if extra_tag else ''}"
            split_diagnostics = _build_split_diagnostics(
                split_truths=split_truths,
                split_predictions=split_predictions,
                tier_id=tier_id,
                model_name=model_label,
                dataset=dataset,
                trait=trait,
            )
            split_metrics = {split: diagnostics["scalar_metrics"] for split, diagnostics in split_diagnostics.items()}
            flat = {f"{split}/{k}": v for split, m in split_metrics.items() for k, v in m.items()}
            # Final scalars → summary tab only (curves come from training-time
            # callbacks; test/* is end-of-training scalar by definition).
            trait_logger.update_summary(flat)

            preds = split_predictions["test"]
            save_predictions(
                test_texts,
                test_labels,
                preds,
                f"{tier_id}_{model_short}",
                dataset,
                trait + ("_weighted" if extra_tag else ""),
            )
            _log_split_diagnostics(
                logger_obj=trait_logger,
                split_diagnostics=split_diagnostics,
                run_name=run_name,
                tier_id=tier_id,
                model_name=model_label,
                dataset=dataset,
                trait=trait,
            )
            _pretty_log_split_metrics(
                tier_id=tier_id,
                model_name=model_label,
                dataset=dataset,
                trait=trait,
                split_metrics=split_metrics,
                y_true_test=list(test_labels),
                y_pred_test=list(preds),
                run_name=run_name,
            )
            per_trait_metrics[trait] = {k.replace("/", "_"): v for k, v in flat.items()}

        # Cross-trait aggregates → run summary tab only (no chart panels).
        aggregates = MultiBackendLogger.aggregate_per_trait(per_trait_metrics)
        parent.update_summary(aggregates)
        _pretty_log_aggregates(aggregates=aggregates, run_name=run_name)
    finally:
        parent.finish()

    return {
        "per_trait": per_trait_metrics,
        "aggregates": aggregates if "aggregates" in locals() else {},
    }


def _run_tier_frozen(
    *,
    tier_id: str,
    tier_cfg: dict,
    dataset: str,
    task: str,
    args,
) -> dict:
    """Tier 2b — frozen RoBERTa + MLP head (chunked encoding)."""
    from src.baselines.frozen_transformer_baselines import RobertaMlpBaseline

    model_type = tier_cfg.get("model", {}).get("type", "roberta_mlp")
    encoder_block = tier_cfg.get("model", {}).get("encoder", {})
    head_block = tier_cfg.get("model", {}).get("head", {})
    training_block = tier_cfg.get("training", {})

    legacy_cfg = {
        "encoder": dict(encoder_block),
        "head": dict(head_block),
        "training": dict(training_block),
        "seed": args.seed,
    }
    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    parent, run_name, _ = _setup_tier_logger(
        tier_id=tier_id,
        tier_cfg=tier_cfg,
        model_name=model_type,
        dataset=dataset,
        task=task,
        seed=args.seed,
        wandb_project=wandb_project,
    )

    traits = _expand_traits_for_tier(dataset, task)
    per_trait_metrics: dict[str, dict[str, float]] = {}
    try:
        for trait in traits:
            logger.info(f"\n=== {tier_id} | {model_type} | {dataset} | trait={trait} ===")
            train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = get_task_data(
                dataset, trait, {}
            )
            enc_name = encoder_block.get("model_name", "roberta-base")
            ck_train = f"{dataset}_train"
            ck_val = f"{dataset}_val"
            ck_test = f"{dataset}_test"

            trainer = RobertaMlpBaseline(config=legacy_cfg, model_name=model_type)
            child = parent.with_prefix(f"trait_{trait}")
            trainer.fit(
                train_texts,
                train_labels,
                val_texts=val_texts,
                val_labels=val_labels,
                cache_key_train=ck_train,
                cache_key_val=ck_val,
                mb_logger=child,
            )
            split_truths = {"train": train_labels, "eval": val_labels, "test": test_labels}
            split_predictions = {
                "train": _as_list(trainer.predict(train_texts, cache_key=ck_train)),
                "eval": _as_list(trainer.predict(val_texts, cache_key=ck_val)),
                "test": _as_list(trainer.predict(test_texts, cache_key=ck_test)),
            }
            split_diagnostics = _build_split_diagnostics(
                split_truths=split_truths,
                split_predictions=split_predictions,
                tier_id=tier_id,
                model_name=model_type,
                dataset=dataset,
                trait=trait,
            )
            split_metrics = {split: diagnostics["scalar_metrics"] for split, diagnostics in split_diagnostics.items()}
            flat = {f"{split}/{k}": v for split, m in split_metrics.items() for k, v in m.items()}
            # Final scalars → summary tab only. Train/eval curves come from
            # `RobertaMlpBaseline.fit` epoch_log; test/* is end-of-training scalar.
            child.update_summary(flat)

            preds = split_predictions["test"]
            save_predictions(test_texts, test_labels, preds, f"{tier_id}_{model_type}", dataset, trait)
            output_dir = f"outputs/models/{tier_id}_{model_type}_{dataset}_{trait}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            trainer.save(f"{output_dir}/model.pt")
            _log_split_diagnostics(
                logger_obj=child,
                split_diagnostics=split_diagnostics,
                run_name=run_name,
                tier_id=tier_id,
                model_name=model_type,
                dataset=dataset,
                trait=trait,
            )
            _pretty_log_split_metrics(
                tier_id=tier_id,
                model_name=model_type,
                dataset=dataset,
                trait=trait,
                split_metrics=split_metrics,
                y_true_test=list(test_labels),
                y_pred_test=preds,
                run_name=run_name,
            )
            per_trait_metrics[trait] = {k.replace("/", "_"): v for k, v in flat.items()}

        # Cross-trait aggregates → run summary tab only (no chart panels).
        aggregates = MultiBackendLogger.aggregate_per_trait(per_trait_metrics)
        parent.update_summary(aggregates)
        _pretty_log_aggregates(aggregates=aggregates, run_name=run_name)
    finally:
        parent.finish()

    return {
        "per_trait": per_trait_metrics,
        "aggregates": aggregates if "aggregates" in locals() else {},
    }


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply --set KEY=VALUE overrides via dot-path notation."""
    import ast

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"--set requires KEY=VALUE format, got: {override!r}")
        key_path, _, raw_value = override.partition("=")
        keys = key_path.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        try:
            node[keys[-1]] = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            node[keys[-1]] = raw_value
    return cfg


def run_tier(args) -> int:
    """Tier-config-driven orchestration. Returns exit code (0 = success)."""
    cfg = load_config(args.config)
    cfg = _apply_overrides(cfg, args.overrides)
    tier_id = cfg.get("tier")
    if not tier_id:
        raise ValueError(f"Config {args.config} must define `tier:` for tier orchestration")
    if not tier_id.startswith("tier"):
        raise ValueError(f"Unexpected tier id: {tier_id}")

    setup_logging()
    set_seed(args.seed)

    # Decide tier flow by model type
    model_block = cfg.get("model", {})
    model_type = model_block.get("type") or model_block.get("model_name", "")
    if model_type in {"logistic_regression", "svm", "naive_bayes", "xgboost", "random_forest"}:
        flow = "ml"
    elif tier_id.startswith("tier2a") or model_type in {"roberta", "distilbert"}:
        flow = "transformer"
    elif tier_id.startswith("tier2b") or model_type in {"roberta_mlp", "frozen_bert_svm"}:
        flow = "frozen"
    else:
        raise ValueError(f"Cannot infer training flow from tier={tier_id}, model_type={model_type}")

    if flow == "ml":
        _run_tier_ml(tier_id=tier_id, tier_cfg=cfg, dataset=args.dataset, task=args.task, args=args)
    elif flow == "transformer":
        _run_tier_transformer(tier_id=tier_id, tier_cfg=cfg, dataset=args.dataset, task=args.task, args=args)
    elif flow == "frozen":
        _run_tier_frozen(tier_id=tier_id, tier_cfg=cfg, dataset=args.dataset, task=args.task, args=args)

    return 0


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument(
        "--model",
        required=False,
        help="Model name or 'all_ml' (legacy mode); inferred from tier config in tier mode",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "mbti",
            "mbti_uncleaned",
            "mbti_augmented",
            "essays",
            "essays_augmented",
            "pandora",
            "pandora_augmented",
            "personality_evd",
            "personality_evd_augmented",
        ],
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
        "--set",
        metavar="KEY=VALUE",
        action="append",
        dest="overrides",
        default=[],
        help=(
            "Override a config value using dot-path notation, e.g. "
            "--set transformer.distilbert.loss_weighting=sqrt_balanced"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: small subset, 1-2 epochs (Phase 1.5.5 verification)",
    )
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)
    config = load_config(args.config)

    # Tier-orchestration mode: config defines `tier:` ⇒ single-W&B-run flow
    if isinstance(config, dict) and config.get("tier"):
        if args.smoke:
            # Smoke: shrink budget per-tier flow handles in run_tier (TBD)
            os.environ["RAGXPR_SMOKE_MODE"] = "1"
        return run_tier(args)

    for override in args.overrides:
        if "=" not in override:
            raise ValueError(f"--set requires KEY=VALUE format, got: {override!r}")
        key_path, _, raw_value = override.partition("=")
        keys = key_path.split(".")
        node = config
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        try:
            node[keys[-1]] = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            node[keys[-1]] = raw_value

    tasks = expand_tasks(args.dataset, args.task)
    models = ML_MODELS if args.model == "all_ml" else [args.model]

    all_results = {}
    for task in tasks:
        for model in models:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Training {model} on {args.dataset}/{task}")
            logger.info(f"{'=' * 60}")
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
                scalar_results = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                logger.info(f"Results: {json.dumps(scalar_results, indent=2)}")
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
