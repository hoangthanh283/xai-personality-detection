"""HuggingFace Trainer-based fine-tuning for DistilBERT / RoBERTa.

Supports two classification modes:
  - "16class": Single 16-way softmax (CrossEntropyLoss)
  - "4dim":    Four binary classifiers (one per MBTI dimension)
  - "ocean_binary": Five binary classifiers (one per Big Five trait)
"""

import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset as HFDataset
from loguru import logger
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainerCallback, TrainingArguments)

import wandb


def _uses_multi_backend_callback(callbacks: list | None) -> bool:
    return any(cb.__class__.__name__ == "MultiBackendCallback" for cb in (callbacks or []))


def _build_report_to(config: "TransformerConfig", wandb_project: str | None, uses_multi_backend: bool):
    report_to_list: list[str] = []
    if wandb_project and not uses_multi_backend:
        report_to_list.append("wandb")
    extra_routes = getattr(config, "report_to_extra", None) or []
    for route in extra_routes:
        if route and route not in report_to_list:
            report_to_list.append(route)
    if getattr(config, "tensorboard_dir", None) and "tensorboard" not in report_to_list:
        report_to_list.append("tensorboard")
    return report_to_list if report_to_list else "none"


def _steps_per_epoch(num_examples: int, batch_size: int, gradient_accumulation_steps: int) -> int:
    batches = math.ceil(max(1, num_examples) / max(1, batch_size))
    return max(1, math.ceil(batches / max(1, gradient_accumulation_steps)))


def _cap_steps_to_epoch(configured_steps: int, steps_per_epoch: int) -> int:
    """Ensure step-based logging/eval emits at least one point per epoch."""
    return max(1, min(int(configured_steps), int(steps_per_epoch)))


@dataclass
class TransformerConfig:
    model_name: str = "distilbert-base-uncased"
    use_pretrained: bool = True
    max_length: int = 512
    truncation_strategy: str = "head_tail"
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_ratio: float = 0.1
    lr_scheduler: str = "linear"
    early_stopping_patience: int = 5
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    dropout: float | None = None
    seed: int = 42
    output_dir: str = "outputs/models/transformer"
    loss_weighting: str = "sqrt_balanced"  # none | balanced | sqrt_balanced | focal
    focal_gamma: float = 2.0  # only used when loss_weighting == "focal"
    use_balanced_sampler: bool = False  # WeightedRandomSampler over training set
    metric_for_best_model: str = "eval_accuracy"
    # Tier-2 step-level training observability
    logging_strategy: str = "epoch"  # "epoch" | "steps"
    logging_steps: int = 50
    eval_strategy: str = "epoch"  # "epoch" | "steps"
    eval_steps: int = 200
    save_strategy: str = "epoch"  # "epoch" | "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    max_grad_norm: float = 1.0
    dataloader_num_workers: int = 0
    tensorboard_dir: str | None = None  # if set, HF Trainer writes TB events here
    report_to_extra: list[str] | None = None  # ["tensorboard"] etc., merged with wandb routing


class _LRLoggerCallback(TrainerCallback):
    """Log learning rate and step-level train loss to W&B at each logging step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if not wandb.run:
            return
        step_log = {"train/global_step": state.global_step}
        if "learning_rate" in logs:
            step_log["train/learning_rate"] = logs["learning_rate"]
        if "loss" in logs:
            step_log["train/loss_step"] = logs["loss"]
        if step_log:
            wandb.log(step_log, step=state.global_step)


class WeightedClassificationTrainer(Trainer):
    """Trainer with optional class-weighted CE, focal loss, and balanced sampling.

    Loss modes (selected by `loss_mode`):
      - "ce" / "weighted_ce" — torch.nn.CrossEntropyLoss with optional class weights.
      - "focal" — focal loss (Lin et al. 2017): (1-pt)^gamma * CE. Down-weights
        easy/majority examples; up-weights hard/minority. `class_weights` act
        as alpha (per-class scaling).

    Balanced sampling (`use_balanced_sampler=True`): overrides `get_train_dataloader`
    to use `WeightedRandomSampler` so each batch sees balanced class distribution
    (replacement=True). Combine with focal loss for strongest minority handling.
    """

    def __init__(
        self,
        *args,
        class_weights: "torch.Tensor | None" = None,
        loss_mode: str = "weighted_ce",
        focal_gamma: float = 2.0,
        use_balanced_sampler: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_mode = loss_mode
        self.focal_gamma = focal_gamma
        self.use_balanced_sampler = use_balanced_sampler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits

        if labels is None:
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs.loss
        else:
            num_labels = model.config.num_labels
            weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
            logits_flat = logits.view(-1, num_labels)
            labels_flat = labels.view(-1)

            if self.loss_mode == "focal":
                # Focal loss: (1-pt)^gamma * weighted_CE per-sample
                ce_per_sample = torch.nn.functional.cross_entropy(
                    logits_flat, labels_flat, weight=weight, reduction="none"
                )
                pt = torch.exp(-ce_per_sample)
                focal_factor = (1.0 - pt) ** self.focal_gamma
                loss = (focal_factor * ce_per_sample).mean()
            else:
                loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
                loss = loss_fct(logits_flat, labels_flat)

        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        """Override Trainer's default DataLoader to inject WeightedRandomSampler.

        Only applies when `use_balanced_sampler=True`. Each sample is weighted by
        `1 / class_count` so the sampler draws an approximately balanced batch.
        replacement=True since class counts can be very imbalanced (e.g. MBTI SN
        86:14 → minority needs to be drawn ~6x more often).
        """
        if not self.use_balanced_sampler:
            return super().get_train_dataloader()
        from torch.utils.data import DataLoader, WeightedRandomSampler

        train_dataset = self.train_dataset
        labels_tensor = torch.tensor(
            train_dataset["labels"]
            if "labels" in train_dataset.column_names
            else [int(row["labels"]) for row in train_dataset]
        )
        class_counts = torch.bincount(labels_tensor)
        inv_freq = 1.0 / class_counts.float().clamp(min=1)
        sample_weights = inv_freq[labels_tensor]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


class TransformerBaseline:
    """HuggingFace-based fine-tuning for personality classification."""

    def __init__(self, config: TransformerConfig | None = None):
        self.config = config or TransformerConfig()
        self.tokenizer = None
        self.model = None
        self.label2id: dict = {}
        self.id2label: dict = {}

    def _setup_labels(self, labels: list[str]) -> None:
        unique_labels = sorted(set(labels))
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def _tokenize(self, texts: list[str]) -> dict:
        prepared_texts = [self._prepare_text_for_tokenizer(text) for text in texts]
        return self.tokenizer(
            prepared_texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

    def _prepare_text_for_tokenizer(self, text: str) -> str:
        if self.config.truncation_strategy not in {"head", "head_tail"}:
            raise ValueError(f"Unknown truncation strategy: {self.config.truncation_strategy}")

        tokens = self.tokenizer.tokenize(text)
        special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        max_content_tokens = max(1, self.config.max_length - special_tokens)

        if len(tokens) <= max_content_tokens:
            return text

        if self.config.truncation_strategy == "head_tail" and max_content_tokens > 1:
            head_tokens = max_content_tokens // 2
            tail_tokens = max_content_tokens - head_tokens
            kept_tokens = tokens[:head_tokens] + tokens[-tail_tokens:]
        else:
            kept_tokens = tokens[:max_content_tokens]

        return self.tokenizer.convert_tokens_to_string(kept_tokens)

    def _prepare_texts_for_tokenizer(self, texts: list[str]) -> list[str]:
        return [self._prepare_text_for_tokenizer(text) for text in texts]

    def _make_hf_dataset(self, texts: list[str], labels: list[str]) -> "HFDataset":
        encodings = self.tokenizer(
            self._prepare_texts_for_tokenizer(texts),
            truncation=True,
            max_length=self.config.max_length,
        )
        label_ids = [self.label2id[label] for label in labels]
        data = dict(encodings)
        data["labels"] = label_ids
        return HFDataset.from_dict(data)

    def _compute_class_weights(self, labels: list[str]) -> "torch.Tensor | None":
        """Compute class weights based on loss_weighting config.

        Modes:
          - none:           uniform (returns None)
          - balanced:       inverse-frequency (can be extreme at high imbalance)
          - sqrt_balanced:  sqrt-dampened, clipped to [0.5, 2.0] (default — safe)
          - focal:          alpha weights for focal loss (sqrt_balanced clipping)
                            applied multiplicatively against (1-pt)^gamma factor
        """
        weighting = getattr(self.config, "loss_weighting", "sqrt_balanced")
        if weighting is None or weighting == "none":
            return None

        from sklearn.utils.class_weight import compute_class_weight

        n_classes = len(self.id2label)
        label_ids = np.array([self.label2id[label] for label in labels], dtype=np.int64)
        class_ids = np.arange(n_classes)
        weights = compute_class_weight(class_weight="balanced", classes=class_ids, y=label_ids)

        if weighting in ("sqrt_balanced", "focal"):
            counts = np.bincount(label_ids, minlength=n_classes).astype(float)
            counts = np.where(counts == 0, 1.0, counts)
            raw = counts.sum() / (n_classes * counts)
            weights = np.sqrt(raw)
            weights = np.clip(weights, 0.5, 2.0)

        return torch.tensor(weights, dtype=torch.float32)

    def _compute_metrics(self, eval_pred) -> dict:
        from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                     recall_score)

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
            "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
            "precision_weighted": precision_score(labels, preds, average="weighted", zero_division=0),
            "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
            "recall_weighted": recall_score(labels, preds, average="weighted", zero_division=0),
        }
        # Per-class metrics — surfaces minority-class collapse on imbalanced
        # tasks (e.g. PersonalityEvd E=HIGH 97.7%, MBTI SN N=86%).
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
        per_class_recall = recall_score(labels, preds, average=None, zero_division=0)
        for i, (f1_i, rec_i) in enumerate(zip(per_class_f1, per_class_recall)):
            metrics[f"f1_class_{i}"] = float(f1_i)
            metrics[f"recall_class_{i}"] = float(rec_i)
        logger.info(
            f"eval | acc={metrics['accuracy']:.4f} | f1_macro={metrics['f1_macro']:.4f} | "
            f"f1_weighted={metrics['f1_weighted']:.4f} | "
            f"prec_macro={metrics['precision_macro']:.4f} | "
            f"rec_macro={metrics['recall_macro']:.4f} | per_class_f1={per_class_f1.tolist()}"
        )
        return metrics

    def train(
        self,
        train_texts: list[str],
        train_labels: list[str],
        val_texts: list[str],
        val_labels: list[str],
        output_dir: str | None = None,
        wandb_project: str | None = None,
        external_callbacks: list | None = None,
    ) -> None:
        """Fine-tune the transformer on personality classification."""
        self._setup_labels(train_labels + val_labels)
        output_dir = output_dir or self.config.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save label mapping
        with open(Path(output_dir) / "label_map.json", "w") as f:
            json.dump({"label2id": self.label2id, "id2label": self.id2label}, f)

        if not self.config.use_pretrained:
            raise ValueError("Transformer baselines must use pretrained checkpoints. Set use_pretrained=true.")

        # Load tokenizer and model
        logger.info(f"Loading pretrained tokenizer/model from {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model_kwargs = {
            "num_labels": len(self.label2id),
            "id2label": self.id2label,
            "label2id": self.label2id,
        }
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        if self.config.dropout is not None:
            mc = self.model.config
            for attr in (
                "dropout",
                "seq_classif_dropout",
                "hidden_dropout_prob",
                "attention_probs_dropout_prob",
                "classifier_dropout",
            ):
                if hasattr(mc, attr):
                    setattr(mc, attr, self.config.dropout)
        logger.info(
            f"Loaded pretrained checkpoint: {getattr(self.model.config, '_name_or_path', self.config.model_name)}"
        )

        # Create HF datasets
        train_dataset = self._make_hf_dataset(train_texts, train_labels)
        val_dataset = self._make_hf_dataset(val_texts, val_labels)
        class_weights = self._compute_class_weights(train_labels)
        logger.info(
            f"Loss weighting: {getattr(self.config, 'loss_weighting', 'sqrt_balanced')} | "
            f"weights={'None' if class_weights is None else class_weights.numpy().round(3).tolist()}"
        )
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8 if self.config.fp16 and torch.cuda.is_available() else None,
        )

        steps_per_epoch = _steps_per_epoch(
            num_examples=len(train_texts),
            batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )
        logging_strategy = getattr(self.config, "logging_strategy", "epoch")
        eval_strategy = getattr(self.config, "eval_strategy", "epoch")
        save_strategy = getattr(self.config, "save_strategy", "epoch")
        logging_steps = _cap_steps_to_epoch(getattr(self.config, "logging_steps", 50), steps_per_epoch)
        eval_steps = _cap_steps_to_epoch(getattr(self.config, "eval_steps", 200), steps_per_epoch)
        save_steps = _cap_steps_to_epoch(getattr(self.config, "save_steps", 200), steps_per_epoch)

        # Validate save_strategy compatibility with eval_strategy when load_best_model_at_end=True.
        if eval_strategy != save_strategy:
            logger.warning(
                f"eval_strategy={eval_strategy} != save_strategy={save_strategy}; "
                "HF Trainer requires they match when load_best_model_at_end=True. "
                f"Forcing save_strategy={eval_strategy}."
            )
            save_strategy = eval_strategy
            save_steps = eval_steps

        uses_multi_backend = _uses_multi_backend_callback(external_callbacks)
        report_to = _build_report_to(self.config, wandb_project, uses_multi_backend)

        training_args_kwargs = {
            "output_dir": output_dir,
            "num_train_epochs": self.config.num_epochs,
            "per_device_train_batch_size": self.config.batch_size,
            "per_device_eval_batch_size": self.config.batch_size * 2,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "warmup_ratio": self.config.warmup_ratio,
            "lr_scheduler_type": self.config.lr_scheduler,
            "fp16": self.config.fp16 and torch.cuda.is_available(),
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "max_grad_norm": getattr(self.config, "max_grad_norm", 1.0),
            "save_strategy": save_strategy,
            "save_steps": save_steps,
            "save_total_limit": getattr(self.config, "save_total_limit", 3),
            "load_best_model_at_end": True,
            "metric_for_best_model": getattr(self.config, "metric_for_best_model", "eval_accuracy"),
            "greater_is_better": True,
            "report_to": report_to,
            "run_name": Path(output_dir).name,
            "seed": self.config.seed,
            "logging_strategy": logging_strategy,
            "logging_steps": logging_steps,
            "logging_first_step": True,
            "dataloader_num_workers": getattr(self.config, "dataloader_num_workers", 0),
        }
        tb_dir = getattr(self.config, "tensorboard_dir", None)
        if tb_dir:
            training_args_kwargs["logging_dir"] = tb_dir
        if getattr(self.config, "gradient_checkpointing", False):
            training_args_kwargs["gradient_checkpointing"] = True

        training_args_params = inspect.signature(TrainingArguments.__init__).parameters
        if "evaluation_strategy" in training_args_params:
            training_args_kwargs["evaluation_strategy"] = eval_strategy
        else:
            training_args_kwargs["eval_strategy"] = eval_strategy
        if eval_strategy == "steps":
            training_args_kwargs["eval_steps"] = eval_steps

        training_args = TrainingArguments(**training_args_kwargs)

        callbacks = []
        if not uses_multi_backend:
            callbacks.append(_LRLoggerCallback())
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience))
        if external_callbacks:
            callbacks.extend(external_callbacks)

        loss_weighting = getattr(self.config, "loss_weighting", "sqrt_balanced")
        trainer = WeightedClassificationTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=callbacks,
            class_weights=class_weights,
            data_collator=data_collator,
            loss_mode="focal" if loss_weighting == "focal" else "weighted_ce",
            focal_gamma=getattr(self.config, "focal_gamma", 2.0),
            use_balanced_sampler=getattr(self.config, "use_balanced_sampler", False),
        )
        for callback in callbacks:
            set_trainer = getattr(callback, "set_trainer", None)
            if callable(set_trainer):
                set_trainer(trainer)

        logger.info("Starting training...")
        trainer.train()

        best_checkpoint = trainer.state.best_model_checkpoint
        if best_checkpoint:
            logger.info(f"Reloading best checkpoint explicitly from {best_checkpoint}")
            self.model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint)
            if torch.cuda.is_available():
                self.model.to("cuda")
        else:
            self.model = trainer.model

        logger.info("Training complete. Saving best model...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def predict(self, texts: list[str], batch_size: int = 32) -> list[str]:
        """Generate predictions for a list of texts."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")

        self.model.eval()
        device = next(self.model.parameters()).device
        all_preds = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            prepared_batch_texts = self._prepare_texts_for_tokenizer(batch_texts)
            inputs = self.tokenizer(
                prepared_batch_texts,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend([self.id2label[p] for p in preds])

        return all_preds

    def evaluate(self, texts: list[str], labels: list[str]) -> dict:
        from sklearn.metrics import (accuracy_score, classification_report,
                                     f1_score, precision_score, recall_score)

        preds = self.predict(texts)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
            "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
            "precision_weighted": precision_score(labels, preds, average="weighted", zero_division=0),
            "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
            "recall_weighted": recall_score(labels, preds, average="weighted", zero_division=0),
            "classification_report": classification_report(labels, preds, zero_division=0),
        }

    @classmethod
    def load(cls, checkpoint_dir: str) -> "TransformerBaseline":
        """Load a trained model from a checkpoint directory."""
        trainer = cls()
        trainer.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        trainer.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)

        label_map_file = Path(checkpoint_dir) / "label_map.json"
        if label_map_file.exists():
            with open(label_map_file, encoding="utf-8") as f:
                mapping = json.load(f)
            trainer.label2id = mapping["label2id"]
            trainer.id2label = {int(k): v for k, v in mapping["id2label"].items()}

        logger.info(f"Loaded model from {checkpoint_dir}")
        return trainer
