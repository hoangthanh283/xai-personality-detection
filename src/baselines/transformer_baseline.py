"""HuggingFace Trainer-based fine-tuning for DistilBERT / RoBERTa.

Supports two classification modes:
  - "16class": Single 16-way softmax (CrossEntropyLoss)
  - "4dim":    Four binary classifiers (one per MBTI dimension)
  - "ocean_binary": Five binary classifiers (one per Big Five trait)
"""

import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

try:
    import torch
    from datasets import Dataset as HFDataset
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, DataCollatorWithPadding,
                              EarlyStoppingCallback, Trainer,
                              TrainingArguments)

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers/torch not installed")


@dataclass
class TransformerConfig:
    model_name: str = "distilbert-base-uncased"
    use_pretrained: bool = True
    max_length: int = 512
    truncation_strategy: str = "head_tail"
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    lr_scheduler: str = "linear"
    early_stopping_patience: int = 3
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    dropout: float | None = None
    seed: int = 42
    output_dir: str = "outputs/models/transformer"
    loss_weighting: str = "sqrt_balanced"  # none | balanced | sqrt_balanced
    metric_for_best_model: str = "eval_accuracy"
    # Tier-2 step-level training observability
    logging_strategy: str = "epoch"          # "epoch" | "steps"
    logging_steps: int = 50
    eval_strategy: str = "epoch"             # "epoch" | "steps"
    eval_steps: int = 200
    save_strategy: str = "epoch"             # "epoch" | "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    max_grad_norm: float = 1.0
    dataloader_num_workers: int = 0
    tensorboard_dir: str | None = None       # if set, HF Trainer writes TB events here
    report_to_extra: list[str] | None = None # ["tensorboard"] etc., merged with wandb routing


if HAS_TRANSFORMERS:
    from transformers import TrainerCallback

    class _LRLoggerCallback(TrainerCallback):
        """Log learning rate and step-level train loss to W&B at each logging step."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            try:
                import wandb as _wandb
                if not _wandb.run:
                    return
                step_log = {"train/global_step": state.global_step}
                if "learning_rate" in logs:
                    step_log["train/learning_rate"] = logs["learning_rate"]
                if "loss" in logs:
                    step_log["train/loss_step"] = logs["loss"]
                if step_log:
                    _wandb.log(step_log, step=state.global_step)
            except Exception:
                pass

    class WeightedClassificationTrainer(Trainer):
        """Trainer that applies label-frequency class weights for single-label classification."""

        def __init__(self, *args, class_weights: "torch.Tensor | None" = None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits

            if labels is None:
                loss = outputs.get("loss") if isinstance(outputs, dict) else outputs.loss
            else:
                weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
                loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

            return (loss, outputs) if return_outputs else loss


class TransformerBaseline:
    """HuggingFace-based fine-tuning for personality classification."""

    def __init__(self, config: TransformerConfig | None = None):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "Please install torch and transformers: pip install torch transformers"
            )
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

        none: no weighting (uniform)
        balanced: inverse-frequency weights (can be extreme at high imbalance ratios)
        sqrt_balanced: sqrt-dampened weights, capped at 4x (safer for imbalanced multi-class)
        """
        weighting = getattr(self.config, "loss_weighting", "sqrt_balanced")
        if weighting is None or weighting == "none":
            return None

        from sklearn.utils.class_weight import compute_class_weight

        n_classes = len(self.id2label)
        label_ids = np.array([self.label2id[label] for label in labels], dtype=np.int64)
        class_ids = np.arange(n_classes)
        weights = compute_class_weight(class_weight="balanced", classes=class_ids, y=label_ids)

        if weighting == "sqrt_balanced":
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
            "precision_weighted": precision_score(
                labels, preds, average="weighted", zero_division=0
            ),
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
            f"f1_weighted={metrics['f1_weighted']:.4f} | prec_macro={metrics['precision_macro']:.4f} | "
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
            raise ValueError(
                "Transformer baselines must use pretrained checkpoints. Set use_pretrained=true."
            )

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
            for attr in ("dropout", "seq_classif_dropout", "hidden_dropout_prob",
                         "attention_probs_dropout_prob", "classifier_dropout"):
                if hasattr(mc, attr):
                    setattr(mc, attr, self.config.dropout)
        logger.info(
            "Loaded pretrained checkpoint: "
            f"{getattr(self.model.config, '_name_or_path', self.config.model_name)}"
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

        # Training arguments — config-driven step strategies for full observability
        report_to_list: list[str] = []
        if wandb_project:
            report_to_list.append("wandb")
        extra_routes = getattr(self.config, "report_to_extra", None) or []
        for r in extra_routes:
            if r and r not in report_to_list:
                report_to_list.append(r)
        # if a logging_dir was set we always want tensorboard events
        if getattr(self.config, "tensorboard_dir", None) and "tensorboard" not in report_to_list:
            report_to_list.append("tensorboard")
        report_to = report_to_list if report_to_list else "none"

        # Validate save_strategy compatibility with eval_strategy when load_best_model_at_end=True
        eval_strategy = getattr(self.config, "eval_strategy", "epoch")
        save_strategy = getattr(self.config, "save_strategy", "epoch")
        if eval_strategy != save_strategy:
            logger.warning(
                f"eval_strategy={eval_strategy} != save_strategy={save_strategy}; "
                "HF Trainer requires they match when load_best_model_at_end=True. "
                f"Forcing save_strategy={eval_strategy}."
            )
            save_strategy = eval_strategy

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
            "save_steps": getattr(self.config, "save_steps", 200),
            "save_total_limit": getattr(self.config, "save_total_limit", 3),
            "load_best_model_at_end": True,
            "metric_for_best_model": getattr(
                self.config, "metric_for_best_model", "eval_accuracy"
            ),
            "greater_is_better": True,
            "report_to": report_to,
            "run_name": f"{Path(output_dir).name}",
            "seed": self.config.seed,
            "logging_strategy": getattr(self.config, "logging_strategy", "epoch"),
            "logging_steps": getattr(self.config, "logging_steps", 50),
            "logging_first_step": True,
            "dataloader_num_workers": getattr(self.config, "dataloader_num_workers", 0),
        }
        # Route TensorBoard events into the per-tier dir if provided
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
            training_args_kwargs["eval_steps"] = getattr(self.config, "eval_steps", 200)

        training_args = TrainingArguments(**training_args_kwargs)

        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience),
            _LRLoggerCallback(),
        ]
        if external_callbacks:
            callbacks.extend(external_callbacks)
        trainer = WeightedClassificationTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=callbacks,
            class_weights=class_weights,
            data_collator=data_collator,
        )

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
            batch_texts = texts[i: i + batch_size]
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
            "precision_weighted": precision_score(
                labels, preds, average="weighted", zero_division=0
            ),
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
            with open(label_map_file) as f:
                mapping = json.load(f)
            trainer.label2id = mapping["label2id"]
            trainer.id2label = {int(k): v for k, v in mapping["id2label"].items()}

        logger.info(f"Loaded model from {checkpoint_dir}")
        return trainer
