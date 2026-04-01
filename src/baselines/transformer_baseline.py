"""HuggingFace Trainer-based fine-tuning for DistilBERT / RoBERTa.

Supports two classification modes:
  - "16class": Single 16-way softmax (CrossEntropyLoss)
  - "4dim":    Four binary classifiers (one per MBTI dimension)
  - "ocean_binary": Five binary classifiers (one per Big Five trait)
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

try:
    import torch
    from torch import nn
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
    from datasets import Dataset as HFDataset
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers/torch not installed")


@dataclass
class TransformerConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 512
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    lr_scheduler: str = "linear"
    early_stopping_patience: int = 3
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    seed: int = 42
    output_dir: str = "outputs/models/transformer"


class TransformerBaseline:
    """HuggingFace-based fine-tuning for personality classification."""

    def __init__(self, config: TransformerConfig | None = None):
        if not HAS_TRANSFORMERS:
            raise ImportError("Please install torch and transformers: pip install torch transformers")
        self.config = config or TransformerConfig()
        self.tokenizer = None
        self.model = None
        self.label2id: dict = {}
        self.id2label: dict = {}

    def _setup_labels(self, labels: list[str]) -> None:
        unique_labels = sorted(set(labels))
        self.label2id = {l: i for i, l in enumerate(unique_labels)}
        self.id2label = {i: l for l, i in self.label2id.items()}

    def _tokenize(self, texts: list[str]) -> dict:
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

    def _make_hf_dataset(self, texts: list[str], labels: list[str]) -> "HFDataset":
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
        )
        label_ids = [self.label2id[l] for l in labels]
        data = dict(encodings)
        data["labels"] = label_ids
        return HFDataset.from_dict(data)

    def _compute_metrics(self, eval_pred) -> dict:
        from sklearn.metrics import accuracy_score, f1_score
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        }

    def train(
        self,
        train_texts: list[str],
        train_labels: list[str],
        val_texts: list[str],
        val_labels: list[str],
        output_dir: str | None = None,
        wandb_project: str | None = None,
    ) -> None:
        """Fine-tune the transformer on personality classification."""
        self._setup_labels(train_labels)
        output_dir = output_dir or self.config.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save label mapping
        with open(Path(output_dir) / "label_map.json", "w") as f:
            json.dump({"label2id": self.label2id, "id2label": self.id2label}, f)

        # Load tokenizer and model
        logger.info(f"Loading {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
        )

        # Create HF datasets
        train_dataset = self._make_hf_dataset(train_texts, train_labels)
        val_dataset = self._make_hf_dataset(val_texts, val_labels)

        # Training arguments
        report_to = "wandb" if wandb_project else "none"
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size * 2,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler,
            fp16=self.config.fp16 and torch.cuda.is_available(),
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            report_to=report_to,
            run_name=f"{Path(output_dir).name}",
            seed=self.config.seed,
            logging_steps=50,
        )

        callbacks = [EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)]
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=callbacks,
        )

        logger.info("Starting training...")
        trainer.train()
        logger.info("Training complete. Saving best model...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def predict(self, texts: list[str], batch_size: int = 32) -> list[str]:
        """Generate predictions for a list of texts."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")

        self.model.eval()
        device = next(self.model.parameters()).device
        all_preds = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
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
        from sklearn.metrics import accuracy_score, f1_score
        preds = self.predict(texts)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
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
