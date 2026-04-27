"""Frozen-encoder transformer baselines for personality detection.

Implements the two paradigms that outperform end-to-end fine-tuning on
small-data, imbalanced, long-document social-media personality tasks:

  (A) FrozenBertSvmBaseline  — Kazameini et al. 2020 (arXiv:2010.01309).
      Frozen BERT/RoBERTa + bagged LinearSVC. Pool = mean of last 4 hidden
      layers over non-[PAD] tokens. Long docs are chunked into 512-token
      windows (stride 256) and per-chunk embeddings are mean-pooled.

  (B) RobertaMlpBaseline     — Gao et al. 2024 (arXiv:2406.16223) paradigm.
      Frozen RoBERTa + 2-layer MLP head (GELU + Dropout + LayerNorm).
      CrossEntropy with sqrt_balanced class weights, AdamW lr=1e-3,
      early stopping on val accuracy.

Both share `FrozenTransformerEncoder` — a cached, chunking encoder that writes
document-level embeddings to ``outputs/embeddings/{model}/{dataset}_{split}.npy``
so the same split isn't re-encoded across 4 MBTI binary tasks + 16-class (a
single encode of 6K MBTI docs takes ~6 min on CPU).
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers import logging as hf_logging

import wandb

hf_logging.set_verbosity_error()


# ---------------------------------------------------------------------------
# Shared frozen encoder with chunking + caching
# ---------------------------------------------------------------------------


@dataclass
class EncoderConfig:
    model_name: str = "roberta-base"
    pooling: str = "mean_last4"  # "mean_last4" | "cls"
    chunk_size: int = 512
    stride: int = 256
    batch_size: int = 16
    device: str | None = None
    cache_dir: str = "outputs/embeddings"


class FrozenTransformerEncoder:
    """Encode texts with a frozen transformer, returning doc-level vectors."""

    def __init__(self, config: EncoderConfig | Mapping[str, Any] | None = None):
        cfg = EncoderConfig(**dict(config or {})) if not isinstance(config, EncoderConfig) else config
        self.config = cfg
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading frozen encoder [{cfg.model_name}] on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModel.from_pretrained(cfg.model_name, output_hidden_states=(cfg.pooling == "mean_last4")).to(
            self.device
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.hidden_dim = self.model.config.hidden_size

    # ---- cache helpers -----------------------------------------------------

    def _cache_path(self, cache_key: str) -> Path:
        slug = self.config.model_name.replace("/", "__")
        return Path(self.config.cache_dir) / slug / f"{cache_key}.npy"

    def _hash_texts(self, texts: list[str]) -> str:
        h = hashlib.sha256()
        for t in texts[:20]:
            h.update(t.encode("utf-8", errors="ignore"))
        h.update(str(len(texts)).encode())
        return h.hexdigest()[:16]

    # ---- encoding ----------------------------------------------------------

    def _pool_chunk(
        self,
        last_hidden: "torch.Tensor",
        hidden_states: list,
        attn_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        """Pool one batch of chunks → (B, hidden_dim). attn_mask shape (B, L)."""
        if self.config.pooling == "cls":
            return last_hidden[:, 0, :]
        # mean_last4: average last 4 hidden-state layers, then mean over non-pad tokens
        last4 = torch.stack(hidden_states[-4:], dim=0).mean(dim=0)  # (B, L, H)
        mask = attn_mask.unsqueeze(-1).float()  # (B, L, 1)
        summed = (last4 * mask).sum(dim=1)  # (B, H)
        counts = mask.sum(dim=1).clamp(min=1e-6)  # (B, 1)
        return summed / counts

    def _encode_one(self, text: str) -> np.ndarray:
        """Encode a single long text — chunk it, embed each chunk, mean-pool chunks."""
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.chunk_size,
            stride=self.config.stride,
            return_overflowing_tokens=True,
            return_tensors="pt",
            padding="max_length",
        )
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)

        chunk_embs = []
        for start in range(0, input_ids.size(0), self.config.batch_size):
            b_ids = input_ids[start : start + self.config.batch_size]
            b_attn = attn[start : start + self.config.batch_size]
            with torch.no_grad():
                out = self.model(input_ids=b_ids, attention_mask=b_attn)
            hs = out.hidden_states if self.config.pooling == "mean_last4" else None
            pooled = self._pool_chunk(out.last_hidden_state, hs, b_attn)  # (B, H)
            chunk_embs.append(pooled.cpu().numpy())
        chunks = np.concatenate(chunk_embs, axis=0)  # (n_chunks, H)
        return chunks.mean(axis=0)  # (H,)

    def encode(self, texts: list[str], cache_key: str | None = None) -> np.ndarray:
        """Encode a list of texts. If cache_key provided, hit/save the cache."""
        if cache_key:
            path = self._cache_path(cache_key)
            if path.exists():
                cached = np.load(path)
                if cached.shape[0] == len(texts):
                    logger.info(f"Encoder cache hit: {path} shape={cached.shape}")
                    return cached
                logger.warning(f"Cache shape mismatch ({cached.shape[0]} vs {len(texts)}); re-encoding")
            path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Encoding {len(texts)} texts with {self.config.model_name} (pooling={self.config.pooling})")
        iterator = tqdm(texts, desc="encode", ncols=80)

        out = np.zeros((len(texts), self.hidden_dim), dtype=np.float32)
        for i, t in enumerate(iterator):
            out[i] = self._encode_one(t or " ")

        if cache_key:
            np.save(path, out)
            logger.info(f"Encoder cache saved: {path} shape={out.shape}")
        return out


# ---------------------------------------------------------------------------
# Baseline A: Frozen encoder + Bagged LinearSVC  (Kazameini 2020)
# ---------------------------------------------------------------------------


class FrozenBertSvmBaseline:
    """Frozen transformer embeddings + BaggingClassifier over LinearSVC."""

    def __init__(self, config: Mapping[str, Any] | None = None, model_name: str = "frozen_bert_svm"):
        cfg = dict(config or {})
        self.config = cfg
        self.model_name = model_name
        self.encoder = FrozenTransformerEncoder(cfg.get("encoder"))
        clf_cfg = cfg.get("classifier", {}) or {}
        svc_cfg = clf_cfg.get("svc", {}) or {}
        self.bag = BaggingClassifier(
            estimator=LinearSVC(
                C=svc_cfg.get("C", 1.0),
                class_weight=svc_cfg.get("class_weight", "balanced"),
                max_iter=svc_cfg.get("max_iter", 10000),
                dual="auto",
            ),
            n_estimators=clf_cfg.get("n_estimators", 10),
            max_samples=clf_cfg.get("max_samples", 0.8),
            random_state=cfg.get("seed", 42),
            n_jobs=clf_cfg.get("n_jobs", 1),
        )
        self._label_encoder: LabelEncoder | None = None
        self.is_fitted = False

    def fit(
        self,
        train_texts: list[str],
        train_labels: list[str],
        cache_key_train: str | None = None,
    ) -> "FrozenBertSvmBaseline":
        self._label_encoder = LabelEncoder().fit(train_labels)
        y = self._label_encoder.transform(train_labels)
        X = self.encoder.encode(train_texts, cache_key=cache_key_train)
        logger.info(f"Fitting Bagged LinearSVC on X={X.shape}, y={y.shape}")
        self.bag.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, texts: list[str], cache_key: str | None = None) -> np.ndarray:
        X = self.encoder.encode(texts, cache_key=cache_key)
        preds = self.bag.predict(X)
        return self._label_encoder.inverse_transform(preds)

    def evaluate(
        self,
        texts: list[str],
        labels: list[str],
        cache_key: str | None = None,
    ) -> dict:
        preds = self.predict(texts, cache_key=cache_key)
        return _classification_metrics(labels, preds)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bag": self.bag, "label_encoder": self._label_encoder, "config": self.config}, f)
        logger.info(f"FrozenBertSvmBaseline saved to {path}")

    @classmethod
    def load(cls, path: str, model_name: str = "frozen_bert_svm") -> "FrozenBertSvmBaseline":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(config=state["config"], model_name=model_name)
        obj.bag = state["bag"]
        obj._label_encoder = state["label_encoder"]
        obj.is_fitted = True
        return obj


# ---------------------------------------------------------------------------
# Baseline B: Frozen encoder + 2-layer MLP head  (Gao 2024 paradigm)
# ---------------------------------------------------------------------------


class _MlpHead(nn.Module):
    """2-layer MLP classifier head: hidden_dim → num_classes."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class RobertaMlpBaseline:
    """Frozen RoBERTa + 2-layer MLP head trained with AdamW + early stopping."""

    def __init__(self, config: Mapping[str, Any] | None = None, model_name: str = "roberta_mlp"):
        cfg = dict(config or {})
        self.config = cfg
        self.model_name = model_name
        self.encoder = FrozenTransformerEncoder(cfg.get("encoder"))
        self.head: "_MlpHead | None" = None
        self._label_encoder: LabelEncoder | None = None
        self.is_fitted = False
        self.device = self.encoder.device

    # ---- training ---------------------------------------------------------

    def _compute_class_weights(self, y: np.ndarray, num_classes: int, scheme: str) -> "torch.Tensor":
        counts = np.bincount(y, minlength=num_classes).astype(np.float64)
        counts = np.clip(counts, 1.0, None)
        if scheme == "balanced":
            w = counts.sum() / (num_classes * counts)
        elif scheme == "sqrt_balanced":
            w = np.sqrt(counts.sum() / (num_classes * counts))
        else:
            w = np.ones(num_classes)
        return torch.tensor(w, dtype=torch.float32, device=self.device)

    def fit(
        self,
        train_texts: list[str],
        train_labels: list[str],
        val_texts: list[str] | None = None,
        val_labels: list[str] | None = None,
        cache_key_train: str | None = None,
        cache_key_val: str | None = None,
        mb_logger=None,
    ) -> "RobertaMlpBaseline":
        self._label_encoder = LabelEncoder().fit(train_labels)
        y_train = self._label_encoder.transform(train_labels)
        num_classes = len(self._label_encoder.classes_)

        logger.info(f"Encoding train ({len(train_texts)}) and val ({len(val_texts or [])}) ...")
        X_train = self.encoder.encode(train_texts, cache_key=cache_key_train)
        X_val = self.encoder.encode(val_texts, cache_key=cache_key_val) if val_texts else None
        y_val = self._label_encoder.transform(val_labels) if val_labels is not None else None

        training = self.config.get("training", {}) or {}
        head_cfg = self.config.get("head", {}) or {}
        self.head = _MlpHead(
            input_dim=self.encoder.hidden_dim,
            hidden_dim=head_cfg.get("hidden_dim", 256),
            num_classes=num_classes,
            dropout=head_cfg.get("dropout", 0.3),
        ).to(self.device)

        class_weights = self._compute_class_weights(
            y_train, num_classes, training.get("loss_weighting", "sqrt_balanced")
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=training.get("learning_rate", 1e-3),
            weight_decay=training.get("weight_decay", 0.01),
        )

        ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long())
        loader = DataLoader(ds, batch_size=training.get("batch_size", 128), shuffle=True)

        max_epochs = training.get("num_epochs", 30)
        patience = training.get("early_stopping_patience", 5)
        best_metric = -np.inf
        best_state: dict | None = None
        waited = 0
        global_step = 0

        for epoch in range(1, max_epochs + 1):
            self.head.train()
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.head(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
                global_step += 1
                # Route per-batch loss through MultiBackendLogger if provided
                # (writes to W&B + TensorBoard with shared step axis); fallback
                # to global wandb.log so legacy paths still log somewhere.
                if mb_logger is not None:
                    mb_logger.log_dict(
                        {"train/loss_step": float(loss.item()), "train/global_step": float(global_step)},
                        step=global_step,
                    )
                elif wandb.run:
                    wandb.log(
                        {"train/loss_step": loss.item(), "train/global_step": global_step},
                        step=global_step,
                    )
            train_running_loss = total_loss / len(ds)  # noqa: F841 (kept for legacy log)

            # Full train + val metric dicts (avoid re-encoding via cached X_*)
            train_metrics = self._eval_on_encoded_full(X_train, y_train)
            val_metrics = self._eval_on_encoded_full(X_val, y_val) if X_val is not None else {}
            current_lr = optimizer.param_groups[0]["lr"]

            msg = (
                f"epoch {epoch:02d} | lr={current_lr:.2e} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"train_acc={train_metrics['accuracy']:.4f} | "
                f"train_f1={train_metrics['f1_macro']:.4f}"
            )
            if val_metrics:
                msg += (
                    f" | val_loss={val_metrics['loss']:.4f} | "
                    f"val_acc={val_metrics['accuracy']:.4f} | "
                    f"val_f1={val_metrics['f1_macro']:.4f}"
                )
            logger.info(msg)

            epoch_log: dict[str, float] = {
                "epoch": float(epoch),
                "train/learning_rate": float(current_lr),
                **{f"train/{k}": float(v) for k, v in train_metrics.items()},
                **{f"eval/{k}": float(v) for k, v in val_metrics.items()},
            }
            if mb_logger is not None:
                mb_logger.log_dict(epoch_log, step=global_step)
            elif wandb.run:
                wandb.log(epoch_log, step=global_step)

            # Early stop on val_loss (consistent with Tier 1 + Tier 2a). Lower
            # is better, so negate for the max-better comparison below.
            if val_metrics:
                metric = -val_metrics["loss"]
            else:
                metric = -train_metrics["loss"]
            if metric > best_metric + 1e-6:
                best_metric = metric
                best_state = {k: v.detach().cpu().clone() for k, v in self.head.state_dict().items()}
                waited = 0
            else:
                waited += 1
                if waited >= patience:
                    logger.info(f"Early stop at epoch {epoch} (best metric={best_metric:.4f})")
                    break

        if best_state is not None:
            self.head.load_state_dict(best_state)
        self.is_fitted = True
        return self

    def _eval_on_encoded(self, X: np.ndarray, y: np.ndarray) -> float:
        return self._eval_on_encoded_full(X, y).get("accuracy", 0.0)

    def _eval_on_encoded_full(self, X: np.ndarray, y: np.ndarray) -> dict:
        self.head.eval()
        with torch.no_grad():
            logits = self.head(torch.from_numpy(X).to(self.device))
            ce_loss = float(F.cross_entropy(logits, torch.from_numpy(y).long().to(self.device)).item())
            preds = logits.argmax(dim=-1).cpu().numpy()
        return {
            "loss": ce_loss,
            "accuracy": float((preds == y).mean()),
            "f1_macro": float(f1_score(y, preds, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y, preds, average="weighted", zero_division=0)),
            "precision_macro": float(precision_score(y, preds, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(y, preds, average="weighted", zero_division=0)),
            "recall_macro": float(recall_score(y, preds, average="macro", zero_division=0)),
            "recall_weighted": float(recall_score(y, preds, average="weighted", zero_division=0)),
        }

    # ---- inference --------------------------------------------------------

    def predict(self, texts: list[str], cache_key: str | None = None) -> np.ndarray:
        X = self.encoder.encode(texts, cache_key=cache_key)
        self.head.eval()
        with torch.no_grad():
            logits = self.head(torch.from_numpy(X).to(self.device))
            preds = logits.argmax(dim=-1).cpu().numpy()
        return self._label_encoder.inverse_transform(preds)

    def evaluate(
        self,
        texts: list[str],
        labels: list[str],
        cache_key: str | None = None,
    ) -> dict:
        preds = self.predict(texts, cache_key=cache_key)
        return _classification_metrics(labels, preds)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "head_state": self.head.state_dict(),
                "head_cfg": {
                    "input_dim": self.encoder.hidden_dim,
                    "hidden_dim": self.config.get("head", {}).get("hidden_dim", 256),
                    "num_classes": len(self._label_encoder.classes_),
                    "dropout": self.config.get("head", {}).get("dropout", 0.3),
                },
                "label_classes": list(self._label_encoder.classes_),
                "config": self.config,
            },
            path,
        )
        logger.info(f"RobertaMlpBaseline saved to {path}")

    @classmethod
    def load(cls, path: str, model_name: str = "roberta_mlp") -> "RobertaMlpBaseline":
        state = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(config=state["config"], model_name=model_name)
        obj.head = _MlpHead(**state["head_cfg"]).to(obj.device)
        obj.head.load_state_dict(state["head_state"])
        obj._label_encoder = LabelEncoder().fit(state["label_classes"])
        obj.is_fitted = True
        return obj


# ---------------------------------------------------------------------------
# Shared metrics helper
# ---------------------------------------------------------------------------


def _classification_metrics(labels: list[str], preds: np.ndarray) -> dict:
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "precision_weighted": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "recall_weighted": recall_score(labels, preds, average="weighted", zero_division=0),
        "classification_report": classification_report(labels, preds, zero_division=0),
    }
    logger.info(
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"F1-macro: {metrics['f1_macro']:.4f} | "
        f"F1-weighted: {metrics['f1_weighted']:.4f}"
    )
    return metrics
