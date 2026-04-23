"""Frozen-BERT+SVM sentence scorer for evidence selection.

Implements the same public API as RoBERTaEvidenceScorer but uses
FrozenBertSvmBaseline (Kazameini 2020 paradigm) as the supervised backbone.

Key advantages over fine-tuned RoBERTa scorer:
- class_weight='balanced' → minority recall 56% on MBTI SN (vs RoBERTa 1%)
- BaggingClassifier.decision_function gives well-calibrated confidence
- No GPU required — frozen encoder + fast SVM inference
- Confidence scale: sigmoid(|SVM margin|) ∈ [0.5, 1.0] — same as RoBERTa softmax

Used by RAGXPRPipeline when evidence_retrieval.backbone = "frozen_svm".
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from loguru import logger

from src.baselines.frozen_transformer_baselines import (
    FrozenBertSvmBaseline,
    FrozenTransformerEncoder,
)
from src.retrieval.roberta_scorer import ScoredSentence


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _margin_to_confidence(margin: float) -> float:
    """Map a signed SVM margin to a confidence in [0.5, 1.0]."""
    return _sigmoid(abs(margin))


class FrozenSvmEvidenceScorer:
    """Score sentences using frozen-encoder + bagged LinearSVC baselines.

    Loads one checkpoint per MBTI dimension (or OCEAN trait). Shares a single
    FrozenTransformerEncoder across all dims to avoid redundant model loads.
    """

    MIN_SENTENCE_TOKENS = 4

    def __init__(
        self,
        checkpoint_paths: dict[str, str],
        encoder_cfg: Mapping[str, Any] | None = None,
        batch_size: int = 32,
    ):
        """
        Args:
            checkpoint_paths: {dim: path_to_model.pkl}, e.g. {"IE": "outputs/models/.../model.pkl"}
            encoder_cfg: optional encoder config override (model_name, pooling, etc.)
            batch_size: batch size for encoder inference (not used by SVM itself)
        """
        self.batch_size = batch_size
        self.baselines: dict[str, FrozenBertSvmBaseline] = {}

        # Build shared encoder once upfront (respects encoder_cfg device override).
        # This avoids each FrozenBertSvmBaseline.load() creating its own GPU encoder.
        shared_encoder: FrozenTransformerEncoder | None = None
        if encoder_cfg is not None and checkpoint_paths:
            shared_encoder = FrozenTransformerEncoder(encoder_cfg)

        for dim, ckpt_path in checkpoint_paths.items():
            if not Path(ckpt_path).exists():
                logger.warning(f"FrozenSVM checkpoint missing for {dim}: {ckpt_path} (skipping)")
                continue
            logger.info(f"Loading FrozenBertSvmBaseline[{dim}] from {ckpt_path}")
            # Patch device into saved config before constructing encoder to avoid CUDA OOM.
            import pickle
            with open(ckpt_path, "rb") as _f:
                state = pickle.load(_f)
            if encoder_cfg is not None and "device" in encoder_cfg:
                enc_cfg = dict(state["config"].get("encoder") or {})
                enc_cfg["device"] = encoder_cfg["device"]
                state["config"] = {**state["config"], "encoder": enc_cfg}
            baseline = FrozenBertSvmBaseline(config=state["config"])
            baseline.bag = state["bag"]
            baseline._label_encoder = state["label_encoder"]
            baseline.is_fitted = True

            # Share encoder across dims to avoid loading roberta-base 4× times.
            if shared_encoder is None:
                shared_encoder = baseline.encoder
            else:
                baseline.encoder = shared_encoder

            self.baselines[dim] = baseline

        if not self.baselines:
            raise RuntimeError("No FrozenSVM checkpoints were loaded")
        logger.info(f"FrozenSvmEvidenceScorer loaded dims: {list(self.baselines.keys())}")

    def _score_dim(self, X: np.ndarray, dim: str) -> list[tuple[str, float]]:
        """Return (label, confidence) for each row of X using the dim classifier."""
        baseline = self.baselines[dim]
        bag = baseline.bag
        le = baseline._label_encoder

        margins = bag.decision_function(X)  # (N,) for binary; (N, C) for multiclass
        out: list[tuple[str, float]] = []
        for i in range(len(X)):
            if margins.ndim == 1:
                margin = float(margins[i])
                # BaggingClassifier binary: positive margin → classes_[1]
                label_idx = 1 if margin > 0 else 0
                label = le.inverse_transform([label_idx])[0]
                conf = _margin_to_confidence(margin)
            else:
                # Multiclass: pick class with largest margin
                label_idx = int(np.argmax(margins[i]))
                label = le.inverse_transform([label_idx])[0]
                conf = _margin_to_confidence(float(margins[i, label_idx]))
            out.append((str(label), conf))
        return out

    def score_sentences(self, sentences: list[str]) -> list[ScoredSentence]:
        """Score each sentence by max SVM confidence across all dims."""
        if not sentences:
            return []

        valid_idxs = [
            i for i, s in enumerate(sentences) if len(s.split()) >= self.MIN_SENTENCE_TOKENS
        ]
        valid_sents = [sentences[i] for i in valid_idxs]

        if not valid_sents:
            return [ScoredSentence(text=s, sentence_idx=i, score=0.0) for i, s in enumerate(sentences)]

        # Encode once — shared encoder, all valid sentences.
        shared_encoder = next(iter(self.baselines.values())).encoder
        X = shared_encoder.encode(valid_sents)

        per_dim_preds: dict[str, list[tuple[str, float]]] = {}
        for dim in self.baselines:
            per_dim_preds[dim] = self._score_dim(X, dim)

        scored_valid: list[ScoredSentence] = []
        for local_i, (orig_i, sent) in enumerate(zip(valid_idxs, valid_sents)):
            dim_labels = {dim: per_dim_preds[dim][local_i] for dim in per_dim_preds}
            max_conf = max(conf for _, conf in dim_labels.values())
            scored_valid.append(
                ScoredSentence(
                    text=sent,
                    sentence_idx=orig_i,
                    score=max_conf,
                    predicted_labels=dim_labels,
                )
            )

        scored_all = list(scored_valid)
        filtered_out = set(range(len(sentences))) - set(valid_idxs)
        for i in filtered_out:
            scored_all.append(ScoredSentence(text=sentences[i], sentence_idx=i, score=0.0))
        return scored_all

    def predict_doc_level(self, text: str) -> dict[str, tuple[str, float]]:
        """Predict per-dim label + confidence for the full document."""
        shared_encoder = next(iter(self.baselines.values())).encoder
        X = shared_encoder.encode([text])  # (1, hidden_dim) — chunking handled internally
        preds: dict[str, tuple[str, float]] = {}
        for dim in self.baselines:
            label, conf = self._score_dim(X, dim)[0]
            preds[dim] = (label, round(conf, 3))
        return preds


def default_mbti_svm_checkpoints(models_dir: str = "outputs/models") -> dict[str, str]:
    return {
        "IE": f"{models_dir}/frozen_bert_svm_mbti_IE/model.pkl",
        "SN": f"{models_dir}/frozen_bert_svm_mbti_SN/model.pkl",
        "TF": f"{models_dir}/frozen_bert_svm_mbti_TF/model.pkl",
        "JP": f"{models_dir}/frozen_bert_svm_mbti_JP/model.pkl",
    }


def default_ocean_svm_checkpoints(models_dir: str = "outputs/models", dataset: str = "essays") -> dict[str, str]:
    dims = ["O", "C", "E", "A", "N"]
    ckpts: dict[str, str] = {}
    for d in dims:
        primary = Path(f"{models_dir}/frozen_bert_svm_{dataset}_{d}/model.pkl")
        fallback = Path(f"{models_dir}/frozen_bert_svm_essays_{d}/model.pkl")
        ckpts[d] = str(primary if primary.exists() else fallback)
    return ckpts
