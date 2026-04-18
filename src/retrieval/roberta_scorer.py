"""Supervised RoBERTa sentence scorer for evidence selection.

Uses fine-tuned RoBERTa classifiers (one per MBTI dimension or OCEAN trait) to:
1. Score each sentence by max per-dim classifier confidence (= personality signal).
2. Optionally produce doc-level predictions to inject as a prior in CoPE Step 3.

Unlike the keyword-based EvidenceRetriever, this scorer uses supervised
features learned from the training data — sentences the fine-tuned model
classifies confidently are the ones with strongest personality signal.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              logging as hf_logging)
    hf_logging.set_verbosity_error()
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class ScoredSentence:
    text: str
    sentence_idx: int
    score: float  # max confidence across dims (in [0.5, 1.0])
    predicted_labels: dict[str, tuple[str, float]] = field(default_factory=dict)
    matched_keywords: list[str] = field(default_factory=list)  # kept for API compat


class RoBERTaEvidenceScorer:
    """Score sentences using an ensemble of fine-tuned RoBERTa classifiers."""

    MIN_SENTENCE_TOKENS = 4  # skip fragments below this

    def __init__(
        self,
        checkpoint_dirs: dict[str, str],
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 256,
    ):
        if not HAS_TORCH:
            raise ImportError("torch/transformers required for RoBERTaEvidenceScorer")

        # Prefer CPU to avoid VRAM contention with the LLM — inference is fast enough.
        self.device = device or ("cuda" if torch.cuda.is_available() and _has_free_vram() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length

        self.classifiers: dict[str, dict[str, Any]] = {}
        for dim, ckpt in checkpoint_dirs.items():
            if not Path(ckpt).exists():
                logger.warning(f"RoBERTa checkpoint missing for {dim}: {ckpt} (skipping)")
                continue
            logger.info(f"Loading RoBERTa[{dim}] from {ckpt} on {self.device}")
            tok = AutoTokenizer.from_pretrained(ckpt)
            model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(self.device)
            model.eval()
            lmap_path = Path(ckpt) / "label_map.json"
            if lmap_path.exists():
                id2label = json.load(open(lmap_path))["id2label"]
                id2label = {int(k): v for k, v in id2label.items()}
            else:
                id2label = {i: l for i, l in model.config.id2label.items()}
            self.classifiers[dim] = {"tokenizer": tok, "model": model, "id2label": id2label}

        if not self.classifiers:
            raise RuntimeError("No RoBERTa classifiers were loaded")
        logger.info(f"RoBERTaEvidenceScorer loaded dims: {list(self.classifiers.keys())}")

    def _predict_batch(self, texts: list[str], dim: str) -> list[tuple[str, float]]:
        """Return (label, confidence) for each text using classifier for `dim`."""
        clf = self.classifiers[dim]
        tok, model, id2label = clf["tokenizer"], clf["model"], clf["id2label"]
        out = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = tok(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for p in probs:
                idx = int(p.argmax())
                out.append((id2label[idx], float(p[idx])))
        return out

    def score_sentences(self, sentences: list[str]) -> list[ScoredSentence]:
        """Score each sentence by max classifier confidence across all dims."""
        if not sentences:
            return []

        # Filter too-short sentences (keep index mapping)
        valid_idxs = [
            i for i, s in enumerate(sentences) if len(s.split()) >= self.MIN_SENTENCE_TOKENS
        ]
        valid_sents = [sentences[i] for i in valid_idxs]
        if not valid_sents:
            # All too short — return all with zero score rather than dropping
            return [ScoredSentence(text=s, sentence_idx=i, score=0.0) for i, s in enumerate(sentences)]

        # Run each classifier on the valid sentences
        per_dim_preds: dict[str, list[tuple[str, float]]] = {}
        for dim in self.classifiers:
            per_dim_preds[dim] = self._predict_batch(valid_sents, dim)

        # Build ScoredSentence per valid_sent
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

        # Include short sentences at the end with zero score so top-k still finds something if needed
        scored_all = list(scored_valid)
        filtered_out_idxs = set(range(len(sentences))) - set(valid_idxs)
        for i in filtered_out_idxs:
            scored_all.append(ScoredSentence(text=sentences[i], sentence_idx=i, score=0.0))
        return scored_all

    def predict_doc_level(self, text: str) -> dict[str, tuple[str, float]]:
        """Predict per-dim label + confidence for the full document."""
        preds = {}
        for dim in self.classifiers:
            label, conf = self._predict_batch([text], dim)[0]
            preds[dim] = (label, round(conf, 3))
        return preds


def _has_free_vram(threshold_gb: float = 1.0) -> bool:
    """Check if GPU has at least threshold_gb free (heuristic)."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return False
    try:
        free, _ = torch.cuda.mem_get_info(0)
        return free / (1024**3) >= threshold_gb
    except Exception:
        return False


def default_mbti_checkpoints(models_dir: str = "outputs/models") -> dict[str, str]:
    return {
        "IE": f"{models_dir}/roberta_mbti_IE",
        "SN": f"{models_dir}/roberta_mbti_SN",
        "TF": f"{models_dir}/roberta_mbti_TF",
        "JP": f"{models_dir}/roberta_mbti_JP",
    }


def default_ocean_checkpoints(models_dir: str = "outputs/models", dataset: str = "essays") -> dict[str, str]:
    # Prefer dataset-specific; fall back to essays for any missing dim.
    dims = ["O", "C", "E", "A", "N"]
    ckpts = {}
    for d in dims:
        primary = Path(f"{models_dir}/roberta_{dataset}_{d}")
        fallback = Path(f"{models_dir}/roberta_essays_{d}")
        ckpts[d] = str(primary if primary.exists() else fallback)
    return ckpts
