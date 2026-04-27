"""Unit tests for frozen-encoder transformer baselines.

Covers FrozenTransformerEncoder + FrozenBertSvmBaseline + RobertaMlpBaseline.
Uses `prajjwal1/bert-tiny` — a 4MB model — to keep tests fast. No network
access needed after the first run (HF cache).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

TINY_MODEL = "sshleifer/tiny-distilroberta-base"  # 2MB RoBERTa — ships full tokenizer


@pytest.fixture(scope="module")
def tiny_encoder_cfg(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("emb_cache")
    return {
        "encoder": {
            "model_name": TINY_MODEL,
            "pooling": "cls",
            "chunk_size": 64,
            "stride": 32,
            "batch_size": 4,
            "cache_dir": str(tmpdir),
        }
    }


@pytest.fixture(scope="module")
def tiny_encoder_cfg_mean4(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("emb_cache_mean4")
    return {
        "encoder": {
            "model_name": TINY_MODEL,
            "pooling": "mean_last4",
            "chunk_size": 64,
            "stride": 32,
            "batch_size": 4,
            "cache_dir": str(tmpdir),
        }
    }


def _synthetic_binary_dataset(n: int = 50):
    """Deterministic 2-class synthetic data. Class `I` = short intro-style phrases,
    `E` = longer extraverted phrases. Enough signal for a tiny model to separate."""
    rng = np.random.RandomState(0)
    i_templates = [
        "I prefer to read quietly at home.",
        "Solitude recharges me after a long day.",
        "I enjoy deep one-on-one conversations.",
        "Crowds and parties drain my energy.",
        "I spend weekends alone with a book.",
    ]
    e_templates = [
        "I love going to big social parties every weekend and meeting strangers.",
        "Large group events always energise me and bring me joy across many friends.",
        "I thrive at networking events full of lively people and new conversations.",
        "Being the centre of attention at a concert crowd is exhilarating every time.",
        "I invite many colleagues over for dinner parties most Saturday evenings.",
    ]
    texts, labels = [], []
    for i in range(n):
        if i % 2 == 0:
            texts.append(rng.choice(i_templates))
            labels.append("I")
        else:
            texts.append(rng.choice(e_templates))
            labels.append("E")
    return texts, labels


# ---------------------------------------------------------------------------
# FrozenTransformerEncoder
# ---------------------------------------------------------------------------


def test_frozen_encoder_shape(tiny_encoder_cfg):
    from src.baselines.frozen_transformer_baselines import \
        FrozenTransformerEncoder

    enc = FrozenTransformerEncoder(tiny_encoder_cfg["encoder"])
    out = enc.encode(["hello world", "another short text", "third one here"])
    assert out.shape == (3, enc.hidden_dim)
    assert out.dtype == np.float32


def test_embedding_cache(tiny_encoder_cfg):
    from src.baselines.frozen_transformer_baselines import \
        FrozenTransformerEncoder

    enc = FrozenTransformerEncoder(tiny_encoder_cfg["encoder"])
    texts = ["cache this", "cache me too", "and me"]
    out1 = enc.encode(texts, cache_key="unit_test_split")

    # Second call should load from cache and return identical embeddings
    out2 = enc.encode(texts, cache_key="unit_test_split")
    np.testing.assert_array_equal(out1, out2)

    # Sanity: cache file exists where we expected
    expected_path = Path(tiny_encoder_cfg["encoder"]["cache_dir"])
    expected_path = expected_path / TINY_MODEL.replace("/", "__") / "unit_test_split.npy"
    assert expected_path.exists()


# ---------------------------------------------------------------------------
# FrozenBertSvmBaseline
# ---------------------------------------------------------------------------


def test_frozen_bert_svm_fit_predict(tiny_encoder_cfg_mean4):
    from src.baselines.frozen_transformer_baselines import \
        FrozenBertSvmBaseline

    texts, labels = _synthetic_binary_dataset(n=30)

    cfg = {**tiny_encoder_cfg_mean4, "classifier": {"n_estimators": 3, "max_samples": 0.8, "svc": {"C": 1.0}}}
    b = FrozenBertSvmBaseline(config=cfg)
    b.fit(texts, labels)
    preds = b.predict(texts)
    assert set(preds).issubset({"I", "E"})
    # Tiny 2-dim model — just assert the pipeline produces valid output (≥ chance).
    # Real-model accuracy is tested in integration, not unit tests.
    acc = float((np.array(preds) == np.array(labels)).mean())
    assert acc >= 0.4, f"Predictions degenerate: {acc}"

    # Save/load round-trip
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/model.pkl"
        b.save(path)
        b2 = FrozenBertSvmBaseline.load(path)
        preds2 = b2.predict(texts)
        np.testing.assert_array_equal(preds, preds2)


# ---------------------------------------------------------------------------
# RobertaMlpBaseline
# ---------------------------------------------------------------------------


def test_roberta_mlp_fit_predict(tiny_encoder_cfg):
    from src.baselines.frozen_transformer_baselines import RobertaMlpBaseline

    texts, labels = _synthetic_binary_dataset(n=30)
    train_texts, train_labels = texts[:20], labels[:20]
    val_texts, val_labels = texts[20:], labels[20:]

    cfg = {
        **tiny_encoder_cfg,
        "head": {"hidden_dim": 16, "dropout": 0.2},
        "training": {
            "learning_rate": 5e-3,
            "weight_decay": 0.01,
            "batch_size": 8,
            "num_epochs": 10,
            "early_stopping_patience": 3,
            "loss_weighting": "balanced",
        },
    }
    m = RobertaMlpBaseline(config=cfg)
    m.fit(train_texts, train_labels, val_texts=val_texts, val_labels=val_labels)
    preds = m.predict(val_texts)
    assert set(preds).issubset({"I", "E"})
    acc = float((np.array(preds) == np.array(val_labels)).mean())
    assert acc >= 0.5, f"Val accuracy too low: {acc}"

    # Save/load round-trip
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/model.pt"
        m.save(path)
        m2 = RobertaMlpBaseline.load(path)
        preds2 = m2.predict(val_texts)
        np.testing.assert_array_equal(preds, preds2)
