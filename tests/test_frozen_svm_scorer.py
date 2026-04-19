"""Unit tests for FrozenSvmEvidenceScorer.

Uses the real MBTI checkpoints from outputs/models/frozen_bert_svm_mbti_*/model.pkl.
All 4 tests run in < 30 s on a warm HF cache because the SVM is trivially fast
and the encoder is shared across dims.
"""
from __future__ import annotations

import pytest

CHECKPOINT_ROOT = "outputs/models"
DIMS = ["IE", "SN", "TF", "JP"]


def _ckpt_paths() -> dict[str, str]:
    return {dim: f"{CHECKPOINT_ROOT}/frozen_bert_svm_mbti_{dim}/model.pkl" for dim in DIMS}


def _skip_if_missing():
    """Skip if any MBTI checkpoint is absent (e.g. CI without model files)."""
    from pathlib import Path
    missing = [p for p in _ckpt_paths().values() if not Path(p).exists()]
    return pytest.mark.skipif(bool(missing), reason=f"Missing checkpoints: {missing}")


# ---------------------------------------------------------------------------
# Test 1: Load
# ---------------------------------------------------------------------------


@_skip_if_missing()
def test_frozen_svm_scorer_loads():
    from src.retrieval.frozen_svm_scorer import FrozenSvmEvidenceScorer
    scorer = FrozenSvmEvidenceScorer(_ckpt_paths())
    assert set(scorer.baselines.keys()) == set(DIMS)
    # All dims share the same encoder object
    encoders = [b.encoder for b in scorer.baselines.values()]
    assert all(e is encoders[0] for e in encoders), "Encoder not shared across dims"


# ---------------------------------------------------------------------------
# Test 2: score_sentences shape + confidence bounds
# ---------------------------------------------------------------------------


@_skip_if_missing()
def test_score_sentences_shape_and_bounds():
    from src.retrieval.frozen_svm_scorer import FrozenSvmEvidenceScorer
    scorer = FrozenSvmEvidenceScorer(_ckpt_paths())

    sentences = [
        "I love spending time alone reading books at home.",
        "She enjoys large social gatherings and meeting new people.",
        "He thinks carefully before making any decision.",
        "They tend to follow schedules and structured plans.",
        "Hi",  # short — below MIN_SENTENCE_TOKENS, score should be 0.0
    ]
    results = scorer.score_sentences(sentences)

    assert len(results) == len(sentences)
    for r in results:
        assert 0.0 <= r.score <= 1.0, f"score out of range: {r.score}"
        if len(r.text.split()) >= scorer.MIN_SENTENCE_TOKENS:
            # Non-trivial sentences: confidence must be in [0.5, 1.0]
            assert r.score >= 0.5, f"SVM confidence below 0.5 for '{r.text}': {r.score}"
        for dim_label, conf in r.predicted_labels.items():
            assert 0.5 <= conf[1] <= 1.0, f"Per-dim confidence out of [0.5, 1.0]: dim={dim_label}, conf={conf}"


# ---------------------------------------------------------------------------
# Test 3: predict_doc_level label set
# ---------------------------------------------------------------------------


@_skip_if_missing()
def test_predict_doc_level_matches_label_set():
    from src.retrieval.frozen_svm_scorer import FrozenSvmEvidenceScorer
    scorer = FrozenSvmEvidenceScorer(_ckpt_paths())

    doc = (
        "I usually prefer quiet evenings at home with a good book. "
        "I tend to think things through carefully before acting. "
        "I like to have a clear plan for the week ahead."
    )
    prior = scorer.predict_doc_level(doc)

    valid_labels = {"IE": {"I", "E"}, "SN": {"S", "N"}, "TF": {"T", "F"}, "JP": {"J", "P"}}
    assert set(prior.keys()) == set(DIMS)
    for dim, (label, conf) in prior.items():
        assert label in valid_labels[dim], f"dim={dim} returned invalid label '{label}'"
        assert 0.5 <= conf <= 1.0, f"dim={dim} confidence out of [0.5, 1.0]: {conf}"


# ---------------------------------------------------------------------------
# Test 4: Balanced predictions on balanced introverted vs extraverted input
# ---------------------------------------------------------------------------


@_skip_if_missing()
def test_balanced_predictions_on_balanced_input():
    """Regression guard against majority-class collapse.

    Pass 5 clearly-introverted sentences + 5 clearly-extraverted sentences.
    The IE classifier should predict at least 2 of each label — if it predicts
    all 'I' or all 'E' the balanced training has failed.
    """
    from src.retrieval.frozen_svm_scorer import FrozenSvmEvidenceScorer
    scorer = FrozenSvmEvidenceScorer(_ckpt_paths())

    introverted = [
        "I prefer to read quietly at home alone in the evenings.",
        "Solitude and peace recharge me after a long day at work.",
        "I enjoy deep one-on-one conversations rather than big parties.",
        "Crowds and loud social events drain my energy completely.",
        "I spend my weekends alone with a book and some music.",
    ]
    extraverted = [
        "I love going to big social parties every weekend and meeting many strangers.",
        "Large group events always energise me and bring me joy.",
        "I thrive at networking events full of lively people and new conversations.",
        "Being the centre of attention at a concert crowd is exhilarating every time.",
        "I invite many colleagues over for dinner parties most Saturday evenings.",
    ]

    all_sentences = introverted + extraverted
    results = scorer.score_sentences(all_sentences)
    ie_preds = [r.predicted_labels.get("IE", ("?", 0.0))[0] for r in results if r.predicted_labels]

    n_I = sum(1 for lbl in ie_preds if lbl == "I")
    n_E = sum(1 for lbl in ie_preds if lbl == "E")
    assert n_I >= 2, f"Only {n_I} 'I' predictions — possible majority collapse (preds: {ie_preds})"
    assert n_E >= 2, f"Only {n_E} 'E' predictions — possible majority collapse (preds: {ie_preds})"
