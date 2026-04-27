"""Tests for evaluation modules."""

import numpy as np

from src.evaluation.classification_metrics import \
    compute_classification_metrics
from src.evaluation.statistical_tests import (bootstrap_confidence_interval,
                                              paired_bootstrap_test)
from src.evaluation.xai_metrics import (evidence_grounding_score,
                                        evidence_relevance_f1, fuzzy_match)


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = ["INTP", "INFJ", "ENFP"]
        y_pred = ["INTP", "INFJ", "ENFP"]
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0
        assert metrics["f1_weighted"] == 1.0

    def test_zero_predictions(self):
        y_true = ["INTP", "INFJ", "ENFP"]
        y_pred = ["INTJ", "INTP", "ISTJ"]
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

    def test_returns_required_keys(self):
        y_true = ["INTP", "INFJ"]
        y_pred = ["INTP", "INTP"]
        metrics = compute_classification_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "f1_weighted" in metrics
        assert "kappa" in metrics
        assert "per_class" in metrics
        assert "confusion_matrix" in metrics


class TestXAIMetrics:
    def test_fuzzy_match_exact(self):
        assert fuzzy_match("hello world", "hello world and more", threshold=0.85)

    def test_fuzzy_match_no_match(self):
        assert not fuzzy_match("completely different", "hello world something", threshold=0.85)

    def test_evidence_grounding_perfect(self):
        predictions = [
            {
                "text": "I love thinking about abstract ideas and theories.",
                "evidence_chain": [{"evidence": "I love thinking about abstract ideas"}],
            }
        ]
        score = evidence_grounding_score(predictions)
        assert score > 0.0

    def test_evidence_grounding_empty(self):
        score = evidence_grounding_score([])
        assert score == 0.0

    def test_evidence_relevance_f1_perfect(self):
        pred = ["I love thinking alone"]
        gold = ["I love thinking alone"]
        score = evidence_relevance_f1(pred, gold)
        assert score == 1.0

    def test_evidence_relevance_f1_empty(self):
        score = evidence_relevance_f1([], [])
        assert score == 0.0


class TestStatisticalTests:
    def test_bootstrap_ci_returns_valid_dict(self):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
        from sklearn.metrics import accuracy_score

        ci = bootstrap_confidence_interval(y_true, y_pred, lambda a, b: accuracy_score(a, b), n_bootstrap=100)
        assert "mean" in ci
        assert "ci_lower" in ci
        assert "ci_upper" in ci
        assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]

    def test_paired_bootstrap_equal_methods(self):
        y_true = np.array(["INTP"] * 50 + ["INFJ"] * 50)
        pred_a = np.copy(y_true)
        pred_b = np.copy(y_true)
        result = paired_bootstrap_test(y_true, pred_a, pred_b, lambda a, b: (a == b).mean(), n_bootstrap=100)
        # When methods are identical, p-value should be high (not significant)
        assert "p_value" in result
        assert "delta" in result
        assert result["delta"] == 0.0
