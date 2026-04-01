"""Evaluation modules."""
from .classification_metrics import compute_classification_metrics
from .xai_metrics import evidence_grounding_score, faithfulness_score, explanation_consistency
from .statistical_tests import mcnemar_test, bootstrap_confidence_interval, paired_bootstrap_test

__all__ = [
    "compute_classification_metrics",
    "evidence_grounding_score",
    "faithfulness_score",
    "explanation_consistency",
    "mcnemar_test",
    "bootstrap_confidence_interval",
    "paired_bootstrap_test",
]
