"""Evaluation modules."""

from .classification_metrics import compute_classification_metrics
from .statistical_tests import (bootstrap_confidence_interval, mcnemar_test,
                                paired_bootstrap_test)
from .xai_metrics import (evidence_grounding_score, explanation_consistency,
                          faithfulness_score)

__all__ = [
    "compute_classification_metrics",
    "evidence_grounding_score",
    "faithfulness_score",
    "explanation_consistency",
    "mcnemar_test",
    "bootstrap_confidence_interval",
    "paired_bootstrap_test",
]
