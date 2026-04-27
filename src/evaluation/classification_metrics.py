"""Classification metrics: accuracy, F1, per-class, Cohen's Kappa."""

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score)


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict:
    """
    Compute standard classification metrics.

    Returns:
        dict with accuracy, f1_macro, f1_weighted, kappa,
              per_class (classification_report dict), confusion_matrix
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def compute_dimension_metrics(
    y_true_dims: dict[str, list[str]],
    y_pred_dims: dict[str, list[str]],
) -> dict:
    """
    Compute per-dimension metrics for MBTI 4-dim classification.

    Args:
        y_true_dims / y_pred_dims: {dimension → list of labels}
    """
    results = {}
    for dim in ["IE", "SN", "TF", "JP"]:
        if dim not in y_true_dims or dim not in y_pred_dims:
            continue
        y_true = y_true_dims[dim]
        y_pred = y_pred_dims[dim]
        results[dim] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }
    if results:
        results["avg_accuracy"] = np.mean([v["accuracy"] for v in results.values()])
    return results


def compute_ocean_metrics(
    y_true_ocean: dict[str, list[str]],
    y_pred_ocean: dict[str, list[str]],
) -> dict:
    """
    Compute per-trait metrics for Big Five binary classification.

    Args:
        y_true_ocean / y_pred_ocean: {trait → list of HIGH/LOW labels}
    """
    results = {}
    for trait in ["O", "C", "E", "A", "N"]:
        if trait not in y_true_ocean or trait not in y_pred_ocean:
            continue
        y_true = y_true_ocean[trait]
        y_pred = y_pred_ocean[trait]
        results[trait] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }
    if results:
        results["avg_accuracy"] = np.mean([v["accuracy"] for v in results.values()])
    return results
