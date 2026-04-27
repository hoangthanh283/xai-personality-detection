"""Shared classification diagnostics for local reports and W&B tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)

matplotlib.use("Agg")

REPORT_COLUMNS = ["class", "precision", "recall", "f1-score", "support"]


def _labels(y_true: list[Any], y_pred: list[Any], labels: list[str] | None = None) -> list[str]:
    if labels is not None:
        return [str(label) for label in labels]
    return sorted({str(v) for v in y_true} | {str(v) for v in y_pred})


def _safe_support(report: dict[str, Any]) -> int:
    weighted = report.get("weighted avg")
    if isinstance(weighted, dict):
        return int(weighted.get("support", 0))
    return 0


def build_classification_diagnostics(
    y_true: list[Any],
    y_pred: list[Any],
    *,
    labels: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build scalar metrics plus table-friendly report and confusion matrix data."""
    y_true_str = [str(v) for v in y_true]
    y_pred_str = [str(v) for v in y_pred]
    classes = _labels(y_true_str, y_pred_str, labels)

    report = classification_report(
        y_true_str,
        y_pred_str,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true_str, y_pred_str, labels=classes)
    support = [int(sum(row)) for row in matrix.tolist()]

    report_rows: list[list[Any]] = []
    for cls in classes:
        row = report.get(cls, {})
        report_rows.append(
            [
                cls,
                float(row.get("precision", 0.0)),
                float(row.get("recall", 0.0)),
                float(row.get("f1-score", 0.0)),
                int(row.get("support", 0)),
            ]
        )
    total_support = _safe_support(report)
    if "accuracy" in report:
        report_rows.append(["accuracy", None, None, float(report["accuracy"]), total_support])
    for avg_key in ("macro avg", "weighted avg"):
        row = report.get(avg_key)
        if isinstance(row, dict):
            report_rows.append(
                [
                    avg_key,
                    float(row.get("precision", 0.0)),
                    float(row.get("recall", 0.0)),
                    float(row.get("f1-score", 0.0)),
                    int(row.get("support", 0)),
                ]
            )

    cm_columns = ["actual \\ predicted", *classes, "support"]
    cm_rows = [[cls, *[int(v) for v in matrix[i].tolist()], support[i]] for i, cls in enumerate(classes)]

    scalar_metrics = {
        "accuracy": float(accuracy_score(y_true_str, y_pred_str)),
        "f1_macro": float(f1_score(y_true_str, y_pred_str, labels=classes, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true_str, y_pred_str, labels=classes, average="weighted", zero_division=0)),
        "precision_macro": float(
            precision_score(y_true_str, y_pred_str, labels=classes, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_true_str, y_pred_str, labels=classes, average="weighted", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true_str, y_pred_str, labels=classes, average="macro", zero_division=0)),
        "recall_weighted": float(
            recall_score(y_true_str, y_pred_str, labels=classes, average="weighted", zero_division=0)
        ),
    }

    return {
        "metadata": metadata or {},
        "classes": classes,
        "scalar_metrics": scalar_metrics,
        "classification_report": report,
        "report_columns": REPORT_COLUMNS[:],
        "report_rows": report_rows,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_columns": cm_columns,
        "confusion_matrix_rows": cm_rows,
        "support": support,
        "n_samples": len(y_true_str),
    }


def render_confusion_matrix_png(
    diagnostics: dict[str, Any],
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path | None:
    """Render a confusion matrix PNG."""
    classes = diagnostics["classes"]
    matrix = np.array(diagnostics["confusion_matrix"])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig_size = max(4, len(classes) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ConfusionMatrixDisplay(matrix, display_labels=classes).plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )
    if title:
        ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output


def diagnostics_to_markdown(diagnostics: dict[str, Any], *, title: str) -> str:
    """Render classification report and confusion matrix tables as Markdown."""
    lines = [f"# {title}", ""]
    lines.extend(
        [
            "## Classification report",
            "",
            "| " + " | ".join(REPORT_COLUMNS) + " |",
            "|" + "---|" * len(REPORT_COLUMNS),
        ]
    )
    for row in diagnostics["report_rows"]:
        formatted = [
            str(row[0]),
            "" if row[1] is None else f"{float(row[1]):.4f}",
            "" if row[2] is None else f"{float(row[2]):.4f}",
            "" if row[3] is None else f"{float(row[3]):.4f}",
            str(row[4]),
        ]
        lines.append("| " + " | ".join(formatted) + " |")

    cm_columns = diagnostics["confusion_matrix_columns"]
    lines.extend(
        [
            "",
            "## Confusion matrix",
            "",
            "Rows = gold class. Columns = predicted class.",
            "",
            "| " + " | ".join(cm_columns) + " |",
            "|" + "---|" * len(cm_columns),
        ]
    )
    for row in diagnostics["confusion_matrix_rows"]:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def persist_classification_diagnostics(
    diagnostics: dict[str, Any],
    output_dir: str | Path,
    stem: str,
    *,
    title: str | None = None,
    render_png: bool = True,
) -> dict[str, str]:
    """Write JSON, Markdown, and optionally PNG diagnostics to disk."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    title = title or stem

    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    md_path.write_text(diagnostics_to_markdown(diagnostics, title=title))

    paths = {"json": str(json_path), "markdown": str(md_path)}
    if render_png:
        png_path = render_confusion_matrix_png(diagnostics, out_dir / f"{stem}.png", title=title)
        if png_path is not None:
            paths["png"] = str(png_path)
    return paths


def load_prediction_jsonl(path: str | Path) -> tuple[list[str], list[str], list[str]]:
    """Load ids, gold labels, and predicted labels from a prediction JSONL file."""
    ids: list[str] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            gold = record.get("gold_label")
            pred = record.get("predicted_label")
            if gold is None or pred is None:
                continue
            ids.append(str(record.get("id", idx)))
            y_true.append(str(gold))
            y_pred.append(str(pred))
    return ids, y_true, y_pred
