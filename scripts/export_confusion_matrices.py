#!/usr/bin/env python
"""Compute confusion matrices for every prediction JSONL under outputs/predictions/.

Reads each `*.jsonl` (gold_label + predicted_label per record), builds a
multi-class confusion matrix comparing all observed classes, and writes both
JSON and Markdown sibling files under outputs/reports/confusion_matrices/.

Use this for Tier 2-5 runs that finished before inline CM logging was wired
into train_baseline.py — no re-run required.

Usage:
    uv run --no-project --python 3.12 --with-requirements requirements.txt \
        python scripts/export_confusion_matrices.py

    # Filter by prefix:
    python scripts/export_confusion_matrices.py --pattern 'tier3_*.jsonl'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

matplotlib.use("Agg")


def load_jsonl(path: Path) -> tuple[list[str], list[str]]:
    gold: list[str] = []
    pred: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            g = rec.get("gold_label")
            p = rec.get("predicted_label")
            if g is None or p is None:
                continue
            gold.append(str(g))
            pred.append(str(p))
    return gold, pred


def write_outputs(stem: str, classes: list[str], matrix: list[list[int]], meta: dict, out_dir: Path) -> None:
    payload = {**meta, "classes": classes, "matrix": matrix, "support": [int(sum(row)) for row in matrix]}
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    md_lines = [
        f"# Confusion matrix — {stem}",
        "",
        "Rows = gold class · Columns = predicted class",
        "",
        "| gold \\\\ pred | " + " | ".join(classes) + " | support |",
        "|" + "---|" * (len(classes) + 2),
    ]
    for i, cls in enumerate(classes):
        row_vals = [str(matrix[i][j]) for j in range(len(classes))]
        md_lines.append(f"| **{cls}** | " + " | ".join(row_vals) + f" | {int(sum(matrix[i]))} |")
    (out_dir / f"{stem}.md").write_text("\n".join(md_lines) + "\n")

    fig_size = max(4, len(classes) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ConfusionMatrixDisplay(np.array(matrix), display_labels=classes).plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )
    ax.set_title(stem)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def process_file(jsonl_path: Path, out_dir: Path) -> None:
    gold, pred = load_jsonl(jsonl_path)
    if not gold:
        print(f"[skip] empty: {jsonl_path.name}")
        return
    classes = sorted(set(gold) | set(pred))
    cm = sk_confusion_matrix(gold, pred, labels=classes).tolist()
    stem = jsonl_path.stem
    meta = {"source": str(jsonl_path), "n_records": len(gold)}
    write_outputs(stem, classes, cm, meta, out_dir)
    print(f"[ok] {jsonl_path.name}: classes={classes}, n={len(gold)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="outputs/predictions")
    ap.add_argument("--out_dir", default="outputs/reports/confusion_matrices")
    ap.add_argument("--pattern", default="*.jsonl")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(pred_dir.glob(args.pattern))
    if not files:
        print(f"No files matching {args.pattern} in {pred_dir}")
        return
    for f in files:
        process_file(f, out_dir)
    print(f"\nWrote {len(files)} confusion matrices to {out_dir}/")


if __name__ == "__main__":
    main()
