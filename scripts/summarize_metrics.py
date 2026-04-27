"""Aggregate per-trait metrics from prediction JSONL files into one summary table.

Reads outputs/predictions/*.jsonl, computes accuracy / macro-F1 / per-class F1
for binary classification predictions, and writes:
- outputs/reports/metrics_summary.json
- outputs/reports/metrics_summary.md
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "outputs" / "predictions"
REPORT_DIR = ROOT / "outputs" / "reports"

# Map dataset prefix to (dataset, trait) parsed from filename.
TRAITS = {"A", "C", "E", "N", "O", "IE", "SN", "TF", "JP"}

# Tier classification by filename prefix.
TIER_PATTERNS = [
    ("tier1_lr", re.compile(r"^logistic_regression_(\w+?)_([A-Z]+)\.jsonl$")),
    ("tier2a_roberta", re.compile(r"^tier2a_roberta_(\w+?)_([A-Z]+)\.jsonl$")),
    ("tier2a_weighted", re.compile(r"^tier2a_weighted_roberta_(\w+?)_([A-Z]+)_weighted\.jsonl$")),
    ("tier2b_mlp", re.compile(r"^tier2b_roberta_mlp_(\w+?)_([A-Z]+)\.jsonl$")),
    ("tier3_zeroshot", re.compile(r"^tier3_zeroshot_(\w+)\.jsonl$")),
    ("tier4_cope", re.compile(r"^tier4_cope_(no_kb|no_evidence_filter|no_cope|full)_(\w+)\.jsonl$")),
]


def parse_filename(name: str):
    for tier, pat in TIER_PATTERNS:
        m = pat.match(name)
        if not m:
            continue
        if tier in {"tier3_zeroshot"}:
            return tier, m.group(1), None
        if tier == "tier4_cope":
            ablation = m.group(1)
            dataset = m.group(2)
            return f"{tier}_{ablation}", dataset, None
        return tier, m.group(1), m.group(2)
    return None, None, None


def load_predictions(path: Path):
    """Yield (gold, pred) pairs. Per-trait files have scalar labels;
    multi-trait (Qwen) files have dict-of-traits → return dict per record."""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield rec


def metrics_for_binary(rows):
    gold = [r["gold_label"] for r in rows]
    pred = [r.get("predicted_label") for r in rows]
    valid = [(g, p) for g, p in zip(gold, pred) if p is not None]
    if not valid:
        return None
    g, p = zip(*valid)
    return {
        "n": len(valid),
        "accuracy": round(accuracy_score(g, p), 4),
        "macro_f1": round(f1_score(g, p, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(g, p, average="weighted", zero_division=0), 4),
        "precision_macro": round(precision_score(g, p, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(g, p, average="macro", zero_division=0), 4),
    }


TRAIT_STRING_RE = re.compile(r"([A-Z]+):(HIGH|LOW|UNKNOWN)")


def parse_trait_string(s):
    """Parse 'O:HIGH,C:LOW,E:HIGH,A:LOW,N:HIGH' into {'O': 'HIGH', ...}."""
    if not isinstance(s, str):
        return {}
    return {m.group(1): m.group(2) for m in TRAIT_STRING_RE.finditer(s)}


def metrics_for_multitrait(rows, traits):
    """For Qwen-style records with predicted dict {trait: HIGH/LOW}."""
    out = {}
    for trait in traits:
        gold, pred = [], []
        for r in rows:
            g_raw = r.get("gold_labels") or r.get("gold") or r.get("gold_label")
            p_raw = r.get("predicted_labels") or r.get("predicted") or r.get("predicted_label")
            g_dict = g_raw if isinstance(g_raw, dict) else parse_trait_string(g_raw)
            p_dict = p_raw if isinstance(p_raw, dict) else parse_trait_string(p_raw)
            g = g_dict.get(trait)
            p = p_dict.get(trait)
            if g is None or p is None:
                continue
            gold.append(g)
            pred.append(p)
        if not gold:
            continue
        out[trait] = {
            "n": len(gold),
            "accuracy": round(accuracy_score(gold, pred), 4),
            "macro_f1": round(f1_score(gold, pred, average="macro", zero_division=0), 4),
        }
    return out


def detect_traits(sample):
    """Pull trait keys from a Qwen-style record."""
    for key in ("predicted_labels", "predicted", "gold_labels", "gold"):
        if key in sample and isinstance(sample[key], dict):
            return list(sample[key].keys())
    for key in ("gold_label", "predicted_label"):
        s = sample.get(key)
        if isinstance(s, str):
            d = parse_trait_string(s)
            if d:
                return list(d.keys())
    return []


def main():
    summary = defaultdict(lambda: defaultdict(dict))  # tier -> dataset -> trait -> metrics
    file_count = 0
    skipped = []

    for jsonl in sorted(PRED_DIR.glob("*.jsonl")):
        tier, dataset, trait = parse_filename(jsonl.name)
        if tier is None:
            skipped.append(jsonl.name)
            continue
        rows = list(load_predictions(jsonl))
        if not rows:
            skipped.append(f"{jsonl.name} (empty)")
            continue
        if trait is not None:
            m = metrics_for_binary(rows)
            if m is None:
                skipped.append(f"{jsonl.name} (no valid preds)")
                continue
            summary[tier][dataset][trait] = m
        else:
            traits = detect_traits(rows[0])
            if not traits:
                m = metrics_for_binary(rows)
                if m is not None:
                    summary[tier][dataset]["__all__"] = m
                else:
                    skipped.append(f"{jsonl.name} (unknown schema)")
                continue
            per_trait = metrics_for_multitrait(rows, traits)
            for t, m in per_trait.items():
                summary[tier][dataset][t] = m
        file_count += 1

    # Write JSON.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORT_DIR / "metrics_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

    # Write Markdown.
    lines = ["# Metrics Summary", "", f"Files aggregated: {file_count}", ""]
    if skipped:
        lines.append("## Skipped files")
        lines += [f"- {s}" for s in skipped]
        lines.append("")
    for tier in sorted(summary):
        lines.append(f"## {tier}")
        lines.append("")
        lines.append("| Dataset | Trait | N | Accuracy | Macro-F1 | Weighted-F1 |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for dataset in sorted(summary[tier]):
            for trait in sorted(summary[tier][dataset]):
                m = summary[tier][dataset][trait]
                lines.append(
                    f"| {dataset} | {trait} | {m.get('n', '-')} | "
                    f"{m.get('accuracy', '-')} | {m.get('macro_f1', '-')} | "
                    f"{m.get('weighted_f1', '-')} |"
                )
        # Mean per dataset.
        for dataset in sorted(summary[tier]):
            f1s = [m["macro_f1"] for m in summary[tier][dataset].values() if "macro_f1" in m]
            accs = [m["accuracy"] for m in summary[tier][dataset].values() if "accuracy" in m]
            if f1s:
                lines.append(
                    f"| {dataset} | **MEAN** | - | "
                    f"{round(sum(accs) / len(accs), 4)} | {round(sum(f1s) / len(f1s), 4)} | - |"
                )
        lines.append("")
    out_md = REPORT_DIR / "metrics_summary.md"
    out_md.write_text("\n".join(lines))

    print(f"Wrote {out_json} and {out_md}")
    print(f"Tiers: {sorted(summary)}")


if __name__ == "__main__":
    main()
