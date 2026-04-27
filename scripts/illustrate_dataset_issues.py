"""Extract illustrative samples + generate visualization charts for each dataset issue.

For each of 4 datasets, finds 2-3 samples that vividly demonstrate the problems
identified in the previous analysis (imbalance, leakage, length, evidence sparsity, etc.).

Also produces matplotlib charts saved to outputs/analysis/charts/.

Outputs:
  - outputs/analysis/illustrative_samples.md  (samples + chart references)
  - outputs/analysis/charts/*.png             (visualization charts)
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_ROOT = Path("data/processed")
CHARTS_DIR = Path("outputs/analysis/charts")
SAMPLES_OUT = Path("outputs/analysis/illustrative_samples.md")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

OCEAN = ["O", "C", "E", "A", "N"]
MBTI_TYPES = [
    "INFJ",
    "INFP",
    "INTJ",
    "INTP",
    "ISFJ",
    "ISFP",
    "ISTJ",
    "ISTP",
    "ENFJ",
    "ENFP",
    "ENTJ",
    "ENTP",
    "ESFJ",
    "ESFP",
    "ESTJ",
    "ESTP",
]
MBTI_RE = re.compile(r"\b(" + "|".join(MBTI_TYPES) + r")(s|es)?\b", re.IGNORECASE)
TAXONOMY_RE = re.compile(
    r"\b(mbti|myers[- ]?briggs|introvert(ed)?|introversion|extravert(ed)?|extrovert(ed)?|"
    r"extraversion|extroversion|personality type|personality types)\b",
    re.IGNORECASE,
)
OCEAN_TAX_RE = re.compile(
    r"\b(openness|conscientious(ness)?|neurotic(ism)?|agreeable(ness)?|extraver(t|sion))\b",
    re.IGNORECASE,
)


def stream(path: Path, cap: int | None = None) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open() as f:
        for i, line in enumerate(f):
            if cap is not None and i >= cap:
                return
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def excerpt(text: str, n: int = 220) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[:n] + "…"


def write_section(out: list[str], title: str, body: str = "") -> None:
    out.append(title)
    if body:
        out.append("")
        out.append(body)
    out.append("")


# ============================================================
# ILLUSTRATIVE SAMPLE EXTRACTION
# ============================================================


def extract_mbti_samples() -> list[dict]:
    """MBTI: imbalance, taxonomy leakage, truncation."""
    found = {"taxonomy_leak": [], "rare_class": [], "very_long": []}
    rare_classes = {"ESTJ", "ESFJ", "ESFP", "ESTP"}
    for rec in stream(DATA_ROOT / "mbti" / "train.jsonl"):
        text = rec.get("text", "")
        words = text.split()
        # Taxonomy leak (mention "introvert/extravert/personality type")
        if len(found["taxonomy_leak"]) < 2:
            m = TAXONOMY_RE.search(text)
            if m:
                # find context window around match
                start = max(0, m.start() - 100)
                end = min(len(text), m.end() + 100)
                ctx = text[start:end]
                found["taxonomy_leak"].append(
                    {
                        "id": rec["id"],
                        "label": rec.get("label_mbti"),
                        "match": m.group(),
                        "context": excerpt(ctx, 280),
                        "n_words": len(words),
                    }
                )
        # Rare class
        if rec.get("label_mbti") in rare_classes and len(found["rare_class"]) < 2:
            found["rare_class"].append(
                {
                    "id": rec["id"],
                    "label": rec.get("label_mbti"),
                    "n_words": len(words),
                    "excerpt": excerpt(text, 240),
                }
            )
        # Very long (>1700 words → truncation problem)
        if len(words) > 1700 and len(found["very_long"]) < 2:
            found["very_long"].append(
                {
                    "id": rec["id"],
                    "label": rec.get("label_mbti"),
                    "n_words": len(words),
                    "head": excerpt(text[:300], 280),
                    "tail": excerpt(text[-300:], 280),
                }
            )
        if all(len(v) >= 2 for v in found.values()):
            break
    return found


def extract_essays_samples() -> list[dict]:
    """Essays: balanced, but old/narrow domain — show domain quirk."""
    found = {"balanced_examples": [], "extreme_short": [], "extreme_long": []}
    short_min = 9999
    long_max = 0
    for rec in stream(DATA_ROOT / "essays" / "train.jsonl"):
        text = rec.get("text", "")
        wl = len(text.split())
        if len(found["balanced_examples"]) < 3:
            ocean = rec.get("label_ocean", {})
            # find a record with mixed labels (not all HIGH or all LOW)
            highs = sum(1 for v in ocean.values() if v == "HIGH")
            if 1 <= highs <= 4:
                found["balanced_examples"].append(
                    {
                        "id": rec["id"],
                        "ocean": ocean,
                        "n_words": wl,
                        "excerpt": excerpt(text, 280),
                    }
                )
        if wl < short_min:
            short_min = wl
            found["extreme_short"] = [
                {
                    "id": rec["id"],
                    "n_words": wl,
                    "ocean": rec.get("label_ocean"),
                    "excerpt": excerpt(text, 240),
                }
            ]
        if wl > long_max:
            long_max = wl
            found["extreme_long"] = [
                {
                    "id": rec["id"],
                    "n_words": wl,
                    "ocean": rec.get("label_ocean"),
                    "excerpt": excerpt(text[:400], 380),
                }
            ]
    return found


def extract_pandora_samples() -> list[dict]:
    """Pandora: 2000-word cap, taxonomy leakage, missing OCEAN."""
    found = {"capped_2000": [], "taxonomy_leak": [], "missing_ocean": []}
    for rec in stream(DATA_ROOT / "pandora" / "train.jsonl"):
        text = rec.get("text", "")
        wl = len(text.split())
        if wl >= 1990 and len(found["capped_2000"]) < 2:
            found["capped_2000"].append(
                {
                    "id": rec["id"],
                    "n_words": wl,
                    "n_total_comments": rec.get("metadata", {}).get("num_total_comments"),
                    "n_sampled": rec.get("metadata", {}).get("num_comments"),
                    "label_mbti": rec.get("label_mbti"),
                    "label_ocean": rec.get("label_ocean"),
                    "head": excerpt(text[:250], 240),
                    "tail": excerpt(text[-250:], 240),
                }
            )
        if len(found["taxonomy_leak"]) < 2:
            m = TAXONOMY_RE.search(text)
            if m:
                start = max(0, m.start() - 80)
                end = min(len(text), m.end() + 120)
                found["taxonomy_leak"].append(
                    {
                        "id": rec["id"],
                        "label_mbti": rec.get("label_mbti"),
                        "match": m.group(),
                        "context": excerpt(text[start:end], 280),
                    }
                )
        if rec.get("label_ocean") is None and rec.get("label_mbti") and len(found["missing_ocean"]) < 2:
            found["missing_ocean"].append(
                {
                    "id": rec["id"],
                    "label_mbti": rec.get("label_mbti"),
                    "label_ocean": rec.get("label_ocean"),
                    "excerpt": excerpt(text, 200),
                }
            )
        if all(len(v) >= 2 for v in found.values()):
            break
    return found


def extract_personality_evd_samples() -> list[dict]:
    """Personality-Evd: short text per record, evidence richness, UNKNOWN-heavy, E=HIGH-heavy."""
    found = {
        "rich_evidence": [],
        "unknown_heavy": [],
        "all_high_extreme": [],
        "rare_combo": [],
    }
    rare_combos_seen = Counter()
    for rec in stream(DATA_ROOT / "personality_evd" / "train.jsonl"):
        ocean = rec.get("label_ocean", {})
        evidence = rec.get("evidence_gold") or []
        # Records with all 5 traits having NON-empty evidence quotes
        non_empty = sum(1 for e in evidence if (e.get("quote") or "").strip())
        if non_empty >= 4 and len(found["rich_evidence"]) < 2:
            found["rich_evidence"].append(
                {
                    "id": rec["id"],
                    "speaker": rec.get("metadata", {}).get("speaker"),
                    "ocean": ocean,
                    "text": excerpt(rec.get("text", ""), 220),
                    "evidence_count": non_empty,
                    "evidence_examples": [
                        {
                            "trait": e["trait"],
                            "level": e["level"],
                            "quote": excerpt(e.get("quote", ""), 100),
                            "reasoning": excerpt(e.get("reasoning", ""), 180),
                        }
                        for e in evidence
                        if (e.get("quote") or "").strip()
                    ][:3],
                }
            )
        # Records with mostly UNKNOWN evidence
        unknown_count = sum(1 for e in evidence if e.get("level") == "UNKNOWN")
        if unknown_count >= 4 and len(found["unknown_heavy"]) < 2:
            found["unknown_heavy"].append(
                {
                    "id": rec["id"],
                    "speaker": rec.get("metadata", {}).get("speaker"),
                    "ocean": ocean,
                    "text": excerpt(rec.get("text", ""), 200),
                    "n_unknown": unknown_count,
                    "n_total_evidence": len(evidence),
                }
            )
        # All HIGH (the dominant 31% combo)
        if all(ocean.get(t) == "HIGH" for t in OCEAN) and len(found["all_high_extreme"]) < 2:
            found["all_high_extreme"].append(
                {
                    "id": rec["id"],
                    "speaker": rec.get("metadata", {}).get("speaker"),
                    "ocean": ocean,
                    "text": excerpt(rec.get("text", ""), 220),
                }
            )
        # Rare combo (not in top 4)
        combo = "".join("1" if ocean.get(t) == "HIGH" else "0" if ocean.get(t) == "LOW" else "X" for t in OCEAN)
        if combo not in {"11111", "11110", "11101", "11100"} and len(found["rare_combo"]) < 2:
            if rare_combos_seen[combo] == 0:
                found["rare_combo"].append(
                    {
                        "id": rec["id"],
                        "speaker": rec.get("metadata", {}).get("speaker"),
                        "combo": combo,
                        "ocean": ocean,
                        "text": excerpt(rec.get("text", ""), 200),
                    }
                )
                rare_combos_seen[combo] += 1
    return found


# ============================================================
# CHART GENERATION
# ============================================================


def chart_mbti_imbalance():
    """Chart 1: MBTI 16-class imbalance bar chart (MBTI vs Pandora side by side)."""
    mbti_counts = Counter()
    for rec in stream(DATA_ROOT / "mbti" / "train.jsonl"):
        if rec.get("label_mbti"):
            mbti_counts[rec["label_mbti"]] += 1
    pandora_counts = Counter()
    for rec in stream(DATA_ROOT / "pandora" / "train.jsonl"):
        if rec.get("label_mbti"):
            pandora_counts[rec["label_mbti"]] += 1

    types_sorted = sorted(MBTI_TYPES, key=lambda t: -mbti_counts.get(t, 0))
    mbti_vals = [mbti_counts.get(t, 0) for t in types_sorted]
    pandora_vals = [pandora_counts.get(t, 0) for t in types_sorted]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(types_sorted))
    w = 0.4
    ax.bar(x - w / 2, mbti_vals, w, label="MBTI (Personality Café)", color="#3a86ff")
    ax.bar(x + w / 2, pandora_vals, w, label="Pandora (Reddit)", color="#ff006e")
    ax.set_xticks(x)
    ax.set_xticklabels(types_sorted, rotation=45)
    ax.set_ylabel("Number of users (train split)")
    ax.set_title("MBTI 16-class imbalance: ratios of 47x (MBTI) and 80x (Pandora)")
    ax.legend()
    # Annotate min/max
    ax.annotate(
        f"Max: {types_sorted[0]} ({mbti_vals[0]:,})",
        xy=(0, mbti_vals[0]),
        xytext=(2, mbti_vals[0]),
        fontsize=9,
        color="#3a86ff",
    )
    ax.annotate(
        f"Min: {types_sorted[-1]} ({mbti_vals[-1]:,})",
        xy=(15, mbti_vals[-1]),
        xytext=(10, max(mbti_vals) * 0.4),
        fontsize=9,
        color="#3a86ff",
        arrowprops=dict(arrowstyle="->", color="#3a86ff"),
    )
    plt.tight_layout()
    out = CHARTS_DIR / "01_mbti_imbalance.png"
    plt.savefig(out, dpi=110)
    plt.close()
    return out


def chart_mbti_dim_balance():
    """Chart 2: 4-dim binary balance comparison MBTI vs Pandora — show T/F flip."""
    datasets = ["mbti", "pandora"]
    dims = ["IE", "SN", "TF", "JP"]
    pos_keys = {"IE": "I", "SN": "N", "TF": "F", "JP": "P"}

    counts = {ds: {d: Counter() for d in dims} for ds in datasets}
    for ds in datasets:
        for rec in stream(DATA_ROOT / ds / "train.jsonl"):
            md = rec.get("label_mbti_dimensions") or {}
            for d in dims:
                v = md.get(d)
                if v:
                    counts[ds][d][v] += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(dims))
    w = 0.35
    mbti_vals = [100 * counts["mbti"][d][pos_keys[d]] / max(sum(counts["mbti"][d].values()), 1) for d in dims]
    pand_vals = [100 * counts["pandora"][d][pos_keys[d]] / max(sum(counts["pandora"][d].values()), 1) for d in dims]
    ax.bar(x - w / 2, mbti_vals, w, label="MBTI", color="#3a86ff")
    ax.bar(x + w / 2, pand_vals, w, label="Pandora", color="#ff006e")
    ax.axhline(50, ls="--", color="gray", lw=0.7, label="Balance line (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(%{pos_keys[d]})" for d in dims])
    ax.set_ylabel("% positive class")
    ax.set_title("MBTI 4-dim binary balance — note T/F flips between datasets")
    ax.legend(loc="upper right")
    for i, v in enumerate(mbti_vals):
        ax.text(i - w / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    for i, v in enumerate(pand_vals):
        ax.text(i + w / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    plt.tight_layout()
    out = CHARTS_DIR / "02_mbti_dim_balance.png"
    plt.savefig(out, dpi=110)
    plt.close()
    return out


def chart_ocean_distribution():
    """Chart 3: OCEAN HIGH% across datasets - show how distributions diverge."""
    datasets = ["essays", "pandora", "personality_evd"]
    counts = {ds: {t: Counter() for t in OCEAN} for ds in datasets}
    for ds in datasets:
        for rec in stream(DATA_ROOT / ds / "train.jsonl"):
            ocean = rec.get("label_ocean") or {}
            for t in OCEAN:
                v = ocean.get(t)
                if v:
                    counts[ds][t][v] += 1

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(OCEAN))
    w = 0.2
    colors = ["#06d6a0", "#118ab2", "#ef476f"]
    for i, ds in enumerate(datasets):
        vals = []
        for t in OCEAN:
            c = counts[ds][t]
            hi = c.get("HIGH", 0)
            lo = c.get("LOW", 0)
            vals.append(100 * hi / (hi + lo) if (hi + lo) else 0)
        ax.bar(x + i * w - 1.5 * w, vals, w, label=ds, color=colors[i])
    ax.axhline(50, ls="--", color="gray", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(OCEAN)
    ax.set_ylabel("% HIGH (excluding UNKNOWN)")
    ax.set_title("OCEAN %HIGH per trait — Personality-Evd extreme E=98%")
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    out = CHARTS_DIR / "03_ocean_distribution.png"
    plt.savefig(out, dpi=110)
    plt.close()
    return out


def chart_text_length():
    """Chart 4: Text length histograms (log y) per dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.flatten()
    datasets = ["mbti", "essays", "pandora", "personality_evd"]
    for idx, ds in enumerate(datasets):
        wl = []
        for rec in stream(DATA_ROOT / ds / "train.jsonl"):
            wl.append(len(rec.get("text", "").split()))
        ax = axes[idx]
        # Use log scale for a fair visual
        if max(wl) > 100:
            bins = np.logspace(0, math.log10(max(wl) + 1), 50)
            ax.hist(wl, bins=bins, color="#118ab2", edgecolor="black")
            ax.set_xscale("log")
        else:
            ax.hist(wl, bins=30, color="#118ab2", edgecolor="black")
        ax.axvline(395, color="red", ls="--", lw=1.2, label="~512 BPE token threshold")
        ax.set_title(f"{ds}\n(n={len(wl):,}, median={int(np.median(wl))} words)")
        ax.set_xlabel("Words per record")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    axes[-1].axis("off")
    plt.suptitle("Word-length distribution per dataset (log x for >100)", fontsize=12)
    plt.tight_layout()
    out = CHARTS_DIR / "04_text_length.png"
    plt.savefig(out, dpi=110)
    plt.close()
    return out


def chart_leakage():
    """Chart 5: Leakage (taxonomy + exact MBTI) bar chart."""
    datasets = ["mbti", "essays", "pandora", "personality_evd"]
    exact_pct = []
    tax_pct = []
    ocean_pct = []
    for ds in datasets:
        n = 0
        ex = 0
        tx = 0
        oc = 0
        for rec in stream(DATA_ROOT / ds / "train.jsonl"):
            n += 1
            t = rec.get("text", "")
            if MBTI_RE.search(t):
                ex += 1
            if TAXONOMY_RE.search(t):
                tx += 1
            if OCEAN_TAX_RE.search(t):
                oc += 1
        exact_pct.append(100 * ex / max(n, 1))
        tax_pct.append(100 * tx / max(n, 1))
        ocean_pct.append(100 * oc / max(n, 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(datasets))
    w = 0.27
    ax.bar(x - w, exact_pct, w, label="MBTI exact (INFJ, INTP, …)", color="#06d6a0")
    ax.bar(x, tax_pct, w, label="MBTI taxonomy (introvert, MBTI, …)", color="#ef476f")
    ax.bar(x + w, ocean_pct, w, label="OCEAN taxonomy (openness, …)", color="#ffd166")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("% records containing leakage term")
    ax.set_title("Residual leakage after preprocessing — taxonomy terms still rampant in MBTI/Pandora")
    ax.legend()
    for i, v in enumerate(tax_pct):
        if v > 1:
            ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9, color="#ef476f")
    plt.tight_layout()
    out = CHARTS_DIR / "05_leakage.png"
    plt.savefig(out, dpi=110)
    plt.close()
    return out


def chart_evd_evidence_levels():
    """Chart 6: Personality-Evd evidence level distribution per trait."""
    counts = {t: Counter() for t in OCEAN}
    for rec in stream(DATA_ROOT / "personality_evd" / "train.jsonl"):
        for ev in rec.get("evidence_gold") or []:
            t = ev.get("trait")
            lvl = ev.get("level")
            if t in OCEAN and lvl:
                counts[t][lvl] += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    levels = ["HIGH", "LOW", "UNKNOWN"]
    colors = {"HIGH": "#ef476f", "LOW": "#06d6a0", "UNKNOWN": "#aaaaaa"}
    bottoms = np.zeros(len(OCEAN))
    for lvl in levels:
        vals = [counts[t].get(lvl, 0) for t in OCEAN]
        ax.bar(OCEAN, vals, bottom=bottoms, label=lvl, color=colors[lvl])
        for i, v in enumerate(vals):
            if v > 50:
                ax.text(i, bottoms[i] + v / 2, str(v), ha="center", fontsize=9, color="white")
        bottoms += np.array(vals)
    ax.set_ylabel("# evidence items (train)")
    ax.set_title("Personality-Evd evidence levels per trait — UNKNOWN dominates, HIGH > LOW × 3")
    ax.legend()
    plt.tight_layout()
    out = CHARTS_DIR / "06_evd_evidence_levels.png"
    plt.savefig(out, dpi=110)
    plt.close()
    return out


def chart_evd_combo():
    """Chart 7: Personality-Evd OCEAN combo distribution — only 14 of 32 used."""
    combos = Counter()
    for rec in stream(DATA_ROOT / "personality_evd" / "train.jsonl"):
        ocean = rec.get("label_ocean") or {}
        combo = "".join("1" if ocean.get(t) == "HIGH" else "0" if ocean.get(t) == "LOW" else "X" for t in OCEAN)
        combos[combo] += 1
    items = combos.most_common()
    labels = [c for c, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(labels)), vals, color=["#ef476f" if v >= 100 else "#118ab2" for v in vals])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, fontsize=9, family="monospace")
    ax.set_ylabel("# train records")
    ax.set_title(f"Personality-Evd: OCEAN combo distribution — only {len(labels)}/32 combos appear, top 4 cover 78.7%")
    for bar, v in zip(bars, vals):
        if v >= 50:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 5, str(v), ha="center", fontsize=8)
    plt.tight_layout()
    out = CHARTS_DIR / "07_evd_combo.png"
    plt.savefig(out, dpi=110)
    plt.close()
    return out


def chart_dataset_size_log():
    """Chart 9: Dataset sizes on log scale - shows the 5-orders-of-magnitude span."""
    datasets = ["personality_evd", "essays", "mbti", "pandora"]
    sizes = []
    for ds in datasets:
        n = 0
        for sp in ["train", "val", "test"]:
            n += sum(1 for _ in stream(DATA_ROOT / ds / f"{sp}.jsonl"))
        sizes.append(n)
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(datasets, sizes, color=["#06d6a0", "#118ab2", "#3a86ff", "#ef476f", "#8338ec"])
    ax.set_xscale("log")
    ax.set_xlabel("# records (total, log scale)")
    ax.set_title("4 datasets — Pandora and MBTI are 4-5x larger than Essays / Personality-Evd")
    for bar, n in zip(bars, sizes):
        ax.text(n * 1.1, bar.get_y() + bar.get_height() / 2, f"{n:,}", va="center", fontsize=9)
    plt.tight_layout()
    out = CHARTS_DIR / "09_dataset_sizes.png"
    plt.savefig(out, dpi=110)
    plt.close()
    return out


# ============================================================
# REPORT GENERATION
# ============================================================


def render_report(samples: dict, charts: dict) -> str:
    out = ["# Illustrative Samples & Visualization Charts", ""]
    out.append(
        "Generated by `scripts/illustrate_dataset_issues.py`. "
        "Each section pairs concrete sample records with a chart that quantifies the issue."
    )
    out.append("")
    out.append("## Charts overview")
    out.append("")
    for name, path in charts.items():
        out.append(f"- **{name}**: `{path}`")
    out.append("")

    # ---- Overall ----
    out.append("## 0. Dataset scale comparison")
    out.append("")
    out.append(f"![Dataset sizes](charts/{Path(charts['Dataset sizes (log)']).name})")
    out.append("")
    out.append("→ Dataset sizes range from Personality-Evd (1.8K) to Pandora (8.9K records).")
    out.append("")

    # ---- MBTI ----
    out.append("## 1. MBTI — class imbalance + taxonomy leakage + truncation")
    out.append("")
    out.append(f"![MBTI imbalance](charts/{Path(charts['MBTI imbalance']).name})")
    out.append("")
    out.append(f"![MBTI dim balance](charts/{Path(charts['MBTI 4-dim balance']).name})")
    out.append("")
    out.append("**Issue 1.1 — Rare class examples** (ESTJ/ESFJ/ESFP/ESTP only ~0.5-1% each):")
    out.append("")
    for s in samples["mbti"]["rare_class"]:
        out.append(f"- `{s['id']}` | **{s['label']}** | {s['n_words']} words")
        out.append(f"  > _{s['excerpt']}_")
    out.append("")

    out.append("**Issue 1.2 — Taxonomy leakage** (model can string-match without learning personality):")
    out.append("")
    for s in samples["mbti"]["taxonomy_leak"]:
        out.append(f"- `{s['id']}` | label=**{s['label']}** | matched=`{s['match']}`")
        # bold the match in context (case-insensitive)
        ctx = s["context"]
        out.append(f"  > _…{ctx}…_")
    out.append("")

    out.append("**Issue 1.3 — Truncation problem** (>1700 words, 96% records exceed BERT 512 tokens):")
    out.append("")
    for s in samples["mbti"]["very_long"]:
        out.append(f"- `{s['id']}` | **{s['label']}** | **{s['n_words']:,} words**")
        out.append(f"  > **HEAD**: _{s['head']}_")
        out.append(f"  > **TAIL** (lost by 512-tok truncation): _{s['tail']}_")
    out.append("")

    # ---- Essays ----
    out.append("## 2. Essays — small, balanced, but old/narrow domain")
    out.append("")
    out.append("**Issue 2.1 — Mixed-label samples** (good for multi-label learning):")
    out.append("")
    for s in samples["essays"]["balanced_examples"]:
        out.append(f"- `{s['id']}` | OCEAN={s['ocean']} | {s['n_words']} words")
        out.append(f"  > _{s['excerpt']}_")
    out.append("")

    out.append("**Issue 2.2 — Length range** (44 to 2000 words; most ~620):")
    out.append("")
    for s in samples["essays"]["extreme_short"]:
        out.append(f"- (shortest) `{s['id']}` | {s['n_words']} words | OCEAN={s['ocean']}")
        out.append(f"  > _{s['excerpt']}_")
    for s in samples["essays"]["extreme_long"]:
        out.append(f"- (longest) `{s['id']}` | {s['n_words']} words | OCEAN={s['ocean']}")
        out.append(f"  > _{s['excerpt']}_")
    out.append("")

    # ---- Pandora ----
    out.append("## 3. Pandora — 2000-word cap, taxonomy leakage, missing OCEAN")
    out.append("")
    out.append(f"![OCEAN distribution](charts/{Path(charts['OCEAN distribution']).name})")
    out.append("")
    out.append("**Issue 3.1 — Hard-capped at 2000 words** (info loss for prolific users):")
    out.append("")
    for s in samples["pandora"]["capped_2000"]:
        out.append(
            f"- `{s['id']}` | mbti={s['label_mbti']} | "
            f"sampled {s['n_sampled']} of {s['n_total_comments']:,} comments | "
            f"**{s['n_words']} words after concat (capped)**"
        )
        out.append(f"  > **HEAD**: _{s['head']}_")
        out.append(f"  > **TAIL**: _{s['tail']}_")
    out.append("")

    out.append("**Issue 3.2 — Taxonomy leakage on Reddit (~24% records):**")
    out.append("")
    for s in samples["pandora"]["taxonomy_leak"]:
        out.append(f"- `{s['id']}` | mbti={s['label_mbti']} | matched=`{s['match']}`")
        out.append(f"  > _…{s['context']}…_")
    out.append("")

    out.append("**Issue 3.3 — Many users have MBTI but no OCEAN** (82% missing OCEAN):")
    out.append("")
    for s in samples["pandora"]["missing_ocean"]:
        out.append(f"- `{s['id']}` | mbti=**{s['label_mbti']}** | ocean={s['label_ocean']}")
        out.append(f"  > _{s['excerpt']}_")
    out.append("")

    # ---- Personality-Evd ----
    out.append("## 4. Personality-Evd — rich evidence but very skewed labels")
    out.append("")
    out.append(f"![Evidence levels](charts/{Path(charts['Evd evidence levels']).name})")
    out.append("")
    out.append(f"![Evd combo](charts/{Path(charts['Evd OCEAN combo']).name})")
    out.append("")

    out.append("**Issue 5.1 — Rich evidence sample** (the gold standard for XAI):")
    out.append("")
    for s in samples["personality_evd"]["rich_evidence"]:
        out.append(
            f"- `{s['id']}` | speaker={s['speaker']} | OCEAN={s['ocean']} | "
            f"{s['evidence_count']}/5 traits with evidence"
        )
        out.append(f"  > **Text**: _{s['text']}_")
        for ev in s["evidence_examples"]:
            out.append(
                f'  > → **{ev["trait"]}={ev["level"]}** | quote: _"{ev["quote"]}"_  \n'
                f"  >   reasoning: _{ev['reasoning']}_"
            )
    out.append("")

    out.append("**Issue 5.2 — UNKNOWN-heavy records** (4-5 of 5 traits have no evidence):")
    out.append("")
    for s in samples["personality_evd"]["unknown_heavy"]:
        out.append(
            f"- `{s['id']}` | speaker={s['speaker']} | **{s['n_unknown']}/{s['n_total_evidence']} evidence = UNKNOWN**"
        )
        out.append(f"  > _{s['text']}_")
    out.append("")

    out.append("**Issue 5.3 — Dominant 'all HIGH' combo** (31% of train, makes accuracy meaningless):")
    out.append("")
    for s in samples["personality_evd"]["all_high_extreme"]:
        out.append(f"- `{s['id']}` | speaker={s['speaker']} | OCEAN={s['ocean']}")
        out.append(f"  > _{s['text']}_")
    out.append("")

    out.append("**Issue 5.4 — Rare combo** (only ~3% of train rolls outside the top-4 patterns):")
    out.append("")
    for s in samples["personality_evd"]["rare_combo"]:
        out.append(f"- `{s['id']}` | speaker={s['speaker']} | combo=`{s['combo']}` | OCEAN={s['ocean']}")
        out.append(f"  > _{s['text']}_")
    out.append("")

    # ---- Leakage chart ----
    out.append("## 5. Cross-dataset leakage view")
    out.append("")
    out.append(f"![Leakage](charts/{Path(charts['Leakage residual']).name})")
    out.append("")
    out.append(
        "**Takeaway**: exact MBTI types are clean, but taxonomy terms (introvert, MBTI, personality type) "
        "still appear in 24-56% of records → must be added to the leakage filter regex."
    )
    out.append("")

    return "\n".join(out)


def main():
    print("[*] Extracting illustrative samples ...", flush=True)
    samples = {
        "mbti": extract_mbti_samples(),
        "essays": extract_essays_samples(),
        "pandora": extract_pandora_samples(),
        "personality_evd": extract_personality_evd_samples(),
    }
    print("[*] Generating charts ...", flush=True)
    charts = {}
    charts["MBTI imbalance"] = chart_mbti_imbalance()
    charts["MBTI 4-dim balance"] = chart_mbti_dim_balance()
    charts["OCEAN distribution"] = chart_ocean_distribution()
    charts["Text length distributions"] = chart_text_length()
    charts["Leakage residual"] = chart_leakage()
    charts["Evd evidence levels"] = chart_evd_evidence_levels()
    charts["Evd OCEAN combo"] = chart_evd_combo()
    charts["Dataset sizes (log)"] = chart_dataset_size_log()
    print("[*] Rendering report ...", flush=True)
    md = render_report(samples, charts)
    SAMPLES_OUT.write_text(md)
    print(f"\n[done] {SAMPLES_OUT}")
    print(f"  charts in: {CHARTS_DIR}")
    for k, v in charts.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
