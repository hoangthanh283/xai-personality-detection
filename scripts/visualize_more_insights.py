"""Generate additional visualization charts for analyses not yet covered.

Existing charts (01-09) cover:
  01 MBTI imbalance, 02 4-dim balance, 03 OCEAN distribution, 04 text length,
  05 leakage, 06 evidence levels, 07 Evd combo, 09 dataset sizes.

New charts (10-17) added here cover:
  10 OCEAN combo distribution comparison (3 datasets side by side)
  11 Personality-Evd state vs trait disagreement per trait
  12 Train/Val/Test split ratios per dataset
  13 Issue severity heatmap (13 issues × 4 datasets)
  14 Personality-Evd evidence quote length distribution
  15 Taxonomy leakage breakdown by term type (MBTI/Pandora)
  16 MBTI 16-class per-split consistency
  17 Dataset role matrix (XAI capability vs benchmark size)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

DATA_ROOT = Path("data/processed")
CHARTS_DIR = Path("outputs/analysis/charts")
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


def stream(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# =====================================================================
# CHART 10: OCEAN combo distribution comparison
# =====================================================================
def chart_10_ocean_combo_compare():
    """Compare OCEAN combo coverage across essays, pandora, personality_evd."""
    datasets = ["essays", "pandora", "personality_evd"]
    combos_per_ds = {}
    for ds in datasets:
        c = Counter()
        for rec in stream(DATA_ROOT / ds / "train.jsonl"):
            ocean = rec.get("label_ocean") or {}
            combo = "".join("1" if ocean.get(t) == "HIGH" else "0" if ocean.get(t) == "LOW" else "X" for t in OCEAN)
            c[combo] += 1
        combos_per_ds[ds] = c

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    colors = ["#06d6a0", "#118ab2", "#ef476f"]
    for ax, (ds, color) in zip(axes, zip(datasets, colors)):
        c = combos_per_ds[ds]
        items = c.most_common(20)
        labels = [k for k, _ in items]
        vals = [v for _, v in items]
        total = sum(c.values())
        n_unique = len([k for k, v in c.items() if v > 0])
        ax.barh(range(len(labels)), vals, color=color)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8, family="monospace")
        ax.invert_yaxis()
        ax.set_xlabel("# train records")
        ax.set_title(f"{ds}\n{n_unique}/32 combos, top 20 = {sum(vals) / total * 100:.0f}%", fontsize=11)
    plt.suptitle("OCEAN combo distribution — Personality-Evd has lowest diversity (14/32)", fontsize=12)
    plt.tight_layout()
    out = CHARTS_DIR / "10_ocean_combo_compare.png"
    plt.savefig(out, dpi=110)
    plt.close()
    return out


# =====================================================================
# CHART 11: Personality-Evd state vs trait disagreement
# =====================================================================
def chart_11_evd_state_trait_disagreement():
    """For each record's evidence_gold, count how many evidence levels disagree
    with the record-level OCEAN label."""
    disagreements = {t: 0 for t in OCEAN}
    totals = {t: 0 for t in OCEAN}
    for rec in stream(DATA_ROOT / "personality_evd" / "train.jsonl"):
        ocean = rec.get("label_ocean") or {}
        for ev in rec.get("evidence_gold") or []:
            t = ev.get("trait")
            ev_level = ev.get("level")
            rec_level = ocean.get(t)
            if t in OCEAN and ev_level in {"HIGH", "LOW"} and rec_level in {"HIGH", "LOW"}:
                totals[t] += 1
                if ev_level != rec_level:
                    disagreements[t] += 1

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(OCEAN))
    w = 0.35
    agree = [totals[t] - disagreements[t] for t in OCEAN]
    disag = [disagreements[t] for t in OCEAN]
    ax.bar(x, agree, w, label="state ↔ trait agree", color="#06d6a0")
    ax.bar(x, disag, w, bottom=agree, label="DISAGREE", color="#ef476f")
    for i, t in enumerate(OCEAN):
        if totals[t] > 0:
            pct = 100 * disagreements[t] / totals[t]
            ax.text(i, totals[t] + 5, f"{disagreements[t]}\n({pct:.0f}%)", ha="center", fontsize=9, color="#ef476f")
    ax.set_xticks(x)
    ax.set_xticklabels(OCEAN)
    ax.set_ylabel("# evidence items (HIGH/LOW only, train)")
    ax.set_title(
        "Personality-Evd: state-level vs trait-level disagreements\n"
        "A and N show most fluctuation — matches Big-Five theory"
    )
    ax.legend()
    plt.tight_layout()
    out = CHARTS_DIR / "11_evd_state_trait_disagreement.png"
    plt.savefig(out, dpi=110)
    plt.close()
    return out


# =====================================================================
# CHART 12: Train/Val/Test split ratios per dataset
# =====================================================================
def chart_12_split_ratios():
    """Bar chart of split sizes per dataset."""
    datasets = ["mbti", "essays", "pandora", "personality_evd"]
    splits = ["train", "val", "test"]
    counts = {ds: {sp: sum(1 for _ in stream(DATA_ROOT / ds / f"{sp}.jsonl")) for sp in splits} for ds in datasets}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: stacked absolute
    ax = axes[0]
    bottom = np.zeros(len(datasets))
    colors = {"train": "#3a86ff", "val": "#ffd166", "test": "#ef476f"}
    for sp in splits:
        vals = [counts[ds][sp] for ds in datasets]
        ax.bar(datasets, vals, bottom=bottom, label=sp, color=colors[sp])
        for i, v in enumerate(vals):
            if v > 100:
                ax.text(i, bottom[i] + v / 2, f"{v:,}", ha="center", color="white", fontsize=9)
        bottom += np.array(vals)
    ax.set_ylabel("# records")
    ax.set_title("Split sizes — absolute")
    ax.legend()

    # Right: stacked percent
    ax = axes[1]
    bottom = np.zeros(len(datasets))
    for sp in splits:
        pcts = [100 * counts[ds][sp] / sum(counts[ds].values()) for ds in datasets]
        ax.bar(datasets, pcts, bottom=bottom, label=sp, color=colors[sp])
        for i, pct in enumerate(pcts):
            if pct > 4:
                ax.text(i, bottom[i] + pct / 2, f"{pct:.0f}%", ha="center", color="white", fontsize=9)
        bottom += np.array(pcts)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of total")
    ax.set_title("Split ratios — Pandora's test only ~3% (just 232 records)")
    ax.legend()

    plt.tight_layout()
    out = CHARTS_DIR / "12_split_ratios.png"
    plt.savefig(out, dpi=110)
    plt.close()
    return out


# =====================================================================
# CHART 13: Issue severity heatmap (13 issues × 4 datasets)
# =====================================================================
def chart_13_issue_heatmap():
    """Heatmap of issue severity per dataset.
    0=N/A, 1=LOW, 2=MEDIUM, 3=HIGH, 4=CRITICAL."""
    issues = [
        "MBTI 16-class imbalance 47-80x",
        "Taxonomy leakage 24-56%",
        "Personality-Evd E=HIGH 97.7%",
        "T/F flip MBTI vs Pandora",
        "MBTI/Pandora truncation 96-98%",
        "Domain bias (Café/Reddit/TV)",
        "Pandora test = 232 records",
        "Personality-Evd 76% UNK for O",
        "Quote mismatch 30% in Personality-Evd",
        "Pandora OCEAN coverage 17.5%",
        "Personality-Evd 14/32 combos",
        "Pandora cap 2000 từ",
        "Essays cũ 1999 narrow domain",
        "No KB attached (cross-cutting)",
    ]
    datasets = ["mbti", "essays", "pandora", "personality_evd"]
    # severity matrix, indexed by [issue_idx][dataset_idx]
    M = np.array(
        [
            [4, 0, 4, 0],  # 1: 16-class imbalance (MBTI, Pandora)
            [4, 0, 3, 0],  # 2: taxonomy leakage
            [0, 0, 0, 4],  # 3: E=HIGH
            [3, 0, 3, 0],  # 4: T/F flip (cross-domain)
            [3, 1, 3, 0],  # 5: truncation (MBTI/Pandora; Essays moderate)
            [3, 0, 3, 3],  # 6: domain bias (all except Essays)
            [0, 0, 2, 0],  # 7: Pandora test small
            [0, 0, 0, 2],  # 8: UNKNOWN heavy
            [0, 0, 0, 2],  # 9: quote mismatch
            [0, 0, 2, 0],  # 10: OCEAN coverage
            [0, 0, 0, 2],  # 11: 14/32 combos
            [0, 0, 1, 0],  # 12: cap 2000
            [0, 1, 0, 0],  # 13: Essays narrow
            [4, 4, 4, 4],  # 14: no KB cross-cutting
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = LinearSegmentedColormap.from_list(
        "sev", [(0, "#f5f5f5"), (0.25, "#ffe5b4"), (0.5, "#ffa500"), (0.75, "#ff4500"), (1.0, "#8b0000")]
    )
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=4, aspect="auto")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_yticks(range(len(issues)))
    ax.set_yticklabels(issues, fontsize=10)

    sev_label = {0: "—", 1: "LOW", 2: "MED", 3: "HIGH", 4: "CRIT"}
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, sev_label[int(v)], ha="center", va="center", color="white" if v >= 3 else "black", fontsize=9)
    ax.set_title(
        "Issue severity per dataset — 14 issues × 4 datasets\n"
        "Note row 14: KB construction is CRITICAL for ALL datasets",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, label="severity (0=NA, 4=CRITICAL)", shrink=0.8)
    plt.tight_layout()
    out = CHARTS_DIR / "13_issue_heatmap.png"
    plt.savefig(out, dpi=110)
    plt.close()
    return out


# =====================================================================
# CHART 14: Personality-Evd evidence quote length distribution
# =====================================================================
def chart_14_evd_quote_length():
    """Histogram of evidence quote lengths (chars + words) for non-empty quotes."""
    quote_words = []
    quote_chars = []
    reasoning_words = []
    for rec in stream(DATA_ROOT / "personality_evd" / "train.jsonl"):
        for ev in rec.get("evidence_gold") or []:
            q = ev.get("quote", "") or ""
            r = ev.get("reasoning", "") or ""
            if q.strip():
                quote_words.append(len(q.split()))
                quote_chars.append(len(q))
            if r.strip():
                reasoning_words.append(len(r.split()))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].hist(quote_words, bins=40, color="#118ab2", edgecolor="black")
    axes[0].axvline(
        np.median(quote_words),
        color="red",
        ls="--",
        label=f"median={int(np.median(quote_words))}",
    )
    axes[0].set_xlabel("Words in quote")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Evidence quote length\n(n={len(quote_words):,} non-empty)")
    axes[0].legend()

    axes[1].hist(quote_chars, bins=40, color="#06d6a0", edgecolor="black")
    axes[1].axvline(
        np.median(quote_chars),
        color="red",
        ls="--",
        label=f"median={int(np.median(quote_chars))}",
    )
    axes[1].set_xlabel("Characters in quote")
    axes[1].set_title("Evidence quote length (chars)")
    axes[1].legend()

    axes[2].hist(reasoning_words, bins=40, color="#ef476f", edgecolor="black")
    axes[2].axvline(
        np.median(reasoning_words),
        color="darkred",
        ls="--",
        label=f"median={int(np.median(reasoning_words))}",
    )
    axes[2].set_xlabel("Words in reasoning")
    axes[2].set_title(f"Reasoning text length\n(n={len(reasoning_words):,})")
    axes[2].legend()

    plt.suptitle("Personality-Evd: evidence quote + reasoning length distributions", fontsize=12)
    plt.tight_layout()
    out = CHARTS_DIR / "14_evd_quote_length.png"
    plt.savefig(out, dpi=110)
    plt.close()
    return out


# =====================================================================
# CHART 15: Taxonomy leakage breakdown by term type
# =====================================================================
def chart_15_taxonomy_breakdown():
    """For MBTI and Pandora, show which taxonomy terms are most common."""
    terms = {
        "MBTI": r"\bmbti\b",
        "Myers-Briggs": r"\bmyers[- ]?briggs\b",
        "Introvert(ed)": r"\bintrovert(ed)?\b",
        "Extravert/Extrovert": r"\bextra[vo]ert(ed)?\b",
        "Introversion": r"\bintroversion\b",
        "Extraversion/Extroversion": r"\bextra[vo]ersion\b",
        "Personality type(s)": r"\bpersonality types?\b",
    }
    counts = {ds: {term: 0 for term in terms} for ds in ["mbti", "pandora"]}
    for ds in ["mbti", "pandora"]:
        for rec in stream(DATA_ROOT / ds / "train.jsonl"):
            text = rec.get("text", "") or ""
            for term, pattern in terms.items():
                if re.search(pattern, text, re.IGNORECASE):
                    counts[ds][term] += 1

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(terms))
    w = 0.35
    ax.bar(x - w / 2, [counts["mbti"][t] for t in terms], w, label="MBTI (n=6071)", color="#3a86ff")
    ax.bar(x + w / 2, [counts["pandora"][t] for t in terms], w, label="Pandora (n=7180)", color="#ef476f")
    ax.set_xticks(x)
    ax.set_xticklabels(terms.keys(), rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("# records containing term")
    ax.set_title("Taxonomy leakage breakdown — 'introvert' is dominant in both MBTI and Pandora")
    ax.legend()
    for i, t in enumerate(terms):
        v_m = counts["mbti"][t]
        v_p = counts["pandora"][t]
        if v_m > 100:
            ax.text(i - w / 2, v_m + 30, f"{v_m}", ha="center", fontsize=8)
        if v_p > 100:
            ax.text(i + w / 2, v_p + 30, f"{v_p}", ha="center", fontsize=8)
    plt.tight_layout()
    out = CHARTS_DIR / "15_taxonomy_leakage_breakdown.png"
    plt.savefig(out, dpi=110)
    plt.close()
    return out


# =====================================================================
# CHART 16: MBTI 16-class per-split consistency
# =====================================================================
def chart_16_mbti_per_split():
    """Show MBTI 16-class distribution is consistent across train/val/test (stratified split sanity)."""
    splits = ["train", "val", "test"]
    counts = {sp: Counter() for sp in splits}
    for sp in splits:
        for rec in stream(DATA_ROOT / "mbti" / f"{sp}.jsonl"):
            label = rec.get("label_mbti")
            if label:
                counts[sp][label] += 1

    types_sorted = sorted(MBTI_TYPES, key=lambda t: -counts["train"].get(t, 0))
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(types_sorted))
    w = 0.27
    colors = {"train": "#3a86ff", "val": "#ffd166", "test": "#ef476f"}
    for i, sp in enumerate(splits):
        total = sum(counts[sp].values())
        pcts = [100 * counts[sp].get(t, 0) / total for t in types_sorted]
        ax.bar(x + (i - 1) * w, pcts, w, label=sp, color=colors[sp])
    ax.set_xticks(x)
    ax.set_xticklabels(types_sorted, rotation=45, fontsize=9)
    ax.set_ylabel("% within split")
    ax.set_title(
        "MBTI 16-class distribution per split — stratified split is consistent\n(<1% deviation between train/val/test)"
    )
    ax.legend()
    plt.tight_layout()
    out = CHARTS_DIR / "16_mbti_per_split.png"
    plt.savefig(out, dpi=110)
    plt.close()
    return out


# =====================================================================
# CHART 17: Dataset role matrix (XAI capability vs benchmark size)
# =====================================================================
def chart_17_role_matrix():
    """2D scatter: x=size, y=XAI capability score; color/size = role.
    Visualize the Primary vs Supporting framing."""
    datasets = {
        "MBTI": {"size": 8673, "xai": 0, "role": "supporting", "note": "Big-scale MBTI bench"},
        "Essays": {"size": 2467, "xai": 0, "role": "supporting", "note": "Balanced OCEAN"},
        "Pandora": {"size": 8951, "xai": 0, "role": "supporting", "note": "Long context"},
        "Personality-Evd": {"size": 1846, "xai": 100, "role": "PRIMARY", "note": "Has evidence_gold"},
    }
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, info in datasets.items():
        x = info["size"]
        y = info["xai"]
        color = "#ef476f" if info["role"] == "PRIMARY" else "#118ab2"
        size = 700 if info["role"] == "PRIMARY" else 400
        ax.scatter(
            x,
            y,
            s=size,
            color=color,
            alpha=0.7,
            edgecolors="black",
            linewidths=2,
            label=info["role"] if name in ["MBTI", "Personality-Evd"] else None,
        )
        ax.annotate(
            f"{name}\n{info['note']}",
            (x, y),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
        )
    ax.set_xscale("log")
    ax.set_xlabel("# records (log scale)")
    ax.set_ylabel("XAI capability (% records with evidence_gold)")
    ax.set_title(
        "Dataset roles — Personality-Evd is the ONLY primary XAI benchmark;\n"
        "MBTI/Pandora/Essays prove generalization (high size, no XAI)",
        fontsize=11,
    )
    ax.set_ylim(-10, 110)
    ax.axhline(50, ls="--", color="gray", lw=0.7)
    ax.text(1500, 52, "XAI threshold", color="gray", fontsize=9)
    ax.legend(loc="center right")
    ax.grid(True, ls=":", alpha=0.4)
    plt.tight_layout()
    out = CHARTS_DIR / "17_role_matrix.png"
    plt.savefig(out, dpi=110)
    plt.close()
    return out


def main():
    chart_funcs = [
        chart_10_ocean_combo_compare,
        chart_11_evd_state_trait_disagreement,
        chart_12_split_ratios,
        chart_13_issue_heatmap,
        chart_14_evd_quote_length,
        chart_15_taxonomy_breakdown,
        chart_16_mbti_per_split,
        chart_17_role_matrix,
    ]
    for fn in chart_funcs:
        print(f"  - {fn.__name__} ...", flush=True)
        out = fn()
        print(f"    {out}")
    print("\n[done] All charts saved to outputs/analysis/charts/")


if __name__ == "__main__":
    main()
