"""Supplementary analyses NOT covered by the existing analyze_datasets.py:

1. Cross-split duplicate detection (exact + near-dup)
2. Big-Five Pearson correlation matrix from raw scores
3. Per-split MBTI 16-class & 4-dim balance
4. OCEAN multi-label combo distribution per dataset
5. Length percentiles per split
6. Per-dataset 5 sample records (sanity check)

Outputs: outputs/analysis/supplementary_report.md
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

DATA_ROOT = Path("data/processed")
OUT = Path("outputs/analysis/supplementary_report.md")
DATASETS = ["mbti", "essays", "pandora", "personality_evd"]
SPLITS = ["train", "val", "test"]
OCEAN = ["O", "C", "E", "A", "N"]
SAMPLE_CAP: dict[str, int] = {}


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


def text_hash(t: str) -> str:
    return hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest()


def near_hash(t: str, n: int = 200) -> str:
    norm = re.sub(r"\s+", " ", t.lower().strip())[:n]
    return hashlib.md5(norm.encode("utf-8", errors="ignore")).hexdigest()


def percentiles(vals: list[float], qs=(0, 50, 90, 95, 99, 100)) -> dict[str, float]:
    if not vals:
        return {f"p{q}": 0 for q in qs} | {"mean": 0}
    s = sorted(vals)
    n = len(s)
    out = {}
    for q in qs:
        if q == 0:
            out[f"p{q}"] = float(s[0])
        elif q == 100:
            out[f"p{q}"] = float(s[-1])
        else:
            idx = max(0, min(n - 1, int(math.ceil(q / 100 * n)) - 1))
            out[f"p{q}"] = float(s[idx])
    out["mean"] = sum(s) / n
    return out


def pearson(xs, ys):
    n = min(len(xs), len(ys))
    if n < 2:
        return None
    xs, ys = xs[:n], ys[:n]
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def analyze_dataset(ds: str) -> dict[str, Any]:
    cap = SAMPLE_CAP.get(ds)
    res = {"dataset": ds, "splits": {}, "_text_hashes": {}, "_near_hashes": {}}
    raw_scores = {t: [] for t in OCEAN}
    samples = []

    for sp in SPLITS:
        path = DATA_ROOT / ds / f"{sp}.jsonl"
        if not path.exists():
            continue
        text_hashes = []
        near_hashes = []
        word_lens = []
        ocean_combo = Counter()
        mbti16 = Counter()
        mbti_dim = {d: Counter() for d in ["IE", "SN", "TF", "JP"]}
        ocean_per_trait = {t: Counter() for t in OCEAN}
        n_recs = 0
        for rec in stream(path, cap):
            n_recs += 1
            text = rec.get("text", "") or ""
            word_lens.append(len(text.split()))
            text_hashes.append(text_hash(text))
            near_hashes.append(near_hash(text))

            if rec.get("label_mbti"):
                mbti16[rec["label_mbti"]] += 1
            if rec.get("label_mbti_dimensions"):
                for d, v in rec["label_mbti_dimensions"].items():
                    mbti_dim[d][v] += 1

            ocean = rec.get("label_ocean")
            if ocean:
                for t in OCEAN:
                    v = ocean.get(t)
                    if v is not None:
                        ocean_per_trait[t][v] += 1
                combo = "".join("1" if ocean.get(t) == "HIGH" else "0" if ocean.get(t) == "LOW" else "X" for t in OCEAN)
                ocean_combo[combo] += 1

            meta = rec.get("metadata") or {}
            bf = meta.get("bigfive_raw")
            if bf:
                for t, ln in [
                    ("O", "openness"),
                    ("C", "conscientiousness"),
                    ("E", "extraversion"),
                    ("A", "agreeableness"),
                    ("N", "neuroticism"),
                ]:
                    v = bf.get(ln)
                    if v is not None:
                        raw_scores[t].append(float(v))

            if sp == "train" and len(samples) < 3:
                samples.append(
                    {
                        "id": rec.get("id"),
                        "text_excerpt": (text[:160] + "…") if len(text) > 160 else text,
                        "label_mbti": rec.get("label_mbti"),
                        "label_ocean": rec.get("label_ocean"),
                    }
                )

        wp = percentiles(word_lens)
        within_exact = len(text_hashes) - len(set(text_hashes))
        within_near = len(near_hashes) - len(set(near_hashes))

        res["splits"][sp] = {
            "n": n_recs,
            "word_pct": {k: round(v, 1) for k, v in wp.items()},
            "mbti16": dict(mbti16.most_common()),
            "mbti_dim": {d: dict(c) for d, c in mbti_dim.items()},
            "ocean_per_trait": {t: dict(c) for t, c in ocean_per_trait.items()},
            "ocean_combo_top": ocean_combo.most_common(8),
            "within_exact_dup": within_exact,
            "within_near_dup": within_near,
            "n_unique_combos": len(ocean_combo),
        }
        res["_text_hashes"][sp] = set(text_hashes)
        res["_near_hashes"][sp] = set(near_hashes)

    # Cross-split duplicates
    res["cross_split"] = {}
    splits_present = list(res["_text_hashes"].keys())
    for i, s1 in enumerate(splits_present):
        for s2 in splits_present[i + 1 :]:
            res["cross_split"][f"{s1}∩{s2}_exact"] = len(res["_text_hashes"][s1] & res["_text_hashes"][s2])
            res["cross_split"][f"{s1}∩{s2}_near"] = len(res["_near_hashes"][s1] & res["_near_hashes"][s2])
    res.pop("_text_hashes")
    res.pop("_near_hashes")

    # Pearson on raw bigfive
    pmat = {}
    if any(raw_scores[t] for t in OCEAN):
        for t1 in OCEAN:
            pmat[t1] = {}
            for t2 in OCEAN:
                pmat[t1][t2] = pearson(raw_scores[t1], raw_scores[t2])
    res["pearson_ocean"] = pmat
    res["raw_scores_n"] = {t: len(raw_scores[t]) for t in OCEAN}
    res["samples_train"] = samples
    return res


def render(results: dict[str, dict]) -> str:
    out = ["# Supplementary Dataset Analysis", ""]
    out.append("Computed by `scripts/analyze_datasets_supplement.py`.")
    out.append("")

    # 1. Per-split sample counts
    out.append("## 1. Per-split record counts")
    out.append("")
    out.append("| Dataset | Train | Val | Test |")
    out.append("|---|---:|---:|---:|")
    for ds in DATASETS:
        sp = results.get(ds, {}).get("splits", {})
        tr = sp.get("train", {}).get("n", 0)
        va = sp.get("val", {}).get("n", 0)
        te = sp.get("test", {}).get("n", 0)
        cap = " *(sampled)*" if SAMPLE_CAP.get(ds) else ""
        out.append(f"| {ds}{cap} | {tr:,} | {va:,} | {te:,} |")
    out.append("")

    # 2. Length percentiles per split
    out.append("## 2. Word-length percentiles per split")
    out.append("")
    for ds in DATASETS:
        out.append(f"### {ds}")
        out.append("")
        out.append("| Split | min | mean | p50 | p90 | p95 | p99 | max |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for sp in SPLITS:
            s = results[ds]["splits"].get(sp)
            if not s:
                continue
            w = s["word_pct"]
            out.append(
                f"| {sp} | {w['p0']:.0f} | {w['mean']:.0f} | {w['p50']:.0f} "
                f"| {w['p90']:.0f} | {w['p95']:.0f} | {w['p99']:.0f} | {w['p100']:.0f} |"
            )
        out.append("")

    # 3. Cross-split duplicates
    out.append("## 3. Cross-split duplicate detection")
    out.append("")
    out.append("| Dataset | train∩val (exact / near) | train∩test | val∩test |")
    out.append("|---|---|---|---|")
    for ds in DATASETS:
        cs = results[ds].get("cross_split", {})
        out.append(
            f"| {ds} | {cs.get('train∩val_exact', 0)} / {cs.get('train∩val_near', 0)} "
            f"| {cs.get('train∩test_exact', 0)} / {cs.get('train∩test_near', 0)} "
            f"| {cs.get('val∩test_exact', 0)} / {cs.get('val∩test_near', 0)} |"
        )
    out.append("")

    out.append("## 4. Within-split duplicate detection")
    out.append("")
    out.append("| Dataset | Split | Exact dup | Near dup (200ch) |")
    out.append("|---|---|---:|---:|")
    for ds in DATASETS:
        for sp in SPLITS:
            s = results[ds]["splits"].get(sp)
            if not s:
                continue
            out.append(f"| {ds} | {sp} | {s['within_exact_dup']} | {s['within_near_dup']} |")
    out.append("")

    # 5. MBTI per-split balance
    out.append("## 5. MBTI 4-dimension balance across splits")
    out.append("")
    for ds in ["mbti", "pandora"]:
        out.append(f"### {ds}")
        out.append("")
        out.append("| Split | I/E (%I) | S/N (%N) | T/F (%F) | J/P (%P) |")
        out.append("|---|---|---|---|---|")
        for sp in SPLITS:
            s = results[ds]["splits"].get(sp)
            if not s or not s["mbti_dim"]["IE"]:
                continue
            d = s["mbti_dim"]
            ie = d["IE"]
            sn = d["SN"]
            tf = d["TF"]
            jp = d["JP"]

            def pct(c, k):
                tot = sum(c.values())
                return f"{100 * c.get(k, 0) / tot:.1f}%" if tot else "—"

            out.append(
                f"| {sp} | {ie.get('I', 0)}/{ie.get('E', 0)} ({pct(ie, 'I')}) "
                f"| {sn.get('S', 0)}/{sn.get('N', 0)} ({pct(sn, 'N')}) "
                f"| {tf.get('T', 0)}/{tf.get('F', 0)} ({pct(tf, 'F')}) "
                f"| {jp.get('J', 0)}/{jp.get('P', 0)} ({pct(jp, 'P')}) |"
            )
        out.append("")

    # 6. OCEAN per-split balance
    out.append("## 6. OCEAN HIGH/LOW per split (% HIGH)")
    out.append("")
    for ds in ["essays", "pandora", "personality_evd"]:
        out.append(f"### {ds}")
        out.append("")
        out.append("| Split | %HIGH(O) | %HIGH(C) | %HIGH(E) | %HIGH(A) | %HIGH(N) | %UNK(any) |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for sp in SPLITS:
            s = results[ds]["splits"].get(sp)
            if not s:
                continue
            row = [sp]
            unk_total = 0
            denom_total = 0
            for t in OCEAN:
                c = s["ocean_per_trait"].get(t, {})
                hi = c.get("HIGH", 0)
                lo = c.get("LOW", 0)
                uk = c.get("UNKNOWN", 0)
                tot = hi + lo + uk
                unk_total += uk
                denom_total += tot
                if hi + lo > 0:
                    row.append(f"{100 * hi / (hi + lo):.1f}%")
                else:
                    row.append("—")
            row.append(f"{100 * unk_total / denom_total:.1f}%" if denom_total else "—")
            out.append("| " + " | ".join(row) + " |")
        out.append("")

    # 7. OCEAN combo distribution
    out.append("## 7. OCEAN multi-label combo distribution (train, top 8)")
    out.append("")
    out.append("Format: `OCEAN` (1=HIGH, 0=LOW, X=UNKNOWN). 32 possible combos if balanced.")
    out.append("")
    for ds in ["essays", "pandora", "personality_evd"]:
        s = results[ds]["splits"].get("train")
        if not s or not s["ocean_combo_top"]:
            continue
        out.append(f"### {ds} ({s['n_unique_combos']} unique combos)")
        out.append("")
        out.append("| Combo | Count | % of train |")
        out.append("|---|---:|---:|")
        tr_n = s["n"]
        for combo, cnt in s["ocean_combo_top"]:
            out.append(f"| `{combo}` | {cnt:,} | {100 * cnt / tr_n:.1f}% |")
        out.append("")

    # 8. Pearson correlation
    out.append("## 8. Pearson correlation on raw Big-Five scores (pooled across splits)")
    out.append("")
    for ds in DATASETS:
        pm = results[ds].get("pearson_ocean", {})
        if not pm or not any(pm.values()):
            continue
        n_obs = max(results[ds].get("raw_scores_n", {}).values())
        out.append(f"### {ds} (n={n_obs:,} observations)")
        out.append("")
        out.append("| | O | C | E | A | N |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for t1 in OCEAN:
            row = [f"**{t1}**"]
            for t2 in OCEAN:
                v = pm.get(t1, {}).get(t2)
                row.append("—" if v is None else f"{v:+.2f}")
            out.append("| " + " | ".join(row) + " |")
        out.append("")

    # 9. Sample records
    out.append("## 9. Train sample records (sanity check)")
    out.append("")
    for ds in DATASETS:
        out.append(f"### {ds}")
        out.append("")
        for s in results[ds].get("samples_train", []):
            out.append(f"- `{s['id']}` | mbti={s['label_mbti']} | ocean={s['label_ocean']}")
            out.append(f"    > _{s['text_excerpt']}_")
        out.append("")

    return "\n".join(out)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for ds in DATASETS:
        print(f"[*] {ds}", flush=True)
        results[ds] = analyze_dataset(ds)
    md = render(results)
    OUT.write_text(md)
    print(f"\n[done] {OUT}")
    print(md)


if __name__ == "__main__":
    main()
