"""Analyze processed personality datasets and their raw sources.

Outputs:
  - outputs/reports/dataset_analysis.json
  - outputs/reports/dataset_analysis.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
OCEAN_TRAITS = ("O", "C", "E", "A", "N")
MBTI_DIMS = ("IE", "SN", "TF", "JP")
MBTI_TYPES = (
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
)

MBTI_PATTERN = re.compile(r"\b(" + "|".join(MBTI_TYPES) + r")(s|es)?\b", re.IGNORECASE)
MBTI_TAXONOMY_PATTERN = re.compile(
    r"\b(mbti|myers[- ]?briggs|introvert|introversion|extravert|extrovert|"
    r"extraversion|extroversion|personality type|personality types)\b",
    re.IGNORECASE,
)
OCEAN_TAXONOMY_PATTERN = re.compile(
    r"\b(big five|ocean|openness|conscientiousness|extraversion|extroversion|"
    r"agreeableness|neuroticism)\b",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_processed(dataset: str, processed_root: Path) -> list[dict[str, Any]]:
    records = []
    for split in SPLITS:
        for rec in load_jsonl(processed_root / dataset / f"{split}.jsonl"):
            rec = dict(rec)
            rec["_split_file"] = split
            records.append(rec)
    return records


def pct(n: int | float, d: int | float) -> float:
    return round((100.0 * n / d), 2) if d else 0.0


def quantile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return values[idx]


def describe_numbers(values: list[int]) -> dict[str, Any]:
    if not values:
        return {}
    return {
        "min": min(values),
        "p25": quantile(values, 0.25),
        "median": int(median(values)),
        "mean": round(mean(values), 2),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": max(values),
    }


def counter_to_sorted_dict(counter: Counter) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], str(item[0]))))


def distribution_stats(counter: Counter) -> dict[str, Any]:
    total = sum(counter.values())
    if not total:
        return {"total": 0, "classes": {}, "majority": None, "minority": None}
    nonzero = {k: v for k, v in counter.items() if v > 0}
    majority_label, majority_count = max(nonzero.items(), key=lambda item: item[1])
    minority_label, minority_count = min(nonzero.items(), key=lambda item: item[1])
    imbalance_ratio = round(majority_count / minority_count, 2) if minority_count else None
    return {
        "total": total,
        "classes": counter_to_sorted_dict(counter),
        "majority": {
            "label": majority_label,
            "count": majority_count,
            "pct": pct(majority_count, total),
        },
        "minority": {
            "label": minority_label,
            "count": minority_count,
            "pct": pct(minority_count, total),
        },
        "imbalance_ratio": imbalance_ratio,
    }


def split_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return {split: sum(1 for r in records if r.get("split") == split) for split in SPLITS}


def analyze_text(records: list[dict[str, Any]]) -> dict[str, Any]:
    word_counts = [len(str(r.get("text", "")).split()) for r in records]
    char_counts = [len(str(r.get("text", ""))) for r in records]
    return {
        "words": describe_numbers(word_counts),
        "chars": describe_numbers(char_counts),
        "empty_text": sum(1 for n in word_counts if n == 0),
        "over_512_words": sum(1 for n in word_counts if n > 512),
        "over_512_words_pct": pct(sum(1 for n in word_counts if n > 512), len(records)),
        "over_2000_words": sum(1 for n in word_counts if n > 2000),
        "over_2000_words_pct": pct(sum(1 for n in word_counts if n > 2000), len(records)),
    }


def analyze_leakage(records: list[dict[str, Any]]) -> dict[str, Any]:
    any_mbti = own_mbti = mbti_taxonomy = ocean_taxonomy = urls = mentions = 0
    examples = {"own_mbti": [], "any_mbti": [], "ocean_taxonomy": []}

    for rec in records:
        text = str(rec.get("text", ""))
        label = rec.get("label_mbti")
        mbti_hits = {m.group(1).upper() for m in MBTI_PATTERN.finditer(text)}
        if mbti_hits:
            any_mbti += 1
            if len(examples["any_mbti"]) < 5:
                examples["any_mbti"].append({"id": rec.get("id"), "hits": sorted(mbti_hits)})
        if label and str(label).upper() in mbti_hits:
            own_mbti += 1
            if len(examples["own_mbti"]) < 5:
                examples["own_mbti"].append({"id": rec.get("id"), "label": label})
        if MBTI_TAXONOMY_PATTERN.search(text):
            mbti_taxonomy += 1
        if OCEAN_TAXONOMY_PATTERN.search(text):
            ocean_taxonomy += 1
            if len(examples["ocean_taxonomy"]) < 5:
                hit = OCEAN_TAXONOMY_PATTERN.search(text)
                examples["ocean_taxonomy"].append({"id": rec.get("id"), "hit": hit.group(0)})
        if URL_PATTERN.search(text):
            urls += 1
        if MENTION_PATTERN.search(text):
            mentions += 1

    total = len(records)
    return {
        "any_mbti_type_mention": any_mbti,
        "any_mbti_type_mention_pct": pct(any_mbti, total),
        "own_mbti_type_mention": own_mbti,
        "own_mbti_type_mention_pct": pct(own_mbti, total),
        "mbti_taxonomy_terms": mbti_taxonomy,
        "mbti_taxonomy_terms_pct": pct(mbti_taxonomy, total),
        "ocean_taxonomy_terms": ocean_taxonomy,
        "ocean_taxonomy_terms_pct": pct(ocean_taxonomy, total),
        "url_leftovers": urls,
        "url_leftovers_pct": pct(urls, total),
        "mention_leftovers": mentions,
        "mention_leftovers_pct": pct(mentions, total),
        "examples": examples,
    }


def analyze_mbti_labels(records: list[dict[str, Any]]) -> dict[str, Any]:
    full = Counter(r.get("label_mbti") for r in records if r.get("label_mbti"))
    dims: dict[str, Any] = {}
    for dim in MBTI_DIMS:
        counts = Counter()
        for rec in records:
            label = rec.get("label_mbti_dimensions") or {}
            if label.get(dim):
                counts[label[dim]] += 1
        dims[dim] = distribution_stats(counts)
    return {"full_type": distribution_stats(full), "dimensions": dims}


def analyze_ocean_labels(records: list[dict[str, Any]]) -> dict[str, Any]:
    traits: dict[str, Any] = {}
    complete = 0
    partial = 0
    missing = 0
    for rec in records:
        labels = rec.get("label_ocean") or {}
        values = [labels.get(t) for t in OCEAN_TRAITS]
        if all(v in {"HIGH", "LOW"} for v in values):
            complete += 1
        elif any(v is not None for v in values):
            partial += 1
        else:
            missing += 1

    for trait in OCEAN_TRAITS:
        counts = Counter()
        for rec in records:
            labels = rec.get("label_ocean") or {}
            counts[str(labels.get(trait, "MISSING")).upper()] += 1
        traits[trait] = distribution_stats(counts)

    return {
        "complete_ocean_labels": complete,
        "partial_ocean_labels": partial,
        "missing_ocean_labels": missing,
        "traits": traits,
    }


def analyze_evidence(records: list[dict[str, Any]]) -> dict[str, Any]:
    with_evidence = [r for r in records if r.get("evidence_gold")]
    evidence_items = []
    for rec in with_evidence:
        for ev in rec.get("evidence_gold") or []:
            evidence_items.append((rec, ev))

    trait_counts = Counter(str(ev.get("trait", "MISSING")).upper() for _, ev in evidence_items)
    level_counts = Counter(str(ev.get("level", "MISSING")).upper() for _, ev in evidence_items)
    nonempty_quote = 0
    quote_found = 0
    utterance_ids_nonempty = 0
    disagreements = Counter()

    for rec, ev in evidence_items:
        quote = str(ev.get("quote", "")).strip()
        if quote:
            nonempty_quote += 1
            if quote in str(rec.get("text", "")):
                quote_found += 1
        if ev.get("utterance_ids"):
            utterance_ids_nonempty += 1

        trait = str(ev.get("trait", "")).upper()
        ev_level = str(ev.get("level", "")).upper()
        record_level = str((rec.get("label_ocean") or {}).get(trait, "")).upper()
        if trait in OCEAN_TRAITS and ev_level in {"HIGH", "LOW"} and record_level in {"HIGH", "LOW"}:
            if ev_level != record_level:
                disagreements[trait] += 1

    return {
        "records_with_evidence": len(with_evidence),
        "records_with_evidence_pct": pct(len(with_evidence), len(records)),
        "total_evidence_items": len(evidence_items),
        "evidence_items_per_record": round(len(evidence_items) / len(with_evidence), 2) if with_evidence else 0,
        "trait_distribution": counter_to_sorted_dict(trait_counts),
        "level_distribution": counter_to_sorted_dict(level_counts),
        "nonempty_quote_items": nonempty_quote,
        "nonempty_quote_items_pct": pct(nonempty_quote, len(evidence_items)),
        "quote_found_in_processed_text": quote_found,
        "quote_found_in_processed_text_pct": pct(quote_found, nonempty_quote),
        "items_with_utterance_ids": utterance_ids_nonempty,
        "items_with_utterance_ids_pct": pct(utterance_ids_nonempty, len(evidence_items)),
        "evidence_record_label_disagreements": counter_to_sorted_dict(disagreements),
    }


def analyze_duplicates(records: list[dict[str, Any]]) -> dict[str, Any]:
    ids = Counter(str(r.get("id")) for r in records)
    texts = Counter(str(r.get("text", "")) for r in records)
    duplicate_ids = sum(1 for v in ids.values() if v > 1)
    duplicate_texts = sum(1 for v in texts.values() if v > 1)
    return {
        "unique_ids": len(ids),
        "duplicate_id_values": duplicate_ids,
        "duplicate_id_records": sum(v for v in ids.values() if v > 1),
        "unique_texts": len(texts),
        "duplicate_text_values": duplicate_texts,
        "duplicate_text_records": sum(v for v in texts.values() if v > 1),
    }


def analyze_common(dataset: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "total_records": len(records),
        "split_counts": split_counts(records),
        "text": analyze_text(records),
        "labels_mbti": analyze_mbti_labels(records),
        "labels_ocean": analyze_ocean_labels(records),
        "leakage": analyze_leakage(records),
        "duplicates": analyze_duplicates(records),
        "evidence": analyze_evidence(records),
    }


def count_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    result = subprocess.run(
        ["wc", "-l", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip().split()[0])


def analyze_raw_mbti(raw_root: Path) -> dict[str, Any]:
    path = raw_root / "mbti" / "mbti_1.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    users = len(df)
    post_counts = []
    any_leak_users = own_leak_users = 0
    any_leak_posts = own_leak_posts = 0
    total_posts = 0

    for _, row in df.iterrows():
        label = str(row.get("type", "")).upper()
        posts = str(row.get("posts", "")).split("|||")
        post_counts.append(len(posts))
        text = " ".join(posts)
        hits = {m.group(1).upper() for m in MBTI_PATTERN.finditer(text)}
        if hits:
            any_leak_users += 1
        if label in hits:
            own_leak_users += 1
        for post in posts:
            total_posts += 1
            post_hits = {m.group(1).upper() for m in MBTI_PATTERN.finditer(post)}
            if post_hits:
                any_leak_posts += 1
            if label in post_hits:
                own_leak_posts += 1

    return {
        "raw_rows_users": users,
        "raw_post_snippets": total_posts,
        "posts_per_user": describe_numbers(post_counts),
        "raw_label_distribution": distribution_stats(Counter(df["type"].str.upper())),
        "users_with_any_mbti_mention": any_leak_users,
        "users_with_any_mbti_mention_pct": pct(any_leak_users, users),
        "users_with_own_type_mention": own_leak_users,
        "users_with_own_type_mention_pct": pct(own_leak_users, users),
        "post_snippets_with_any_mbti_mention": any_leak_posts,
        "post_snippets_with_any_mbti_mention_pct": pct(any_leak_posts, total_posts),
        "post_snippets_with_own_type_mention": own_leak_posts,
        "post_snippets_with_own_type_mention_pct": pct(own_leak_posts, total_posts),
    }


def analyze_raw_essays(raw_root: Path) -> dict[str, Any]:
    path = raw_root / "essays" / "essays.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding="latin-1")
    label_cols = {"cOPN": "O", "cCON": "C", "cEXT": "E", "cAGR": "A", "cNEU": "N"}
    labels = {}
    for col, trait in label_cols.items():
        if col in df:
            labels[trait] = distribution_stats(Counter(df[col].astype(str).str.lower()))
    return {
        "raw_rows": len(df),
        "raw_columns": list(df.columns),
        "raw_trait_label_distribution": labels,
        "unique_authors": df["#AUTHID"].nunique() if "#AUTHID" in df else None,
    }


def analyze_raw_pandora(raw_root: Path) -> dict[str, Any]:
    profile_path = raw_root / "pandora" / "author_profiles.csv"
    comments_path = raw_root / "pandora" / "all_comments_since_2015.csv"
    if not profile_path.exists():
        return {}
    df = pd.read_csv(profile_path)
    valid_mbti = df.get("mbti", pd.Series(dtype=object)).astype(str).str.upper().isin(MBTI_TYPES)
    ocean_cols = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    has_ocean = df[ocean_cols].notna().all(axis=1) if all(c in df for c in ocean_cols) else pd.Series()
    return {
        "raw_author_profiles": len(df),
        "raw_comment_rows_including_header": count_lines(comments_path),
        "raw_comment_rows_excluding_header": (count_lines(comments_path) - 1) if comments_path.exists() else None,
        "profiles_with_valid_mbti": int(valid_mbti.sum()),
        "profiles_with_valid_mbti_pct": pct(int(valid_mbti.sum()), len(df)),
        "profiles_with_complete_ocean": int(has_ocean.sum()) if len(has_ocean) else None,
        "profiles_with_complete_ocean_pct": pct(int(has_ocean.sum()), len(df)) if len(has_ocean) else None,
    }


def analyze_raw_personality_evd(raw_root: Path) -> dict[str, Any]:
    root = raw_root / "personality_evd"
    if not root.exists():
        return {}
    split_lines = {}
    speakers_per_item = []
    dialogue_turns = []
    evidence_items = 0
    for split in SPLITS:
        path = root / f"{split}.jsonl"
        rows = load_jsonl(path)
        split_lines[split] = len(rows)
        for row in rows:
            dialogue = row.get("dialogue", [])
            dialogue_turns.append(len(dialogue))
            speakers = {turn.get("speaker") for turn in dialogue if turn.get("speaker")}
            speakers_per_item.append(len(speakers))
            evidence_items += len(row.get("evidence", []) or [])
    return {
        "raw_dialogue_rows_by_split": split_lines,
        "raw_dialogue_rows_total": sum(split_lines.values()),
        "speakers_per_dialogue_row": describe_numbers(speakers_per_item),
        "turns_per_dialogue_row": describe_numbers(dialogue_turns),
        "raw_evidence_items": evidence_items,
    }


def add_dataset_specific_stats(dataset: str, records: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    if dataset == "mbti":
        num_posts = [
            int((r.get("metadata") or {}).get("num_posts", 0))
            for r in records
            if (r.get("metadata") or {}).get("num_posts") is not None
        ]
        avg_post_lengths = [
            float((r.get("metadata") or {}).get("avg_post_length", 0))
            for r in records
            if (r.get("metadata") or {}).get("avg_post_length") is not None
        ]
        stats["mbti_specific"] = {
            "num_clean_posts_per_user": describe_numbers(num_posts),
            "avg_clean_post_length_words": describe_numbers([round(x) for x in avg_post_lengths]),
        }

    if dataset == "pandora":
        sampled = [
            int((r.get("metadata") or {}).get("num_comments", 0))
            for r in records
            if (r.get("metadata") or {}).get("num_comments") is not None
        ]
        total_comments = [
            int((r.get("metadata") or {}).get("num_total_comments", 0))
            for r in records
            if (r.get("metadata") or {}).get("num_total_comments") is not None
        ]
        with_mbti = sum(1 for r in records if r.get("label_mbti"))
        with_ocean = sum(1 for r in records if r.get("label_ocean"))
        stats["pandora_specific"] = {
            "records_with_mbti": with_mbti,
            "records_with_mbti_pct": pct(with_mbti, len(records)),
            "records_with_ocean": with_ocean,
            "records_with_ocean_pct": pct(with_ocean, len(records)),
            "sampled_comments_per_user": describe_numbers(sampled),
            "total_available_comments_per_user": describe_numbers(total_comments),
            "records_capped_at_100_comments": sum(1 for n in sampled if n >= 100),
            "records_capped_at_100_comments_pct": pct(sum(1 for n in sampled if n >= 100), len(sampled)),
        }

    if dataset == "personality_evd":
        utterances = [
            int((r.get("metadata") or {}).get("num_utterances", 0))
            for r in records
            if (r.get("metadata") or {}).get("num_utterances") is not None
        ]
        speakers = {str((r.get("metadata") or {}).get("speaker")) for r in records}
        dialogues = {str((r.get("metadata") or {}).get("dialogue_id")) for r in records}
        stats["personality_evd_specific"] = {
            "unique_speakers": len(speakers),
            "unique_dialogue_ids": len(dialogues),
            "utterances_per_speaker_record": describe_numbers(utterances),
        }


def build_report(analysis: dict[str, Any]) -> str:
    lines = [
        "# Dataset Analysis Report",
        "",
        "Generated from local `data/raw` and `data/processed` files.",
        "",
        "## Dataset Overview",
        "",
        "| Dataset | Records | Train | Val | Test | Median words | >512 words | Primary labels | Evidence records |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for name, stats in analysis["datasets"].items():
        splits = stats["split_counts"]
        text = stats["text"]
        mbti_total = stats["labels_mbti"]["full_type"]["total"]
        ocean_complete = stats["labels_ocean"]["complete_ocean_labels"]
        labels = []
        if mbti_total:
            labels.append(f"MBTI={mbti_total}")
        if ocean_complete:
            labels.append(f"OCEAN={ocean_complete}")
        evidence_records = stats["evidence"]["records_with_evidence"]
        lines.append(
            f"| {name} | {stats['total_records']} | {splits['train']} | {splits['val']} | "
            f"{splits['test']} | {text['words'].get('median')} | "
            f"{text['over_512_words']} ({text['over_512_words_pct']}%) | "
            f"{', '.join(labels) or 'n/a'} | {evidence_records} |"
        )

    lines.extend(
        [
            "",
            "## Research Framing",
            "",
            "`personality_evd` is the only dataset with gold evidence annotations, so it should "
            "be the main benchmark for RAG-XPR's explainability claim. `mbti`, `pandora`, "
            "and `essays` are best treated as accuracy/generalization benchmarks because "
            "they do not provide ground-truth evidence labels.",
            "",
            "None of the four datasets includes an attached psychology knowledge base. RAG-XPR "
            "must therefore build an external KB from validated sources such as BFI-2, "
            "NEO-PI-R/NEO-PI-3, MBTI manuals, and personality facet/behavioral-marker papers. "
            "This KB work should be planned early because retrieval quality directly affects "
            "state identification and explanation quality.",
        ]
    )

    lines.extend(["", "## Leakage Overview", ""])
    lines.extend(
        [
            "| Dataset | Any MBTI type mention | Own MBTI mention | MBTI taxonomy terms | "
            "OCEAN taxonomy terms | URL leftovers | Mention leftovers |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, stats in analysis["datasets"].items():
        leak = stats["leakage"]
        lines.append(
            f"| {name} | {leak['any_mbti_type_mention']} ({leak['any_mbti_type_mention_pct']}%) | "
            f"{leak['own_mbti_type_mention']} ({leak['own_mbti_type_mention_pct']}%) | "
            f"{leak['mbti_taxonomy_terms']} ({leak['mbti_taxonomy_terms_pct']}%) | "
            f"{leak['ocean_taxonomy_terms']} ({leak['ocean_taxonomy_terms_pct']}%) | "
            f"{leak['url_leftovers']} ({leak['url_leftovers_pct']}%) | "
            f"{leak['mention_leftovers']} ({leak['mention_leftovers_pct']}%) |"
        )

    lines.extend(["", "## MBTI Distributions", ""])
    for name, stats in analysis["datasets"].items():
        full = stats["labels_mbti"]["full_type"]
        if not full["total"]:
            continue
        lines.append(f"### {name}")
        lines.append(
            f"- Full type: {full['total']} labeled; majority `{full['majority']['label']}` "
            f"{full['majority']['pct']}%; minority `{full['minority']['label']}` "
            f"{full['minority']['pct']}%; imbalance {full['imbalance_ratio']}x."
        )
        lines.append(f"- Counts: `{json.dumps(full['classes'], ensure_ascii=False)}`")
        for dim, dim_stats in stats["labels_mbti"]["dimensions"].items():
            if dim_stats["total"]:
                lines.append(
                    f"- {dim}: `{json.dumps(dim_stats['classes'], ensure_ascii=False)}`; "
                    f"majority {dim_stats['majority']['pct']}%."
                )
        lines.append("")

    lines.extend(["## OCEAN Distributions", ""])
    for name, stats in analysis["datasets"].items():
        ocean = stats["labels_ocean"]
        if not ocean["complete_ocean_labels"] and not ocean["partial_ocean_labels"]:
            continue
        lines.append(f"### {name}")
        lines.append(
            f"- Complete labels: {ocean['complete_ocean_labels']}; partial: "
            f"{ocean['partial_ocean_labels']}; missing: {ocean['missing_ocean_labels']}."
        )
        for trait, trait_stats in ocean["traits"].items():
            lines.append(
                f"- {trait}: `{json.dumps(trait_stats['classes'], ensure_ascii=False)}`; "
                f"majority {trait_stats['majority']['label']}={trait_stats['majority']['pct']}%."
            )
        lines.append("")

    lines.extend(["## Evidence Coverage", ""])
    for name, stats in analysis["datasets"].items():
        ev = stats["evidence"]
        if not ev["records_with_evidence"]:
            continue
        lines.append(f"### {name}")
        lines.append(f"- Records with evidence: {ev['records_with_evidence']} ({ev['records_with_evidence_pct']}%).")
        lines.append(
            f"- Evidence items: {ev['total_evidence_items']}; nonempty quotes: "
            f"{ev['nonempty_quote_items']} ({ev['nonempty_quote_items_pct']}%); quote found "
            f"in processed text: {ev['quote_found_in_processed_text_pct']}%."
        )
        lines.append(f"- Levels: `{json.dumps(ev['level_distribution'], ensure_ascii=False)}`")
        lines.append(
            f"- Evidence vs record-label disagreements: "
            f"`{json.dumps(ev['evidence_record_label_disagreements'], ensure_ascii=False)}`"
        )
        lines.append("")

    lines.extend(["## Raw Source Summary", ""])
    lines.append("```json")
    lines.append(json.dumps(analysis["raw_sources"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.extend(["## Main Data Risks", ""])
    lines.extend(derive_risks(analysis))
    lines.append("")
    return "\n".join(lines)


def derive_risks(analysis: dict[str, Any]) -> list[str]:
    risks = []
    mbti = analysis["datasets"].get("mbti")
    if mbti:
        full = mbti["labels_mbti"]["full_type"]
        risks.append(
            f"- MBTI is strongly imbalanced: majority `{full['majority']['label']}` is "
            f"{full['majority']['pct']}%, minority `{full['minority']['label']}` is "
            f"{full['minority']['pct']}%, ratio {full['imbalance_ratio']}x. Treat 16-class "
            "accuracy as fragile; prefer 4 binary axes plus macro-F1/balanced accuracy."
        )
        leak = mbti["leakage"]
        risks.append(
            f"- Processed MBTI has {leak['own_mbti_type_mention']} own-type exact mentions, "
            "but raw MBTI leakage is substantial. Keep type-stripping mandatory before any "
            "benchmark."
        )

    pandora = analysis["datasets"].get("pandora")
    if pandora:
        ps = pandora.get("pandora_specific", {})
        risks.append(
            f"- Pandora has many user records, but only {ps.get('records_with_ocean')} "
            f"({ps.get('records_with_ocean_pct')}%) have complete OCEAN labels. OCEAN experiments "
            "are sample-limited despite the huge raw Reddit comment file."
        )
        risks.append(
            "- Pandora texts are long user aggregates; transformer 512-token truncation discards "
            f"signal for {pandora['text']['over_512_words_pct']}% of records."
        )

    essays = analysis["datasets"].get("essays")
    if essays:
        risks.append(
            "- Essays is clean and near-balanced compared with social data, but it is small and "
            "single-document per user; it is a weak test for long-history retrieval."
        )

    evd = analysis["datasets"].get("personality_evd")
    if evd:
        ev = evd["evidence"]
        risks.append(
            f"- PersonalityEvd is the only processed dataset with gold evidence "
            f"({ev['records_with_evidence_pct']}% coverage), but many evidence levels are UNKNOWN; "
            "evaluate explanation quality separately from trait classification."
        )
        e_trait = evd["labels_ocean"]["traits"]["E"]
        risks.append(
            f"- PersonalityEvd has severe E-trait skew: majority `{e_trait['majority']['label']}` "
            f"is {e_trait['majority']['pct']}%. Accuracy will overstate model quality."
        )

    return risks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze personality datasets.")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/reports")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mbti", "essays", "pandora", "personality_evd"],
        choices=["mbti", "essays", "pandora", "personality_evd"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = REPO_ROOT / args.processed_root
    raw_root = REPO_ROOT / args.raw_root
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis: dict[str, Any] = {"datasets": {}, "raw_sources": {}}

    for dataset in args.datasets:
        records = load_processed(dataset, processed_root)
        stats = analyze_common(dataset, records)
        add_dataset_specific_stats(dataset, records, stats)
        analysis["datasets"][dataset] = stats

    analysis["raw_sources"] = {
        "mbti": analyze_raw_mbti(raw_root),
        "essays": analyze_raw_essays(raw_root),
        "pandora": analyze_raw_pandora(raw_root),
        "personality_evd": analyze_raw_personality_evd(raw_root),
    }

    json_path = output_dir / "dataset_analysis.json"
    md_path = output_dir / "dataset_analysis.md"
    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_report(analysis), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
