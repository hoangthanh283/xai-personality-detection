#!/usr/bin/env python
"""Audit the psychology KB schema, provenance, coverage, and leakage risk."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_base.schema import summarize_records, validate_chunk_record


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def audit_records(records: list[dict]) -> dict:
    errors = {}
    chunk_ids = Counter(r.get("chunk_id") for r in records)
    duplicate_ids = sorted(cid for cid, count in chunk_ids.items() if cid and count > 1)
    token_like_lengths = [len((r.get("text") or "").split()) for r in records]
    short = [r.get("chunk_id") for r in records if len((r.get("text") or "").split()) < 12]
    long = [r.get("chunk_id") for r in records if len((r.get("text") or "").split()) > 450]

    for idx, record in enumerate(records):
        validation = validate_chunk_record(record)
        if validation:
            errors[record.get("chunk_id") or f"row_{idx}"] = validation

    trait_category = defaultdict(Counter)
    for record in records:
        meta = record.get("metadata", {})
        if meta.get("framework") == "ocean":
            trait = meta.get("trait") or "UNSPECIFIED"
            trait_category[trait][meta.get("category", "<missing>")] += 1

    return {
        "summary": summarize_records(records),
        "validation": {
            "num_invalid": len(errors),
            "errors": errors,
            "duplicate_chunk_ids": duplicate_ids,
            "short_chunks": short[:100],
            "long_chunks": long[:100],
            "min_words": min(token_like_lengths) if token_like_lengths else 0,
            "max_words": max(token_like_lengths) if token_like_lengths else 0,
        },
        "ocean_trait_category_coverage": {trait: dict(counts) for trait, counts in sorted(trait_category.items())},
    }


def _collect_strings(value, strings: list[str]) -> None:
    if isinstance(value, str):
        text = " ".join(value.split())
        if len(text) >= 30:
            strings.append(text)
    elif isinstance(value, dict):
        for child in value.values():
            _collect_strings(child, strings)
    elif isinstance(value, list):
        for child in value:
            _collect_strings(child, strings)


def check_exact_leakage(records: list[dict], dataset_path: Path) -> dict:
    """Check whether KB chunks copy long strings from a held-out dataset file."""
    if not dataset_path.exists():
        return {"checked": False, "reason": f"missing file: {dataset_path}", "matches": []}

    heldout_strings: list[str] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                _collect_strings(json.loads(line), heldout_strings)
            except json.JSONDecodeError:
                text = " ".join(line.split())
                if len(text) >= 30:
                    heldout_strings.append(text)

    matches = []
    for record in records:
        text = " ".join((record.get("text") or "").split())
        if not text:
            continue
        for heldout in heldout_strings:
            if heldout in text:
                matches.append(
                    {
                        "chunk_id": record.get("chunk_id"),
                        "matched_text": heldout[:200],
                    }
                )
                break
    return {
        "checked": True,
        "dataset_path": str(dataset_path),
        "num_heldout_strings": len(heldout_strings),
        "matches": matches,
    }


def write_markdown(report: dict, path: Path) -> None:
    summary = report["summary"]
    validation = report["validation"]
    lines = [
        "# Psychology KB Audit",
        "",
        f"- Chunks: {summary['num_chunks']}",
        f"- Invalid chunks: {validation['num_invalid']}",
        f"- Duplicate chunk ids: {len(validation['duplicate_chunk_ids'])}",
        f"- Word length range: {validation['min_words']}–{validation['max_words']}",
        "",
        "## Quality Tier",
    ]
    for key, value in sorted(summary["quality_tier"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Category"])
    for key, value in sorted(summary["category"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## OCEAN Trait Coverage"])
    for trait, counts in report["ocean_trait_category_coverage"].items():
        total = sum(counts.values())
        lines.append(f"- {trait}: {total} chunks ({dict(counts)})")
    if validation["errors"]:
        lines.extend(["", "## Validation Errors"])
        for chunk_id, errors in validation["errors"].items():
            lines.append(f"- {chunk_id}: {', '.join(errors)}")
    leakage = report.get("leakage", {})
    if leakage:
        lines.extend(["", "## Held-Out Leakage Check"])
        lines.append(f"- Checked: {leakage.get('checked')}")
        if leakage.get("dataset_path"):
            lines.append(f"- Dataset: {leakage['dataset_path']}")
        lines.append(f"- Exact matches: {len(leakage.get('matches', []))}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit built psychology KB chunks")
    parser.add_argument("--chunks", default="data/knowledge_base/chunks.jsonl")
    parser.add_argument("--json-output", default="data/knowledge_base/reports/kb_audit.json")
    parser.add_argument("--md-output", default="data/knowledge_base/reports/kb_audit.md")
    parser.add_argument("--leakage-test-jsonl", default="data/processed/personality_evd/test.jsonl")
    args = parser.parse_args()

    records = load_jsonl(Path(args.chunks))
    report = audit_records(records)
    report["leakage"] = check_exact_leakage(records, Path(args.leakage_test_jsonl))

    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, Path(args.md_output))

    invalid = report["validation"]["num_invalid"]
    duplicates = len(report["validation"]["duplicate_chunk_ids"])
    leaks = len(report["leakage"].get("matches", []))
    print(f"KB audit complete: {len(records)} chunks, {invalid} invalid, {duplicates} duplicates")
    if leaks:
        print(f"Potential held-out leakage matches: {leaks}")
    if invalid or duplicates or leaks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
