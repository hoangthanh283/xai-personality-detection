#!/usr/bin/env python
"""Export normalized KB source views into root-level JSONL snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_base.schema import normalize_metadata


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_source_records(config: dict, *, include_legacy: bool = False) -> list[dict]:
    records: list[dict] = []
    for source in config.get("sources", []):
        name = source.get("name", "")
        if not include_legacy and name.endswith("_legacy"):
            continue
        path = Path(source["path"])
        if path.suffix != ".jsonl" or not path.exists():
            continue
        source_defaults = {
            "name": name,
            "source_id": source.get("source_id", name),
            "source": name,
            "framework": source.get("framework", "both"),
            "category": source.get("category", "behavioral_marker"),
        }
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                item = json.loads(line)
                text = item.get("text", "").strip()
                if not text:
                    continue
                metadata = normalize_metadata(item.get("metadata", {}), source_defaults)
                records.append(
                    {
                        "chunk_id": item.get("chunk_id", f"{path.stem}_{i:05d}"),
                        "text": text,
                        "metadata": metadata,
                    }
                )
    return records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export KB snapshots for review and reuse")
    parser.add_argument("--config", default="configs/kb_config.yaml")
    parser.add_argument("--output-dir", default="data/knowledge_base")
    parser.add_argument("--include-legacy", action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    output_dir = Path(args.output_dir)
    rows = load_source_records(cfg, include_legacy=args.include_legacy)

    ocean_rows = [r for r in rows if r["metadata"].get("framework") in {"ocean", "both"}]
    mbti_rows = [r for r in rows if r["metadata"].get("framework") in {"mbti", "both"}]
    cope_rows = [
        r
        for r in rows
        if r["metadata"].get("category") in {"few_shot_example", "evidence_mapping_example"}
    ]

    write_jsonl(output_dir / "psychology_kb_source_dump_v1.jsonl", rows)
    write_jsonl(output_dir / "ocean_knowledge_v1.jsonl", ocean_rows)
    write_jsonl(output_dir / "mbti_knowledge_v1.jsonl", mbti_rows)
    write_jsonl(output_dir / "cope_examples_v1.jsonl", cope_rows)

    print(
        "Exported KB views:"
        f" full={len(rows)} ocean={len(ocean_rows)} mbti={len(mbti_rows)} cope={len(cope_rows)}"
    )


if __name__ == "__main__":
    main()
