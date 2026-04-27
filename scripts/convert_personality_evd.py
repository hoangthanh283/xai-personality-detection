#!/usr/bin/env python
"""Convert PersonalityEvd raw format into parser-compatible split JSONL files.

This converter transforms the downloaded repository format:
  data/raw/personality_evd/Dataset/
    - dialogue.json
    - EPR-State Task/{train_annotation,valid_annotation,test_annotation}.json
    - EPR-Trait Task/trait_annotation.json

into files expected by src/data/personality_evd_parser.py:
  data/raw/personality_evd/{train,val,test}.jsonl

Usage:
  python scripts/convert_personality_evd.py
  python scripts/convert_personality_evd.py --input_dir data/raw/personality_evd --output_dir data/raw/personality_evd
"""

import argparse
import json
import re
from pathlib import Path

from loguru import logger

TRAIT_SHORT = {
    "openness": "O",
    "conscientiousness": "C",
    "extraversion": "E",
    "agreeableness": "A",
    "neuroticism": "N",
}

TURN_RE_ZH = re.compile(r"^第(?P<utt_id>\d+)句(?P<speaker>.+?)说[：:](?P<utterance>.*)$")
TURN_RE_EN = re.compile(r"^Utterance\s+(?P<utt_id>\d+)\s+(?P<speaker>.+?)\s+said:\s*(?P<utterance>.*)$")


def normalize_level(level: str) -> str:
    text = str(level).strip().lower()
    if "高" in text or "high" in text:
        return "HIGH"
    if "低" in text or "low" in text:
        return "LOW"
    return "UNKNOWN"


def parse_turn(line: str) -> dict:
    line = str(line).strip()
    match = TURN_RE_EN.match(line) or TURN_RE_ZH.match(line)
    if not match:
        return {"utt_id": None, "speaker": "UNKNOWN", "utterance": line}
    return {
        "utt_id": int(match.group("utt_id")),
        "speaker": match.group("speaker").strip(),
        "utterance": match.group("utterance").strip(),
    }


def parse_utt_ids(raw_ids: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", str(raw_ids))]


def build_ocean_labels(trait_entry: dict) -> dict:
    labels = {}
    for trait, short in TRAIT_SHORT.items():
        data = trait_entry.get(trait, {})
        labels[short] = normalize_level(data.get("level", "UNKNOWN"))
    return labels


def build_evidence_for_dialogue(
    target_speaker: str,
    dialogue_id: str,
    turns: list[dict],
    state_annotation: dict,
) -> list[dict]:
    evidences = []
    for trait, ann in state_annotation.items():
        trait_short = TRAIT_SHORT.get(trait, trait)
        utt_ids = parse_utt_ids(ann.get("utt_id", ""))
        id_set = set(utt_ids)
        quote_turns = [t["utterance"] for t in turns if t["utt_id"] in id_set and t["speaker"] == target_speaker]
        if not quote_turns:
            quote_turns = [t["utterance"] for t in turns if t["utt_id"] in id_set]
        evidences.append(
            {
                "speaker": target_speaker,
                "dialogue_id": str(dialogue_id),
                "trait": trait_short,
                "level": normalize_level(ann.get("level", "UNKNOWN")),
                "level_raw": ann.get("level", ""),
                "utterance_ids": utt_ids,
                "quote": " ".join(quote_turns).strip(),
                "reasoning": ann.get("nat_lang", ""),
            }
        )
    return evidences


def convert_split(
    split_name: str,
    state_annotations: dict,
    dialogues: dict,
    trait_annotations: dict,
) -> list[dict]:
    rows = []
    for speaker, split_data in state_annotations.items():
        if speaker not in dialogues:
            logger.warning(f"Speaker '{speaker}' not found in dialogue.json; skipping")
            continue

        dialogue_map = dialogues[speaker].get("dialogue", {})
        state_by_dialogue = split_data.get("annotation", {})
        ocean_labels = build_ocean_labels(trait_annotations.get(speaker, {}))

        for dialogue_id, state_ann in state_by_dialogue.items():
            raw_turns = dialogue_map.get(str(dialogue_id), [])
            turns = [parse_turn(t) for t in raw_turns]
            target_turns = [t for t in turns if t["speaker"] == speaker]
            if not target_turns:
                logger.debug(f"No target-speaker turns for {speaker} dialog {dialogue_id}; using all turns")
                target_turns = turns
            rows.append(
                {
                    "dialogue": [{"speaker": t["speaker"], "utterance": t["utterance"]} for t in target_turns],
                    "personality": {speaker: None},
                    "personality_ocean": {speaker: ocean_labels},
                    "evidence": build_evidence_for_dialogue(speaker, str(dialogue_id), turns, state_ann),
                    "metadata": {
                        "target_speaker": speaker,
                        "dialogue_id": str(dialogue_id),
                        "split": split_name,
                    },
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Convert PersonalityEvd raw format to parser-compatible JSONL")
    parser.add_argument(
        "--input_dir",
        default="data/raw/personality_evd_en",
        help="Root containing 'Dataset/' subdir. Defaults to English version.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/raw/personality_evd",
        help="Where to write {train,val,test}.jsonl. Defaults overwrite existing.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    dataset_dir = input_root / "Dataset"

    dialogue_path = dataset_dir / "dialogue.json"
    trait_path = dataset_dir / "EPR-Trait Task" / "trait_annotation.json"
    split_paths = {
        "train": dataset_dir / "EPR-State Task" / "train_annotation.json",
        "val": dataset_dir / "EPR-State Task" / "valid_annotation.json",
        "test": dataset_dir / "EPR-State Task" / "test_annotation.json",
    }

    for required in [dialogue_path, trait_path, *split_paths.values()]:
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")

    dialogues = json.load(open(dialogue_path, encoding="utf-8"))
    trait_annotations = json.load(open(trait_path, encoding="utf-8"))

    total = 0
    for split_name, split_path in split_paths.items():
        state_annotations = json.load(open(split_path, encoding="utf-8"))
        rows = convert_split(split_name, state_annotations, dialogues, trait_annotations)
        out_path = output_root / f"{split_name}.jsonl"
        write_jsonl(out_path, rows)
        total += len(rows)
        logger.info(f"Wrote {len(rows)} rows to {out_path}")

    logger.info(f"Conversion complete. Total rows: {total}")


if __name__ == "__main__":
    main()
