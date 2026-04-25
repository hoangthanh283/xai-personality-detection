#!/usr/bin/env python
"""Build KB enrichment sources from English PersonalityEvd train/valid data."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

TRAIT_SHORT = {
    "openness": "O",
    "conscientiousness": "C",
    "extraversion": "E",
    "agreeableness": "A",
    "neuroticism": "N",
}

TRAIT_NAME = {
    "O": "Openness",
    "C": "Conscientiousness",
    "E": "Extraversion",
    "A": "Agreeableness",
    "N": "Neuroticism",
}

TURN_RE_EN = re.compile(
    r"^Utterance\s+(?P<utt_id>\d+)\s+(?P<speaker>.+?)\s+said:\s*(?P<utterance>.*)$"
)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def collect_long_strings(value: Any, strings: list[str]) -> None:
    if isinstance(value, str):
        text = " ".join(value.split())
        if len(text) >= 30:
            strings.append(text)
    elif isinstance(value, dict):
        for child in value.values():
            collect_long_strings(child, strings)
    elif isinstance(value, list):
        for child in value:
            collect_long_strings(child, strings)


def load_leakage_strings(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []

    heldout_strings: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                collect_long_strings(json.loads(line), heldout_strings)
            except json.JSONDecodeError:
                text = " ".join(line.split())
                if len(text) >= 30:
                    heldout_strings.append(text)
    return heldout_strings


def filter_exact_leakage(
    records: list[dict[str, Any]], heldout_strings: list[str]
) -> tuple[list[dict[str, Any]], int]:
    if not heldout_strings:
        return records, 0

    kept = []
    removed = 0
    for record in records:
        text = " ".join((record.get("text") or "").split())
        if any(heldout in text for heldout in heldout_strings):
            removed += 1
            continue
        kept.append(record)
    return kept, removed


def normalize_level(level: str) -> str:
    text = str(level or "").strip().lower()
    if "high" in text:
        return "HIGH"
    if "low" in text:
        return "LOW"
    return "UNKNOWN"


def parse_utt_ids(raw_ids: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", str(raw_ids or ""))]


def parse_turn(line: str) -> dict[str, Any]:
    match = TURN_RE_EN.match(str(line).strip())
    if not match:
        return {"utt_id": None, "speaker": None, "utterance": str(line).strip()}
    return {
        "utt_id": int(match.group("utt_id")),
        "speaker": match.group("speaker").strip(),
        "utterance": match.group("utterance").strip(),
    }


def reconstruct_quote(
    speaker: str,
    dialogue_id: str,
    utt_ids: list[int],
    dialogues: dict[str, Any],
) -> str:
    if not utt_ids:
        return ""
    raw_turns = dialogues.get(speaker, {}).get("dialogue", {}).get(str(dialogue_id), [])
    turns = [parse_turn(line) for line in raw_turns]
    id_set = set(utt_ids)
    target_turns = [
        turn["utterance"]
        for turn in turns
        if turn["utt_id"] in id_set and turn["speaker"] == speaker
    ]
    if not target_turns:
        target_turns = [turn["utterance"] for turn in turns if turn["utt_id"] in id_set]
    return " ".join(text for text in target_turns if text).strip()


def iter_state_annotations(
    input_dir: Path,
    splits: tuple[str, ...] = ("train", "valid"),
) -> list[dict[str, Any]]:
    dataset_dir = input_dir / "Dataset"
    dialogues = load_json(dataset_dir / "dialogue.json")
    split_files = {
        "train": dataset_dir / "EPR-State Task" / "train_annotation.json",
        "valid": dataset_dir / "EPR-State Task" / "valid_annotation.json",
    }

    rows: list[dict[str, Any]] = []
    for split in splits:
        state_annotations = load_json(split_files[split])
        dataset_split = "val" if split == "valid" else split
        for speaker, speaker_data in state_annotations.items():
            for dialogue_id, traits in speaker_data.get("annotation", {}).items():
                for trait_name, annotation in traits.items():
                    trait = TRAIT_SHORT.get(trait_name)
                    if not trait:
                        continue
                    utt_ids = parse_utt_ids(annotation.get("utt_id", ""))
                    level = normalize_level(annotation.get("level", ""))
                    quote = reconstruct_quote(speaker, dialogue_id, utt_ids, dialogues)
                    rows.append(
                        {
                            "dataset_split": dataset_split,
                            "speaker": speaker,
                            "dialogue_id": str(dialogue_id),
                            "trait": trait,
                            "trait_name": TRAIT_NAME[trait],
                            "level": level,
                            "level_raw": annotation.get("level", ""),
                            "utterance_ids": utt_ids,
                            "quote": quote,
                            "reasoning": annotation.get("nat_lang", ""),
                        }
                    )
    return rows


def _stable_key(row: dict[str, Any]) -> tuple:
    return (
        row["trait"],
        row["level"],
        row["dataset_split"],
        row["speaker"],
        row["dialogue_id"],
        ",".join(str(i) for i in row["utterance_ids"]),
    )


def sample_evidence_rows(
    rows: list[dict[str, Any]],
    *,
    max_positive_per_trait_level: int,
    max_unknown_per_trait: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["level"] in {"HIGH", "LOW"} and row["quote"]:
            grouped[(row["trait"], row["level"])].append(row)
        elif row["level"] == "UNKNOWN":
            grouped[(row["trait"], "UNKNOWN")].append(row)

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for key in sorted(grouped):
        candidates = sorted(grouped[key], key=_stable_key)
        rng.shuffle(candidates)
        limit = max_unknown_per_trait if key[1] == "UNKNOWN" else max_positive_per_trait_level
        sampled.extend(sorted(candidates[:limit], key=_stable_key))
    return sorted(sampled, key=_stable_key)


def build_evidence_mapping_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for idx, row in enumerate(rows):
        level = row["level"]
        trait = row["trait"]
        trait_name = row["trait_name"]
        if level == "UNKNOWN":
            text = (
                "Insufficient evidence mapping from PersonalityEvd train/valid. "
                f"Trait: {trait_name}. Gold label: cannot be determined. "
                "The dialogue does not contain clear behavior for this trait, so the correct "
                "action is to abstain from forcing HIGH or LOW."
            )
            mapping_type = "insufficient_evidence"
            pole = "BOTH"
        else:
            text = (
                "Evidence mapping from PersonalityEvd train/valid. "
                f'Evidence quote: "{row["quote"]}" '
                f"Gold label: {trait_name} {level}. "
                f"Gold reasoning: {row['reasoning']} "
                "Use this as a grounded example of mapping concrete utterances to an OCEAN "
                "trait signal."
            )
            mapping_type = "gold_evidence_mapping"
            pole = level

        records.append(
            {
                "chunk_id": f"personality_evd_evidence_mapping_{idx:05d}",
                "text": text,
                "metadata": {
                    "source": "personality_evd_trainval_evidence",
                    "source_id": "personality_evd_trainval_evidence",
                    "framework": "ocean",
                    "category": "evidence_mapping_example",
                    "mapping_type": mapping_type,
                    "dataset_split": row["dataset_split"],
                    "speaker": row["speaker"],
                    "dialogue_id": row["dialogue_id"],
                    "utterance_ids": row["utterance_ids"],
                    "trait": trait,
                    "pole": pole,
                    "level_raw": row["level_raw"],
                    "language": "en",
                    "split_safety": "train_val_only_no_test_content",
                },
            }
        )
    return records


def abstention_records() -> list[dict[str, Any]]:
    rules = [
        (
            "empty_quote",
            "When the evidence quote is empty or the cited utterance cannot be reconstructed, "
            "the model should mark the trait as insufficiently supported instead of inferring "
            "HIGH or LOW from the label prior.",
            None,
        ),
        (
            "single_weak_cue",
            "A single weak cue such as a short agreement, greeting, thanks, or apology is not "
            "enough to infer a stable Big Five trait unless it is paired with clearer behavior "
            "or repeated across contexts.",
            None,
        ),
        (
            "social_politeness_agreeableness",
            "Polite social formulas can support an Agreeableness state only weakly. Do not infer "
            "high Agreeableness from routine politeness unless the text shows concern, respect, "
            "helping behavior, forgiveness, or trust beyond etiquette.",
            "A",
        ),
        (
            "situational_anxiety_neuroticism",
            "A situational anxious reaction can indicate a high Neuroticism state, but it should "
            "not be treated as stable high Neuroticism without repeated worry, rumination, "
            "emotional volatility, or stress vulnerability.",
            "N",
        ),
        (
            "single_dialogue_trait_limit",
            "One dialogue can justify a state-level judgment, but trait-level judgment requires "
            "a pattern across states or dialogues. If the available dialogue is narrow and "
            "context-specific, lower confidence or abstain.",
            None,
        ),
    ]
    records = []
    for idx, (rule_id, text, trait) in enumerate(rules, start=1):
        records.append(
            {
                "chunk_id": f"abstention_rule_{idx:03d}_{rule_id}",
                "text": f"Abstention rule. {text}",
                "metadata": {
                    "source": "whole_trait_theory_fleeson_2015",
                    "source_id": "whole_trait_theory_fleeson_2015",
                    "framework": "ocean",
                    "category": "evidence_mapping_example",
                    "mapping_type": "abstention_rule",
                    "condition": rule_id,
                    "trait": trait,
                    "pole": "BOTH",
                    "language": "en",
                    "split_safety": "no_dataset_test_content",
                },
            }
        )
    return records


def aggregation_records() -> list[dict[str, Any]]:
    rules = [
        (
            "state_is_short_term",
            "A personality state is a short-term pattern of thought, feeling, or behavior in a "
            "concrete situation. It should be used as evidence for a trait, not confused with "
            "the trait itself.",
        ),
        (
            "trait_requires_distribution",
            "A trait reflects a distribution of states over time. Repeated states across "
            "different contexts provide stronger trait evidence than an isolated event.",
        ),
        (
            "state_trait_disagreement",
            "State-trait disagreement is valid: a person can show a temporary state that points "
            "opposite to their stable trait. Treat disagreement as uncertainty to aggregate, not "
            "as annotation noise.",
        ),
        (
            "repeated_evidence_priority",
            "When inferring traits, prioritize repeated evidence, cross-context consistency, and "
            "facet coverage over vivid but isolated utterances.",
        ),
        (
            "conflicting_states_confidence",
            "When retrieved states point to conflicting trait polarities, reduce confidence and "
            "explain the conflict instead of forcing a single high-confidence conclusion.",
        ),
    ]
    records = []
    for idx, (rule_id, text) in enumerate(rules, start=1):
        records.append(
            {
                "chunk_id": f"aggregation_rule_{idx:03d}_{rule_id}",
                "text": f"Aggregation rule. {text}",
                "metadata": {
                    "source": "whole_trait_theory_fleeson_2015",
                    "source_id": "whole_trait_theory_fleeson_2015",
                    "framework": "ocean",
                    "category": "evidence_mapping_example",
                    "mapping_type": "aggregation_rule",
                    "condition": rule_id,
                    "pole": "BOTH",
                    "language": "en",
                    "split_safety": "no_dataset_test_content",
                },
            }
        )
    return records


BFI2_FACET_ANCHORS = [
    (
        "O",
        "HIGH",
        "IntellectualCuriosity",
        "seeks complex ideas, asks conceptual questions, and enjoys learning for its own sake",
    ),
    (
        "O",
        "LOW",
        "IntellectualCuriosity",
        "prefers familiar facts and practical answers over abstract or theoretical exploration",
    ),
    (
        "O",
        "HIGH",
        "AestheticSensitivity",
        "notices beauty, art, music, style, or sensory detail and responds with interest",
    ),
    (
        "O",
        "LOW",
        "AestheticSensitivity",
        "shows little interest in art, symbolic meaning, aesthetic nuance, or sensory richness",
    ),
    (
        "O",
        "HIGH",
        "CreativeImagination",
        "generates novel possibilities, counterfactuals, stories, or unconventional solutions",
    ),
    (
        "O",
        "LOW",
        "CreativeImagination",
        "prefers proven routines and concrete reality over imagined alternatives",
    ),
    (
        "C",
        "HIGH",
        "Organization",
        "keeps tasks, spaces, schedules, and materials orderly and easy to track",
    ),
    (
        "C",
        "LOW",
        "Organization",
        "loses track of tasks, leaves materials scattered, or works without a stable system",
    ),
    (
        "C",
        "HIGH",
        "Productiveness",
        "starts work promptly, persists through effort, and finishes obligations on time",
    ),
    (
        "C",
        "LOW",
        "Productiveness",
        "delays starting, misses deadlines, or relies on last-minute pressure to act",
    ),
    (
        "C",
        "HIGH",
        "Responsibility",
        "keeps promises, follows through on duties, and considers obligations seriously",
    ),
    (
        "C",
        "LOW",
        "Responsibility",
        "breaks commitments, forgets duties, or treats obligations as optional",
    ),
    (
        "E",
        "HIGH",
        "Sociability",
        "actively seeks social contact, group interaction, conversation, and shared activity",
    ),
    (
        "E",
        "LOW",
        "Sociability",
        "prefers solitude, small circles, quiet settings, or limited social stimulation",
    ),
    (
        "E",
        "HIGH",
        "Assertiveness",
        "speaks up, directs action, initiates decisions, and takes visible social space",
    ),
    (
        "E",
        "LOW",
        "Assertiveness",
        "holds back, avoids taking charge, or lets others lead social interactions",
    ),
    (
        "E",
        "HIGH",
        "EnergyLevel",
        "shows enthusiasm, high activity, positive affect, and rapid engagement with events",
    ),
    (
        "E",
        "LOW",
        "EnergyLevel",
        "describes low stimulation needs, slower pace, quiet recharge, or limited outward energy",
    ),
    (
        "A",
        "HIGH",
        "Compassion",
        "responds to others' distress with care, help, sympathy, and concern for welfare",
    ),
    (
        "A",
        "LOW",
        "Compassion",
        "shows indifference to others' distress or prioritizes self-interest over care",
    ),
    (
        "A",
        "HIGH",
        "Respectfulness",
        "uses tact, cooperation, patience, and non-hostile disagreement in conflict",
    ),
    (
        "A",
        "LOW",
        "Respectfulness",
        "uses harsh criticism, contempt, antagonism, or dismissive conflict behavior",
    ),
    (
        "A",
        "HIGH",
        "Trust",
        "interprets others charitably and assumes goodwill unless there is clear contrary evidence",
    ),
    (
        "A",
        "LOW",
        "Trust",
        "expects selfish motives, questions intentions, or remains guarded and suspicious",
    ),
    (
        "N",
        "HIGH",
        "Anxiety",
        "anticipates threat, worries repeatedly, and feels tense before uncertain outcomes",
    ),
    (
        "N",
        "LOW",
        "Anxiety",
        "stays calm under uncertainty and rarely dwells on possible negative outcomes",
    ),
    (
        "N",
        "HIGH",
        "Depression",
        "describes sadness, hopelessness, guilt, low self-worth, or persistent low mood",
    ),
    (
        "N",
        "LOW",
        "Depression",
        "recovers from setbacks and describes emotional balance or general contentment",
    ),
    (
        "N",
        "HIGH",
        "EmotionalVolatility",
        "reacts intensely, shifts mood quickly, or struggles to calm after provocation",
    ),
    (
        "N",
        "LOW",
        "EmotionalVolatility",
        "keeps composure, regulates irritation, and returns to baseline quickly",
    ),
]


def bfi2_anchor_records() -> list[dict[str, Any]]:
    records = []
    for idx, (trait, pole, facet, marker) in enumerate(BFI2_FACET_ANCHORS, start=1):
        trait_name = TRAIT_NAME[trait]
        text = (
            f"BFI-2 paraphrased behavioral anchor for {trait_name} {pole}, facet {facet}: "
            f"the person {marker}."
        )
        records.append(
            {
                "chunk_id": f"bfi2_item_anchor_{idx:03d}_{trait}_{pole}_{facet}",
                "text": text,
                "metadata": {
                    "source": "bfi2_paraphrased_item_anchors",
                    "source_id": "bfi2_paraphrased_item_anchors",
                    "framework": "ocean",
                    "category": "behavioral_marker",
                    "trait": trait,
                    "pole": pole,
                    "facet": facet,
                    "trait_signals": [f"{trait}{'+' if pole == 'HIGH' else '-'}"],
                    "domain": "bfi2_facet_anchor",
                    "language": "en",
                    "split_safety": "no_dataset_test_content",
                },
            }
        )
    return records


def build_enrichment_sources(
    input_dir: Path,
    output_dir: Path,
    *,
    max_positive_per_trait_level: int = 20,
    max_unknown_per_trait: int = 10,
    seed: int = 42,
    leakage_test_jsonl: Path | None = None,
) -> dict[str, int]:
    rows = iter_state_annotations(input_dir, splits=("train", "valid"))
    sampled = sample_evidence_rows(
        rows,
        max_positive_per_trait_level=max_positive_per_trait_level,
        max_unknown_per_trait=max_unknown_per_trait,
        seed=seed,
    )
    evidence_mapping_records = build_evidence_mapping_records(sampled)
    evidence_mapping_records, excluded_leakage = filter_exact_leakage(
        evidence_mapping_records,
        load_leakage_strings(leakage_test_jsonl),
    )
    static_abstention_records = abstention_records()
    static_aggregation_records = aggregation_records()
    static_bfi2_records = bfi2_anchor_records()

    paths = {
        "personality_evd_evidence_mappings": (
            output_dir / "personality_evd_evidence_mappings.jsonl",
            evidence_mapping_records,
        ),
        "abstention_and_insufficient_evidence": (
            output_dir / "abstention_and_insufficient_evidence.jsonl",
            static_abstention_records,
        ),
        "state_trait_aggregation_rules": (
            output_dir / "state_trait_aggregation_rules.jsonl",
            static_aggregation_records,
        ),
        "bfi2_item_anchors": (output_dir / "bfi2_item_anchors.jsonl", static_bfi2_records),
    }

    counts = {}
    for name, (path, records) in paths.items():
        write_jsonl(path, records)
        counts[name] = len(records)
    counts["personality_evd_evidence_mappings_excluded_leakage"] = excluded_leakage
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build KB enrichment source JSONL files from English PersonalityEvd train/valid"
    )
    parser.add_argument("--input-dir", default="data/raw/personality_evd_en")
    parser.add_argument("--output-dir", default="data/knowledge_base/sources")
    parser.add_argument("--max-positive-per-trait-level", type=int, default=20)
    parser.add_argument("--max-unknown-per-trait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--leakage-test-jsonl",
        default="data/processed/personality_evd/test.jsonl",
        help=(
            "Optional held-out processed JSONL used only for exact-string leakage filtering. "
            "Set to an empty string to disable."
        ),
    )
    args = parser.parse_args()

    leakage_test_jsonl = Path(args.leakage_test_jsonl) if args.leakage_test_jsonl else None
    counts = build_enrichment_sources(
        Path(args.input_dir),
        Path(args.output_dir),
        max_positive_per_trait_level=args.max_positive_per_trait_level,
        max_unknown_per_trait=args.max_unknown_per_trait,
        seed=args.seed,
        leakage_test_jsonl=leakage_test_jsonl,
    )
    for name, count in counts.items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
