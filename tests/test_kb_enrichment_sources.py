"""Tests for PersonalityEvd English KB enrichment source generation."""

import json
from pathlib import Path

from scripts.build_kb_enrichment_sources import build_enrichment_sources


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_personality_evd_fixture(root: Path) -> None:
    dataset_dir = root / "Dataset"
    annotation_dir = dataset_dir / "EPR-State Task"

    _write_json(
        dataset_dir / "dialogue.json",
        {
            "Alex": {
                "dlg_num": 1,
                "dialogue": {
                    "1": [
                        "Utterance 1 Alex said: I planned the schedule carefully.",
                        "Utterance 2 Alex said: Thanks for helping me.",
                    ]
                },
            }
        },
    )
    _write_json(
        annotation_dir / "train_annotation.json",
        {
            "Alex": {
                "annotation": {
                    "1": {
                        "conscientiousness": {
                            "level": "high level",
                            "utt_id": "1",
                            "nat_lang": "Alex plans carefully and shows task organization.",
                        },
                        "openness": {
                            "level": "cannot be determined",
                            "utt_id": "",
                            "nat_lang": "There is no clear openness evidence.",
                        },
                    }
                }
            }
        },
    )
    _write_json(annotation_dir / "valid_annotation.json", {"Alex": {"annotation": {}}})
    (annotation_dir / "test_annotation.json").write_text("{not valid json", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_build_enrichment_sources_uses_train_valid_without_reading_test(tmp_path):
    input_dir = tmp_path / "personality_evd_en"
    output_dir = tmp_path / "kb_sources"
    _make_personality_evd_fixture(input_dir)

    counts = build_enrichment_sources(
        input_dir,
        output_dir,
        max_positive_per_trait_level=5,
        max_unknown_per_trait=5,
        seed=42,
    )

    assert counts["personality_evd_evidence_mappings"] == 2
    assert counts["abstention_and_insufficient_evidence"] == 5
    assert counts["state_trait_aggregation_rules"] == 5
    assert counts["bfi2_item_anchors"] == 30


def test_positive_examples_require_reconstructed_quote(tmp_path):
    input_dir = tmp_path / "personality_evd_en"
    output_dir = tmp_path / "kb_sources"
    _make_personality_evd_fixture(input_dir)
    build_enrichment_sources(input_dir, output_dir, seed=42)

    records = _read_jsonl(output_dir / "personality_evd_evidence_mappings.jsonl")
    positive = [record for record in records if record["metadata"]["mapping_type"] == "gold_evidence_mapping"]

    assert len(positive) == 1
    assert "I planned the schedule carefully" in positive[0]["text"]
    assert positive[0]["metadata"]["pole"] == "HIGH"
    assert positive[0]["metadata"]["utterance_ids"] == [1]


def test_unknown_examples_preserve_abstention_semantics(tmp_path):
    input_dir = tmp_path / "personality_evd_en"
    output_dir = tmp_path / "kb_sources"
    _make_personality_evd_fixture(input_dir)
    build_enrichment_sources(input_dir, output_dir, seed=42)

    records = _read_jsonl(output_dir / "personality_evd_evidence_mappings.jsonl")
    unknown = [record for record in records if record["metadata"]["mapping_type"] == "insufficient_evidence"]

    assert len(unknown) == 1
    assert "cannot be determined" in unknown[0]["text"]
    assert "abstain" in unknown[0]["text"]
    assert unknown[0]["metadata"]["pole"] == "BOTH"


def test_enrichment_generation_is_deterministic(tmp_path):
    input_dir = tmp_path / "personality_evd_en"
    output_a = tmp_path / "kb_sources_a"
    output_b = tmp_path / "kb_sources_b"
    _make_personality_evd_fixture(input_dir)

    build_enrichment_sources(input_dir, output_a, seed=42)
    build_enrichment_sources(input_dir, output_b, seed=42)

    files = [
        "personality_evd_evidence_mappings.jsonl",
        "abstention_and_insufficient_evidence.jsonl",
        "state_trait_aggregation_rules.jsonl",
        "bfi2_item_anchors.jsonl",
    ]
    for filename in files:
        assert (output_a / filename).read_text(encoding="utf-8") == (output_b / filename).read_text(encoding="utf-8")
