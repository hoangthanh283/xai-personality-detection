"""Tests for KB chunking and embed_text generation."""

import json

from src.knowledge_base.builder import KBBuilder
from src.knowledge_base.embedder import KBEmbedder
from src.retrieval.hybrid_search import BM25Retriever


def test_atomic_behavioral_marker_stays_single_chunk(tmp_path):
    source_path = tmp_path / "markers.jsonl"
    record = {
        "text": "Keeps deadlines, uses checklists, and plans tasks carefully before execution.",
        "metadata": {
            "framework": "ocean",
            "category": "behavioral_marker",
            "trait": "C",
            "pole": "HIGH",
            "domain": "self_regulation",
            "source": "project_seed",
        },
    }
    source_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    builder = KBBuilder({"chunking": {"default": {"mode": "atomic", "max_tokens": 160}}})
    chunks = list(
        builder.parse_jsonl_source(
            source_path,
            {"framework": "ocean", "category": "behavioral_marker"},
        )
    )

    assert len(chunks) == 1
    assert chunks[0].text.startswith("Keeps deadlines")
    assert chunks[0].embed_text.startswith("Behavioral marker.")
    assert "Trait: C+" in chunks[0].embed_text


def test_few_shot_structured_chunking_splits_into_step_blocks(tmp_path):
    source_path = tmp_path / "examples.jsonl"
    record = {
        "text": "\n".join(
            [
                "## INPUT TEXT",
                '"I prefer solitude after social gatherings."',
                "",
                "## STEP 1: EVIDENCE EXTRACTION",
                '[{"quote": "prefer solitude", "behavior_type": "social_selectivity"}]',
                "",
                "## STEP 2: STATE IDENTIFICATION",
                '[{"state_label": "SocialWithdrawal"}]',
                "",
                "## STEP 3: TRAIT INFERENCE",
                '{"prediction": "low_E"}',
            ]
        ),
        "metadata": {
            "framework": "ocean",
            "category": "few_shot_example",
            "target_label": "low_E",
            "example_id": "example_01",
            "source": "synthetic_textbook_example",
        },
    }
    source_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    builder = KBBuilder(
        {
            "chunking": {
                "default": {"mode": "atomic", "max_tokens": 160},
                "by_category": {
                    "few_shot_example": {
                        "mode": "structured_blocks",
                        "block_split": ["## STEP 1", "## STEP 2", "## STEP 3"],
                    }
                },
            }
        }
    )
    chunks = list(
        builder.parse_jsonl_source(
            source_path,
            {"framework": "ocean", "category": "few_shot_example"},
        )
    )

    assert len(chunks) == 3
    assert "## INPUT TEXT" in chunks[0].text
    assert "## STEP 1" in chunks[0].text
    assert chunks[0].metadata["block_label"] == "INPUT+STEP1"
    assert chunks[1].metadata["block_label"] == "STEP 2"
    assert chunks[2].metadata["block_label"] == "STEP 3"


def test_evidence_mapping_embed_text_has_trait_and_level_anchors(tmp_path):
    source_path = tmp_path / "evidence_mappings.jsonl"
    record = {
        "text": (
            'Evidence mapping from PersonalityEvd train/valid. Evidence quote: "I planned '
            'the schedule carefully." Gold label: Conscientiousness HIGH.'
        ),
        "metadata": {
            "framework": "ocean",
            "category": "evidence_mapping_example",
            "source_id": "personality_evd_trainval_evidence",
            "mapping_type": "gold_evidence_mapping",
            "trait": "C",
            "pole": "HIGH",
        },
    }
    source_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    builder = KBBuilder({"chunking": {"default": {"mode": "atomic", "max_tokens": 160}}})
    chunks = list(
        builder.parse_jsonl_source(
            source_path,
            {"framework": "ocean", "category": "evidence_mapping_example"},
        )
    )

    assert len(chunks) == 1
    assert chunks[0].embed_text.startswith("Evidence mapping.")
    assert "Trait: C HIGH" in chunks[0].embed_text
    assert "Source: personality_evd_trainval_evidence" in chunks[0].embed_text


def test_embedder_prefers_embed_text(monkeypatch):
    captured = {}

    def fake_embed_texts(self, texts):
        captured["texts"] = texts
        return texts

    monkeypatch.setattr(KBEmbedder, "embed_texts", fake_embed_texts)

    chunks = [
        type(
            "Chunk",
            (),
            {"text": "human text", "embed_text": "anchor text", "chunk_id": "c1", "metadata": {}},
        )()
    ]

    _, embeddings = KBEmbedder({}).embed_chunks(chunks)
    assert embeddings == ["anchor text"]
    assert captured["texts"] == ["anchor text"]


def test_bm25_category_list_filter(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    records = [
        {
            "chunk_id": "state_1",
            "text": "Social withdrawal reflects preference for solitude after socializing.",
            "metadata": {"framework": "ocean", "category": "state_definition"},
        },
        {
            "chunk_id": "trait_1",
            "text": "Extraversion reflects sociability and social energy.",
            "metadata": {"framework": "ocean", "category": "trait_definition"},
        },
    ]
    chunks_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    retriever = BM25Retriever(str(chunks_path))
    results = retriever.search(
        "withdrawal",
        top_k=5,
        framework="ocean",
        category=["state_definition", "behavioral_marker"],
    )

    assert len(results) == 1
    assert results[0].chunk_id == "state_1"
