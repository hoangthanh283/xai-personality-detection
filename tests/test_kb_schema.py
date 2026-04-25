"""Tests for KB schema normalization, audit, and offline retrieval QA."""

import json
from pathlib import Path

from scripts.audit_kb import audit_records
from scripts.evaluate_kb_retrieval import evaluate
from src.knowledge_base.schema import normalize_metadata, validate_chunk_record
from src.retrieval.hybrid_search import BM25Retriever


def test_normalize_metadata_adds_provenance_defaults():
    metadata = normalize_metadata(
        {"source": "neo_pi_r_costa_mccrae", "framework": "ocean", "trait": "N"},
        {"category": "facet_definition"},
    )
    assert metadata["source_id"] == "neo_pi_r_costa_mccrae"
    assert metadata["citation"]
    assert metadata["quality_tier"] == "A"
    assert metadata["license_status"] == "citable_paraphrase"
    assert metadata["language"] == "en"


def test_validate_chunk_requires_ocean_trait():
    record = {
        "chunk_id": "bad_ocean",
        "text": "A valid-looking chunk without an OCEAN trait.",
        "metadata": {
            "framework": "ocean",
            "category": "trait_definition",
            "source_id": "project_seed",
            "quality_tier": "C",
            "language": "en",
        },
    }
    assert "ocean chunk missing valid trait" in validate_chunk_record(record)


def test_built_kb_has_no_schema_errors():
    chunks_path = Path("data/knowledge_base/chunks.jsonl")
    records = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
    report = audit_records(records)
    assert report["validation"]["num_invalid"] == 0
    assert not report["validation"]["duplicate_chunk_ids"]
    assert report["summary"]["num_chunks"] >= 700


def test_bm25_retrieval_hits_required_ocean_queries():
    chunks_path = "data/knowledge_base/chunks.jsonl"
    queries = [
        {
            "query": "worries constantly before events",
            "framework": "ocean",
            "expected_trait": "N",
            "expected_pole": "HIGH",
            "expected_category": "state_definition",
        },
        {
            "query": "keeps deadlines and plans tasks carefully",
            "framework": "ocean",
            "expected_trait": "C",
            "expected_pole": "HIGH",
            "expected_category": "state_definition",
        },
        {
            "query": "prefers solitude after socializing",
            "framework": "ocean",
            "expected_trait": "E",
            "expected_pole": "LOW",
            "expected_category": "state_definition",
        },
    ]
    report = evaluate(BM25Retriever(chunks_path), queries, top_k=5)
    assert report["recall_at_5"] == 1.0
