#!/usr/bin/env python
"""Build and index the psychology knowledge base.

Usage:
    python scripts/build_kb.py --step all --config configs/kb_config.yaml
    python scripts/build_kb.py --step parse
    python scripts/build_kb.py --step embed
    python scripts/build_kb.py --step index
    python scripts/build_kb.py --step verify
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_base.schema import stable_json_hash, summarize_records, validate_chunk_record
from src.utils.logging_config import setup_logging


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _config_hash(config: dict) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_manifest(
    config: dict, output_dir: Path, *, embeddings_shape: tuple | None = None
) -> None:
    """Write reproducibility metadata for the current built KB."""
    chunks_path = output_dir / "chunks.jsonl"
    records = _read_jsonl(chunks_path)
    validation_errors = {}
    seen = set()
    duplicates = []
    for idx, record in enumerate(records):
        cid = record.get("chunk_id")
        if cid in seen:
            duplicates.append(cid)
        seen.add(cid)
        errors = validate_chunk_record(record)
        if errors:
            validation_errors[f"{cid or idx}"] = errors

    manifest = {
        "kb_version": config.get("kb_version", "psych_kb_unversioned"),
        "collection_name": config.get("qdrant", {}).get("collection_name", "psych_kb"),
        "alias_name": config.get("qdrant", {}).get("alias_name"),
        "embedding": config.get("embedding", {}),
        "chunking": config.get("chunking", {}),
        "sources": config.get("sources", []),
        "config_hash": _config_hash(config),
        "chunks_hash": stable_json_hash(records),
        "summary": summarize_records(records),
        "validation": {
            "num_invalid": len(validation_errors),
            "num_duplicate_chunk_ids": len(duplicates),
            "duplicate_chunk_ids": duplicates[:50],
            "errors": validation_errors,
        },
    }
    if embeddings_shape is not None:
        manifest["embeddings_shape"] = list(embeddings_shape)

    manifest_path = output_dir / "kb_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote KB manifest → {manifest_path}")


def step_parse(config: dict) -> None:
    """Parse source files into chunks."""
    from src.knowledge_base.builder import KBBuilder

    builder = KBBuilder(config)
    source_configs = config.get("sources", [])
    if not source_configs:
        logger.warning("No sources configured in kb_config.yaml")
        # Create minimal demo chunks if no sources found
        logger.info("Creating demo knowledge base chunks...")
        chunks = _create_demo_chunks()
    else:
        chunks = builder.build_from_sources(source_configs)
    output_dir = config.get("output_dir", "data/knowledge_base/")
    output_path = Path(output_dir) / "chunks.jsonl"
    builder.save_chunks(chunks, str(output_path))
    write_manifest(config, Path(output_dir))
    logger.info(f"Parsed {len(chunks)} chunks → {output_path}")


def _create_demo_chunks():
    """Create minimal demonstration KB chunks."""
    from src.knowledge_base.builder import KBChunk

    demo_entries = [
        {
            "chunk_id": "mbti_intp_001",
            "text": (
                "INTPs are known for analytical thinking, love of theory, and desire "
                "to understand underlying principles. They are often described as "
                "logical, precise, and skeptical."
            ),
            "metadata": {
                "source": "demo_kb",
                "framework": "mbti",
                "type": "INTP",
                "category": "type_description",
            },
        },
        {
            "chunk_id": "mbti_infj_001",
            "text": (
                "INFJs are imaginative, empathetic, and deeply value meaningful "
                "relationships. They have a rich inner world and often strive to "
                "understand the deeper meaning of things."
            ),
            "metadata": {
                "source": "demo_kb",
                "framework": "mbti",
                "type": "INFJ",
                "category": "type_description",
            },
        },
        {
            "chunk_id": "ocean_openness_001",
            "text": (
                "Openness to Experience describes the breadth, depth, originality, "
                "and complexity of mental and experiential life. High scorers are "
                "curious, creative, and open to new ideas."
            ),
            "metadata": {
                "source": "demo_kb",
                "framework": "ocean",
                "trait": "O",
                "category": "trait_definition",
            },
        },
        {
            "chunk_id": "ocean_conscientiousness_001",
            "text": (
                "Conscientiousness reflects individual differences in organization, "
                "self-discipline, and dependability. High scorers tend to be "
                "goal-oriented, reliable, and systematic."
            ),
            "metadata": {
                "source": "demo_kb",
                "framework": "ocean",
                "trait": "C",
                "category": "trait_definition",
            },
        },
        {
            "chunk_id": "ocean_extraversion_001",
            "text": (
                "Extraversion is characterized by sociability, assertiveness, and "
                "positive affect. Extraverts gain energy from social interactions, "
                "whereas introverts tend to prefer solitary activities."
            ),
            "metadata": {
                "source": "demo_kb",
                "framework": "both",
                "trait": "E",
                "category": "trait_definition",
            },
        },
        {
            "chunk_id": "behavioral_introversion_001",
            "text": (
                "Introverted behaviors include preferring one-on-one conversations, "
                "needing quiet time to recharge, feeling drained by large social "
                "gatherings, and processing thoughts internally."
            ),
            "metadata": {"source": "demo_kb", "framework": "mbti", "category": "behavioral_marker"},
        },
        {
            "chunk_id": "behavioral_intuition_001",
            "text": (
                "Intuitive individuals prefer abstract ideas over concrete facts, "
                "often follow hunches, enjoy theoretical discussions, and focus on "
                "future possibilities rather than present realities."
            ),
            "metadata": {"source": "demo_kb", "framework": "mbti", "category": "behavioral_marker"},
        },
    ]
    return [
        KBChunk(chunk_id=e["chunk_id"], text=e["text"], metadata=e["metadata"])
        for e in demo_entries
    ]


def step_embed(config: dict) -> None:
    """Embed chunks with Sentence-Transformers."""
    import json

    from src.knowledge_base.builder import KBChunk
    from src.knowledge_base.embedder import KBEmbedder

    output_dir = Path(config.get("output_dir", "data/knowledge_base/"))
    chunks_path = output_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}. Run --step parse first.")

    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            chunks.append(
                KBChunk(
                    chunk_id=data["chunk_id"],
                    text=data["text"],
                    metadata=data.get("metadata", {}),
                )
            )

    embedder = KBEmbedder(config.get("embedding", {}))
    _, embeddings = embedder.embed_chunks(chunks)
    embedder.save_embeddings(embeddings, str(output_dir / "embeddings.npy"))
    write_manifest(config, output_dir, embeddings_shape=embeddings.shape)
    logger.info(f"Embedded {len(chunks)} chunks → {output_dir / 'embeddings.npy'}")


def step_index(config: dict) -> None:
    """Index embeddings into Qdrant."""
    import json

    import numpy as np

    from src.knowledge_base.builder import KBChunk
    from src.knowledge_base.indexer import KBIndexer

    output_dir = Path(config.get("output_dir", "data/knowledge_base/"))
    chunks_path = output_dir / "chunks.jsonl"
    embeddings_path = output_dir / "embeddings.npy"

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}. Run --step embed first.")

    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            chunks.append(
                KBChunk(
                    chunk_id=data["chunk_id"],
                    text=data["text"],
                    metadata=data.get("metadata", {}),
                )
            )

    embeddings = np.load(str(embeddings_path))
    expected_size = config.get("qdrant", {}).get("vector_size")
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunk/embedding count mismatch: {len(chunks)} chunks vs "
            f"{len(embeddings)} embeddings. Run --step embed after --step parse."
        )
    if expected_size and embeddings.shape[1] != expected_size:
        raise ValueError(
            f"Embedding dimension mismatch: embeddings.npy has {embeddings.shape[1]} dims, "
            f"but qdrant.vector_size={expected_size}. Rebuild embeddings with "
            f"{config.get('embedding', {}).get('model', '<configured model>')}."
        )
    qdrant_cfg = config.get("qdrant", {})
    indexer = KBIndexer(qdrant_cfg)
    indexer.create_collection(recreate=False)
    indexer.index_chunks(chunks, embeddings)
    indexer.upsert_alias()
    info = indexer.get_collection_info()
    logger.info(f"Qdrant collection info: {info}")


def step_verify(config: dict) -> None:
    """Verify the KB by running sample queries."""
    from src.knowledge_base.embedder import KBEmbedder
    from src.knowledge_base.indexer import KBIndexer

    qdrant_cfg = config.get("qdrant", {})
    indexer = KBIndexer(qdrant_cfg)
    info = indexer.get_collection_info()
    logger.info(f"Collection info: {info}")

    embedder = KBEmbedder(config.get("embedding", {}))
    sample_queries = [
        "introversion social anxiety quiet",
        "analytical thinking logical decision making",
        "conscientiousness organization planning",
    ]
    for query in sample_queries:
        logger.info(f"\nQuery: '{query}'")
        vec = embedder.encode(query)[0]
        results = indexer.sample_query(vec, top_k=3)
        for r in results:
            logger.info(f"  [score={r['score']:.3f}] {r['text'][:100]}...")


def main():
    parser = argparse.ArgumentParser(description="Build and index psychology knowledge base")
    parser.add_argument("--config", default="configs/kb_config.yaml")
    parser.add_argument(
        "--step", choices=["parse", "embed", "index", "verify", "all"], required=True
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    if args.step in ("parse", "all"):
        step_parse(config)
    if args.step in ("embed", "all"):
        step_embed(config)
    if args.step in ("index", "all"):
        step_index(config)
    if args.step in ("verify",):
        step_verify(config)

    logger.info("Knowledge base build complete!")


if __name__ == "__main__":
    main()
