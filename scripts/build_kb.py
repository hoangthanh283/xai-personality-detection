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
import sys
from pathlib import Path

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


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
    logger.info(f"Parsed {len(chunks)} chunks → {output_path}")


def _create_demo_chunks():
    """Create minimal demonstration KB chunks."""
    from src.knowledge_base.builder import KBChunk
    demo_entries = [
        {
            "chunk_id": "mbti_intp_001",
            "text": "INTPs are known for their analytical thinking, love of theory, and desire to understand the underlying principles of things. They are often described as logical, precise, and skeptical.",
            "metadata": {"source": "demo_kb", "framework": "mbti", "type": "INTP", "category": "type_description"},
        },
        {
            "chunk_id": "mbti_infj_001",
            "text": "INFJs are imaginative, empathetic, and deeply value meaningful relationships. They have a rich inner world and often strive to understand the deeper meaning of things.",
            "metadata": {"source": "demo_kb", "framework": "mbti", "type": "INFJ", "category": "type_description"},
        },
        {
            "chunk_id": "ocean_openness_001",
            "text": "Openness to Experience describes the breadth, depth, originality, and complexity of an individual's mental and experiential life. High scorers are curious, creative, and open to new ideas.",
            "metadata": {"source": "demo_kb", "framework": "ocean", "trait": "O", "category": "trait_definition"},
        },
        {
            "chunk_id": "ocean_conscientiousness_001",
            "text": "Conscientiousness reflects individual differences in organization, self-discipline, and dependability. High scorers tend to be goal-oriented, reliable, and systematic.",
            "metadata": {"source": "demo_kb", "framework": "ocean", "trait": "C", "category": "trait_definition"},
        },
        {
            "chunk_id": "ocean_extraversion_001",
            "text": "Extraversion is characterized by sociability, assertiveness, and positive affect. Extraverts gain energy from social interactions, whereas introverts tend to prefer solitary activities.",
            "metadata": {"source": "demo_kb", "framework": "both", "trait": "E", "category": "trait_definition"},
        },
        {
            "chunk_id": "behavioral_introversion_001",
            "text": "Introverted behaviors include preferring one-on-one conversations, needing quiet time to recharge, feeling drained by large social gatherings, and processing thoughts internally.",
            "metadata": {"source": "demo_kb", "framework": "mbti", "category": "behavioral_marker"},
        },
        {
            "chunk_id": "behavioral_intuition_001",
            "text": "Intuitive individuals prefer abstract ideas over concrete facts, often follow hunches, enjoy theoretical discussions, and focus on future possibilities rather than present realities.",
            "metadata": {"source": "demo_kb", "framework": "mbti", "category": "behavioral_marker"},
        },
    ]
    return [KBChunk(chunk_id=e["chunk_id"], text=e["text"], metadata=e["metadata"]) for e in demo_entries]


def step_embed(config: dict) -> None:
    """Embed chunks with Sentence-Transformers."""
    import json
    import numpy as np
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
            chunks.append(KBChunk(
                chunk_id=data["chunk_id"],
                text=data["text"],
                metadata=data.get("metadata", {}),
            ))

    embedder = KBEmbedder(config.get("embedding", {}))
    _, embeddings = embedder.embed_chunks(chunks)
    embedder.save_embeddings(embeddings, str(output_dir / "embeddings.npy"))
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
            chunks.append(KBChunk(
                chunk_id=data["chunk_id"],
                text=data["text"],
                metadata=data.get("metadata", {}),
            ))

    embeddings = np.load(str(embeddings_path))
    qdrant_cfg = config.get("qdrant", {})
    indexer = KBIndexer(qdrant_cfg)
    indexer.create_collection(recreate=False)
    indexer.index_chunks(chunks, embeddings)
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
    parser.add_argument("--step", choices=["parse", "embed", "index", "verify", "all"], required=True)
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
