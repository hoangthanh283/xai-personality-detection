"""Retrieve psychology definitions from the Qdrant knowledge base."""
from dataclasses import dataclass

import numpy as np
from loguru import logger

from src.knowledge_base.embedder import KBEmbedder


@dataclass
class KBChunkResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict


class KBRetriever:
    """
    Given evidence sentences, retrieves relevant psychology definitions.

    For each evidence sentence:
    1. Embed the sentence
    2. Query Qdrant with semantic search
    3. Optionally apply metadata filter (e.g., only MBTI definitions)
    4. Return top-k KB chunks
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.embedder = KBEmbedder(self.config.get("embedding"))
        self._qdrant = None
        self.collection_name = self.config.get("collection_name", "psych_kb")
        self.qdrant_url = self.config.get("qdrant_url", "http://localhost:6333")

    @property
    def qdrant(self):
        if self._qdrant is None:
            from qdrant_client import QdrantClient
            self._qdrant = QdrantClient(url=self.qdrant_url)
        return self._qdrant

    def _build_filter(self, framework: str | None = None, category: str | None = None):
        """Build Qdrant payload filter."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
        conditions = []
        if framework and framework != "both":
            conditions.append(
                FieldCondition(
                    key="framework",
                    match=MatchAny(any=[framework, "both"]),
                )
            )
        if category:
            conditions.append(
                FieldCondition(key="category", match=MatchValue(value=category))
            )
        if conditions:
            return Filter(must=conditions)
        return None

    def search(
        self,
        query: str,
        top_k: int = 5,
        framework: str | None = None,
        category: str | None = None,
    ) -> list[KBChunkResult]:
        """Search the KB for chunks relevant to a query."""
        # Embed query
        vector = self.embedder.encode(query)
        if vector.ndim > 1:
            vector = vector[0]

        filter_ = self._build_filter(framework, category)
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=vector.tolist(),
            query_filter=filter_,
            limit=top_k,
        )
        return [
            KBChunkResult(
                chunk_id=r.payload.get("chunk_id", ""),
                text=r.payload.get("text", ""),
                score=r.score,
                metadata={k: v for k, v in r.payload.items() if k not in ("chunk_id", "text")},
            )
            for r in results
        ]

    def search_many(
        self,
        queries: list[str],
        top_k: int = 5,
        framework: str | None = None,
        category: str | None = None,
    ) -> list[list[KBChunkResult]]:
        """Batch search for multiple queries."""
        return [self.search(q, top_k=top_k, framework=framework, category=category) for q in queries]


def deduplicate_chunks(chunks: list[KBChunkResult]) -> list[KBChunkResult]:
    """Remove duplicate chunks by chunk_id, keeping highest-scoring version."""
    seen = {}
    for chunk in chunks:
        if chunk.chunk_id not in seen or chunk.score > seen[chunk.chunk_id].score:
            seen[chunk.chunk_id] = chunk
    return sorted(seen.values(), key=lambda c: c.score, reverse=True)
