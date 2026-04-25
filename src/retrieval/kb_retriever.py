"""Retrieve psychology definitions from the Qdrant knowledge base."""

from dataclasses import dataclass
from typing import Iterable

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
        embedding_cfg = self.config.get("embedding", {})
        if "embedding_model" in self.config and "model" not in embedding_cfg:
            embedding_cfg = {**embedding_cfg, "model": self.config["embedding_model"]}
        self.embedder = KBEmbedder(embedding_cfg)
        self._qdrant = None
        qdrant_cfg = self.config.get("qdrant", {})
        self.collection_name = (
            self.config.get("collection_name")
            or self.config.get("collection")
            or qdrant_cfg.get("collection_name")
            or "psych_kb"
        )
        self.qdrant_url = (
            self.config.get("qdrant_url") or qdrant_cfg.get("url") or "http://localhost:6333"
        )

    @property
    def qdrant(self):
        if self._qdrant is None:
            from qdrant_client import QdrantClient

            self._qdrant = QdrantClient(url=self.qdrant_url, check_compatibility=False)
        return self._qdrant

    def _normalize_categories(self, category: str | Iterable[str] | None) -> list[str] | None:
        if category is None:
            return None
        if isinstance(category, str):
            return [category]
        values = [str(value) for value in category if str(value).strip()]
        return values or None

    def _build_filter(
        self,
        framework: str | None = None,
        category: str | Iterable[str] | None = None,
    ):
        """Build Qdrant payload filter."""
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        conditions = []
        if framework and framework != "both":
            conditions.append(
                FieldCondition(
                    key="framework",
                    match=MatchAny(any=[framework, "both"]),
                )
            )
        categories = self._normalize_categories(category)
        if categories:
            matcher = (
                MatchValue(value=categories[0])
                if len(categories) == 1
                else MatchAny(any=categories)
            )
            conditions.append(FieldCondition(key="category", match=matcher))
        if conditions:
            return Filter(must=conditions)
        return None

    def _qdrant_query(self, **kwargs):
        """Robust qdrant query handling for both search() and query_points() APIs."""
        client = self.qdrant
        # New API (v1.10.0+)
        if hasattr(client, "query_points"):
            return client.query_points(**kwargs)
        # Old API (v1.1.0 - v1.9.0)
        elif hasattr(client, "search"):
            # Map query_points arguments to search arguments if necessary
            search_kwargs = {
                "collection_name": kwargs.get("collection_name"),
                "query_vector": kwargs.get("query"),
                "query_filter": kwargs.get("query_filter"),
                "limit": kwargs.get("limit"),
                "with_payload": kwargs.get("with_payload", True),
            }
            # Response format differ slightly, but we only need .points
            from dataclasses import dataclass

            @dataclass
            class MockResponse:
                points: list

            res = client.search(**search_kwargs)
            return MockResponse(points=res)
        else:
            raise AttributeError(
                "QdrantClient object has no attribute 'query_points' or 'search'. "
                f"Available: {dir(client)}"
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        framework: str | None = None,
        category: str | Iterable[str] | None = None,
    ) -> list[KBChunkResult]:
        """Search the KB for chunks relevant to a query."""
        # Embed query
        vector = self.embedder.encode(query)
        if vector.ndim > 1:
            vector = vector[0]

        filter_ = self._build_filter(framework, category)
        response = self._qdrant_query(
            collection_name=self.collection_name,
            query=vector.tolist(),
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
            for r in response.points
        ]

    def search_many(
        self,
        queries: list[str],
        top_k: int = 5,
        framework: str | None = None,
        category: str | Iterable[str] | None = None,
    ) -> list[list[KBChunkResult]]:
        """Batch search for multiple queries (optimized)."""
        if not queries:
            return []

        # 1. Batch embed all queries
        query_vectors = self.embedder.encode(queries)
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)

        filter_ = self._build_filter(framework, category)

        # 2. Qdrant doesn't have a simple multi-query semantic search in one call
        # (unless using the Batch API, but query_points is per-point).
        # However, batching the embedding step is the biggest win.
        results = []
        for vector in query_vectors:
            response = self._qdrant_query(
                collection_name=self.collection_name,
                query=vector.tolist(),
                query_filter=filter_,
                limit=top_k,
            )
            results.append(
                [
                    KBChunkResult(
                        chunk_id=r.payload.get("chunk_id", ""),
                        text=r.payload.get("text", ""),
                        score=r.score,
                        metadata={
                            k: v for k, v in r.payload.items() if k not in ("chunk_id", "text")
                        },
                    )
                    for r in response.points
                ]
            )
        return results


def deduplicate_chunks(chunks: list[KBChunkResult]) -> list[KBChunkResult]:
    """Remove duplicate chunks by chunk_id, keeping highest-scoring version."""
    seen = {}
    for chunk in chunks:
        if chunk.chunk_id not in seen or chunk.score > seen[chunk.chunk_id].score:
            seen[chunk.chunk_id] = chunk
    return sorted(seen.values(), key=lambda c: c.score, reverse=True)
