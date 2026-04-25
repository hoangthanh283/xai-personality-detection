"""Hybrid (semantic + BM25) retrieval.

Score = alpha * semantic_score + (1 - alpha) * bm25_score
"""

from loguru import logger

from src.retrieval.kb_retriever import KBChunkResult, KBRetriever


class BM25Retriever:
    """BM25 keyword-based retrieval over KB chunks."""

    def __init__(self, chunks_path: str | None = None):
        self._bm25 = None
        self._corpus: list[dict] = []
        if chunks_path:
            self._load_corpus(chunks_path)

    def _load_corpus(self, path: str) -> None:
        import json

        with open(path, encoding="utf-8") as f:
            self._corpus = [json.loads(line) for line in f if line.strip()]
        logger.info(f"Loaded {len(self._corpus)} chunks for BM25 index")
        self._build_index()

    def _build_index(self) -> None:
        try:
            from rank_bm25 import BM25Okapi

            tokenized = [doc["text"].lower().split() for doc in self._corpus]
            self._bm25 = BM25Okapi(tokenized)
            logger.info("BM25 index built")
        except ImportError:
            logger.warning("rank_bm25 not installed, BM25 retrieval disabled")

    def search(
        self, query: str, top_k: int = 10, framework: str | None = None, category: str | None = None
    ) -> list[KBChunkResult]:
        if self._bm25 is None or not self._corpus:
            return []
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        candidate_indices = []
        for idx, doc in enumerate(self._corpus):
            meta = doc.get("metadata", {})
            if (
                framework
                and framework != "both"
                and meta.get("framework") not in {framework, "both"}
            ):
                continue
            if category and meta.get("category") != category:
                continue
            candidate_indices.append(idx)
        top_indices = sorted(candidate_indices, key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self._corpus[idx]
                results.append(
                    KBChunkResult(
                        chunk_id=doc.get("chunk_id", f"bm25_{idx}"),
                        text=doc.get("text", ""),
                        score=float(scores[idx]),
                        metadata=doc.get("metadata", {}),
                    )
                )
        return results


class HybridRetriever:
    """
    Combines dense (semantic) and sparse (BM25 keyword) retrieval.
    Uses Reciprocal Rank Fusion (RRF) for result merging.
    """

    def __init__(self, config: dict | None = None, chunks_path: str | None = None):
        self.config = config or {}
        self.alpha = self.config.get("alpha", 0.7)
        self.dense_retriever = KBRetriever(self.config)
        self.sparse_retriever = BM25Retriever(chunks_path)

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[KBChunkResult],
        sparse_results: list[KBChunkResult],
        top_k: int,
        k: int = 60,
    ) -> list[KBChunkResult]:
        """Merge results using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        chunk_map: dict[str, KBChunkResult] = {}

        for rank, chunk in enumerate(dense_results, start=1):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0) + self.alpha * (1.0 / (k + rank))
            chunk_map[cid] = chunk

        for rank, chunk in enumerate(sparse_results, start=1):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0) + (1 - self.alpha) * (1.0 / (k + rank))
            chunk_map[cid] = chunk

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result = []
        for cid, fused_score in ranked[:top_k]:
            chunk = chunk_map[cid]
            result.append(
                KBChunkResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=fused_score,
                    metadata=chunk.metadata,
                )
            )
        return result

    def search(
        self,
        query: str,
        top_k: int = 5,
        framework: str | None = None,
        category: str | None = None,
    ) -> list[KBChunkResult]:
        """Hybrid search combining semantic and BM25 results."""
        dense_results = self.dense_retriever.search(
            query, top_k=top_k * 2, framework=framework, category=category
        )
        sparse_results = self.sparse_retriever.search(
            query, top_k=top_k * 2, framework=framework, category=category
        )
        return self._reciprocal_rank_fusion(dense_results, sparse_results, top_k)

    def search_many(
        self,
        queries: list[str],
        top_k: int = 5,
        framework: str | None = None,
        category: str | None = None,
    ) -> list[list[KBChunkResult]]:
        """Batch hybrid search."""
        if not queries:
            return []

        # 1. Batch dense search
        dense_results_list = self.dense_retriever.search_many(
            queries, top_k=top_k * 2, framework=framework, category=category
        )

        # 2. Sequential sparse search (BM25 is very fast on CPU locally)
        results = []
        for i, query in enumerate(queries):
            sparse_results = self.sparse_retriever.search(
                query, top_k=top_k * 2, framework=framework, category=category
            )
            results.append(
                self._reciprocal_rank_fusion(dense_results_list[i], sparse_results, top_k)
            )
        return results
