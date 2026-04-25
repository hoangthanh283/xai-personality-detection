"""Index KB chunks into Qdrant."""

from uuid import NAMESPACE_URL, uuid5

import numpy as np
from loguru import logger

from src.knowledge_base.builder import KBChunk

QDRANT_CONFIG = {
    "collection_name": "psych_kb",
    "alias_name": None,
    "recreate_collection": False,
    "vector_size": 768,
    "distance": "Cosine",
    "on_disk": False,
    "hnsw_config": {
        "m": 16,
        "ef_construct": 100,
    },
}


class KBIndexer:
    """Indexes KB chunk embeddings into Qdrant."""

    def __init__(self, config: dict | None = None):
        self.config = {**QDRANT_CONFIG, **(config or {})}
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            url = self.config.get("url", "http://localhost:6333")
            logger.info(f"Connecting to Qdrant at {url}")
            self._client = QdrantClient(url=url, check_compatibility=False)
        return self._client

    def create_collection(self, recreate: bool = False) -> None:
        """Create Qdrant collection."""
        from qdrant_client.models import Distance, HnswConfigDiff, VectorParams

        collection_name = self.config["collection_name"]
        distance_map = {
            "Cosine": Distance.COSINE,
            "Dot": Distance.DOT,
            "Euclid": Distance.EUCLID,
        }
        distance = distance_map.get(self.config["distance"], Distance.COSINE)

        existing = [c.name for c in self.client.get_collections().collections]
        if collection_name in existing:
            if recreate:
                logger.info(f"Deleting existing collection: {collection_name}")
                self.client.delete_collection(collection_name)
            else:
                logger.info(f"Collection '{collection_name}' already exists, skipping creation")
                return

        hnsw = self.config.get("hnsw_config", {})
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self.config["vector_size"],
                distance=distance,
                on_disk=self.config.get("on_disk", False),
            ),
            hnsw_config=HnswConfigDiff(
                m=hnsw.get("m", 16),
                ef_construct=hnsw.get("ef_construct", 100),
            ),
        )
        logger.info(f"Created collection '{collection_name}'")

    def upsert_alias(self) -> None:
        """Point alias_name to collection_name when configured."""
        alias_name = self.config.get("alias_name")
        collection_name = self.config["collection_name"]
        if not alias_name or alias_name == collection_name:
            return
        try:
            from qdrant_client.models import (
                CreateAlias,
                CreateAliasOperation,
                DeleteAlias,
                DeleteAliasOperation,
            )

            aliases = self.client.get_aliases().aliases
            existing = [a.alias_name for a in aliases if a.alias_name == alias_name]
            actions = []
            if existing:
                actions.append(
                    DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias_name))
                )
            actions.append(
                CreateAliasOperation(
                    create_alias=CreateAlias(
                        collection_name=collection_name,
                        alias_name=alias_name,
                    )
                )
            )
            self.client.update_collection_aliases(change_aliases_operations=actions)
            logger.info(f"Updated Qdrant alias '{alias_name}' → '{collection_name}'")
        except Exception as exc:
            logger.warning(f"Could not update Qdrant alias '{alias_name}': {exc}")

    def index_chunks(
        self,
        chunks: list[KBChunk],
        embeddings: np.ndarray,
        batch_size: int = 100,
    ) -> None:
        """Upload chunk embeddings + payloads to Qdrant."""
        from qdrant_client.models import PointStruct

        collection_name = self.config["collection_name"]
        total = len(chunks)
        logger.info(f"Indexing {total} chunks into '{collection_name}'")

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_chunks = chunks[start:end]
            batch_embeddings = embeddings[start:end]

            points = [
                PointStruct(
                    id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                    vector=emb.tolist(),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "embed_text": chunk.embed_text or chunk.text,
                        **chunk.metadata,
                    },
                )
                for chunk, emb in zip(batch_chunks, batch_embeddings)
            ]
            self.client.upsert(collection_name=collection_name, points=points)
            logger.debug(f"Indexed chunks {start}-{end}/{total}")

        logger.info(f"Indexed {total} chunks successfully")

    def get_collection_info(self) -> dict:
        """Get info about the current collection."""
        collection_name = self.config["collection_name"]
        info = self.client.get_collection(collection_name)
        return {
            "name": collection_name,
            "vectors_count": getattr(info, "vectors_count", getattr(info, "points_count", 0)),
            "points_count": getattr(info, "points_count", getattr(info, "vectors_count", 0)),
        }

    def sample_query(self, query_vector: np.ndarray, top_k: int = 3) -> list[dict]:
        """Run a sample query to verify the index."""
        response = self.client.query_points(
            collection_name=self.config["collection_name"],
            query=query_vector.tolist(),
            limit=top_k,
        )
        return [
            {
                "score": r.score,
                "text": r.payload.get("text", ""),
                "chunk_id": r.payload.get("chunk_id", ""),
            }
            for r in response.points
        ]
