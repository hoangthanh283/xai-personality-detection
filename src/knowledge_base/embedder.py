"""Embed KB chunks with Sentence-BERT."""
from pathlib import Path

import numpy as np
from loguru import logger

from src.knowledge_base.builder import KBChunk

EMBEDDING_CONFIG = {
    "model": "BAAI/bge-base-en-v1.5",
    "batch_size": 64,
    "normalize": True,
}


class KBEmbedder:
    """Embeds text chunks using Sentence-Transformers."""

    def __init__(self, config: dict | None = None):
        self.config = {**EMBEDDING_CONFIG, **(config or {})}
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            model_name = self.config["model"]
            logger.info(f"Loading embedding model: {model_name}")
            self._model = SentenceTransformer(model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts and return numpy array."""
        batch_size = self.config["batch_size"]
        normalize = self.config["normalize"]
        logger.info(f"Embedding {len(texts)} texts with batch_size={batch_size}")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=True,
        )
        return embeddings

    def embed_chunks(self, chunks: list[KBChunk]) -> tuple[list[KBChunk], np.ndarray]:
        """Embed all chunks and return (chunks, embedding_matrix)."""
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)
        return chunks, embeddings

    def save_embeddings(self, embeddings: np.ndarray, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embeddings)
        logger.info(f"Saved embeddings {embeddings.shape} to {output_path}")

    def load_embeddings(self, path: str) -> np.ndarray:
        embeddings = np.load(path)
        logger.info(f"Loaded embeddings {embeddings.shape} from {path}")
        return embeddings

    def encode(self, text: str | list[str]) -> np.ndarray:
        """Encode a single text or list of texts."""
        if isinstance(text, str):
            text = [text]
        return self.embed_texts(text)
