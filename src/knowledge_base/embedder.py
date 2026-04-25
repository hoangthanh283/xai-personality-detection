"""Embed KB chunks with Sentence-BERT."""

from pathlib import Path

import numpy as np
from loguru import logger

from src.knowledge_base.builder import KBChunk

EMBEDDING_CONFIG = {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 128,  # Can be larger for the smaller model
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
            import torch
            from sentence_transformers import SentenceTransformer

            model_name = self.config["model"]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Support Apple Silicon (MPS)
            if (
                device == "cpu"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                device = "mps"

            logger.info(f"Loading embedding model: {model_name} on {device}")
            self._model = SentenceTransformer(model_name, device=device)
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
        texts = [chunk.embed_text or chunk.text for chunk in chunks]
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
