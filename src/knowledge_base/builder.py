"""Parse psychology sources into chunks for the knowledge base.

Sources: MBTI type descriptions, OCEAN trait definitions, psychology textbook excerpts.
Each chunk gets metadata: {source, trait, category, page}
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from loguru import logger

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    logger.warning("tiktoken not installed, falling back to word-based chunking")

CHUNK_CONFIG = {
    "chunk_size": 512,           # tokens
    "chunk_overlap": 64,         # token overlap
    "tokenizer": "cl100k_base",  # tiktoken
    "split_on": ["\n\n", "\n", ". "],
    "min_chunk_size": 50,
}


@dataclass
class KBChunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


class TextChunker:
    """Splits text into overlapping chunks of a specified token size."""

    def __init__(self, config: dict | None = None):
        self.config = {**CHUNK_CONFIG, **(config or {})}
        if HAS_TIKTOKEN:
            self.enc = tiktoken.get_encoding(self.config["tokenizer"])
        else:
            self.enc = None

    def count_tokens(self, text: str) -> int:
        if self.enc:
            return len(self.enc.encode(text))
        return len(text.split())  # word-based fallback

    def chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        chunk_size = self.config["chunk_size"]
        chunk_overlap = self.config["chunk_overlap"]
        min_size = self.config["min_chunk_size"]

        # Split on preferred boundaries
        paragraphs = re.split(r"\n\n+", text)
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            if current_tokens + para_tokens > chunk_size and current_chunk:
                # Save current chunk
                if self.count_tokens(current_chunk) >= min_size:
                    chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-int(chunk_overlap * len(words) / chunk_size):]
                current_chunk = " ".join(overlap_words) + " " + para
                current_tokens = self.count_tokens(current_chunk)
            else:
                current_chunk += ("\n\n" if current_chunk else "") + para
                current_tokens += para_tokens

        if current_chunk and self.count_tokens(current_chunk) >= min_size:
            chunks.append(current_chunk.strip())

        return chunks


class KBBuilder:
    """Parses markdown/JSON KB source files into chunks."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.chunker = TextChunker(self.config.get("chunking"))

    def parse_jsonl_source(self, file_path: Path) -> Iterator[KBChunk]:
        """Parse a JSONL knowledge base source file."""
        with open(file_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    text = item.get("text", "")
                    metadata = item.get("metadata", {})
                    chunk_id = item.get("chunk_id", f"{file_path.stem}_{i:05d}")
                    # Chunk if needed
                    if self.chunker.count_tokens(text) > self.config.get("chunking", {}).get("chunk_size", 512):
                        sub_chunks = self.chunker.chunk_text(text)
                        for j, chunk_text in enumerate(sub_chunks):
                            yield KBChunk(
                                chunk_id=f"{chunk_id}_c{j}",
                                text=chunk_text,
                                metadata=metadata,
                            )
                    else:
                        yield KBChunk(chunk_id=chunk_id, text=text, metadata=metadata)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Skipping malformed KB record at {file_path}:{i}: {e}")

    def parse_markdown_source(self, file_path: Path, metadata: dict | None = None) -> Iterator[KBChunk]:
        """Parse a markdown file into chunks."""
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        sub_chunks = self.chunker.chunk_text(content)
        base_meta = metadata or {"source": file_path.stem, "format": "markdown"}
        for i, chunk_text in enumerate(sub_chunks):
            yield KBChunk(
                chunk_id=f"{file_path.stem}_{i:05d}",
                text=chunk_text,
                metadata=base_meta,
            )

    def build_from_sources(self, source_configs: list[dict]) -> list[KBChunk]:
        """Build all KB chunks from configured source files."""
        all_chunks = []
        for source in source_configs:
            path = Path(source["path"])
            if not path.exists():
                logger.warning(f"KB source file not found: {path}")
                continue
            metadata = {
                "source": source.get("name", path.stem),
                "framework": source.get("framework", "both"),
                "category": source.get("category", "general"),
            }
            if path.suffix in (".jsonl", ".json"):
                chunks = list(self.parse_jsonl_source(path))
            elif path.suffix in (".md", ".txt"):
                chunks = list(self.parse_markdown_source(path, metadata))
            else:
                logger.warning(f"Unsupported file format: {path}")
                continue
            logger.info(f"Parsed {len(chunks)} chunks from {path}")
            all_chunks.extend(chunks)
        logger.info(f"Total KB chunks: {len(all_chunks)}")
        return all_chunks

    def save_chunks(self, chunks: list[KBChunk], output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                record = {"chunk_id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")
