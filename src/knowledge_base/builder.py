"""Parse psychology sources into chunks for the knowledge base."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import tiktoken
from loguru import logger

from src.knowledge_base.schema import normalize_metadata

CHUNK_CONFIG = {
    "default": {
        "mode": "atomic",
        "max_tokens": 160,
        "chunk_overlap": 0,
    },
    "by_category": {
        "trait_definition": {"mode": "atomic"},
        "facet_definition": {"mode": "atomic"},
        "state_definition": {"mode": "atomic"},
        "behavioral_marker": {"mode": "atomic"},
        "linguistic_correlate": {"mode": "atomic"},
        "type_description": {"mode": "atomic"},
        "cognitive_function": {"mode": "atomic"},
        "few_shot_example": {
            "mode": "structured_blocks",
            "block_split": ["## STEP 1", "## STEP 2", "## STEP 3"],
        },
    },
    "tokenizer": "cl100k_base",  # tiktoken
    "split_on": ["\n\n", "\n", ". "],
    "min_chunk_size": 50,
}


@dataclass
class KBChunk:
    chunk_id: str
    text: str
    embed_text: str | None = None
    metadata: dict = field(default_factory=dict)


class TextChunker:
    """Splits text into overlapping chunks of a specified token size."""

    def __init__(self, config: dict | None = None):
        self.config = {**CHUNK_CONFIG, **(config or {})}
        self.enc = tiktoken.get_encoding(self.config["tokenizer"])

    def count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    def _has_legacy_config(self) -> bool:
        return "chunk_size" in self.config or "chunk_overlap" in self.config

    def category_config(self, category: str | None = None) -> dict[str, Any]:
        if self._has_legacy_config():
            return {
                "mode": "legacy",
                "max_tokens": self.config.get("chunk_size", 512),
                "chunk_overlap": self.config.get("chunk_overlap", 64),
                "min_chunk_size": self.config.get("min_chunk_size", 50),
            }

        merged = dict(self.config.get("default", {}))
        merged.update((self.config.get("by_category", {}) or {}).get(category or "", {}))
        merged.setdefault("mode", "atomic")
        merged.setdefault("max_tokens", 160)
        merged.setdefault("chunk_overlap", 0)
        merged.setdefault("min_chunk_size", self.config.get("min_chunk_size", 50))
        return merged

    def chunk_text(
        self,
        text: str,
        *,
        chunk_size: int,
        chunk_overlap: int,
        min_size: int,
    ) -> list[str]:
        """Split text into overlapping chunks."""
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
                overlap_words = words[-int(chunk_overlap * len(words) / chunk_size) :]
                current_chunk = " ".join(overlap_words) + " " + para
                current_tokens = self.count_tokens(current_chunk)
            else:
                current_chunk += ("\n\n" if current_chunk else "") + para
                current_tokens += para_tokens

        if current_chunk and self.count_tokens(current_chunk) >= min_size:
            chunks.append(current_chunk.strip())

        return chunks

    def _split_structured_blocks(self, text: str, block_split: list[str] | None = None) -> list[dict[str, Any]]:
        headings = [heading.strip() for heading in (block_split or []) if heading.strip()]
        if not headings:
            return [{"text": text.strip(), "metadata": {}}]

        lines = text.splitlines()
        sections: list[tuple[str | None, list[str]]] = []
        current_heading: str | None = None
        current_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            matched_heading = next((heading for heading in headings if stripped.startswith(heading)), None)
            if matched_heading and current_lines:
                sections.append((current_heading, current_lines))
                current_lines = [line]
                current_heading = matched_heading
            else:
                if matched_heading:
                    current_heading = matched_heading
                current_lines.append(line)

        if current_lines:
            sections.append((current_heading, current_lines))

        if not sections:
            return [{"text": text.strip(), "metadata": {}}]

        blocks: list[dict[str, Any]] = []
        current_block_lines: list[str] = []
        current_block_label = "INPUT+STEP1"

        for heading, section_lines in sections:
            section_text = "\n".join(section_lines).strip()
            if not section_text:
                continue
            if heading in {None, "## STEP 1"}:
                current_block_lines.append(section_text)
                continue
            if current_block_lines:
                blocks.append(
                    {
                        "text": "\n\n".join(current_block_lines).strip(),
                        "metadata": {
                            "block_label": current_block_label,
                            "block_kind": "structured_example_block",
                        },
                    }
                )
                current_block_lines = []
            current_block_label = heading.replace("## ", "").strip()
            blocks.append(
                {
                    "text": section_text,
                    "metadata": {
                        "block_label": current_block_label,
                        "block_kind": "structured_example_block",
                    },
                }
            )

        if current_block_lines:
            blocks.insert(
                0,
                {
                    "text": "\n\n".join(current_block_lines).strip(),
                    "metadata": {
                        "block_label": current_block_label,
                        "block_kind": "structured_example_block",
                    },
                },
            )

        return blocks or [{"text": text.strip(), "metadata": {}}]

    def chunk_record(self, text: str, category: str | None = None) -> list[dict[str, Any]]:
        config = self.category_config(category)
        mode = config.get("mode", "atomic")
        max_tokens = int(config.get("max_tokens", 160))
        chunk_overlap = int(config.get("chunk_overlap", 0))
        min_size = int(config.get("min_chunk_size", self.config.get("min_chunk_size", 50)))

        if mode == "structured_blocks":
            return self._split_structured_blocks(text, config.get("block_split"))

        if mode in {"atomic", "legacy"}:
            if self.count_tokens(text) <= max_tokens:
                return [{"text": text.strip(), "metadata": {}}]
            return [
                {"text": chunk_text, "metadata": {}}
                for chunk_text in self.chunk_text(
                    text,
                    chunk_size=max_tokens,
                    chunk_overlap=chunk_overlap,
                    min_size=min_size,
                )
            ]

        raise ValueError(f"Unsupported chunking mode: {mode}")


class KBBuilder:
    """Parses markdown/JSON KB source files into chunks."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.chunker = TextChunker(self.config.get("chunking"))

    def _build_embed_text(self, text: str, metadata: dict[str, Any]) -> str:
        category = metadata.get("category", "general")
        framework = str(metadata.get("framework", "both")).upper()
        trait = metadata.get("trait")
        pole = metadata.get("pole")
        state_label = metadata.get("state_label")
        facet = metadata.get("facet")
        type_label = metadata.get("type") or metadata.get("target_label")
        function = metadata.get("function")
        associated_traits = metadata.get("associated_traits") or metadata.get("trait_signals") or []
        if isinstance(associated_traits, str):
            associated_traits = [associated_traits]
        trait_signal = ", ".join(str(value) for value in associated_traits if str(value).strip())
        if not trait_signal and trait:
            trait_signal = f"{trait}{'+' if pole == 'HIGH' else '-' if pole == 'LOW' else ''}"

        if category == "state_definition":
            label = state_label or metadata.get("name") or "UnlabeledState"
            prefix = f"State: {label}. Trait signals: {trait_signal or 'unspecified'}. Definition:"
        elif category == "behavioral_marker":
            domain = metadata.get("domain")
            domain_text = f" Domain: {domain}." if domain else ""
            prefix = f"Behavioral marker. Framework: {framework}. Trait: {trait_signal or 'unspecified'}.{domain_text}"
        elif category == "linguistic_correlate":
            prefix = f"Linguistic correlate. Framework: {framework}. Trait: {trait_signal or 'unspecified'}."
        elif category == "trait_definition":
            prefix = f"Trait definition. Framework: {framework}. Trait: {trait or type_label or 'unspecified'}."
        elif category == "facet_definition":
            prefix = (
                f"Facet definition. Trait: {trait or 'unspecified'}. "
                f"Facet: {facet or 'unspecified'}. Pole: {pole or 'BOTH'}."
            )
        elif category == "type_description":
            prefix = f"Type description. Framework: {framework}. Type: {type_label or 'unspecified'}."
        elif category == "cognitive_function":
            prefix = f"Cognitive function. Type: {type_label or 'unspecified'}. Function: {function or 'unspecified'}."
        elif category == "evidence_mapping_example":
            mapping_type = metadata.get("mapping_type", "evidence_mapping")
            if mapping_type == "abstention_rule":
                condition = metadata.get("condition", "insufficient evidence")
                prefix = f"Abstention rule. Framework: {framework}. Trait: {trait or 'any'}. Condition: {condition}."
            elif mapping_type == "aggregation_rule":
                condition = metadata.get("condition", "state trait aggregation")
                prefix = f"Aggregation rule. Framework: {framework}. Condition: {condition}."
            else:
                source_id = metadata.get("source_id", "unknown_source")
                level = pole or metadata.get("level") or "BOTH"
                prefix = (
                    f"Evidence mapping. Framework: {framework}. "
                    f"Trait: {trait or 'unspecified'} {level}. Source: {source_id}."
                )
        elif category == "few_shot_example":
            example_id = metadata.get("example_id", "unspecified")
            block_label = metadata.get("block_label")
            block_text = f" Block: {block_label}." if block_label else ""
            prefix = (
                f"Few-shot example. Framework: {framework}. "
                f"Target: {type_label or 'unspecified'}. Example: {example_id}.{block_text}"
            )
        else:
            prefix = f"Knowledge chunk. Framework: {framework}. Category: {category}."

        return f"{prefix} {text.strip()}".strip()

    def parse_jsonl_source(
        self,
        file_path: Path,
        source_defaults: dict | None = None,
    ) -> Iterator[KBChunk]:
        """Parse a JSONL knowledge base source file."""
        with open(file_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    text = item.get("text", "")
                    metadata = normalize_metadata(item.get("metadata", {}), source_defaults)
                    chunk_id = item.get("chunk_id", f"{file_path.stem}_{i:05d}")
                    chunk_entries = self.chunker.chunk_record(text, metadata.get("category"))
                    if len(chunk_entries) == 1:
                        chunk_metadata = {**metadata, **chunk_entries[0].get("metadata", {})}
                        yield KBChunk(
                            chunk_id=chunk_id,
                            text=chunk_entries[0]["text"],
                            embed_text=self._build_embed_text(chunk_entries[0]["text"], chunk_metadata),
                            metadata=chunk_metadata,
                        )
                        continue

                    for j, entry in enumerate(chunk_entries):
                        chunk_metadata = {
                            **metadata,
                            **entry.get("metadata", {}),
                            "subchunk_index": j,
                        }
                        yield KBChunk(
                            chunk_id=f"{chunk_id}_c{j}",
                            text=entry["text"],
                            embed_text=self._build_embed_text(entry["text"], chunk_metadata),
                            metadata=chunk_metadata,
                        )
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Skipping malformed KB record at {file_path}:{i}: {e}")

    def parse_markdown_source(self, file_path: Path, metadata: dict | None = None) -> Iterator[KBChunk]:
        """Parse a markdown file into chunks."""
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        base_meta = normalize_metadata(
            {"source": file_path.stem, "format": "markdown"},
            metadata,
        )
        category_config = self.chunker.category_config(base_meta.get("category"))
        sub_chunks = self.chunker.chunk_text(
            content,
            chunk_size=category_config.get("max_tokens", 160),
            chunk_overlap=category_config.get("chunk_overlap", 0),
            min_size=category_config.get("min_chunk_size", 50),
        )
        for i, chunk_text in enumerate(sub_chunks):
            yield KBChunk(
                chunk_id=f"{file_path.stem}_{i:05d}",
                text=chunk_text,
                embed_text=self._build_embed_text(chunk_text, base_meta),
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
                "name": source.get("name", path.stem),
                "source_id": source.get("source_id", source.get("name", path.stem)),
                "source": source.get("name", path.stem),
                "framework": source.get("framework", "both"),
                "category": source.get("category", "general"),
            }
            if path.suffix in (".jsonl", ".json"):
                chunks = list(self.parse_jsonl_source(path, metadata))
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
                record = {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "embed_text": chunk.embed_text or chunk.text,
                    "metadata": chunk.metadata,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")
