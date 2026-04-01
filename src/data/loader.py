"""Unified DataLoader for all datasets."""
import json
from pathlib import Path

from loguru import logger


class DataLoader:
    """Loads processed JSONL datasets into memory."""

    def __init__(self, processed_dir: str):
        self.processed_dir = Path(processed_dir)

    def load_split(self, dataset: str, split: str = "train") -> list[dict]:
        """Load a specific split of a dataset."""
        file_path = self.processed_dir / dataset / f"{split}.jsonl"
        if not file_path.exists():
            raise FileNotFoundError(f"Split file not found: {file_path}")

        records = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON: {e}")

        logger.info(f"Loaded {len(records)} records from {file_path}")
        return records

    def load_all_splits(self, dataset: str) -> dict[str, list[dict]]:
        """Load all splits of a dataset."""
        return {
            split: self.load_split(dataset, split)
            for split in ["train", "val", "test"]
        }

    def get_texts_and_labels(
        self,
        records: list[dict],
        label_type: str = "mbti",
        dimension: str | None = None,
    ) -> tuple[list[str], list[str]]:
        """Extract texts and labels from records.

        Args:
            records: List of unified JSONL records
            label_type: "mbti" (16-class), "mbti_dim" (per-dimension binary), "ocean" (per-trait binary)
            dimension: For "mbti_dim", specify "IE", "SN", "TF", or "JP"
                       For "ocean", specify "O", "C", "E", "A", or "N"
        """
        texts, labels = [], []
        for rec in records:
            text = rec.get("text", "")
            if not text:
                continue

            if label_type == "mbti":
                label = rec.get("label_mbti")
            elif label_type == "mbti_dim":
                dims = rec.get("label_mbti_dimensions") or {}
                label = dims.get(dimension)
            elif label_type == "ocean":
                ocean = rec.get("label_ocean") or {}
                label = ocean.get(dimension)
            else:
                label = None

            if label is not None:
                texts.append(text)
                labels.append(label)

        return texts, labels

    def get_statistics(self, records: list[dict]) -> dict:
        """Compute basic statistics about a dataset split."""
        from collections import Counter
        stats = {
            "total": len(records),
            "sources": Counter(r.get("source") for r in records),
            "mbti_distribution": Counter(r.get("label_mbti") for r in records if r.get("label_mbti")),
            "avg_text_length": sum(len(r.get("text", "").split()) for r in records) / max(len(records), 1),
        }
        return stats
