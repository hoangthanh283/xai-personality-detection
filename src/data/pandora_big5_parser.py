"""Pandora Big Five Hugging Face mirror -> JSONL parser.

This adapter is intentionally separate from the official Pandora parser.
It consumes the public parquet mirror hosted at ``jingjietan/pandora-big5``
while preserving the official Pandora code path for the request-gated release.

Pipeline:
1. Read the provided parquet splits from ``data/raw/pandora_big5/``
2. Clean each text with the shared text preprocessor
3. Binarize OCEAN scores on the mirror's 0..100 scale
4. Write unified JSONL files to ``data/processed/pandora_big5/{train,val,test}.jsonl``
"""

import hashlib
import json
from pathlib import Path

from loguru import logger

from src.data.preprocessor import PreprocessorConfig, TextPreprocessor

SPLIT_PATTERNS = {
    "train": "train-*.parquet",
    "validation": "validation-*.parquet",
    "test": "test-*.parquet",
}
OUTPUT_SPLITS = {"train": "train", "validation": "val", "test": "test"}
OCEAN_COLUMNS = {
    "O": "openness",
    "C": "conscientiousness",
    "E": "extraversion",
    "A": "agreeableness",
    "N": "neuroticism",
}


def binarize_ocean(score: float, threshold: float = 50.0) -> str:
    """Convert a 0..100 Big Five score into a binary label."""
    return "HIGH" if float(score) > threshold else "LOW"


class PandoraBig5Parser:
    """Parse the public Pandora Big Five mirror into the repo's unified JSONL format."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        preprocessor_cfg = PreprocessorConfig(
            remove_urls=True,
            remove_mentions=True,
            remove_mbti_mentions=True,
            min_words=self.config.get("min_words", 5),
            max_words=self.config.get("max_words", 2000),
        )
        self.preprocessor = TextPreprocessor(preprocessor_cfg)
        self.ocean_threshold = float(self.config.get("ocean_threshold", 50.0))
        self.log_every = int(self.config.get("log_every", 200000))
        self.max_records_per_split = self.config.get("max_records_per_split")
        self.source_name = self.config.get("source_name", "pandora_big5")
        self.source_repo = self.config.get("source_repo", "jingjietan/pandora-big5")

    def _load_rows(self, parquet_files: list[Path]):
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError("datasets package is required to parse pandora_big5 parquet files.") from e

        return load_dataset(
            "parquet",
            data_files={"rows": [str(path) for path in parquet_files]},
            split="rows",
        )

    def _build_record(self, row: dict, split_name: str) -> dict | None:
        text = str(row.get("text", "")).strip()
        cleaned = self.preprocessor.clean_and_validate(text)
        if cleaned is None:
            return None

        ocean_raw = {name: float(row[col]) for col, name in OCEAN_COLUMNS.items()}
        ocean_labels = {
            trait: binarize_ocean(row[trait], self.ocean_threshold)
            for trait in OCEAN_COLUMNS
        }

        raw_row_id = row.get("__index_level_0__")
        record_key = f"{split_name}:{raw_row_id}:{cleaned}"
        record_id = f"pandora_big5_{hashlib.md5(record_key.encode()).hexdigest()[:12]}"

        return {
            "id": record_id,
            "text": cleaned,
            "label_mbti": None,
            "label_mbti_dimensions": None,
            "label_ocean": ocean_labels,
            "source": self.source_name,
            "split": OUTPUT_SPLITS[split_name],
            "metadata": {
                "source_repo": self.source_repo,
                "raw_split": split_name,
                "raw_row_id": raw_row_id,
                "ptype_raw": row.get("ptype"),
                "bigfive_raw": ocean_raw,
            },
            "evidence_gold": None,
        }

    def _process_split(self, data_dir: Path, source_split: str, output_dir: Path) -> None:
        parquet_files = sorted(data_dir.glob(SPLIT_PATTERNS[source_split]))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found for split '{source_split}' in {data_dir} "
                f"matching {SPLIT_PATTERNS[source_split]}"
            )

        dataset = self._load_rows(parquet_files)
        output_split = OUTPUT_SPLITS[source_split]
        output_file = output_dir / f"{output_split}.jsonl"

        written = 0
        skipped = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for row in dataset:
                record = self._build_record(row, source_split)
                if record is None:
                    skipped += 1
                    continue

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

                if self.max_records_per_split and written >= int(self.max_records_per_split):
                    logger.warning(
                        f"Reached max_records_per_split={self.max_records_per_split} for {source_split}; "
                        "stopping early."
                    )
                    break

                if self.log_every > 0 and written % self.log_every == 0:
                    logger.info(f"{source_split}: wrote {written} records so far")

        logger.info(
            f"Saved {written} records to {output_file} from {len(dataset)} raw rows "
            f"(skipped {skipped})"
        )

    def run(self, data_dir: str, output_dir: str) -> None:
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for split_name in ("train", "validation", "test"):
            logger.info(f"Processing pandora_big5 split: {split_name}")
            self._process_split(data_path, split_name, output_path)

        logger.info("Pandora Big Five parsing complete.")
