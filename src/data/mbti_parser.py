"""MBTI CSV → JSONL parser.

Pipeline:
1. Split posts by '|||' delimiter
2. Remove URLs (regex: https?://\\S+)
3. Remove @mentions
4. Strip MBTI type mentions from text (avoid data leakage!)
5. Filter posts < 10 words
6. Aggregate: concatenate user's cleaned posts (max 512 tokens for transformers)
7. Stratified split: 70/15/15 train/val/test

Output: data/processed/mbti/{train,val,test}.jsonl
"""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from src.data.preprocessor import PreprocessorConfig, TextPreprocessor
from src.utils.text_utils import MBTI_TYPES

# MBTI dimension mapping
DIMENSIONS = {
    "IE": ("I", "E"),  # Introversion / Extraversion
    "SN": ("S", "N"),  # Sensing / Intuition
    "TF": ("T", "F"),  # Thinking / Feeling
    "JP": ("J", "P"),  # Judging / Perceiving
}


def parse_mbti_dimensions(mbti_type: str) -> dict[str, str]:
    """Extract dimension labels from a 4-letter MBTI type."""
    if len(mbti_type) != 4:
        return {}
    type_upper = mbti_type.upper()
    return {
        "IE": type_upper[0],
        "SN": type_upper[1],
        "TF": type_upper[2],
        "JP": type_upper[3],
    }


def make_id(text: str, source: str = "mbti") -> str:
    """Generate a stable ID based on content hash."""
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{source}_{h}"


class MBTIParser:
    """Parses MBTI Kaggle CSV dataset into unified JSONL format."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        preprocessor_cfg = PreprocessorConfig(
            remove_urls=True,
            remove_mentions=True,
            remove_mbti_mentions=self.config.get("remove_type_mentions", True),
            min_words=self.config.get("min_words", 10),
            max_words=self.config.get("max_tokens", 512) * 4,  # rough word estimate
        )
        self.preprocessor = TextPreprocessor(preprocessor_cfg)
        self.split_ratio = self.config.get("split_ratio", [0.70, 0.15, 0.15])
        self.seed = self.config.get("seed", 42)

    def parse(self, raw_path: str) -> list[dict]:
        """Parse raw MBTI CSV into list of unified records."""
        df = pd.read_csv(raw_path)
        logger.info(f"Loaded {len(df)} rows from {raw_path}")

        records = []
        skipped = 0
        for _, row in df.iterrows():
            mbti_type = str(row["type"]).strip().upper()
            if mbti_type not in MBTI_TYPES:
                skipped += 1
                continue

            # Split posts by ||| delimiter
            raw_posts = str(row["posts"]).split("|||")

            # Clean each post
            cleaned_posts = []
            for post in raw_posts:
                cleaned = self.preprocessor.clean(post.strip())
                if self.preprocessor.is_valid(cleaned):
                    cleaned_posts.append(cleaned)

            if not cleaned_posts:
                skipped += 1
                continue

            # Concatenate posts for the user
            combined_text = " ".join(cleaned_posts)
            record_id = make_id(combined_text, "mbti")

            record = {
                "id": record_id,
                "text": combined_text,
                "label_mbti": mbti_type,
                "label_mbti_dimensions": parse_mbti_dimensions(mbti_type),
                "label_ocean": None,
                "source": "mbti",
                "split": None,
                "metadata": {
                    "num_posts": len(cleaned_posts),
                    "avg_post_length": sum(len(p.split()) for p in cleaned_posts) / max(len(cleaned_posts), 1),
                },
                "evidence_gold": None,
            }
            records.append(record)

        logger.info(f"Parsed {len(records)} valid records, skipped {skipped}")
        return records

    def split_and_save(self, records: list[dict], output_dir: str) -> None:
        """Stratified split and save to JSONL files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        texts = [r["text"] for r in records]
        labels = [r["label_mbti"] for r in records]

        train_size = self.split_ratio[0]
        val_size = self.split_ratio[1]
        test_size = self.split_ratio[2]

        # First split: train vs (val + test)
        train_idx, valtest_idx = train_test_split(
            range(len(records)),
            test_size=(val_size + test_size),
            stratify=labels,
            random_state=self.seed,
        )

        # Second split: val vs test
        valtest_labels = [labels[i] for i in valtest_idx]
        val_idx, test_idx = train_test_split(
            list(valtest_idx),
            test_size=test_size / (val_size + test_size),
            stratify=valtest_labels,
            random_state=self.seed,
        )

        splits = {"train": train_idx, "val": val_idx, "test": test_idx}
        for split_name, indices in splits.items():
            split_records = []
            for i in indices:
                rec = dict(records[i])
                rec["split"] = split_name
                split_records.append(rec)

            out_file = output_path / f"{split_name}.jsonl"
            with open(out_file, "w", encoding="utf-8") as f:
                for rec in split_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(split_records)} records to {out_file}")

    def run(self, raw_path: str, output_dir: str) -> None:
        """Full pipeline: parse → split → save."""
        records = self.parse(raw_path)
        self.split_and_save(records, output_dir)
        logger.info("MBTI parsing complete.")
