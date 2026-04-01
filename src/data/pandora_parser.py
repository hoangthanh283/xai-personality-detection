"""Pandora Reddit dataset → JSONL parser.

Pipeline:
1. Join user metadata with comments
2. For each user: sample up to 100 comments (balanced across subreddits)
3. Clean: remove Reddit markdown ([link](url)), quotes (> ...), bot text
4. Concatenate sampled comments per user
5. Binarize Big Five: score > 3.0 → High, else → Low (per trait)
6. Stratified split: 70/15/15

Output: data/processed/pandora/{train,val,test}.jsonl
"""
import hashlib
import json
import re
import random
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from src.data.preprocessor import PreprocessorConfig, TextPreprocessor

OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
TRAIT_SHORT = {"openness": "O", "conscientiousness": "C", "extraversion": "E", "agreeableness": "A", "neuroticism": "N"}

REDDIT_QUOTE_PATTERN = re.compile(r"^>.*$", re.MULTILINE)
REDDIT_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
REDDIT_BOT_PHRASES = ["I am a bot", "automoderator", "this action was performed automatically"]


def binarize_ocean(score: float, threshold: float = 3.0) -> str:
    return "HIGH" if score > threshold else "LOW"


def clean_reddit_text(text: str) -> str:
    """Remove Reddit-specific markup."""
    text = REDDIT_QUOTE_PATTERN.sub("", text)
    text = REDDIT_LINK_PATTERN.sub(r"\1", text)
    text = re.sub(r"#+\s*", "", text)  # Remove headings
    return text.strip()


def is_bot_comment(text: str) -> bool:
    text_lower = text.lower()
    return any(phrase.lower() in text_lower for phrase in REDDIT_BOT_PHRASES)


class PandoraParser:
    """Parses Pandora Reddit dataset into unified JSONL format."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        preprocessor_cfg = PreprocessorConfig(
            remove_urls=True,
            remove_mentions=True,
            remove_mbti_mentions=True,
            min_words=5,
        )
        self.preprocessor = TextPreprocessor(preprocessor_cfg)
        self.max_comments = self.config.get("max_comments_per_user", 100)
        self.ocean_threshold = self.config.get("ocean_threshold", 3.0)
        self.split_ratio = self.config.get("split_ratio", [0.70, 0.15, 0.15])
        self.seed = self.config.get("seed", 42)
        random.seed(self.seed)

    def parse(self, data_dir: str) -> list[dict]:
        data_path = Path(data_dir)

        # Load users.json
        users_file = data_path / "users.json"
        if not users_file.exists():
            logger.warning(f"users.json not found at {users_file}")
            return []

        users = {}
        with open(users_file, encoding="utf-8") as f:
            for line in f:
                try:
                    u = json.loads(line)
                    users[u["author"]] = u
                except (json.JSONDecodeError, KeyError):
                    continue

        logger.info(f"Loaded {len(users)} users from {users_file}")

        records = []
        comments_dir = data_path / "comments"

        for author, user_data in users.items():
            bigfive = user_data.get("bigfive", {})
            if not all(t in bigfive for t in OCEAN_TRAITS):
                continue

            # Load user comments
            user_comments_file = comments_dir / f"{author}.json"
            if not user_comments_file.exists():
                continue

            comments = []
            try:
                with open(user_comments_file, encoding="utf-8") as f:
                    for line in f:
                        try:
                            c = json.loads(line)
                            body = c.get("body", "").strip()
                            if not is_bot_comment(body):
                                cleaned = clean_reddit_text(body)
                                cleaned = self.preprocessor.clean(cleaned)
                                if self.preprocessor.is_valid(cleaned):
                                    comments.append(cleaned)
                        except (json.JSONDecodeError, KeyError):
                            continue
            except (OSError, IOError):
                continue

            if not comments:
                continue

            # Sample up to max_comments
            sampled = random.sample(comments, min(len(comments), self.max_comments))
            combined_text = " ".join(sampled)

            # Build OCEAN labels
            ocean_labels = {
                TRAIT_SHORT[t]: binarize_ocean(bigfive[t], self.ocean_threshold)
                for t in OCEAN_TRAITS
            }

            record_id = f"pandora_{hashlib.md5(author.encode()).hexdigest()[:8]}"
            record = {
                "id": record_id,
                "text": combined_text,
                "label_mbti": user_data.get("mbti"),
                "label_mbti_dimensions": None,
                "label_ocean": ocean_labels,
                "source": "pandora",
                "split": None,
                "metadata": {
                    "user_id": author,
                    "num_comments": len(sampled),
                    "bigfive_raw": bigfive,
                },
                "evidence_gold": None,
            }
            records.append(record)

        logger.info(f"Parsed {len(records)} Pandora records")
        return records

    def split_and_save(self, records: list[dict], output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        labels = [r["label_ocean"].get("O", "HIGH") for r in records]
        val_size = self.split_ratio[1]
        test_size = self.split_ratio[2]

        train_idx, valtest_idx = train_test_split(
            range(len(records)), test_size=(val_size + test_size),
            stratify=labels, random_state=self.seed,
        )
        valtest_labels = [labels[i] for i in valtest_idx]
        val_idx, test_idx = train_test_split(
            list(valtest_idx), test_size=test_size / (val_size + test_size),
            stratify=valtest_labels, random_state=self.seed,
        )

        splits = {"train": train_idx, "val": val_idx, "test": test_idx}
        for split_name, indices in splits.items():
            split_records = [dict({**records[i], "split": split_name}) for i in indices]
            out_file = output_path / f"{split_name}.jsonl"
            with open(out_file, "w", encoding="utf-8") as f:
                for rec in split_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(split_records)} records to {out_file}")

    def run(self, data_dir: str, output_dir: str) -> None:
        records = self.parse(data_dir)
        self.split_and_save(records, output_dir)
        logger.info("Pandora parsing complete.")
