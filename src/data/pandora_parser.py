"""Pandora Reddit dataset (CSV) -> JSONL parser.

Reads the public Pandora corpus:
  - author_profiles.csv  (user metadata + Big Five + MBTI)
  - all_comments_since_2015.csv  (~17.6M comments)

Pipeline:
1. Load author profiles with Big Five (percentile 0-100) and/or MBTI labels
2. Stream comments in chunks, filtering for English and quality
3. Clean Reddit markup, remove bot/quote content
4. Sample up to max_comments_per_user comments (balanced across subreddits)
5. Concatenate sampled comments per user
6. Binarize Big Five: percentile > 50 -> HIGH, else LOW
7. Stratified split: 70/15/15

Output: data/processed/pandora/{train,val,test}.jsonl
"""

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from src.data.preprocessor import PreprocessorConfig, TextPreprocessor

OCEAN_COLS = {
    "openness": "O",
    "conscientiousness": "C",
    "extraversion": "E",
    "agreeableness": "A",
    "neuroticism": "N",
}

VALID_MBTI_TYPES = {
    "INFJ",
    "INFP",
    "INTJ",
    "INTP",
    "ISFJ",
    "ISFP",
    "ISTJ",
    "ISTP",
    "ENFJ",
    "ENFP",
    "ENTJ",
    "ENTP",
    "ESFJ",
    "ESFP",
    "ESTJ",
    "ESTP",
}

REDDIT_QUOTE_PATTERN = re.compile(r"^>.*$", re.MULTILINE)
REDDIT_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
REDDIT_HEADING_PATTERN = re.compile(r"#+\s*")
BOT_SUBSTRINGS = ["i am a bot", "automoderator", "this action was performed automatically"]


def binarize_ocean_percentile(score: float, threshold: float = 50.0) -> str:
    return "HIGH" if float(score) > threshold else "LOW"


def parse_mbti_dimensions(mbti_type: str | None) -> dict | None:
    if not mbti_type or not isinstance(mbti_type, str):
        return None
    mbti_type = mbti_type.strip().upper()
    if mbti_type not in VALID_MBTI_TYPES:
        return None
    return {
        "IE": mbti_type[0],
        "SN": mbti_type[1],
        "TF": mbti_type[2],
        "JP": mbti_type[3],
    }


def clean_reddit_text(text: str) -> str:
    text = REDDIT_QUOTE_PATTERN.sub("", text)
    text = REDDIT_LINK_PATTERN.sub(r"\1", text)
    text = REDDIT_HEADING_PATTERN.sub("", text)
    return text.strip()


def is_bot_comment(text: str) -> bool:
    text_lower = text.lower()
    return any(sub in text_lower for sub in BOT_SUBSTRINGS)


class PandoraParser:
    """Parses Pandora Reddit CSV dataset into unified JSONL format."""

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
        self.max_comments = self.config.get("max_comments_per_user", 100)
        self.ocean_threshold = self.config.get("ocean_threshold", 50.0)
        self.split_ratio = self.config.get("split_ratio", [0.70, 0.15, 0.15])
        self.seed = self.config.get("seed", 42)
        self.chunk_size = self.config.get("chunk_size", 500_000)
        self.filter_english = self.config.get("filter_english", True)
        random.seed(self.seed)

    def _load_author_profiles(self, data_dir: Path) -> pd.DataFrame:
        profiles_path = data_dir / "author_profiles.csv"
        logger.info(f"Loading author profiles from {profiles_path}")
        df = pd.read_csv(profiles_path)
        logger.info(f"Loaded {len(df)} author profiles")

        df["author"] = df["author"].astype(str).str.strip()

        mbti_valid = df["mbti"].apply(lambda x: isinstance(x, str) and x.strip().upper() in VALID_MBTI_TYPES)
        has_ocean = (
            df[["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]].notna().all(axis=1)
        )

        logger.info(f"Authors with valid MBTI: {mbti_valid.sum()}")
        logger.info(f"Authors with all OCEAN: {has_ocean.sum()}")
        logger.info(f"Authors with MBTI or OCEAN: {(mbti_valid | has_ocean).sum()}")

        df["_has_mbti"] = mbti_valid
        df["_has_ocean"] = has_ocean
        return df

    def _stream_comments(self, data_dir: Path, target_authors: set[str]) -> dict[str, list[str]]:
        """Stream comments in chunks, keeping only those from target authors.

        Uses bulk pandas filtering for speed, then per-comment cleaning only
        for surviving rows.
        """
        comments_path = data_dir / "all_comments_since_2015.csv"
        if not comments_path.exists():
            logger.error(f"Comments file not found: {comments_path}")
            return {}

        logger.info(f"Streaming comments from {comments_path} (target: {len(target_authors)} authors)")

        author_comments: dict[str, list[str]] = defaultdict(list)
        total_raw = 0
        total_kept = 0

        for chunk in pd.read_csv(
            comments_path,
            chunksize=self.chunk_size,
            usecols=["author", "body", "lang", "subreddit"],
            dtype={"author": str, "body": str, "lang": str, "subreddit": str},
            na_values=[""],
            keep_default_na=True,
        ):
            total_raw += len(chunk)

            chunk["author"] = chunk["author"].astype(str).str.strip()

            chunk = chunk[chunk["author"].isin(target_authors)]
            if chunk.empty:
                continue

            chunk = chunk.dropna(subset=["body"])
            chunk["body"] = chunk["body"].astype(str)

            if self.filter_english:
                chunk = chunk[chunk["lang"] == "en"]

            chunk = chunk[chunk["body"].str.strip().str.len() > 0]

            for row in chunk.itertuples(index=False):
                body = row.body
                if is_bot_comment(body):
                    continue
                cleaned = clean_reddit_text(body)
                cleaned = self.preprocessor.clean(cleaned)
                if self.preprocessor.is_valid(cleaned):
                    author_comments[row.author].append(cleaned)
                    total_kept += 1

            if total_raw % (self.chunk_size * 5) == 0:
                logger.info(
                    f"  Progress: {total_kept} comments kept from {total_raw} raw rows, {len(author_comments)} authors"
                )

        logger.info(
            f"Finished streaming: {total_kept} comments from {total_raw} raw rows, {len(author_comments)} authors"
        )
        return dict(author_comments)

    def parse(self, data_dir: str) -> list[dict]:
        data_path = Path(data_dir)

        profiles = self._load_author_profiles(data_path)

        eligible = profiles[profiles["_has_mbti"] | profiles["_has_ocean"]]
        target_authors = set(eligible["author"].tolist())
        logger.info(f"Target authors with labels: {len(target_authors)}")

        author_comments = self._stream_comments(data_path, target_authors)

        authors_with_comments = set(author_comments.keys()) & target_authors
        logger.info(f"Authors with labels and comments: {len(authors_with_comments)}")

        eligible_with_comments = eligible[eligible["author"].isin(authors_with_comments)]
        logger.info(f"Processing {len(eligible_with_comments)} authors")

        records = []
        skipped_short = 0

        for _, row in eligible_with_comments.iterrows():
            author = row["author"]
            comments = author_comments.get(author)
            if not comments:
                continue

            mbti_type = None
            mbti_dims = None
            ocean_labels = None
            bigfive_raw = None

            if row["_has_mbti"]:
                mbti_raw = str(row.get("mbti", "")).strip().upper()
                mbti_type = mbti_raw
                mbti_dims = parse_mbti_dimensions(mbti_raw)

            if row["_has_ocean"]:
                ocean_labels = {}
                for col, short in OCEAN_COLS.items():
                    val = row.get(col)
                    if pd.notna(val):
                        ocean_labels[short] = binarize_ocean_percentile(val, self.ocean_threshold)
                bigfive_raw = {short: float(row[col]) for col, short in OCEAN_COLS.items() if pd.notna(row.get(col))}

            sampled = random.sample(comments, min(len(comments), self.max_comments))
            combined_text = " ".join(sampled)
            combined_text = self.preprocessor.clean(combined_text)

            if not self.preprocessor.is_valid(combined_text):
                skipped_short += 1
                continue

            record_id = f"pandora_{hashlib.md5(author.encode()).hexdigest()[:8]}"
            metadata = {
                "user_id": author,
                "num_comments": len(sampled),
                "num_total_comments": len(comments),
            }
            if bigfive_raw:
                metadata["bigfive_raw"] = bigfive_raw

            record = {
                "id": record_id,
                "text": combined_text,
                "label_mbti": mbti_type,
                "label_mbti_dimensions": mbti_dims,
                "label_ocean": ocean_labels,
                "source": "pandora",
                "split": None,
                "metadata": metadata,
                "evidence_gold": None,
            }
            records.append(record)

        mbti_count = sum(1 for r in records if r.get("label_mbti"))
        ocean_count = sum(1 for r in records if r.get("label_ocean"))
        logger.info(
            f"Parsed {len(records)} Pandora records "
            f"({mbti_count} with MBTI, {ocean_count} with OCEAN, "
            f"skipped too short: {skipped_short})"
        )
        return records

    def split_and_save(self, records: list[dict], output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        stratify_labels = self._make_stratify_labels(records)

        val_size = self.split_ratio[1]
        test_size = self.split_ratio[2]
        n = len(records)
        indices = list(range(n))

        if stratify_labels is not None:
            train_idx, valtest_idx = train_test_split(
                indices,
                test_size=(val_size + test_size),
                stratify=stratify_labels,
                random_state=self.seed,
            )
            valtest_strat = [stratify_labels[i] for i in valtest_idx]
            val_idx, test_idx = train_test_split(
                list(valtest_idx),
                test_size=test_size / (val_size + test_size),
                stratify=valtest_strat,
                random_state=self.seed,
            )
        else:
            train_idx, valtest_idx = train_test_split(
                indices,
                test_size=(val_size + test_size),
                random_state=self.seed,
            )
            val_idx, test_idx = train_test_split(
                list(valtest_idx),
                test_size=test_size / (val_size + test_size),
                random_state=self.seed,
            )

        splits = {"train": train_idx, "val": val_idx, "test": test_idx}
        for split_name, split_indices in splits.items():
            split_records = [dict({**records[i], "split": split_name}) for i in split_indices]
            out_file = output_path / f"{split_name}.jsonl"
            with open(out_file, "w", encoding="utf-8") as f:
                for rec in split_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(split_records)} records to {out_file}")

        mbti_count = sum(1 for r in records if r.get("label_mbti"))
        ocean_count = sum(1 for r in records if r.get("label_ocean"))
        logger.info(
            f"Pandora dataset: {len(records)} total records, "
            f"{mbti_count} with MBTI labels, {ocean_count} with OCEAN labels"
        )

    @staticmethod
    def _make_stratify_labels(records: list[dict]) -> list[str] | None:
        """Build stratification labels from MBTI types, falling back to OCEAN.

        Returns None if stratification is not feasible.
        """
        from collections import Counter

        mbti_labels = [r.get("label_mbti") for r in records]
        has_mbti = any(label is not None for label in mbti_labels)

        if has_mbti:
            mbti_nonnull = [label for label in mbti_labels if label is not None]
            counts = Counter(mbti_nonnull)
            min_count = min(counts.values()) if counts else 0
            if min_count >= 2 and len(counts) > 1:
                stratify = []
                fallback_idx = 0
                for label in mbti_labels:
                    if label is not None:
                        stratify.append(label)
                    else:
                        ocean = records[fallback_idx].get("label_ocean") or {}
                        stratify.append(f"ocean_{ocean.get('O', 'UNK')}")
                    fallback_idx += 1
                return stratify

        ocean_labels = [r.get("label_ocean") for r in records]
        has_ocean = any(o is not None for o in ocean_labels)
        if has_ocean:
            stratify = []
            for i, ocean in enumerate(ocean_labels):
                if ocean and "O" in ocean:
                    stratify.append(ocean["O"])
                elif mbti_labels[i] is not None:
                    mbti = mbti_labels[i]
                    dim = mbti[0] if mbti else "X"
                    stratify.append(f"mbti_{dim}")
                else:
                    stratify.append("UNK")
            counts = Counter(stratify)
            min_count = min(counts.values())
            if min_count >= 2:
                return stratify

        return None

    def run(self, data_dir: str, output_dir: str) -> None:
        records = self.parse(data_dir)
        if records:
            self.split_and_save(records, output_dir)
        else:
            logger.error("No records parsed from Pandora dataset")
        logger.info("Pandora parsing complete.")
