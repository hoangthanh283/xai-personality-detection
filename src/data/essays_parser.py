"""Essays dataset (Pennebaker & King) → JSONL parser.

Pipeline:
1. Parse CSV with proper encoding (latin-1)
2. Map y/n → HIGH/LOW for each trait
3. Minimal cleaning: fix encoding artifacts, normalize whitespace
4. Stratified split: 70/15/15

Output: data/processed/essays/{train,val,test}.jsonl
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from src.data.preprocessor import PreprocessorConfig, TextPreprocessor

OCEAN_COLUMNS = {
    "cOPN": "O",  # Openness
    "cCON": "C",  # Conscientiousness
    "cEXT": "E",  # Extraversion
    "cAGR": "A",  # Agreeableness
    "cNEU": "N",  # Neuroticism
}


def label_to_binary(label: str) -> str:
    return "HIGH" if str(label).lower() == "y" else "LOW"


class EssaysParser:
    """Parses Essays Big Five dataset into unified JSONL format."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        preprocessor_cfg = PreprocessorConfig(
            remove_urls=False,  # Essays are formal — minimal cleaning
            remove_mentions=False,
            remove_mbti_mentions=False,
            min_words=self.config.get("min_words", 10),
        )
        self.preprocessor = TextPreprocessor(preprocessor_cfg)
        self.encoding = self.config.get("encoding", "latin-1")
        self.split_ratio = self.config.get("split_ratio", [0.70, 0.15, 0.15])
        self.seed = self.config.get("seed", 42)

    def parse(self, raw_path: str) -> list[dict]:
        df = pd.read_csv(raw_path, encoding=self.encoding)
        logger.info(f"Loaded {len(df)} essays from {raw_path}")

        records = []
        skipped = 0
        for _, row in df.iterrows():
            text = str(row.get("TEXT", "")).strip()
            cleaned = self.preprocessor.clean(text)
            if not self.preprocessor.is_valid(cleaned):
                skipped += 1
                continue

            # Extract OCEAN labels
            ocean_labels = {}
            for col, trait in OCEAN_COLUMNS.items():
                if col in row:
                    ocean_labels[trait] = label_to_binary(row[col])

            user_id = str(row.get("#AUTHID", ""))
            record_id = f"essays_{hashlib.md5(user_id.encode()).hexdigest()[:8]}"

            record = {
                "id": record_id,
                "text": cleaned,
                "label_mbti": None,
                "label_mbti_dimensions": None,
                "label_ocean": ocean_labels,
                "source": "essays",
                "split": None,
                "metadata": {
                    "user_id": user_id,
                    "num_words": len(cleaned.split()),
                },
                "evidence_gold": None,
            }
            records.append(record)

        logger.info(f"Parsed {len(records)} valid records, skipped {skipped}")
        return records

    def split_and_save(self, records: list[dict], output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Stratify on Openness as proxy label
        labels = [r["label_ocean"].get("O", "HIGH") for r in records]
        val_size = self.split_ratio[1]
        test_size = self.split_ratio[2]

        train_idx, valtest_idx = train_test_split(
            range(len(records)),
            test_size=(val_size + test_size),
            stratify=labels,
            random_state=self.seed,
        )
        valtest_labels = [labels[i] for i in valtest_idx]
        val_idx, test_idx = train_test_split(
            list(valtest_idx),
            test_size=test_size / (val_size + test_size),
            stratify=valtest_labels,
            random_state=self.seed,
        )

        splits = {"train": train_idx, "val": val_idx, "test": test_idx}
        for split_name, indices in splits.items():
            split_records = [dict({**records[i], "split": split_name}) for i in indices]
            out_file = output_path / f"{split_name}.jsonl"
            with open(out_file, "w", encoding="utf-8") as f:
                for rec in split_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(split_records)} records to {out_file}")

    def run(self, raw_path: str, output_dir: str) -> None:
        records = self.parse(raw_path)
        self.split_and_save(records, output_dir)
        logger.info("Essays parsing complete.")
