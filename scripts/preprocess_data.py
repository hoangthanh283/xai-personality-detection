#!/usr/bin/env python
"""Preprocess all datasets into unified JSONL format.

Usage:
    python scripts/preprocess_data.py --all
    python scripts/preprocess_data.py --dataset mbti
    python scripts/preprocess_data.py --verify
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging
from src.utils.seed import set_seed


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def preprocess_mbti(cfg: dict) -> None:
    from src.data.mbti_parser import MBTIParser

    parser = MBTIParser(cfg)
    raw_path = cfg["raw_path"]
    output_dir = cfg["output_dir"]
    if not Path(raw_path).exists():
        logger.error(f"MBTI raw file not found: {raw_path}. Download from Kaggle: datasnaek/mbti-type")
        return
    parser.run(raw_path, output_dir)


def preprocess_essays(cfg: dict) -> None:
    from src.data.essays_parser import EssaysParser

    parser = EssaysParser(cfg)
    raw_path = cfg["raw_path"]
    output_dir = cfg["output_dir"]
    if not Path(raw_path).exists():
        logger.error(f"Essays raw file not found: {raw_path}")
        return
    parser.run(raw_path, output_dir)


def preprocess_pandora(cfg: dict) -> None:
    from src.data.pandora_parser import PandoraParser

    parser = PandoraParser(cfg)
    raw_path = cfg["raw_path"]
    output_dir = cfg["output_dir"]
    if not Path(raw_path).exists():
        logger.error(f"Pandora data dir not found: {raw_path}")
        return
    parser.run(raw_path, output_dir)


def preprocess_personality_evd(cfg: dict) -> None:
    from src.data.personality_evd_parser import PersonalityEvdParser

    parser = PersonalityEvdParser(cfg)
    raw_path = cfg["raw_path"]
    output_dir = cfg["output_dir"]
    if not Path(raw_path).exists():
        logger.error(f"Personality Evd dir not found: {raw_path}")
        return
    parser.run(raw_path, output_dir)


def verify_outputs(config: dict) -> None:
    """Verify outputs: print row counts and sample records."""
    from src.data.loader import DataLoader

    loader = DataLoader("data/processed")

    for dataset_name, cfg in config["datasets"].items():
        output_dir = Path(cfg["output_dir"])
        logger.info(f"\n=== {dataset_name.upper()} ===")
        for split in ["train", "val", "test"]:
            split_file = output_dir / f"{split}.jsonl"
            if split_file.exists():
                records = loader.load_split(dataset_name, split)
                stats = loader.get_statistics(records)
                logger.info(f"  {split}: {stats['total']} records, avg_len={stats['avg_text_length']:.0f} words")
                if split == "train" and records:
                    logger.info(f"  Sample: {json.dumps(records[0])[:200]}...")
            else:
                logger.warning(f"  {split}: NOT FOUND ({split_file})")


def main():
    parser = argparse.ArgumentParser(description="Preprocess personality datasets")
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument(
        "--dataset",
        choices=["mbti", "mbti_uncleaned", "essays", "pandora", "personality_evd"],
    )
    parser.add_argument("--all", action="store_true", help="Process all datasets")
    parser.add_argument("--verify", action="store_true", help="Verify outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)
    config = load_config(args.config)

    if args.verify:
        verify_outputs(config)
        return

    datasets_to_process = []
    if args.all:
        datasets_to_process = ["mbti", "essays", "pandora", "personality_evd"]
    elif args.dataset:
        datasets_to_process = [args.dataset]
    else:
        parser.print_help()
        return

    for dataset_name in datasets_to_process:
        if dataset_name not in config["datasets"]:
            logger.warning(f"Dataset '{dataset_name}' not in config")
            continue
        cfg = config["datasets"][dataset_name]
        logger.info(f"\nProcessing {dataset_name}...")
        try:
            if dataset_name in ("mbti", "mbti_uncleaned"):
                preprocess_mbti(cfg)
            elif dataset_name == "essays":
                preprocess_essays(cfg)
            elif dataset_name == "pandora":
                preprocess_pandora(cfg)
            elif dataset_name == "personality_evd":
                preprocess_personality_evd(cfg)
        except Exception as e:
            logger.error(f"Failed to process {dataset_name}: {e}")
            raise

    logger.info("\nAll preprocessing complete!")


if __name__ == "__main__":
    main()
