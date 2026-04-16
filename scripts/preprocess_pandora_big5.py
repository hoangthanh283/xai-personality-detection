#!/usr/bin/env python
"""Preprocess the public pandora_big5 mirror into unified JSONL files.

Usage:
    python scripts/preprocess_pandora_big5.py
    python scripts/preprocess_pandora_big5.py --raw_path data/raw/pandora_big5 --output_dir data/processed/pandora_big5
"""

import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pandora_big5_parser import PandoraBig5Parser
from src.utils.logging_config import setup_logging
from src.utils.seed import set_seed


def load_dataset_config(config_path: str, dataset_name: str = "pandora_big5") -> dict:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return dict(config["datasets"][dataset_name])


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess the pandora_big5 dataset")
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--raw_path", help="Override raw data directory")
    parser.add_argument("--output_dir", help="Override processed output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    cfg = load_dataset_config(args.config)
    if args.raw_path:
        cfg["raw_path"] = args.raw_path
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    cfg["seed"] = args.seed

    raw_path = cfg["raw_path"]
    output_dir = cfg["output_dir"]
    if not Path(raw_path).exists():
        logger.error(f"pandora_big5 raw directory not found: {raw_path}")
        return

    parser_impl = PandoraBig5Parser(cfg)
    parser_impl.run(raw_path, output_dir)


if __name__ == "__main__":
    main()
