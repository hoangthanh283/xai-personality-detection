#!/usr/bin/env python3
"""Download pre-trained word embeddings for the LSTM baseline.

Downloads GloVe 6B (822 MB zip, extracts to ~1 GB) into data/embeddings/.

Usage:
    uv run --no-project --python 3.12 --with-requirements requirements.txt \\
        python scripts/download_embeddings.py
    python scripts/download_embeddings.py --dim 100   # smaller, faster
"""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

from loguru import logger

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
EMBED_DIR = Path("data/embeddings")


def _show_progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    pct = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
    mb = downloaded / 1_048_576
    print(f"\r  {pct:5.1f}%  {mb:.0f} MB", end="", flush=True)


def download_glove(dim: int = 300) -> Path:
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    target = EMBED_DIR / f"glove.6B.{dim}d.txt"

    if target.exists():
        logger.info(f"Already exists: {target}")
        return target

    zip_path = EMBED_DIR / "glove.6B.zip"
    if not zip_path.exists():
        logger.info(f"Downloading GloVe 6B from {GLOVE_URL} ...")
        urllib.request.urlretrieve(GLOVE_URL, zip_path, reporthook=_show_progress)
        print()
        logger.info(f"Saved zip to {zip_path}")

    logger.info(f"Extracting glove.6B.{dim}d.txt ...")
    with zipfile.ZipFile(zip_path) as zf:
        filename = f"glove.6B.{dim}d.txt"
        zf.extract(filename, EMBED_DIR)
    logger.info(f"Extracted to {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GloVe word embeddings")
    parser.add_argument(
        "--dim", type=int, choices=[50, 100, 200, 300], default=300, help="Embedding dimension (default: 300)"
    )
    parser.add_argument("--keep_zip", action="store_true", help="Keep the zip file after extraction")
    args = parser.parse_args()

    path = download_glove(args.dim)
    logger.info(f"GloVe {args.dim}d ready at {path}")

    if not args.keep_zip:
        zip_path = EMBED_DIR / "glove.6B.zip"
        if zip_path.exists():
            zip_path.unlink()
            logger.info("Removed zip file")


if __name__ == "__main__":
    sys.exit(main())
