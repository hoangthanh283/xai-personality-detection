#!/usr/bin/env python
"""Download datasets required by this project.

Automates:
  - MBTI (Kaggle): datasnaek/mbti-type
  - Essays CSV (public mirror)
  - Pandora Big Five public mirror: jingjietan/pandora-big5

Usage:
  python scripts/download_data.py --all
  python scripts/download_data.py --mbti
  python scripts/download_data.py --essays
  python scripts/download_data.py --pandora_big5
"""

import argparse
import os
from pathlib import Path

from loguru import logger

ESSAYS_URL = (
    "https://raw.githubusercontent.com/"
    "jkwieser/personality-prediction-from-text/master/data/training/essays.csv"
)


def maybe_load_env(env_file: str) -> None:
    try:
        from dotenv import load_dotenv
        if Path(env_file).exists():
            load_dotenv(env_file)
            logger.info(f"Loaded environment variables from {env_file}")
    except ImportError:
        logger.warning("python-dotenv not installed; skipping .env loading")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_mbti(raw_root: Path, force: bool = False) -> None:
    mbti_dir = raw_root / "mbti"
    ensure_dir(mbti_dir)
    target_csv = mbti_dir / "mbti_1.csv"
    if target_csv.exists() and not force:
        logger.info(f"MBTI already exists: {target_csv} (skip)")
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise RuntimeError("Kaggle package is required. Install with `uv pip install kaggle`.") from e

    logger.info("Downloading MBTI dataset from Kaggle: datasnaek/mbti-type")
    api = KaggleApi()
    api.authenticate()  # Uses KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json
    api.dataset_download_files("datasnaek/mbti-type", path=str(mbti_dir), unzip=True, quiet=False)

    if not target_csv.exists():
        raise FileNotFoundError(f"Download finished but expected file missing: {target_csv}")
    logger.info(f"MBTI ready: {target_csv}")


def download_essays(raw_root: Path, force: bool = False) -> None:
    essays_dir = raw_root / "essays"
    ensure_dir(essays_dir)
    target_csv = essays_dir / "essays.csv"
    if target_csv.exists() and not force:
        logger.info(f"Essays already exists: {target_csv} (skip)")
        return

    try:
        import requests
    except ImportError as e:
        raise RuntimeError("requests package is required.") from e

    logger.info(f"Downloading Essays CSV from: {ESSAYS_URL}")
    with requests.get(ESSAYS_URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(target_csv, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    if not target_csv.exists() or target_csv.stat().st_size == 0:
        raise FileNotFoundError(f"Download finished but file is missing/empty: {target_csv}")
    logger.info(f"Essays ready: {target_csv}")


def main():
    from download_pandora_big5 import download_pandora_big5

    parser = argparse.ArgumentParser(description="Download datasets for RAG-XPR")
    parser.add_argument("--env_file", default=".env", help="Path to .env file")
    parser.add_argument("--raw_dir", default="data/raw", help="Raw data root directory")
    parser.add_argument("--mbti", action="store_true", help="Download MBTI dataset")
    parser.add_argument("--essays", action="store_true", help="Download Essays dataset")
    parser.add_argument("--pandora_big5", action="store_true", help="Download Pandora Big Five public mirror")
    parser.add_argument("--all", action="store_true", help="Download all supported datasets")
    parser.add_argument("--force", action="store_true", help="Re-download even if target file exists")
    args = parser.parse_args()

    maybe_load_env(args.env_file)
    raw_root = Path(args.raw_dir)
    ensure_dir(raw_root)
    ensure_dir(raw_root / "pandora")
    ensure_dir(raw_root / "pandora_big5")
    ensure_dir(raw_root / "personality_evd")

    do_all = args.all or (not args.mbti and not args.essays and not args.pandora_big5)
    do_mbti = do_all or args.mbti
    do_essays = do_all or args.essays
    do_pandora_big5 = do_all or args.pandora_big5

    if do_mbti:
        download_mbti(raw_root, force=args.force)
    if do_essays:
        download_essays(raw_root, force=args.force)
    if do_pandora_big5:
        download_pandora_big5(raw_root / "pandora_big5", force=args.force)

    logger.info("Download step complete.")
    logger.info("Next: run preprocessing, e.g. `python scripts/preprocess_data.py --dataset mbti`")


if __name__ == "__main__":
    main()
