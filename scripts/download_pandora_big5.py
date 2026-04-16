#!/usr/bin/env python
"""Download the public Pandora Big Five mirror from Hugging Face.

Usage:
    python scripts/download_pandora_big5.py
    python scripts/download_pandora_big5.py --raw_dir data/raw/pandora_big5 --force
"""

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

from loguru import logger

DEFAULT_REPO_ID = "jingjietan/pandora-big5"
HF_DATASET_API = "https://huggingface.co/api/datasets/{repo_id}"
HF_RESOLVE_URL = "https://huggingface.co/datasets/{repo_id}/resolve/main/{path}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fetch_repo_files(repo_id: str) -> list[str]:
    api_url = HF_DATASET_API.format(repo_id=repo_id)
    with urllib.request.urlopen(api_url, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))

    return sorted(sibling["rfilename"] for sibling in payload.get("siblings", []))


def stream_download(url: str, target_path: Path) -> None:
    with urllib.request.urlopen(url, timeout=300) as response, open(target_path, "wb") as f:
        shutil.copyfileobj(response, f)


def download_pandora_big5(raw_dir: Path, repo_id: str = DEFAULT_REPO_ID, force: bool = False) -> None:
    ensure_dir(raw_dir)

    repo_files = fetch_repo_files(repo_id)
    target_files = [path for path in repo_files if path.endswith(".parquet") or path == "README.md"]
    if not target_files:
        raise FileNotFoundError(f"No parquet files found in Hugging Face dataset {repo_id}")

    logger.info(f"Downloading {len(target_files)} files from {repo_id} into {raw_dir}")
    for relative_path in target_files:
        target_path = raw_dir / Path(relative_path).name
        if target_path.exists() and not force:
            logger.info(f"File already exists: {target_path} (skip)")
            continue

        download_url = HF_RESOLVE_URL.format(repo_id=repo_id, path=relative_path)
        logger.info(f"Downloading {relative_path}")
        stream_download(download_url, target_path)

        if not target_path.exists() or target_path.stat().st_size == 0:
            raise FileNotFoundError(f"Downloaded file is missing or empty: {target_path}")

    logger.info("Pandora Big Five mirror download complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the pandora_big5 Hugging Face mirror")
    parser.add_argument("--raw_dir", default="data/raw/pandora_big5", help="Target raw-data directory")
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id")
    parser.add_argument("--force", action="store_true", help="Re-download files even if they already exist")
    args = parser.parse_args()

    download_pandora_big5(Path(args.raw_dir), repo_id=args.repo_id, force=args.force)


if __name__ == "__main__":
    main()
