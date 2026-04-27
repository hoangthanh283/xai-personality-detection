#!/usr/bin/env python
"""EDA-style data augmentation for personality datasets (small-train rescue).

Applies four cheap text augmentation operations from Wei & Zou (2019)
"EDA: Easy Data Augmentation Techniques for Boosting Performance on Text
Classification Tasks":
  - Synonym Replacement (SR) via WordNet
  - Random Insertion (RI) of WordNet synonyms
  - Random Swap (RS) of two words
  - Random Deletion (RD) at probability p

Generates `--n_aug` augmented copies per training sample, applying each
operation with a fraction of words modified. Validation/test splits stay
untouched (eval honesty).

Targeted at Pandora (1087 train) and Essays (1726 train) where fine-tune
RoBERTa overfits due to sample size. MBTI (6071 train) and PerEvd (1292,
already evidence-grounded) skipped by default.

Usage:
    uv run --no-project --python 3.12 --with-requirements requirements.txt \
        python scripts/augment_data.py --dataset pandora --n_aug 2

    # Output: data/processed/pandora_augmented/{train,val,test}.jsonl
    # Train file = original + n_aug × original = (n_aug+1) × original size
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import nltk
from loguru import logger
from nltk.corpus import wordnet

# EDA core
_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "as",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
}


def _get_synonyms(word: str) -> list[str]:
    """Look up synonyms via WordNet; return lemma names different from the word."""
    syns = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower() and name.isascii():
                syns.add(name)
    return list(syns)


def synonym_replacement(words: list[str], n: int) -> list[str]:
    candidates = [w for w in words if w.lower() not in _STOPWORDS]
    random.shuffle(candidates)
    out = list(words)
    replaced = 0
    for w in candidates:
        if replaced >= n:
            break
        syns = _get_synonyms(w)
        if not syns:
            continue
        choice = random.choice(syns)
        out = [choice if x == w else x for x in out]
        replaced += 1
    return out


def random_insertion(words: list[str], n: int) -> list[str]:
    out = list(words)
    for _ in range(n):
        candidates = [w for w in out if w.lower() not in _STOPWORDS]
        if not candidates:
            break
        random.shuffle(candidates)
        for w in candidates:
            syns = _get_synonyms(w)
            if syns:
                out.insert(random.randint(0, len(out)), random.choice(syns))
                break
    return out


def random_swap(words: list[str], n: int) -> list[str]:
    out = list(words)
    if len(out) < 2:
        return out
    for _ in range(n):
        i, j = random.sample(range(len(out)), 2)
        out[i], out[j] = out[j], out[i]
    return out


def random_deletion(words: list[str], p: float) -> list[str]:
    if len(words) <= 1:
        return list(words)
    out = [w for w in words if random.random() > p]
    if not out:
        return [random.choice(words)]
    return out


def eda_augment(text: str, alpha: float = 0.1) -> str:
    """Apply one EDA op (chosen uniformly) with fraction alpha of words modified."""
    words = text.split()
    if not words:
        return text
    n = max(1, int(alpha * len(words)))
    op = random.choice(("sr", "ri", "rs", "rd"))
    if op == "sr":
        new_words = synonym_replacement(words, n)
    elif op == "ri":
        new_words = random_insertion(words, n)
    elif op == "rs":
        new_words = random_swap(words, n)
    else:
        new_words = random_deletion(words, alpha)
    return " ".join(new_words)


# Pipeline
def _ensure_wordnet() -> None:
    try:
        wordnet.synsets("test")
    except LookupError:
        logger.info("Downloading WordNet (one-time, ~30MB)...")
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


def augment_split(in_path: Path, out_path: Path, n_aug: int, alpha: float, seed: int) -> None:
    random.seed(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_in += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
            text = rec.get("text", "")
            for k in range(n_aug):
                aug_text = eda_augment(text, alpha=alpha)
                aug_rec = dict(rec)
                aug_rec["text"] = aug_text
                # Preserve label fields; mark provenance for debugging.
                aug_rec["id"] = f"{rec.get('id', n_in)}_aug{k + 1}"
                aug_rec["augmented"] = True
                fout.write(json.dumps(aug_rec, ensure_ascii=False) + "\n")
                n_out += 1
    logger.info(f"  {in_path.name}: {n_in} → {n_out} records ({n_aug}x augmentation)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["pandora", "essays", "mbti", "personality_evd"],
        help="Source dataset (reads from data/processed/{dataset}/)",
    )
    ap.add_argument("--n_aug", type=int, default=2, help="augmented copies per train sample")
    ap.add_argument("--alpha", type=float, default=0.1, help="fraction of words modified per op")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--src_dir", default="data/processed", help="parent dir of source dataset")
    ap.add_argument("--out_suffix", default="_augmented", help="output dataset is {dataset}{out_suffix}")
    args = ap.parse_args()

    _ensure_wordnet()

    src_dir = Path(args.src_dir) / args.dataset
    out_dir = Path(args.src_dir) / f"{args.dataset}{args.out_suffix}"
    if not src_dir.exists():
        raise SystemExit(f"Source not found: {src_dir}")

    logger.info(f"Augmenting {args.dataset} → {out_dir.name} (n_aug={args.n_aug}, alpha={args.alpha})")

    # Only augment train; copy val/test verbatim (eval honesty).
    train_in = src_dir / "train.jsonl"
    if not train_in.exists():
        raise SystemExit(f"train.jsonl not found in {src_dir}")
    augment_split(train_in, out_dir / "train.jsonl", args.n_aug, args.alpha, args.seed)

    for split in ("val", "test"):
        src = src_dir / f"{split}.jsonl"
        if src.exists():
            dst = out_dir / f"{split}.jsonl"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            logger.info(f"  {split}.jsonl: copied verbatim")
        else:
            logger.warning(f"  {split}.jsonl not found in {src_dir}")

    logger.info(f"Done. Output: {out_dir}/")


if __name__ == "__main__":
    main()
