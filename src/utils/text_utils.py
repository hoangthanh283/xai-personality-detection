"""Common text processing utilities."""

import re
import unicodedata

MBTI_TYPES = [
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
]

MBTI_PATTERN = re.compile(r"\b(" + "|".join(MBTI_TYPES) + r")(s|es)?\b", re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
REPEATED_PUNCT_PATTERN = re.compile(r"([!?.]){2,}")
EXTRA_WHITESPACE_PATTERN = re.compile(r"\s+")
MBTI_DELIM_PATTERN = re.compile(r"\|\|\|")


def remove_urls(text: str) -> str:
    return URL_PATTERN.sub(" ", text)


def remove_mentions(text: str) -> str:
    return MENTION_PATTERN.sub(" ", text)


def remove_mbti_mentions(text: str, replace_with: str = "") -> str:
    """Remove MBTI type mentions to prevent data leakage.

    Default replaces with empty string to avoid creating a noisy,
    non-discriminative token that pollutes TF-IDF vocabulary.
    """
    return MBTI_PATTERN.sub(replace_with, text)


def remove_mbti_delimiters(text: str) -> str:
    """Remove MBTI-specific '|||' delimiters."""
    return MBTI_DELIM_PATTERN.sub(" ", text)


def normalize_punctuation(text: str) -> str:
    """Reduce repeated punctuation: '!!!' → '!'"""
    return REPEATED_PUNCT_PATTERN.sub(r"\1", text)


def strip_whitespace(text: str) -> str:
    return EXTRA_WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters."""
    return unicodedata.normalize("NFKC", text)


def count_words(text: str) -> int:
    return len(text.split())


def truncate_to_words(text: str, max_words: int = 2000) -> str:
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text


def clean_text_pipeline(
    text: str,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    remove_mbti: bool = True,
    lowercase: bool = False,
) -> str:
    """Apply standard cleaning pipeline."""
    if remove_urls:
        text = URL_PATTERN.sub(" ", text)
    if remove_mentions:
        text = MENTION_PATTERN.sub(" ", text)

    # Remove MBTI-specific delimiter
    text = remove_mbti_delimiters(text)

    if remove_mbti:
        text = remove_mbti_mentions(text)

    if lowercase:
        text = text.lower()

    text = normalize_punctuation(text)
    text = normalize_unicode(text)
    text = strip_whitespace(text)
    return text


def split_into_sentences(text: str) -> list[str]:
    """Simple sentence splitting (use spaCy for better quality)."""
    # Basic split on sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_simple(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenization."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens
