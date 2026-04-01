"""Text cleaning pipeline per Naz et al. (2025)."""
from dataclasses import dataclass

from src.utils.text_utils import (
    clean_text_pipeline,
    count_words,
    truncate_to_words,
)


@dataclass
class PreprocessorConfig:
    remove_urls: bool = True
    remove_mentions: bool = True
    remove_mbti_mentions: bool = True
    lowercase: bool = False
    min_words: int = 10
    max_words: int = 2000


class TextPreprocessor:
    """
    Text cleaning pipeline per Naz et al. (2025):
    1. Remove URLs, @mentions, ||| delimiters (MBTI-specific)
    2. Lowercase (optional, configurable)
    3. Remove repeated punctuation ('!!!' → '!')
    4. Strip extra whitespace
    5. Filter posts < 10 words or > 2000 words
    """

    def __init__(self, config: PreprocessorConfig | None = None):
        self.config = config or PreprocessorConfig()

    def clean(self, text: str) -> str:
        """Apply the full cleaning pipeline."""
        text = clean_text_pipeline(
            text,
            remove_urls=self.config.remove_urls,
            remove_mentions=self.config.remove_mentions,
            remove_mbti=self.config.remove_mbti_mentions,
            lowercase=self.config.lowercase,
        )
        # Truncate long texts
        text = truncate_to_words(text, self.config.max_words)
        return text

    def is_valid(self, text: str) -> bool:
        """Return True if text meets minimum quality requirements."""
        return count_words(text) >= self.config.min_words

    def clean_and_validate(self, text: str) -> str | None:
        """Clean text and return None if it fails validation."""
        cleaned = self.clean(text)
        if not self.is_valid(cleaned):
            return None
        return cleaned
