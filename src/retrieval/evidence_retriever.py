"""Extract candidate evidence sentences from input text.

Strategy:
1. Split text into sentences (spaCy sentence tokenizer)
2. Score each sentence for "personality signal" using:
   a. Keyword matching against personality-relevant lexicons
   b. Sentence embedding similarity to trait-descriptive anchors
3. Return top-k sentences ranked by score
"""
import re
from dataclasses import dataclass, field

from loguru import logger

# Personality-relevant keywords (LIWC-inspired categories)
PERSONALITY_KEYWORDS = {
    "social": ["friend", "people", "talk", "social", "group", "party", "together", "relationship", "interact"],
    "cognitive": ["think", "idea", "concept", "analyze", "understand", "theory", "logic", "reason", "plan"],
    "emotion": ["feel", "love", "hate", "sad", "happy", "angry", "anxious", "excited", "overwhelm", "worry"],
    "preference": ["prefer", "like", "enjoy", "love", "hate", "interest", "passion", "favorite", "avoid"],
    "lifestyle": ["always", "never", "routine", "schedule", "organize", "spontaneous", "flexible", "deadline"],
    "decision": ["decide", "choose", "weigh", "consider", "judge", "evaluate", "opt", "select", "pick"],
}
ALL_KEYWORDS = set(kw for keywords in PERSONALITY_KEYWORDS.values() for kw in keywords)


@dataclass
class EvidenceSentence:
    text: str
    sentence_idx: int
    score: float
    matched_keywords: list[str] = field(default_factory=list)


class EvidenceRetriever:
    """
    Extracts candidate evidence sentences from input text.
    Pre-filters noise BEFORE sending to LLM, reducing token cost.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.top_k = self.config.get("top_k", 10)
        self._nlp = None

    @property
    def nlp(self):
        """Lazy-load spaCy for sentence splitting."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy en_core_web_sm")
            except (ImportError, OSError):
                logger.warning("spaCy not available, using regex sentence splitting")
                self._nlp = None
        return self._nlp

    def split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        if self.nlp is not None:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Fallback: simple regex split
            sentences = re.split(r"(?<=[.!?])\s+", text)
            return [s.strip() for s in sentences if s.strip()]

    def score_sentences(self, sentences: list[str]) -> list[EvidenceSentence]:
        """Score sentences by personality signal strength."""
        scored = []
        for idx, sent in enumerate(sentences):
            words = set(re.findall(r"\b\w+\b", sent.lower()))
            matched = list(words & ALL_KEYWORDS)
            # Keyword score: fraction of matched keywords, capped
            keyword_score = min(len(matched) / max(len(words), 1) * 10.0, 1.0)

            scored.append(EvidenceSentence(
                text=sent,
                sentence_idx=idx,
                score=keyword_score,
                matched_keywords=matched,
            ))
        return scored

    def extract(self, text: str, top_k: int | None = None) -> list["EvidenceSentence"]:
        """Extract top-k evidence sentences from input text."""
        top_k = top_k or self.top_k
        sentences = self.split_sentences(text)
        if not sentences:
            return []
        scored = self.score_sentences(sentences)
        # Sort by score descending, take top-k
        sorted_sents = sorted(scored, key=lambda s: s.score, reverse=True)
        return sorted_sents[:top_k]
