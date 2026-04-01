"""Tests for retrieval components."""

from src.retrieval.evidence_retriever import (EvidenceRetriever,
                                              EvidenceSentence)


class TestEvidenceRetriever:
    def test_split_sentences_basic(self):
        retriever = EvidenceRetriever()
        text = "I love reading books. Social events drain me. I prefer being alone."
        sentences = retriever.split_sentences(text)
        assert len(sentences) >= 2

    def test_score_sentences(self):
        retriever = EvidenceRetriever()
        sentences = [
            "I love thinking about abstract ideas and theories.",
            "The weather is nice today.",
            "Socializing with many people exhausts me.",
        ]
        scored = retriever.score_sentences(sentences)
        assert len(scored) == 3
        # Sentences with personality keywords should score higher
        # "I love thinking" and "Socializing" have personality keywords
        assert max(scored, key=lambda s: s.score).score > 0

    def test_extract_returns_top_k(self):
        retriever = EvidenceRetriever({"top_k": 3})
        text = " ".join([f"Sentence {i} about thinking and ideas." for i in range(20)])
        results = retriever.extract(text, top_k=3)
        assert len(results) <= 3

    def test_extract_empty_text(self):
        retriever = EvidenceRetriever()
        results = retriever.extract("", top_k=5)
        assert results == []

    def test_evidence_sentence_dataclass(self):
        ev = EvidenceSentence(text="I prefer solitude", sentence_idx=0, score=0.7, matched_keywords=["prefer"])
        assert ev.text == "I prefer solitude"
        assert ev.score == 0.7
