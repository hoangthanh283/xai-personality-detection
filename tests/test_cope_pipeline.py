"""Tests for CoPE pipeline components."""
import json
from unittest.mock import MagicMock

from src.reasoning.evidence_extractor import (EvidenceExtractor,
                                              ExtractedEvidence)
from src.reasoning.state_identifier import IdentifiedState, StateIdentifier
from src.reasoning.trait_inferencer import TraitInferencer
from src.retrieval.evidence_retriever import EvidenceSentence


def make_mock_llm(response: dict):
    """Create a mock LLM client that returns the given dict as JSON."""
    mock = MagicMock()
    mock.generate.return_value = json.dumps(response)
    return mock


class TestEvidenceExtractor:
    def test_extract_valid_response(self):
        evidence_response = [
            {
                "quote": "I love spending time alone thinking",
                "sentence_idx": 0,
                "behavior_type": "lifestyle_preference",
                "description": "Prefers solitary activities",
            }
        ]
        llm = make_mock_llm(evidence_response)
        extractor = EvidenceExtractor(llm)
        candidate = [EvidenceSentence(text="I love spending time alone thinking", sentence_idx=0, score=0.8)]
        result = extractor.extract("I love spending time alone thinking", candidate)
        assert len(result) == 1
        assert result[0].quote == "I love spending time alone thinking"
        assert result[0].behavior_type == "lifestyle_preference"

    def test_extract_handles_bad_json(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "not valid json"
        extractor = EvidenceExtractor(mock_llm)
        result = extractor.extract("Some text", [], max_retries=0)
        assert result == []

    def test_extract_empty_list(self):
        llm = make_mock_llm([])
        extractor = EvidenceExtractor(llm)
        result = extractor.extract("Text", [])
        assert result == []


class TestStateIdentifier:
    def test_identify_valid_response(self):
        state_response = [
            {
                "evidence_idx": 1,
                "quote": "I love spending time alone",
                "state_label": "Social Withdrawal",
                "state_definition": "Preference for solitude",
                "kb_reference": "demo_kb/behavioral_marker",
                "confidence": 0.85,
                "reasoning": "Strong preference for alone time indicates introversion.",
            }
        ]
        llm = make_mock_llm(state_response)
        identifier = StateIdentifier(llm)
        evidence = [ExtractedEvidence("I love spending time alone", 0, "lifestyle_preference", "Prefers solitude")]
        result = identifier.identify(evidence, kb_chunks=[])
        assert len(result) == 1
        assert result[0].state_label == "Social Withdrawal"
        assert result[0].confidence == 0.85

    def test_identify_empty_evidence(self):
        llm = make_mock_llm([])
        identifier = StateIdentifier(llm)
        result = identifier.identify([], kb_chunks=[])
        assert result == []


class TestTraitInferencer:
    def test_infer_mbti(self):
        prediction_response = {
            "prediction": {
                "type": "INTP",
                "dimensions": {
                    "IE": {"label": "I", "confidence": 0.9, "supporting_states": [1]},
                    "SN": {"label": "N", "confidence": 0.85, "supporting_states": [1]},
                    "TF": {"label": "T", "confidence": 0.8, "supporting_states": [1]},
                    "JP": {"label": "P", "confidence": 0.75, "supporting_states": [1]},
                },
            },
            "explanation": "The text reveals strong introversion and analytical thinking.",
            "evidence_chain": [{"evidence": "I love alone time", "state": "Social Withdrawal", "trait_contribution": "I"}],
        }
        llm = make_mock_llm(prediction_response)
        inferencer = TraitInferencer(llm)
        state = IdentifiedState(1, "I love alone time", "Social Withdrawal", "def", "ref", 0.9, "reasoning")
        result = inferencer.infer([state], trait_kb_chunks=[])
        assert result.predicted_label == "INTP"
        assert "introversion" in result.explanation.lower()

    def test_infer_handles_empty_states(self):
        llm = make_mock_llm({})
        inferencer = TraitInferencer(llm)
        result = inferencer.infer([], trait_kb_chunks=[])
        assert result.predicted_label == "UNKNOWN"


class TestCoPEPipeline:
    def test_full_pipeline_run(self):
        from src.reasoning.cope_pipeline import CoPEPipeline

        # Mock LLM returning appropriate responses for each step
        call_count = [0]
        responses = [
            json.dumps([{"quote": "I prefer thinking alone", "sentence_idx": 0, "behavior_type": "lifestyle_preference", "description": "Prefers solitude"}]),
            json.dumps([{"evidence_idx": 1, "quote": "I prefer thinking alone", "state_label": "Social Withdrawal", "state_definition": "def", "kb_reference": "ref", "confidence": 0.85, "reasoning": "reason"}]),
            json.dumps({
                "prediction": {"type": "INTP", "dimensions": {"IE": {"label": "I", "confidence": 0.9, "supporting_states": [1]}, "SN": {"label": "N", "confidence": 0.85, "supporting_states": [1]}, "TF": {"label": "T", "confidence": 0.8, "supporting_states": [1]}, "JP": {"label": "P", "confidence": 0.75, "supporting_states": [1]}}},
                "explanation": "Strong introversion and intuition detected.",
                "evidence_chain": [{"evidence": "I prefer thinking alone", "state": "Social Withdrawal", "trait_contribution": "I"}],
            }),
        ]

        def mock_generate(messages, **kwargs):
            if call_count[0] < len(responses):
                resp = responses[call_count[0]]
            else:
                resp = responses[-1]
            call_count[0] += 1
            return resp

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = mock_generate

        pipeline = CoPEPipeline(mock_llm, kb_retriever=None)
        candidate = [EvidenceSentence("I prefer thinking alone", 0, 0.8)]
        result = pipeline.run("I prefer thinking alone and analyzing ideas.", candidate, framework="mbti")

        assert result["predicted_label"] == "INTP"
        assert "evidence_chain" in result
