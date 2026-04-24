"""Step 1: Extract behavioral evidence from input text using LLM."""
import json
import re
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from loguru import logger

from src.rag_pipeline.llm_client import extract_json
from src.retrieval.evidence_retriever import EvidenceSentence

PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class ExtractedEvidence:
    quote: str
    sentence_idx: int
    behavior_type: str
    description: str


class EvidenceExtractor:
    """CoPE Step 1: Use LLM to extract behavioral evidence from text."""

    def __init__(self, llm_client, config: dict | None = None):
        self.llm = llm_client
        self.config = config or {}
        self.env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)))

    @staticmethod
    def _salvage_json_candidates(response: str) -> list:
        """Best-effort recovery for malformed JSON-like list outputs."""
        if not response:
            return []

        # First try the shared extractor.
        try:
            parsed = json.loads(extract_json(response))
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return next((v for v in parsed.values() if isinstance(v, list)), [])
        except Exception:
            pass

        # Try to find individual object literals and parse them one by one.
        matches = re.findall(r"\{.*?\}", response, flags=re.DOTALL)
        recovered = []
        for m in matches:
            try:
                item = json.loads(m)
                if isinstance(item, dict):
                    recovered.append(item)
            except Exception:
                continue
        return recovered

    def _render_prompt(
        self,
        text: str,
        candidate_sentences: list[EvidenceSentence],
        max_evidence: int,
    ) -> str:
        template = self.env.get_template("evidence_extraction.j2")
        return template.render(
            input_text=text,
            candidate_sentences=candidate_sentences,
            max_evidence=max_evidence,
        )

    def extract(
        self,
        text: str,
        candidate_sentences: list[EvidenceSentence],
        max_evidence: int = 10,
        max_retries: int = 2,
    ) -> list[ExtractedEvidence]:
        """Extract behavioral evidence from text using LLM."""
        prompt = self._render_prompt(text, candidate_sentences, max_evidence)
        base_messages = [{"role": "user", "content": prompt}]
        repair_suffix = (
            "\n\nYour previous reply was not valid JSON. "
            "Return ONLY a JSON array of objects with keys: "
            "quote, sentence_idx, behavior_type, description. "
            "No markdown, no commentary, no extra text."
        )

        last_response = ""
        for attempt in range(max_retries + 1):
            try:
                messages = base_messages if attempt == 0 else [
                    {"role": "user", "content": prompt + repair_suffix}
                ]
                response = self.llm.generate(messages)
                last_response = response
                evidence_list = self._salvage_json_candidates(response)

                return [
                    ExtractedEvidence(
                        quote=item.get("quote", ""),
                        sentence_idx=item.get("sentence_idx", -1),
                        behavior_type=item.get("behavior_type", "general"),
                        description=item.get("description", ""),
                    )
                    for item in evidence_list
                    if isinstance(item, dict) and item.get("quote")
                ]
            except (json.JSONDecodeError, KeyError, AttributeError, TypeError) as e:
                logger.warning(f"Evidence extraction attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    salvaged = self._salvage_json_candidates(last_response)
                    if salvaged:
                        logger.warning(
                            f"Recovered {len(salvaged)} evidence items from malformed JSON response"
                        )
                        return [
                            ExtractedEvidence(
                                quote=item.get("quote", ""),
                                sentence_idx=item.get("sentence_idx", -1),
                                behavior_type=item.get("behavior_type", "general"),
                                description=item.get("description", ""),
                            )
                            for item in salvaged
                            if isinstance(item, dict) and item.get("quote")
                        ]
                    logger.error("All evidence extraction attempts failed, returning empty list")
                    return []
