"""Step 1: Extract behavioral evidence from input text using LLM."""
import json
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from loguru import logger

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
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(max_retries + 1):
            try:
                response = self.llm.generate(messages)
                # Parse JSON response
                # Strip markdown code blocks if present
                content = response.strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]

                evidence_list = json.loads(content)
                if not isinstance(evidence_list, list):
                    evidence_list = []

                return [
                    ExtractedEvidence(
                        quote=item.get("quote", ""),
                        sentence_idx=item.get("sentence_idx", -1),
                        behavior_type=item.get("behavior_type", "general"),
                        description=item.get("description", ""),
                    )
                    for item in evidence_list
                    if item.get("quote")
                ]
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Evidence extraction attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    logger.error("All evidence extraction attempts failed, returning empty list")
                    return []
