"""Step 2: Map behavioral evidence to psychological states using LLM + KB."""
import json
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from loguru import logger

from src.reasoning.evidence_extractor import ExtractedEvidence
from src.retrieval.kb_retriever import KBChunkResult

PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class IdentifiedState:
    evidence_idx: int
    quote: str
    state_label: str
    state_definition: str
    kb_reference: str
    confidence: float
    reasoning: str


class StateIdentifier:
    """CoPE Step 2: Use LLM + KB to map evidence to psychological states."""

    def __init__(self, llm_client, kb_retriever=None, config: dict | None = None):
        self.llm = llm_client
        self.kb = kb_retriever
        self.config = config or {}
        self.env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)))

    def _render_prompt(
        self,
        evidence_list: list[ExtractedEvidence],
        kb_chunks: list[KBChunkResult],
    ) -> str:
        template = self.env.get_template("state_identification.j2")
        return template.render(evidence_list=evidence_list, kb_chunks=kb_chunks)

    def identify(
        self,
        evidence_list: list[ExtractedEvidence],
        kb_chunks: list[KBChunkResult],
        max_retries: int = 2,
    ) -> list[IdentifiedState]:
        """Identify psychological states for each piece of evidence."""
        if not evidence_list:
            return []

        prompt = self._render_prompt(evidence_list, kb_chunks)
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(max_retries + 1):
            try:
                response = self.llm.generate(messages)
                content = response.strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]

                states_list = json.loads(content)
                if not isinstance(states_list, list):
                    states_list = []

                return [
                    IdentifiedState(
                        evidence_idx=item.get("evidence_idx", i),
                        quote=item.get("quote", ""),
                        state_label=item.get("state_label", "Unknown"),
                        state_definition=item.get("state_definition", ""),
                        kb_reference=item.get("kb_reference", ""),
                        confidence=float(item.get("confidence", 0.5)),
                        reasoning=item.get("reasoning", ""),
                    )
                    for i, item in enumerate(states_list)
                ]
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"State identification attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    logger.error("All state identification attempts failed")
                    return []
