"""Step 3: Aggregate psychological states → personality trait labels."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from loguru import logger

from src.reasoning.state_identifier import IdentifiedState
from src.retrieval.kb_retriever import KBChunkResult

PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class PredictionResult:
    """Final output of the CoPE pipeline."""
    predicted_label: str
    prediction_details: dict = field(default_factory=dict)
    explanation: str = ""
    evidence_chain: list[dict] = field(default_factory=list)
    framework: str = "mbti"


class TraitInferencer:
    """CoPE Step 3: Infer personality traits from aggregated psychological states."""

    def __init__(self, llm_client, kb_retriever=None, config: dict | None = None):
        self.llm = llm_client
        self.kb = kb_retriever
        self.config = config or {}
        self.env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)))

    def _render_prompt(
        self,
        states: list[IdentifiedState],
        trait_kb_chunks: list[KBChunkResult],
        framework: str,
    ) -> str:
        template = self.env.get_template("trait_inference.j2")
        return template.render(states=states, trait_kb_chunks=trait_kb_chunks, framework=framework)

    def _extract_mbti_label(self, prediction_details: dict) -> str:
        """Extract full MBTI label from dimension predictions."""
        dims = prediction_details.get("dimensions", {})
        type_str = ""
        for dim in ["IE", "SN", "TF", "JP"]:
            label = dims.get(dim, {}).get("label", "?")
            type_str += label
        return type_str if "?" not in type_str else "INTP"  # safe default

    def _extract_ocean_label(self, prediction_details: dict) -> str:
        """Extract a combined OCEAN label string."""
        traits = prediction_details.get("traits", {})
        parts = []
        for t in ["O", "C", "E", "A", "N"]:
            label = traits.get(t, {}).get("label", "?")
            parts.append(f"{t}:{label}")
        return ",".join(parts)

    def infer(
        self,
        states: list[IdentifiedState],
        trait_kb_chunks: list[KBChunkResult],
        framework: str = "mbti",
        max_retries: int = 2,
    ) -> PredictionResult:
        """Infer personality traits from states and KB definitions."""
        if not states:
            return PredictionResult(
                predicted_label="UNKNOWN",
                explanation="No psychological states were identified.",
                framework=framework,
            )

        prompt = self._render_prompt(states, trait_kb_chunks, framework)
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(max_retries + 1):
            try:
                response = self.llm.generate(messages)
                content = response.strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]

                result_data: dict[str, Any] = json.loads(content)
                prediction = result_data.get("prediction", {})
                explanation = result_data.get("explanation", "")
                evidence_chain = result_data.get("evidence_chain", [])

                if framework == "mbti":
                    label = prediction.get("type") or self._extract_mbti_label(prediction)
                else:
                    label = self._extract_ocean_label(prediction)

                return PredictionResult(
                    predicted_label=label,
                    prediction_details=prediction,
                    explanation=explanation,
                    evidence_chain=evidence_chain,
                    framework=framework,
                )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Trait inference attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    logger.error("All trait inference attempts failed, returning fallback prediction")
                    return PredictionResult(
                        predicted_label="UNKNOWN",
                        explanation="Inference failed due to parsing errors.",
                        framework=framework,
                    )
