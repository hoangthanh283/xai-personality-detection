"""Step 3: Aggregate psychological states → personality trait labels."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from loguru import logger

from src.rag_pipeline.llm_client import extract_json
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
        roberta_prior: dict | None = None,
    ) -> str:
        template = self.env.get_template("trait_inference.j2")
        return template.render(
            states=states,
            trait_kb_chunks=trait_kb_chunks,
            framework=framework,
            roberta_prior=roberta_prior,
        )

    def _extract_mbti_label(self, prediction_details: dict) -> str:
        """Extract full MBTI label from dimension predictions."""
        dims = prediction_details.get("dimensions", {})
        type_str = ""
        for dim in ["IE", "SN", "TF", "JP"]:
            label = dims.get(dim, {}).get("label", "?")
            type_str += label
        return type_str if "?" not in type_str else "INTP"  # safe default

    _VALID_OCEAN_LABELS = {"HIGH", "LOW"}

    def _normalize_ocean_label(self, label: str, prior_label: str | None = None) -> str:
        normalized = str(label).strip().upper()
        if normalized in self._VALID_OCEAN_LABELS:
            return normalized
        return prior_label or "HIGH"

    def _extract_ocean_label(self, prediction_details: dict, roberta_prior: dict | None = None) -> str:
        """Extract a combined OCEAN label string, normalizing invalid values."""
        traits = prediction_details.get("traits", {})
        parts = []
        for t in ["O", "C", "E", "A", "N"]:
            raw = traits.get(t, {}).get("label", "?")
            prior_label = None
            if roberta_prior and t in roberta_prior:
                p = roberta_prior[t]
                prior_label = str(p[0]) if isinstance(p, (list, tuple)) else str(p)
            label = self._normalize_ocean_label(raw, prior_label)
            parts.append(f"{t}:{label}")
        return ",".join(parts)

    def infer(
        self,
        states: list[IdentifiedState],
        trait_kb_chunks: list[KBChunkResult],
        framework: str = "mbti",
        max_retries: int = 2,
        roberta_prior: dict | None = None,
    ) -> PredictionResult:
        """Infer personality traits from states and KB definitions."""
        if not states:
            # If we have a RoBERTa prior, fall back to it rather than UNKNOWN.
            if roberta_prior:
                fallback = _prior_to_label(roberta_prior, framework)
                if fallback:
                    return PredictionResult(
                        predicted_label=fallback,
                        prediction_details={"type": fallback, "source": "roberta_prior_fallback"},
                        explanation="CoPE found no states; using RoBERTa baseline prior as fallback.",
                        framework=framework,
                    )
            return PredictionResult(
                predicted_label="UNKNOWN",
                explanation="No psychological states were identified.",
                framework=framework,
            )

        prompt = self._render_prompt(states, trait_kb_chunks, framework, roberta_prior=roberta_prior)
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(max_retries + 1):
            try:
                response = self.llm.generate(messages)
                content = extract_json(response)
                result_data: dict[str, Any] = json.loads(content)
                prediction = result_data.get("prediction", {})
                explanation = result_data.get("explanation", "")
                evidence_chain = result_data.get("evidence_chain", [])

                if framework == "mbti":
                    # Prefer computed label from dimensions (more reliable than LLM's "type" field).
                    # Fall back to LLM's "type" only if dimensions are missing/invalid.
                    computed = self._extract_mbti_label(prediction)
                    llm_type = (prediction.get("type") or "").strip().upper()
                    if "?" not in computed and len(computed) == 4:
                        label = computed
                    elif len(llm_type) == 4 and all(c in "IESNTFJP" for c in llm_type):
                        label = llm_type
                    else:
                        label = computed  # "INTP" default
                else:
                    label = self._extract_ocean_label(prediction, roberta_prior=roberta_prior)

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
                    # Use RoBERTa prior as final fallback if available.
                    if roberta_prior:
                        fallback = _prior_to_label(roberta_prior, framework)
                        if fallback:
                            return PredictionResult(
                                predicted_label=fallback,
                                prediction_details={"type": fallback, "source": "roberta_prior_fallback"},
                                explanation="Step-3 JSON parse failed; using RoBERTa prior as fallback.",
                                framework=framework,
                            )
                    return PredictionResult(
                        predicted_label="UNKNOWN",
                        explanation="Inference failed due to parsing errors.",
                        framework=framework,
                    )


def _prior_to_label(roberta_prior: dict, framework: str) -> str | None:
    """Convert a RoBERTa doc-level prior dict to the final label string."""
    if not roberta_prior:
        return None
    if framework == "mbti":
        # Expect keys IE, SN, TF, JP → each (label, conf)
        parts = []
        for dim in ["IE", "SN", "TF", "JP"]:
            pair = roberta_prior.get(dim)
            if not pair or not pair[0]:
                return None
            parts.append(pair[0])
        return "".join(parts)
    else:
        parts = []
        for trait in ["O", "C", "E", "A", "N"]:
            pair = roberta_prior.get(trait)
            if not pair or not pair[0]:
                return None
            parts.append(f"{trait}:{pair[0]}")
        return ",".join(parts)
