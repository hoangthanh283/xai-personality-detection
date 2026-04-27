"""Full 3-step CoPE pipeline orchestrator."""

from typing import Any

from loguru import logger

from src.reasoning.evidence_extractor import (EvidenceExtractor,
                                              ExtractedEvidence)
from src.reasoning.state_identifier import IdentifiedState, StateIdentifier
from src.reasoning.trait_inferencer import PredictionResult, TraitInferencer
from src.retrieval.evidence_retriever import EvidenceSentence
from src.retrieval.kb_retriever import KBChunkResult, deduplicate_chunks

STATE_RETRIEVAL_CATEGORIES = [
    "state_definition",
    "behavioral_marker",
    "linguistic_correlate",
    "evidence_mapping_example",
]
TRAIT_RETRIEVAL_CATEGORIES = [
    "trait_definition",
    "facet_definition",
    "evidence_mapping_example",
]


class CoPEPipeline:
    """
    Orchestrates the 3-step Chain-of-Personality-Evidence reasoning:

    Step 1: EvidenceExtractor  → Identify behavioral quotations from text
    Step 2: StateIdentifier    → Map evidence → psychological states (KB-grounded)
    Step 3: TraitInferencer    → Aggregate states → personality label (KB-grounded)
    """

    def __init__(self, llm_client, kb_retriever, config: dict | None = None):
        self.llm = llm_client
        self.kb = kb_retriever
        self.config = config or {}
        self.evidence_extractor = EvidenceExtractor(llm_client, config)
        self.state_identifier = StateIdentifier(llm_client, kb_retriever, config)
        self.trait_inferencer = TraitInferencer(llm_client, kb_retriever, config)

        self.num_evidence = self.config.get("num_evidence", 10)
        self.num_kb_chunks = self.config.get("num_kb_chunks", 5)
        self.max_retries = self.config.get("max_retries_per_step", 2)
        self.skip_steps = set(self.config.get("skip_steps", []))

    def run(
        self,
        text: str,
        candidate_evidence: list[EvidenceSentence],
        framework: str = "mbti",
        save_intermediate: bool = True,
        yield_steps: bool = False,
        roberta_prior: dict | None = None,
    ) -> dict | Any:
        """
        Run the 3-step reasoning chain.
        If yield_steps=True, returns a generator yielding (step_name, current_result).
        Otherwise returns a dict directly.
        """
        if yield_steps:
            return self._run_stream(text, candidate_evidence, framework, save_intermediate, roberta_prior)
        return self._run_collect(text, candidate_evidence, framework, save_intermediate, roberta_prior)

    def _run_collect(
        self,
        text: str,
        candidate_evidence: list[EvidenceSentence],
        framework: str,
        save_intermediate: bool,
        roberta_prior: dict | None = None,
    ) -> dict:
        """Non-streaming: run all steps and return final dict."""
        last: dict = {}
        for _, payload in self._run_stream(text, candidate_evidence, framework, save_intermediate, roberta_prior):
            if isinstance(payload, dict) and "predicted_label" in payload:
                last = payload
        return last

    def _run_stream(
        self,
        text: str,
        candidate_evidence: list[EvidenceSentence],
        framework: str = "mbti",
        save_intermediate: bool = True,
        roberta_prior: dict | None = None,
    ):
        """Generator version — yields (step_name, payload) tuples."""
        # Step 1: Evidence Extraction
        logger.info("Step 1: Extracting behavioral evidence")
        evidence: list[ExtractedEvidence] = self.evidence_extractor.extract(
            text,
            candidate_evidence,
            max_evidence=self.num_evidence,
            max_retries=self.max_retries,
        )
        logger.info(f"Extracted {len(evidence)} evidence items")
        yield "Step 1: Extracting Evidence", evidence

        # Step 2: State Identification
        if 2 in self.skip_steps and 3 in self.skip_steps:
            logger.info("Skipping Steps 2 and 3 (Direct RAG ablation)")
            import json

            kb_texts = []
            if self.kb is not None:
                trait_kb = self.kb.search(
                    f"{framework} personality trait definitions",
                    top_k=5,
                    framework=framework,
                    category="trait_definition",
                )
                kb_texts = [c.text for c in trait_kb]
            evidence_text = "\n".join([e.quote for e in evidence])
            prompt = f"Predict the {framework} personality type based on the text.\n\nText:\n{evidence_text}\n"
            if kb_texts:
                prompt += "\nKnowledge Base Context:\n" + "\n".join(kb_texts) + "\n"
            prompt += '\nProvide output as JSON: {"prediction": {"type": "XXXX"}, "explanation": "..."}'

            response = self.llm.generate([{"role": "user", "content": prompt}])
            try:
                data = json.loads(response.strip().strip("```json").strip("```"))
                pred = data.get("prediction", {}).get("type", "UNKNOWN")
                expl = data.get("explanation", "")
            except Exception:
                pred = "UNKNOWN"
                expl = response

            output = {
                "predicted_label": pred,
                "prediction_details": {"type": pred},
                "explanation": expl,
                "evidence_chain": [],
            }
            yield "Final Result", output
            return

        if 2 in self.skip_steps:
            logger.info("Skipping Step 2: Extracting state directly from evidence")
            states = [
                IdentifiedState(
                    state_label=e.behavior_type,
                    confidence=1.0,
                    quote=e.quote,
                    reasoning="Skipped Step 2",
                )
                for e in evidence
            ]
            kb_chunks = []
        else:
            logger.info("Step 2: Identifying psychological states")
            kb_chunks: list[KBChunkResult] = []
            if self.kb is not None:
                queries = [ev.description or ev.quote for ev in evidence]
                results = self.kb.search_many(
                    queries,
                    top_k=self.num_kb_chunks,
                    framework=framework,
                    category=STATE_RETRIEVAL_CATEGORIES,
                )
                for chunks in results:
                    kb_chunks.extend(chunks)
                kb_chunks = deduplicate_chunks(kb_chunks)
                logger.info(f"Retrieved {len(kb_chunks)} KB chunks for state identification via batch search")

            states: list[IdentifiedState] = self.state_identifier.identify(
                evidence, kb_chunks, max_retries=self.max_retries
            )
            logger.info(f"Identified {len(states)} states")
            yield "Step 2: Identifying Psychological States", states

        # Step 3: Trait Inference
        logger.info("Step 3: Inferring personality traits")
        trait_kb: list[KBChunkResult] = []
        if self.kb is not None:
            trait_kb = self.kb.search(
                f"{framework} personality trait definitions",
                top_k=5,
                framework=framework,
                category=TRAIT_RETRIEVAL_CATEGORIES,
            )

        result: PredictionResult = self.trait_inferencer.infer(
            states,
            trait_kb,
            framework=framework,
            max_retries=self.max_retries,
            roberta_prior=roberta_prior,
        )
        logger.info(f"Predicted: {result.predicted_label}")
        yield "Step 3: Predicting Traits", result

        output = {
            "predicted_label": result.predicted_label,
            "prediction_details": result.prediction_details,
            "explanation": result.explanation,
            "evidence_chain": result.evidence_chain,
        }

        if save_intermediate:
            output["intermediate"] = {
                "step1_evidence": [
                    {
                        "quote": e.quote,
                        "behavior_type": e.behavior_type,
                        "description": e.description,
                    }
                    for e in evidence
                ],
                "step2_states": [
                    {
                        "state_label": s.state_label,
                        "confidence": s.confidence,
                        "quote": s.quote,
                        "reasoning": s.reasoning,
                    }
                    for s in states
                ],
                "kb_chunks_used": [
                    {"chunk_id": c.chunk_id, "score": c.score, "text": c.text[:200] + "..."}
                    for c in (kb_chunks + trait_kb)[:20]
                ],
            }

        yield "Final Result", output
