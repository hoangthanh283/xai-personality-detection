"""Full 3-step CoPE pipeline orchestrator."""
from loguru import logger

from src.reasoning.evidence_extractor import EvidenceExtractor, ExtractedEvidence
from src.reasoning.state_identifier import StateIdentifier, IdentifiedState
from src.reasoning.trait_inferencer import PredictionResult, TraitInferencer
from src.retrieval.evidence_retriever import EvidenceSentence
from src.retrieval.kb_retriever import KBChunkResult, deduplicate_chunks


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

    def run(
        self,
        text: str,
        candidate_evidence: list[EvidenceSentence],
        framework: str = "mbti",
        save_intermediate: bool = False,
    ) -> dict:
        """
        Run the full CoPE pipeline.

        Returns a dict with keys:
        - predicted_label
        - prediction_details
        - explanation
        - evidence_chain
        - intermediate (if save_intermediate=True)
        """
        # ── Step 1: Evidence Extraction ──────────────────────────────────────
        logger.info("Step 1: Extracting behavioral evidence")
        evidence: list[ExtractedEvidence] = self.evidence_extractor.extract(
            text,
            candidate_evidence,
            max_evidence=self.num_evidence,
            max_retries=self.max_retries,
        )
        logger.info(f"Extracted {len(evidence)} evidence items")

        # ── Step 2: State Identification ─────────────────────────────────────
        logger.info("Step 2: Identifying psychological states")
        kb_chunks: list[KBChunkResult] = []
        if self.kb is not None:
            for ev in evidence:
                chunks = self.kb.search(
                    ev.description or ev.quote,
                    top_k=self.num_kb_chunks,
                    framework=framework,
                    category="behavioral_marker",
                )
                kb_chunks.extend(chunks)
            kb_chunks = deduplicate_chunks(kb_chunks)
            logger.info(f"Retrieved {len(kb_chunks)} KB chunks for state identification")

        states: list[IdentifiedState] = self.state_identifier.identify(
            evidence, kb_chunks, max_retries=self.max_retries
        )
        logger.info(f"Identified {len(states)} states")

        # ── Step 3: Trait Inference ───────────────────────────────────────────
        logger.info("Step 3: Inferring personality traits")
        trait_kb: list[KBChunkResult] = []
        if self.kb is not None:
            trait_kb = self.kb.search(
                f"{framework} personality trait definitions",
                top_k=10,
                framework=framework,
                category="trait_definition",
            )

        result: PredictionResult = self.trait_inferencer.infer(
            states, trait_kb, framework=framework, max_retries=self.max_retries
        )
        logger.info(f"Predicted: {result.predicted_label}")

        output = {
            "predicted_label": result.predicted_label,
            "prediction_details": result.prediction_details,
            "explanation": result.explanation,
            "evidence_chain": result.evidence_chain,
        }

        if save_intermediate:
            output["intermediate"] = {
                "step1_evidence": [
                    {"quote": e.quote, "behavior_type": e.behavior_type, "description": e.description}
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

        return output
