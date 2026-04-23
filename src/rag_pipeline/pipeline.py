"""Main RAG-XPR pipeline orchestrator.

Ties together:
1. Text preprocessing
2. Evidence retrieval from input text
3. KB retrieval (psychology definitions)
4. CoPE reasoning (3-step LLM chain)
"""
from typing import Any

from loguru import logger

from src.data.preprocessor import PreprocessorConfig, TextPreprocessor
from src.rag_pipeline.llm_client import LLMClient, build_llm_client
from src.reasoning.cope_pipeline import CoPEPipeline
from src.retrieval.evidence_retriever import EvidenceRetriever
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.kb_retriever import KBRetriever


class RAGXPRPipeline:
    """
    Main orchestrator: input text → prediction + explanation.

    predict(text) lifecycle:
        1. Preprocess text
        2. Retrieve evidence sentences from text
        3. Retrieve KB context (psychology definitions)
        4. Run CoPE 3-step reasoning
        → Returns structured PredictionResult dict
    """

    def __init__(self, config: dict):
        self.config = config
        cope_config = config.get("cope", {})
        retrieval_config = config.get("retrieval", {})
        evidence_config = config.get("evidence_retrieval", {})

        # Build components
        self.preprocessor = TextPreprocessor(PreprocessorConfig())

        # Optional RoBERTa scorer for supervised evidence selection + Step-3 prior.
        self.roberta_scorer = self._build_roberta_scorer(
            evidence_config, cope_config.get("framework", "mbti")
        )
        self.use_roberta_prior = evidence_config.get("use_roberta_prior", False) and self.roberta_scorer is not None

        self.evidence_retriever = EvidenceRetriever(evidence_config, roberta_scorer=self.roberta_scorer)

        # KB retriever (semantic, BM25, or hybrid)
        skip_kb = retrieval_config.get("skip_kb", False)
        if skip_kb:
            self.kb_retriever = None
        else:
            method = evidence_config.get("method", "hybrid")
            if method == "hybrid":
                chunks_path = config.get("kb_chunks_path", "data/knowledge_base/chunks.jsonl")
                self.kb_retriever = HybridRetriever(retrieval_config, chunks_path)
            elif method == "semantic":
                self.kb_retriever = KBRetriever(retrieval_config)
            elif method == "keyword":
                chunks_path = config.get("kb_chunks_path", "data/knowledge_base/chunks.jsonl")
                from src.retrieval.hybrid_search import BM25Retriever
                self.kb_retriever = BM25Retriever(chunks_path)

        # LLM client
        llm_config = config.get(
            "llm",
            {"provider": "openrouter", "model": "qwen/qwen3.6-plus-preview:free"},
        )
        self.llm_client: LLMClient = build_llm_client(llm_config)

        # CoPE pipeline
        self.cope_pipeline = CoPEPipeline(self.llm_client, self.kb_retriever, cope_config)

        self.framework = cope_config.get("framework", "mbti")
        self.save_intermediate = config.get("output", {}).get("save_intermediate", True)

    def _build_roberta_scorer(self, evidence_config: dict, framework: str):
        """Instantiate supervised scorer based on evidence_retrieval.backbone config.

        backbone: "frozen_svm" (default) → FrozenSvmEvidenceScorer
                  "roberta"              → RoBERTaEvidenceScorer (legacy)
        Returns None when scorer is keyword-only and no prior is requested.
        """
        backbone = evidence_config.get("backbone", "frozen_svm")
        scorer_type = evidence_config.get("scorer", "keyword")
        use_prior = evidence_config.get("use_roberta_prior", False)

        if scorer_type not in ("roberta", "hybrid") and not use_prior:
            return None

        if backbone == "frozen_svm":
            return self._build_frozen_svm_scorer(evidence_config, framework)
        return self._build_finetuned_roberta_scorer(evidence_config, framework)

    def _build_frozen_svm_scorer(self, evidence_config: dict, framework: str):
        """Load FrozenSvmEvidenceScorer with pre-trained MBTI or OCEAN checkpoints."""
        try:
            from src.retrieval.frozen_svm_scorer import (FrozenSvmEvidenceScorer,
                                                          default_mbti_svm_checkpoints,
                                                          default_ocean_svm_checkpoints)
        except ImportError as e:
            logger.warning(f"FrozenSvmEvidenceScorer unavailable ({e}); falling back to keyword")
            return None

        ckpts = evidence_config.get("frozen_svm_checkpoints")
        if not ckpts:
            dataset_hint = evidence_config.get("roberta_dataset", "essays")
            ckpts = (default_mbti_svm_checkpoints() if framework == "mbti"
                     else default_ocean_svm_checkpoints(dataset=dataset_hint))
        device = evidence_config.get("roberta_device", "cpu")
        encoder_cfg = {"device": device} if device else None
        try:
            return FrozenSvmEvidenceScorer(
                checkpoint_paths=ckpts,
                encoder_cfg=encoder_cfg,
                batch_size=evidence_config.get("roberta_batch_size", 32),
            )
        except Exception as e:
            logger.warning(f"Failed to load FrozenSVM scorer: {e}; falling back to keyword")
            return None

    def _build_finetuned_roberta_scorer(self, evidence_config: dict, framework: str):
        """Load RoBERTaEvidenceScorer (fine-tuned checkpoints, legacy backbone)."""
        try:
            from src.retrieval.roberta_scorer import (RoBERTaEvidenceScorer,
                                                      default_mbti_checkpoints,
                                                      default_ocean_checkpoints)
        except ImportError as e:
            logger.warning(f"RoBERTa scorer unavailable ({e}); falling back to keyword")
            return None

        ckpts = evidence_config.get("roberta_checkpoints")
        if not ckpts:
            dataset_hint = evidence_config.get("roberta_dataset", "essays")
            ckpts = (default_mbti_checkpoints() if framework == "mbti"
                     else default_ocean_checkpoints(dataset=dataset_hint))
        try:
            return RoBERTaEvidenceScorer(
                checkpoint_dirs=ckpts,
                device=evidence_config.get("roberta_device"),
                batch_size=evidence_config.get("roberta_batch_size", 32),
            )
        except Exception as e:
            logger.warning(f"Failed to load RoBERTa scorer: {e}; falling back to keyword")
            return None

    @staticmethod
    def _is_non_english(text: str, threshold: float = 0.3) -> bool:
        """Return True if text is predominantly non-Latin (e.g., Chinese, Japanese, Korean)."""
        if not text:
            return False
        cjk = sum(1 for c in text if '一' <= c <= '鿿' or '぀' <= c <= 'ヿ' or '가' <= c <= '힯')
        return cjk / len(text) > threshold

    def predict(self, text: str, yield_steps: bool = False) -> dict | Any:
        """
        Run the full RAG-XPR pipeline on a single text.
        If yield_steps=True, returns a generator of results.
        """
        # 1. Preprocess
        clean_text = self.preprocessor.clean(text)

        # For non-English text (e.g., Chinese TV dialogue), the English KB and
        # evidence extraction are unreliable. Skip Steps 1-2 and use the supervised
        # prior directly in Step 3 — this recovers baseline-level accuracy.
        if self._is_non_english(clean_text) and self.use_roberta_prior and self.roberta_scorer is not None:
            logger.debug("Non-English text detected — skipping Steps 1-2, using prior-only mode")
            try:
                roberta_prior = self.roberta_scorer.predict_doc_level(clean_text)
            except Exception as e:
                logger.warning(f"Prior failed on non-English text: {e}")
                roberta_prior = None
            result = self.cope_pipeline.run(
                clean_text,
                candidate_evidence=[],
                framework=self.framework,
                save_intermediate=self.save_intermediate,
                roberta_prior=roberta_prior,
            )
            if roberta_prior is not None and isinstance(result, dict):
                result.setdefault("roberta_prior", roberta_prior)
            return result

        # 2. Retrieve evidence sentences from text
        top_k_evidence = self.config.get("evidence_retrieval", {}).get("top_k", 10)
        pre_filter = self.config.get("evidence_retrieval", {}).get("pre_filter", True)
        if pre_filter:
            candidate_evidence = self.evidence_retriever.extract(clean_text, top_k=top_k_evidence)
            logger.debug(f"Found {len(candidate_evidence)} candidate evidence sentences")
        else:
            from src.retrieval.evidence_retriever import EvidenceSentence
            candidate_evidence = [EvidenceSentence(text=clean_text[:2000], sentence_idx=0, score=1.0)]
            logger.debug("Skipping evidence pre-filter (ablation)")

        # Optional: doc-level RoBERTa prior for Step 3
        roberta_prior = None
        if self.use_roberta_prior and self.roberta_scorer is not None:
            try:
                roberta_prior = self.roberta_scorer.predict_doc_level(clean_text)
                logger.debug(f"RoBERTa prior: {roberta_prior}")
            except Exception as e:
                logger.warning(f"RoBERTa prior failed: {e}")

        # 3 + 4. CoPE reasoning (retrieves KB context internally)
        if yield_steps:
            return self.cope_pipeline.run(
                clean_text,
                candidate_evidence,
                framework=self.framework,
                save_intermediate=self.save_intermediate,
                yield_steps=True,
                roberta_prior=roberta_prior,
            )

        result = self.cope_pipeline.run(
            clean_text,
            candidate_evidence,
            framework=self.framework,
            save_intermediate=self.save_intermediate,
            roberta_prior=roberta_prior,
        )
        if roberta_prior is not None and isinstance(result, dict):
            result.setdefault("roberta_prior", roberta_prior)
        return result

    def predict_batch(self, texts: list[str], show_progress: bool = True) -> list[dict]:
        """Run the pipeline on a batch of texts."""
        try:
            from tqdm import tqdm
            iterator = tqdm(texts, desc="RAG-XPR inference") if show_progress else texts
        except ImportError:
            iterator = texts

        results = []
        for text in iterator:
            try:
                result = self.predict(text)
            except Exception as e:
                logger.error(f"Pipeline failed for text: {e}")
                result = {"predicted_label": "UNKNOWN", "explanation": str(e), "evidence_chain": []}
            results.append(result)
        return results

    @classmethod
    def from_config_file(cls, config_path: str) -> "RAGXPRPipeline":
        """Build pipeline from a YAML config file."""
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(config)
