"""Main RAG-XPR pipeline orchestrator.

Ties together:
1. Text preprocessing
2. Evidence retrieval from input text
3. KB retrieval (psychology definitions)
4. CoPE reasoning (3-step LLM chain)
"""
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
        self.evidence_retriever = EvidenceRetriever(evidence_config)

        # KB retriever (semantic, BM25, or hybrid)
        method = evidence_config.get("method", "semantic")
        if method == "hybrid":
            chunks_path = config.get("kb_chunks_path")
            self.kb_retriever = HybridRetriever(retrieval_config, chunks_path)
        else:
            self.kb_retriever = KBRetriever(retrieval_config)

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

    def predict(self, text: str) -> dict:
        """
        Run the full RAG-XPR pipeline on a single text.

        Returns:
            dict with keys: predicted_label, prediction_details,
                           explanation, evidence_chain, [intermediate]
        """
        # 1. Preprocess
        clean_text = self.preprocessor.clean(text)

        # 2. Retrieve evidence sentences from text
        top_k_evidence = self.config.get("evidence_retrieval", {}).get("top_k", 10)
        candidate_evidence = self.evidence_retriever.extract(clean_text, top_k=top_k_evidence)
        logger.debug(f"Found {len(candidate_evidence)} candidate evidence sentences")

        # 3 + 4. CoPE reasoning (retrieves KB context internally)
        result = self.cope_pipeline.run(
            clean_text,
            candidate_evidence,
            framework=self.framework,
            save_intermediate=self.save_intermediate,
        )
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
