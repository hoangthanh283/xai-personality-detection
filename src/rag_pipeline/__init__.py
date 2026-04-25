"""RAG pipeline integration modules."""

from .llm_client import OllamaClient, OpenAIClient, OpenRouterClient, VLLMClient

__all__ = ["RAGXPRPipeline", "OpenAIClient", "OpenRouterClient", "VLLMClient", "OllamaClient"]


def __getattr__(name: str):
    """Lazily import the heavy pipeline to avoid CoPE import cycles."""
    if name == "RAGXPRPipeline":
        from .pipeline import RAGXPRPipeline

        return RAGXPRPipeline
    raise AttributeError(name)
