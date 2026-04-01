"""Unified LLM interface supporting OpenRouter/OpenAI, vLLM, and Ollama backends."""
import json
import os
import time
from abc import ABC, abstractmethod

from loguru import logger


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, messages: list[dict], **kwargs) -> str:
        """Generate a response given a list of chat messages."""
        ...

    def generate_json(self, messages: list[dict], **kwargs) -> dict:
        """Generate and parse a JSON response."""
        response = self.generate(messages, **kwargs)
        content = response.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)


class OpenAIClient(LLMClient):
    """OpenAI GPT-4o / GPT-4o-mini client via API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: int = 60,
        retry_attempts: int = 3,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        import openai
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, messages: list[dict], **kwargs) -> str:
        for attempt in range(self.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                    **kwargs,
                )
                return response.choices[0].message.content
            except Exception as e:
                wait_time = 2 ** attempt
                logger.warning(
                    f"LLM API error (attempt {attempt + 1}/{self.retry_attempts}): "
                    f"{e}. Retrying in {wait_time}s"
                )
                if attempt < self.retry_attempts - 1:
                    time.sleep(wait_time)
                else:
                    raise


class OpenRouterClient(OpenAIClient):
    """OpenRouter client via OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "qwen/qwen3.6-plus-preview:free",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: int = 60,
        retry_attempts: int = 3,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retry_attempts=retry_attempts,
            api_key=api_key,
            base_url=base_url,
        )


def _resolve_api_key(provider: str, explicit_api_key: str | None = None) -> str | None:
    """Resolve API key with universal env var first, provider-specific fallbacks second."""
    if explicit_api_key:
        return explicit_api_key

    universal_key = os.getenv("LLM_API_KEY")
    if universal_key:
        return universal_key

    if provider == "openrouter":
        return os.getenv("OPENROUTER_API_KEY")
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    return None


class VLLMClient(LLMClient):
    """Local open-source models via vLLM OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        import openai
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(base_url=base_url, api_key="dummy")

    def generate(self, messages: list[dict], **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content


class OllamaClient(LLMClient):
    """Local quick experiments via Ollama server."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434/api/chat",
        temperature: float = 0.1,
    ):
        import requests
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self._requests = requests

    def generate(self, messages: list[dict], **kwargs) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        response = self._requests.post(self.base_url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["message"]["content"]


def build_llm_client(config: dict) -> LLMClient:
    """Factory function to build an LLM client from config."""
    provider = config.get("provider", "openrouter")
    model = os.getenv("LLM_MODEL_NAME") or config.get("model", "qwen/qwen3.6-plus-preview:free")
    temperature = config.get("temperature", 0.1)
    max_tokens = config.get("max_tokens", 2048)
    api_key = _resolve_api_key(provider, config.get("api_key"))

    if provider == "openrouter":
        return OpenRouterClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=config.get("timeout", 60),
            retry_attempts=config.get("retry_attempts", 3),
            api_key=api_key,
            base_url=config.get("base_url", "https://openrouter.ai/api/v1"),
        )
    elif provider == "openai":
        return OpenAIClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=config.get("timeout", 60),
            retry_attempts=config.get("retry_attempts", 3),
            api_key=api_key,
            base_url=config.get("base_url"),
        )
    elif provider == "vllm":
        return VLLMClient(
            base_url=config.get("base_url", "http://localhost:8000/v1"),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "ollama":
        return OllamaClient(
            model=model,
            base_url=config.get("base_url", "http://localhost:11434/api/chat"),
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
