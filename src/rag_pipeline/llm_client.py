"""Unified LLM interface supporting OpenRouter/OpenAI, vLLM, and Ollama backends."""
import json
import os
import re
import time
from abc import ABC, abstractmethod

from loguru import logger


def extract_json(text: str) -> str:
    """Robustly extract a JSON value (array or object) from an LLM response.

    Strips markdown code fences, prose before/after JSON, and handles nested structures.
    Returns the raw JSON string, ready for json.loads().
    """
    s = text.strip()
    # Strip ```json or ``` fences
    if s.startswith("```"):
        # remove opening fence
        s = re.sub(r"^```(?:json)?\s*", "", s)
        # remove trailing fence
        s = re.sub(r"\s*```\s*$", "", s)
        s = s.strip()
    # Find first JSON start character
    first_brace = s.find("{")
    first_bracket = s.find("[")
    candidates = [c for c in (first_brace, first_bracket) if c >= 0]
    if not candidates:
        return s  # let json.loads fail with original
    start = min(candidates)
    # Find matching last close
    opener = s[start]
    closer = "}" if opener == "{" else "]"
    # Walk from end to find last closer
    last = s.rfind(closer)
    if last <= start:
        return s
    return s[start : last + 1]


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


class LocalTransformersClient(LLMClient):
    """Local small LLM (e.g., Qwen, Phi-3) using HuggingFace Transformers."""

    def __init__(
        self,
        model: str = "microsoft/Phi-3.5-mini-instruct",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer, logging,
                                  pipeline)

        # Silence noisy transformers warnings
        logging.set_verbosity_error()

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Loading local LLM [{model}] on {device}...")

        # Configure quantization if bitsandbytes is available and on GPU
        model_kwargs = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "device_map": "auto" if device == "cuda" else None,
            "trust_remote_code": True,
        }

        try:
            import bitsandbytes  # noqa
            if device == "cuda":
                logger.info("Enabling 4-bit quantization for local model")
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
        except ImportError:
            logger.debug("bitsandbytes not found, skipping quantization")

        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model_obj = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)

        self.pipe = pipeline(
            "text-generation",
            model=self.model_obj,
            tokenizer=self.tokenizer,
        )

    def generate(self, messages: list[dict], **kwargs) -> str:
        # Use tokenizer.apply_chat_template for robustness (Phi-3, Qwen, etc.)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        start_time = time.perf_counter()
        outputs = self.pipe(
            prompt,
            max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            do_sample=True if self.temperature > 0 else False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        latency = time.perf_counter() - start_time
        logger.debug(f"Local LLM [{self.model}] responded in {latency:.2f}s")

        generated_text = outputs[0]["generated_text"]
        # Extract response from the end of the prompt
        if prompt in generated_text:
            return generated_text[len(prompt):].strip()
        return generated_text


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
                start_time = time.perf_counter()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                    **kwargs,
                )
                latency = time.perf_counter() - start_time
                logger.info(f"LLM [{self.model}] responded in {latency:.2f}s")
                return response.choices[0].message.content
            except Exception as e:
                # Specialized handling for 429 Rate Limit
                is_rate_limit = "429" in str(e) or "rate limit" in str(e).lower()
                wait_time = (5 if is_rate_limit else 2) ** attempt

                logger.warning(
                    f"LLM API error (attempt {attempt + 1}/{self.retry_attempts}): "
                    f"{e}. {'Rate limit detected.' if is_rate_limit else ''} Retrying in {wait_time}s"
                )
                if attempt < self.retry_attempts - 1:
                    time.sleep(wait_time)
                else:
                    raise


class OpenRouterClient(OpenAIClient):
    """OpenRouter client via OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "qwen/qwen3.6-plus:free",
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
        timeout: int = 300,
        seed: int | None = None,
    ):
        import requests
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        # Deterministic decoding: forward seed into Ollama `options.seed`.
        # Falls back to OLLAMA_SEED env var when caller does not set one.
        if seed is None:
            env_seed = os.getenv("OLLAMA_SEED")
            seed = int(env_seed) if env_seed else None
        self.seed = seed
        self._requests = requests

    def generate(self, messages: list[dict], **kwargs) -> str:
        start = time.perf_counter()
        options: dict = {"temperature": self.temperature}
        if self.seed is not None:
            options["seed"] = self.seed
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        response = self._requests.post(self.base_url, json=payload, timeout=kwargs.get("timeout", self.timeout))
        response.raise_for_status()
        latency = time.perf_counter() - start
        logger.info(f"LLM [{self.model}] responded in {latency:.2f}s")
        return response.json()["message"]["content"]


def build_llm_client(config: dict) -> LLMClient:
    """Factory function to build an LLM client from config."""
    # DEFAULT TO LOCAL if not specified
    provider = config.get("provider", "local")
    # LLM_MODEL_NAME env var only overrides API-based providers, not local Ollama/vLLM
    # (where the model string is a local name, not an API identifier).
    api_providers = {"openrouter", "openai"}
    if provider in api_providers:
        model = os.getenv("LLM_MODEL_NAME") or config.get("model", "microsoft/Phi-3.5-mini-instruct")
    else:
        model = config.get("model", "microsoft/Phi-3.5-mini-instruct")
    temperature = config.get("temperature", 0.1)
    max_tokens = config.get("max_tokens", 2048)
    api_key = _resolve_api_key(provider, config.get("api_key"))

    if provider == "local":
        return LocalTransformersClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "openrouter":
        return OpenRouterClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=config.get("timeout", 60),
            retry_attempts=config.get("retry_attempts", 5),
            api_key=api_key,
            base_url=config.get("base_url", "https://openrouter.ai/api/v1"),
        )
    elif provider == "openai":
        return OpenAIClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=config.get("timeout", 60),
            retry_attempts=config.get("retry_attempts", 5),
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
            timeout=config.get("timeout", 300),
            seed=config.get("seed"),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
