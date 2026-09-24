"""LLM chat clients for Anthropic and OpenAI.

Both providers expose the same minimal interface — ``complete(system, user)``
— so the RAG pipeline is provider-agnostic. The SDKs are imported lazily so
that installing only one provider's package is sufficient to run.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

# Current Anthropic models (as of mid-2026). Versioned strings are used
# deliberately so behaviour is reproducible.
ANTHROPIC_MODELS = {
    "claude-sonnet-4-6": "Balanced quality and cost (recommended default)",
    "claude-opus-4-8": "Most capable; best for hard reasoning",
    "claude-haiku-4-5-20251001": "Fastest and cheapest",
}

OPENAI_MODELS = {
    "gpt-4o-mini": "Fast and inexpensive (recommended default)",
    "gpt-4o": "Higher quality, higher cost",
}


class LLMClient(ABC):
    """Abstract chat client."""

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Return the assistant's text response for a single-turn prompt."""


class AnthropicClient(LLMClient):
    """Anthropic Claude chat client."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        from anthropic import Anthropic  # lazy import

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self.client = Anthropic(api_key=key)
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Concatenate all text blocks in the response.
        return "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        ).strip()


class OpenAIClient(LLMClient):
    """OpenAI chat client."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        from openai import OpenAI  # lazy import

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=key)
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


def build_client(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
) -> LLMClient:
    """Factory that builds the appropriate client for a provider.

    Args:
        provider: "anthropic" or "openai".
        model: Optional model override; falls back to the provider default.
        api_key: Optional explicit API key.
    """
    provider = provider.lower()
    if provider == "anthropic":
        return AnthropicClient(model=model or "claude-sonnet-4-6", api_key=api_key)
    if provider == "openai":
        return OpenAIClient(model=model or "gpt-4o-mini", api_key=api_key)
    raise ValueError(f"Unknown provider '{provider}'. Use 'anthropic' or 'openai'.")
