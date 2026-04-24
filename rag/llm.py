"""LLM provider abstraction with OpenAI (default) and Groq (fallback)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()


OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"

PROVIDER_OPENAI = "openai"
PROVIDER_GROQ = "groq"
SUPPORTED_PROVIDERS = (PROVIDER_OPENAI, PROVIDER_GROQ)


class LLMError(RuntimeError):
    """Raised when no LLM provider can fulfill the request."""


@dataclass
class LLMResponse:
    """Result of an LLM generation call."""

    text: str
    provider: str
    model: str
    fallback_used: bool = False
    fallback_reason: Optional[str] = None


def available_providers() -> List[str]:
    """Return providers whose API key is present in the environment."""
    available: List[str] = []
    if os.getenv("OPENAI_API_KEY"):
        available.append(PROVIDER_OPENAI)
    if os.getenv("GROQ_API_KEY"):
        available.append(PROVIDER_GROQ)
    return available


def _default_model_for(provider: str) -> str:
    return OPENAI_DEFAULT_MODEL if provider == PROVIDER_OPENAI else GROQ_DEFAULT_MODEL


def _call_openai(system_prompt: str, user_prompt: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content or ""


def _call_groq(system_prompt: str, user_prompt: str, model: str) -> str:
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content or ""


def _resolve_provider(requested: str) -> Tuple[str, bool, Optional[str]]:
    """Pick a provider, falling back to the other if the requested key is missing."""
    if requested not in SUPPORTED_PROVIDERS:
        raise LLMError(f"Unknown provider: {requested!r}")

    available = available_providers()
    if not available:
        raise LLMError(
            "No LLM provider is configured. "
            "Set OPENAI_API_KEY or GROQ_API_KEY in your .env file."
        )

    if requested in available:
        return requested, False, None

    fallback = available[0]
    reason = f"{requested.upper()}_API_KEY is not set; using {fallback} instead."
    return fallback, True, reason


def generate(
    system_prompt: str,
    user_prompt: str,
    provider: str = PROVIDER_OPENAI,
    model: Optional[str] = None,
) -> LLMResponse:
    """Generate an answer using `provider`, falling back to the other if needed."""
    chosen, fallback_used, reason = _resolve_provider(provider)
    chosen_model = model or _default_model_for(chosen)

    if chosen == PROVIDER_OPENAI:
        text = _call_openai(system_prompt, user_prompt, chosen_model)
    else:
        text = _call_groq(system_prompt, user_prompt, chosen_model)

    return LLMResponse(
        text=text,
        provider=chosen,
        model=chosen_model,
        fallback_used=fallback_used,
        fallback_reason=reason,
    )
