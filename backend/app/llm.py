"""Groq chat-completions wrapper (Llama 3.3 70B by default).

Kept deliberately thin: the agent loop in agent.py owns the tool-calling logic;
this module only handles the raw Groq call and surfaces a clean error if the
API key is missing.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import get_settings

_client = None


class LLMNotConfigured(RuntimeError):
    pass


def _get_client():
    global _client
    settings = get_settings()
    if not settings.groq_api_key:
        raise LLMNotConfigured(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and put it in backend/.env"
        )
    if _client is None:
        from groq import Groq

        _client = Groq(api_key=settings.groq_api_key)
    return _client


def chat(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    temperature: Optional[float] = None,
) -> Any:
    """Single chat completion. Returns the raw Groq message object."""
    settings = get_settings()
    client = _get_client()
    kwargs: Dict[str, Any] = {
        "model": settings.groq_model,
        "messages": messages,
        "temperature": settings.llm_temperature if temperature is None else temperature,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    completion = client.chat.completions.create(**kwargs)
    return completion
