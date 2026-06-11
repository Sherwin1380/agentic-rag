"""Langfuse observability with a graceful no-op fallback.

If LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set, every call here is a
cheap no-op so the app runs identically without an account. When they *are*
set, each request produces a trace with nested spans for retrieval, tool calls,
and the LLM generations — which is what makes the system look production-grade.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .config import get_settings

_client = None


def _get_client():
    global _client
    settings = get_settings()
    if not settings.langfuse_enabled:
        return None
    if _client is None:
        try:
            from langfuse import Langfuse

            _client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        except Exception:
            return None
    return _client


class _NoOpSpan:
    def update(self, **kwargs: Any) -> None: ...
    def end(self, **kwargs: Any) -> None: ...


class Trace:
    """Thin wrapper around a Langfuse trace (or a no-op)."""

    def __init__(self, name: str, user_input: str, metadata: Optional[Dict] = None):
        self._client = _get_client()
        self._trace = None
        if self._client is not None:
            try:
                self._trace = self._client.trace(
                    name=name, input=user_input, metadata=metadata or {}
                )
            except Exception:
                self._trace = None

    @property
    def id(self) -> Optional[str]:
        return getattr(self._trace, "id", None) if self._trace else None

    def span(self, name: str, **kwargs: Any):
        if self._trace is None:
            return _NoOpSpan()
        try:
            return self._trace.span(name=name, **kwargs)
        except Exception:
            return _NoOpSpan()

    def generation(self, name: str, **kwargs: Any):
        if self._trace is None:
            return _NoOpSpan()
        try:
            return self._trace.generation(name=name, **kwargs)
        except Exception:
            return _NoOpSpan()

    def update(self, **kwargs: Any) -> None:
        if self._trace is not None:
            try:
                self._trace.update(**kwargs)
            except Exception:
                pass

    def flush(self) -> None:
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:
                pass
