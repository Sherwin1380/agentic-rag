"""Langfuse observability with a graceful no-op fallback.

If LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set, every call here is a
cheap no-op so the app runs identically without an account. When they *are*
set, each request produces a trace with nested spans for retrieval, tool calls,
and the LLM generations — which is what makes the system look production-grade.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

from .config import get_settings

_log = logging.getLogger(__name__)

_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    settings = get_settings()
    if not settings.langfuse_enabled:
        return None
    # Double-checked locking: without the lock, two concurrent first-requests
    # each build a Langfuse client, orphaning one queue + consumer thread so
    # half the events never ship.
    if _client is None:
        with _client_lock:
            if _client is None:
                try:
                    from langfuse import Langfuse

                    _client = Langfuse(
                        public_key=settings.langfuse_public_key,
                        secret_key=settings.langfuse_secret_key,
                        host=settings.langfuse_host,
                        # Batch a request's events into ONE POST (flush_at=1
                        # forces a separate slow round-trip per event — ~4x
                        # slower to drain).
                        flush_interval=0.5,
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
        # Flush SYNCHRONOUSLY within the request.  A fire-and-forget daemon
        # thread (or relying on the background consumer) proved unreliable on
        # Render: events sat queued until the process got a SIGTERM on deploy,
        # which then drained them via the SDK's atexit handler — so traces only
        # appeared after a redeploy.  Blocking here completes the HTTP POST
        # while the container is provably awake.  Healthy flush is ~0.3-0.8s.
        #
        # Logged (not silently swallowed) so Render logs reveal whether the
        # flush actually completes per-request: a slow/failing flush here is
        # the cause of "events from 10 min ago all appear at once" — the queue
        # backs up and a later successful flush drains the whole backlog.
        if self._client is None:
            return
        t0 = time.perf_counter()
        try:
            self._client.flush()
            dt = time.perf_counter() - t0
            if dt > 3.0:
                _log.warning("langfuse flush slow: %.1fs (trace %s)", dt, self.id)
            else:
                _log.debug("langfuse flush ok: %.2fs (trace %s)", dt, self.id)
        except Exception as exc:
            _log.error(
                "langfuse flush FAILED after %.1fs (trace %s): %s",
                time.perf_counter() - t0, self.id, exc,
            )
