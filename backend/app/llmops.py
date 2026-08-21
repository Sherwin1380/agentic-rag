"""LLMOps infrastructure: prompt versioning, cost tracking, model routing, response logging.

Enterprise MLOps/LLMOps layer covering:
  - Prompt versioning:     every system prompt is tagged with a semantic version
  - Cost tracking:         token counts mapped to Groq pricing for spend visibility
  - Model routing:         primary model with explicit fallback on failure
  - Response logging:      every request/response appended to a JSONL audit file
  - Deployment manifest:   version snapshot for rollback support
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import STORAGE_DIR, get_settings


# ---------------------------------------------------------------------------
# Prompt Registry — semantic versioning for system prompts
# ---------------------------------------------------------------------------

PROMPT_VERSION = "v1.3.1"
PROMPT_REGISTRY: Dict[str, str] = {}


def register_prompt(version: str, text: str) -> None:
    PROMPT_REGISTRY[version] = text


def get_prompt(version: str = PROMPT_VERSION) -> Optional[str]:
    return PROMPT_REGISTRY.get(version)


# ---------------------------------------------------------------------------
# Cost Tracker — token-level cost estimation (USD per million tokens, Groq pricing)
# ---------------------------------------------------------------------------

_COST_PER_MILLION: Dict[str, Dict[str, float]] = {
    "openai/gpt-oss-20b":      {"input": 0.075, "output": 0.30},
    "openai/gpt-oss-120b":     {"input": 0.15, "output": 0.60},
    "qwen/qwen3.6-27b":        {"input": 0.60, "output": 3.00},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama3-8b-8192":          {"input": 0.05, "output": 0.08},
    "gemma2-9b-it":            {"input": 0.20, "output": 0.20},
    "mixtral-8x7b-32768":      {"input": 0.24, "output": 0.24},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost given token counts."""
    rates = _COST_PER_MILLION.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# Model Router — primary + explicit fallback strategy
# ---------------------------------------------------------------------------

@dataclass
class ModelRoute:
    primary: str
    fallback: str


def get_model_route() -> ModelRoute:
    settings = get_settings()
    return ModelRoute(primary=settings.groq_model, fallback=settings.fallback_model)


# ---------------------------------------------------------------------------
# Response Logger — JSONL audit trail for all requests/responses
# ---------------------------------------------------------------------------

_LOG_PATH = STORAGE_DIR / "response_log.jsonl"


@dataclass
class ResponseLogEntry:
    request_id: str
    timestamp: float
    prompt_version: str
    model_used: str
    fallback_used: bool
    user_message: str
    answer_preview: str
    sources_cited: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    trace_id: Optional[str]
    grounding_score: float
    hallucination_risk: str


def log_response(entry: ResponseLogEntry) -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry)) + "\n")


def read_response_log(limit: int = 100) -> List[Dict[str, Any]]:
    if not _LOG_PATH.exists():
        return []
    lines = _LOG_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[-limit:] if line.strip()]


# ---------------------------------------------------------------------------
# Deployment Manifest — rollback-ready version snapshot
# ---------------------------------------------------------------------------

_MANIFEST_PATH = STORAGE_DIR / "deployment_manifest.json"


def write_deployment_manifest(extra: Optional[Dict[str, Any]] = None) -> None:
    settings = get_settings()
    manifest = {
        "deployed_at": time.time(),
        "prompt_version": PROMPT_VERSION,
        "primary_model": settings.groq_model,
        "fallback_model": settings.fallback_model,
        "embedding_model": settings.embedding_model,
        "collection_name": settings.collection_name,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        **(extra or {}),
    }
    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_deployment_manifest() -> Optional[Dict[str, Any]]:
    if not _MANIFEST_PATH.exists():
        return None
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
