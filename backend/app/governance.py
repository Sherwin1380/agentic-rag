"""Governance and risk layer: hallucination mitigation, grounding validation, audit logging.

Implements enterprise AI governance requirements:
  - Source-grounded generation enforcement (citations must map to retrieved chunks)
  - Hallucination risk scoring (detect answers that lack retrieval grounding)
  - Retrieval traceability (full audit trail: queries issued, chunks retrieved, scores)
  - Citation validation (every [n] in the answer must map to a real retrieved source)
  - Compliance auditability (append-only JSONL audit log; never mutated after write)
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import STORAGE_DIR

_AUDIT_LOG_PATH = STORAGE_DIR / "audit_log.jsonl"

_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Grounding validation
# ---------------------------------------------------------------------------

def compute_grounding_score(answer: str, sources: List[Any]) -> float:
    """Return 0–1 score: fraction of answer sentences that contain a citation.

    1.0 = every sentence is grounded in a retrieved source.
    < 0.5 = model is likely drawing on parametric knowledge.
    """
    if not answer.strip() or not sources:
        return 0.0
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer) if s.strip()]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if _CITATION_RE.search(s))
    return round(cited / len(sentences), 3)


def assess_hallucination_risk(
    answer: str, sources: List[Any], grounding_score: float
) -> str:
    """Classify hallucination risk as 'low', 'medium', or 'high'.

    Heuristics (in priority order):
      no sources retrieved        → high  (zero factual grounding)
      grounding_score < 0.30      → medium (most sentences uncited)
      otherwise                   → low
    """
    if not sources:
        return "high"
    if grounding_score < 0.30:
        return "medium"
    return "low"


def validate_citations(answer: str, sources: List[Any]) -> Dict[str, Any]:
    """Check that every [n] citation in the answer maps to a real source object."""
    cited_nums = {int(m) for m in _CITATION_RE.findall(answer)}
    valid_nums = {s.n for s in sources}
    orphaned = cited_nums - valid_nums
    uncited = valid_nums - cited_nums
    return {
        "cited": sorted(cited_nums),
        "valid": sorted(valid_nums),
        "orphaned_citations": sorted(orphaned),
        "uncited_sources": sorted(uncited),
        "all_citations_valid": len(orphaned) == 0,
    }


# ---------------------------------------------------------------------------
# Retrieval traceability
# ---------------------------------------------------------------------------

def build_retrieval_trace(steps: List[Any], sources: List[Any]) -> Dict[str, Any]:
    """Summarise the full retrieval chain for explainability and auditability."""
    retrieval_steps = [s for s in steps if s.tool == "search_documentation"]
    return {
        "num_retrieval_calls": len(retrieval_steps),
        "total_sources_retrieved": len(sources),
        "queries_issued": [s.arguments.get("query") for s in retrieval_steps],
        "source_ids": [s.source for s in sources],
        "dense_scores": [s.dense_score for s in sources],
        "sparse_scores": [s.sparse_score for s in sources],
    }


# ---------------------------------------------------------------------------
# Audit log — append-only, immutable record for compliance
# ---------------------------------------------------------------------------

def audit_log(
    *,
    request_id: str,
    user_message: str,
    answer: str,
    grounding_score: float,
    hallucination_risk: str,
    citation_check: Dict[str, Any],
    retrieval_trace: Dict[str, Any],
    model: str,
    prompt_version: str,
    trace_id: Optional[str],
) -> None:
    _AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.time(),
        "request_id": request_id,
        "model": model,
        "prompt_version": prompt_version,
        "trace_id": trace_id,
        "user_message": user_message[:500],
        "answer_preview": answer[:300],
        "grounding_score": grounding_score,
        "hallucination_risk": hallucination_risk,
        "citation_check": citation_check,
        "retrieval_trace": retrieval_trace,
    }
    with open(_AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_audit_log(limit: int = 50) -> List[Dict[str, Any]]:
    if not _AUDIT_LOG_PATH.exists():
        return []
    lines = _AUDIT_LOG_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[-limit:] if line.strip()]
