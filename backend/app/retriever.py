"""Hybrid retrieval: dense (Chroma) + sparse (BM25), fused with RRF.

Reciprocal Rank Fusion (RRF) combines two ranked lists without needing the
scores to be on the same scale, which makes it robust when mixing cosine
similarity with BM25 term scores.
"""
from __future__ import annotations

import re
import threading
from typing import Any, Dict, List, Optional

from .config import get_settings
from . import embeddings, vectorstore

_bm25 = None
_bm25_ids: List[str] = []
_bm25_docs: List[Dict[str, Any]] = []  # parallel to _bm25_ids: {id,text,metadata}
_bm25_lock = threading.Lock()

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _build_bm25() -> None:
    """(Re)build the in-memory BM25 index from everything in Chroma."""
    global _bm25, _bm25_ids, _bm25_docs
    from rank_bm25 import BM25Okapi

    data = vectorstore.get_all()
    ids = data["ids"]
    docs = data["documents"]
    metas = data["metadatas"]

    _bm25_ids = ids
    _bm25_docs = [
        {"id": ids[i], "text": docs[i], "metadata": metas[i]} for i in range(len(ids))
    ]
    if not ids:
        _bm25 = None
        return
    tokenized = [_tokenize(d) for d in docs]
    _bm25 = BM25Okapi(tokenized)


def ensure_bm25() -> None:
    if _bm25 is None and not _bm25_ids:
        with _bm25_lock:
            if _bm25 is None and not _bm25_ids:
                _build_bm25()


def refresh() -> None:
    """Force a rebuild — call after re-ingesting."""
    global _bm25, _bm25_ids, _bm25_docs
    with _bm25_lock:
        _bm25 = None
        _bm25_ids = []
        _bm25_docs = []
        _build_bm25()


def _sparse_search(query: str, k: int) -> List[Dict[str, Any]]:
    ensure_bm25()
    if _bm25 is None:
        return []
    scores = _bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {**_bm25_docs[i], "score": float(scores[i])}
        for i in ranked
        if scores[i] > 0
    ]


def _dense_search(query: str, k: int) -> List[Dict[str, Any]]:
    qv = embeddings.embed_query(query)
    return vectorstore.query(qv, k)


def hybrid_search(
    query: str, top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Run both retrievers and fuse with RRF. Returns top_k chunk dicts."""
    settings = get_settings()
    top_k = top_k or settings.top_k

    dense = _dense_search(query, settings.dense_k)
    sparse = _sparse_search(query, settings.sparse_k) if settings.enable_sparse_bm25 else []

    fused: Dict[str, Dict[str, Any]] = {}

    def add(results: List[Dict[str, Any]], source: str) -> None:
        for rank, item in enumerate(results):
            _id = item["id"]
            entry = fused.setdefault(
                _id,
                {
                    "id": _id,
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "rrf": 0.0,
                    "dense_score": None,
                    "sparse_score": None,
                },
            )
            entry["rrf"] += 1.0 / (settings.rrf_k + rank + 1)
            entry[f"{source}_score"] = round(item["score"], 4)

    add(dense, "dense")
    add(sparse, "sparse")

    ranked = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)
    return ranked[:top_k]
