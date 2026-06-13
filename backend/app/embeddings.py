"""Embedding model wrapper.

Uses sentence-transformers on CPU. The model is downloaded once on first use and
cached by HuggingFace. Some retrieval models, notably E5 and BGE, require query
and passage prefixes; this module applies those consistently for ingest/query.
"""
from __future__ import annotations

import threading
from typing import List

from .config import get_settings

_model = None
_lock = threading.Lock()


def _get_model():
    """Lazily load the embedding model exactly once (thread-safe)."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                # Imported lazily so that importing this module (e.g. for type
                # hints or config) does not pull in torch until embeddings are
                # actually needed.
                from sentence_transformers import SentenceTransformer

                settings = get_settings()
                _model = SentenceTransformer(settings.embedding_model, device="cpu")
    return _model


def _prefix(kind: str) -> str:
    settings = get_settings()
    model_name = settings.embedding_model.lower()

    if kind == "query" and settings.embedding_query_prefix:
        return _normalize_prefix(settings.embedding_query_prefix)
    if kind == "passage" and settings.embedding_passage_prefix:
        return _normalize_prefix(settings.embedding_passage_prefix)

    if "e5" in model_name:
        return "query: " if kind == "query" else "passage: "
    if "bge" in model_name and kind == "query":
        return "Represent this sentence for searching relevant passages: "
    return ""


def _normalize_prefix(prefix: str) -> str:
    if prefix.endswith(":"):
        return f"{prefix} "
    return prefix


def _embed(texts: List[str], kind: str) -> List[List[float]]:
    prefix = _prefix(kind)
    if prefix:
        texts = [prefix + text for text in texts]
    model = _get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of documents."""
    return _embed(texts, kind="passage")


def embed_query(text: str) -> List[float]:
    """Embed a single query string."""
    return _embed([text], kind="query")[0]
