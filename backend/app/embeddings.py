"""Local embedding model wrapper.

Uses sentence-transformers on CPU. The model (~90 MB) is downloaded once on
first use and cached by HuggingFace under ~/.cache. No API key required.
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


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of documents."""
    model = _get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_query(text: str) -> List[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
