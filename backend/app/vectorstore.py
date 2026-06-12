"""Local Chroma vector store wrapper.

We supply our own embeddings (from app.embeddings) rather than letting Chroma
call an embedding function, so the same model is used for ingest and query.
The collection is persisted on disk under STORAGE_DIR/chroma.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .config import get_settings

_client = None
_collection = None


def _get_client():
    global _client
    if _client is None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        settings = get_settings()
        _client = chromadb.PersistentClient(
            path=settings.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
    return _client


def get_collection():
    global _collection
    if _collection is None:
        settings = get_settings()
        _collection = _get_client().get_collection(name=settings.collection_name)
    return _collection


def reset_collection() -> None:
    """Drop and recreate the collection (used by the ingest script)."""
    global _collection
    settings = get_settings()
    client = _get_client()
    try:
        client.delete_collection(settings.collection_name)
    except Exception:
        pass
    _collection = client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(
    ids: List[str],
    texts: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
) -> None:
    get_collection().add(
        ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas
    )


def query(query_embedding: List[float], k: int) -> List[Dict[str, Any]]:
    """Return up to k nearest chunks as dicts with id/text/metadata/score."""
    col = get_collection()
    if col.count() == 0:
        return []
    n = min(k, col.count())
    res = col.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    out: List[Dict[str, Any]] = []
    if not res["ids"] or not res["ids"][0]:
        return out
    for i, _id in enumerate(res["ids"][0]):
        distance = res["distances"][0][i]
        out.append(
            {
                "id": _id,
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                # cosine distance -> similarity in [0, 1]
                "score": 1.0 - float(distance),
            }
        )
    return out


def get_all() -> Dict[str, List[Any]]:
    """Fetch every chunk (id/text/metadata) — used to build the BM25 index."""
    col = get_collection()
    if col.count() == 0:
        return {"ids": [], "documents": [], "metadatas": []}
    return col.get(include=["documents", "metadatas"])


def count() -> int:
    return get_collection().count()
