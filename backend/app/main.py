"""FastAPI entrypoint for the Agentic RAG backend."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import agent, retriever, vectorstore
from .config import get_settings
from .llm import LLMNotConfigured
from .models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # BM25 warming loads all chunk text into memory. Keep startup lightweight on
    # small hosts like Render free tier so the port can bind before retrieval.
    if get_settings().warm_bm25_on_startup:
        try:
            retriever.ensure_bm25()
        except Exception:
            pass
    yield


app = FastAPI(
    title="Agentic RAG over Claude API docs",
    version="1.0.0",
    description="Hybrid-search RAG with an agent that decides when to retrieve, "
    "calculate, or search the web.",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    sqlite_path = None
    sqlite_exists = False
    sqlite_size_mb = 0.0
    collection_error = None
    settings = get_settings()
    try:
        from pathlib import Path

        sqlite_path = Path(settings.chroma_path) / "chroma.sqlite3"
        sqlite_exists = sqlite_path.exists()
        sqlite_size_mb = (
            round(sqlite_path.stat().st_size / 1024 / 1024, 2)
            if sqlite_exists
            else 0.0
        )
    except Exception as exc:  # noqa: BLE001
        collection_error = f"chroma file check failed: {exc}"

    try:
        n = vectorstore.count()
    except Exception as exc:  # noqa: BLE001
        n = 0
        collection_error = str(exc)
    return HealthResponse(
        status="ok",
        documents_indexed=n,
        llm_configured=bool(settings.groq_api_key),
        langfuse_enabled=settings.langfuse_enabled,
        model=settings.groq_model,
        chroma_path=settings.chroma_path,
        collection_name=settings.collection_name,
        chroma_sqlite_exists=sqlite_exists,
        chroma_sqlite_size_mb=sqlite_size_mb,
        collection_error=collection_error,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message is required")
    try:
        result = agent.run_agent(req.message, req.history)
    except LLMNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return ChatResponse(**result)


@app.get("/search")
def search(q: str, k: int | None = None):
    """Debug endpoint: see raw hybrid-retrieval results without the LLM."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="q is required")
    chunks = retriever.hybrid_search(q, top_k=k)
    return {
        "query": q,
        "results": [
            {
                "id": c["id"],
                "title": (c.get("metadata") or {}).get("title"),
                "source": (c.get("metadata") or {}).get("source"),
                "url": (c.get("metadata") or {}).get("url"),
                "dense_score": c.get("dense_score"),
                "sparse_score": c.get("sparse_score"),
                "rrf": round(c.get("rrf", 0), 5),
                "text": c["text"][:400],
            }
            for c in chunks
        ],
    }


@app.get("/experiments")
def experiments():
    """Serve the latest embedding x chunking benchmark results for the UI page."""
    import json

    from .config import DATA_DIR

    exp_dir = DATA_DIR / "experiments"
    path = exp_dir / "results.json"
    if not path.exists():
        path = exp_dir / "reeval_chroma_results.json"
    if not path.exists():
        return {"status": "empty", "results": [], "completed": 0, "total_configs": 0}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "error", "results": [], "completed": 0, "total_configs": 0}


@app.get("/")
def root():
    return {"service": "agentic-rag", "docs": "/docs", "health": "/health"}
