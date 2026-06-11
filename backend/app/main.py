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
    # Warm the BM25 index on startup so the first query isn't slow. Safe even
    # when the collection is empty (it just builds an empty index).
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
    try:
        n = vectorstore.count()
    except Exception:
        n = 0
    return HealthResponse(
        status="ok",
        documents_indexed=n,
        llm_configured=bool(settings.groq_api_key),
        langfuse_enabled=settings.langfuse_enabled,
        model=settings.groq_model,
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

    path = DATA_DIR / "experiments" / "results.json"
    if not path.exists():
        return {"status": "empty", "results": [], "completed": 0, "total_configs": 0}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "error", "results": [], "completed": 0, "total_configs": 0}


@app.get("/")
def root():
    return {"service": "agentic-rag", "docs": "/docs", "health": "/health"}
