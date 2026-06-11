"""Central configuration, loaded from environment / .env file.

Nothing here requires a paid account. The only key needed to *run* the agent is
GROQ_API_KEY (free at https://console.groq.com). Embeddings run locally, the
vector store is local Chroma, and Langfuse observability is optional.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve important paths relative to the backend/ directory so the app behaves
# the same whether launched from the repo root, backend/, or a container.
BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
EVAL_DIR = DATA_DIR / "eval"
STORAGE_DIR = BACKEND_DIR / "storage"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (Groq) ---
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1
    max_agent_steps: int = 5  # safety cap on the tool-use loop

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Vector store / retrieval ---
    chroma_path: str = str(STORAGE_DIR / "chroma")
    collection_name: str = "banking_regs"
    chunk_size: int = 900       # characters per chunk (~220 tokens)
    chunk_overlap: int = 150
    top_k: int = 5              # final chunks passed to the LLM
    dense_k: int = 12           # candidates from vector search before fusion
    sparse_k: int = 12          # candidates from BM25 before fusion
    rrf_k: int = 60             # reciprocal-rank-fusion constant

    # --- Tools ---
    enable_web_search: bool = True

    # --- Observability (optional, free Langfuse cloud tier) ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- API ---
    cors_origins: str = "*"  # comma-separated; "*" allows the local Next.js dev server

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def cors_origins_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
