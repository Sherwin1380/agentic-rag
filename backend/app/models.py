"""Pydantic request/response schemas for the API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's question")
    history: List[ChatMessage] = Field(default_factory=list)


class Source(BaseModel):
    n: int                      # citation number used in the answer, e.g. [1]
    title: str
    source: str                 # relative path / doc id
    url: Optional[str] = None
    snippet: str
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None


class AgentStep(BaseModel):
    tool: str
    arguments: Dict[str, Any]
    summary: str                # short human-readable result summary


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source] = Field(default_factory=list)
    steps: List[AgentStep] = Field(default_factory=list)
    trace_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    llm_configured: bool
    langfuse_enabled: bool
    model: str
    chroma_path: Optional[str] = None
    collection_name: Optional[str] = None
    chroma_sqlite_exists: Optional[bool] = None
    chroma_sqlite_size_mb: Optional[float] = None
    collection_error: Optional[str] = None
