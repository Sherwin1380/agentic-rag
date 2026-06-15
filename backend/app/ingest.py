"""Corpus loading, chunking, and index building.

Corpus files are Markdown with a tiny frontmatter block:

    ---
    title: Models overview
    url: https://www.ecfr.gov/current/title-12/section-1.1
    ---
    # body...

Chunking is paragraph-aware: we accumulate paragraphs up to ~chunk_size
characters with a sliding overlap, which keeps related sentences together while
staying small enough for precise retrieval.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from .config import CORPUS_DIR, DATA_DIR, get_settings
from . import embeddings, vectorstore, retriever
from .warehouse import get_warehouse

BANKING_JSONL = DATA_DIR / "banking" / "sections.jsonl"


def parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    meta: Dict[str, str] = {}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            block = text[3:end].strip()
            body = text[end + 4 :].lstrip("\n")
            for line in block.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
    return meta, body


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > size:
                # Hard-split an oversized paragraph.
                start = 0
                while start < len(para):
                    chunks.append(para[start : start + size])
                    start += size - overlap
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)

    # Add a sliding character overlap between adjacent chunks for context bleed.
    if overlap > 0 and len(chunks) > 1:
        overlapped: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append(f"{tail} {chunks[i]}".strip())
        chunks = overlapped
    return chunks


def load_corpus() -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw)
        docs.append(
            {
                "source": path.name,
                "title": meta.get("title", path.stem),
                "url": meta.get("url", ""),
                "text": body,
            }
        )
    return docs


def build_index(verbose: bool = True) -> int:
    settings = get_settings()
    docs = load_corpus()
    if not docs:
        raise RuntimeError(f"No .md files found in {CORPUS_DIR}")

    vectorstore.reset_collection()

    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, str]] = []
    for doc in docs:
        chunks = chunk_text(doc["text"], settings.chunk_size, settings.chunk_overlap)
        for i, chunk in enumerate(chunks):
            ids.append(f"{doc['source']}::chunk-{i}")
            texts.append(chunk)
            metadatas.append(
                {
                    "source": doc["source"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "chunk_index": i,
                }
            )
        if verbose:
            print(f"  {doc['source']}: {len(chunks)} chunks")

    if verbose:
        print(f"Embedding {len(texts)} chunks with {settings.embedding_model} ...")

    # Embed in batches to keep memory bounded.
    batch = 64
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch):
        vectors.extend(embeddings.embed_texts(texts[i : i + batch]))

    vectorstore.add_documents(ids, texts, vectors, metadatas)
    retriever.refresh()

    if verbose:
        print(f"Indexed {len(texts)} chunks from {len(docs)} documents.")
    return len(texts)


def load_jsonl_corpus(path: Path) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def build_banking_index(
    jsonl_path: Path = BANKING_JSONL, verbose: bool = True, batch: int = 128
) -> int:
    """Build the Chroma index from the large eCFR banking-regulations corpus.

    Each JSONL record is one CFR section; we chunk it and carry category/part/
    section metadata so retrieval can be evaluated per category. Embedding and
    indexing are streamed in batches to keep memory bounded at scale.
    """
    settings = get_settings()
    docs = load_jsonl_corpus(jsonl_path)
    if not docs:
        raise RuntimeError(f"No records found in {jsonl_path}. Run fetch_ecfr.py first.")

    vectorstore.reset_collection()

    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, object]] = []
    for doc in docs:
        chunks = chunk_text(doc["text"], settings.chunk_size, settings.chunk_overlap)
        for i, chunk in enumerate(chunks):
            ids.append(f"{doc['id']}::c{i}")
            texts.append(chunk)
            metadatas.append(
                {
                    "source": doc["id"],
                    "title": doc["title"],
                    "url": doc.get("url", ""),
                    "category": doc.get("category", ""),
                    "part": str(doc.get("part", "")),
                    "section": str(doc.get("section", "")),
                    "chunk_index": i,
                }
            )

    total = len(texts)
    if verbose:
        print(f"{len(docs)} sections -> {total} chunks; embedding in batches of {batch} ...")

    for start in range(0, total, batch):
        end = min(start + batch, total)
        vectors = embeddings.embed_texts(texts[start:end])
        vectorstore.add_documents(
            ids[start:end], texts[start:end], vectors, metadatas[start:end]
        )
        if verbose and (start // batch) % 25 == 0:
            print(f"  embedded {end}/{total}")

    retriever.refresh()

    # --- Warehouse pipeline: RAW → STAGING → ENRICHED ---
    # Mirrors a Snowflake 3-tier data architecture so corpus lineage is auditable.
    try:
        wh = get_warehouse()
        wh.write_raw(docs)
        wh.write_staging(docs)
        enriched = [
            {
                "chunk_id": ids[i],
                "section_id": metadatas[i].get("source", ""),
                "title": metadatas[i].get("title", ""),
                "category": metadatas[i].get("category", ""),
                "part": metadatas[i].get("part", ""),
                "section": metadatas[i].get("section", ""),
                "url": metadatas[i].get("url", ""),
                "chunk_index": metadatas[i].get("chunk_index", 0),
                "char_count": len(texts[i]),
                "embedding_model": settings.embedding_model,
            }
            for i in range(len(ids))
        ]
        wh.write_enriched(enriched)
        if verbose:
            print(
                f"Warehouse: raw={wh.count('raw')} "
                f"staging={wh.count('staging')} "
                f"enriched={wh.count('enriched')}"
            )
    except Exception as exc:  # noqa: BLE001 - warehouse failures never block indexing
        if verbose:
            print(f"[warehouse] skipped: {exc}")

    if verbose:
        print(f"Indexed {total} chunks from {len(docs)} sections.")
    return total
