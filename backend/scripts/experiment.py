"""Embedding-model x chunking-strategy benchmark over the banking eval.

For each (embedding model, chunking strategy) we:
  1. chunk a bounded experiment corpus (all eval ground-truth sections + a fixed
     pool of distractor sections),
  2. embed the chunks (passages) and the 100 eval queries,
  3. persist the passage vectors to a per-config Chroma collection,
  4. retrieve with dense-only and with hybrid (dense + BM25 + RRF),
  5. score MRR / Hit@k / Recall@k overall and per category.

Results stream to data/experiments/results.json after every config so the UI can
show progress live and a crash never loses completed work.

Each config is stored in its own experiment Chroma collection so the winning
index can be promoted without re-paying to embed the same corpus. The eval itself
still uses an in-memory NumPy matrix and BM25 index for fast, repeatable scoring.

Run from backend/:
    python scripts/experiment.py                  # full sweep
    python scripts/experiment.py --models bge-small-en-v1.5 --distractors 500
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATA_DIR, STORAGE_DIR, get_settings  # noqa: E402
from app.ingest import chunk_text, load_jsonl_corpus  # noqa: E402

SECTIONS = DATA_DIR / "banking" / "sections.jsonl"
QA = DATA_DIR / "banking" / "qa.jsonl"
OUT_DIR = DATA_DIR / "experiments"
RESULTS = OUT_DIR / "results.json"
WINNER = OUT_DIR / "winner.json"
EXPERIMENT_CHROMA_PATH = STORAGE_DIR / "experiment_chroma"

K = 5
DENSE_K = 12
SPARSE_K = 12
RRF_K = 60

# Embedding models. `kind` is "st" (local sentence-transformers) or "openai".
# query_prefix / passage_prefix implement each model's recommended retrieval
# instructions (BGE and E5 need them; MiniLM/GTE/OpenAI do not).
MODELS = [
    {"name": "all-MiniLM-L6-v2", "kind": "st",
     "model_id": "sentence-transformers/all-MiniLM-L6-v2"},
    {"name": "bge-small-en-v1.5", "kind": "st", "model_id": "BAAI/bge-small-en-v1.5",
     "query_prefix": "Represent this sentence for searching relevant passages: "},
    {"name": "gte-small", "kind": "st", "model_id": "thenlper/gte-small"},
    {"name": "e5-small-v2", "kind": "st", "model_id": "intfloat/e5-small-v2",
     "query_prefix": "query: ", "passage_prefix": "passage: "},
    {"name": "text-embedding-3-small", "kind": "openai",
     "model_id": "text-embedding-3-small"},
    {"name": "text-embedding-3-large", "kind": "openai",
     "model_id": "text-embedding-3-large"},
]

# Chunking grid: 3 sizes x {overlap on, overlap off}.
CHUNKINGS = []
for size in (400, 900, 1500):
    CHUNKINGS.append({"size": size, "overlap": int(round(size * 0.17))})
    CHUNKINGS.append({"size": size, "overlap": 0})

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())


def slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return value[:48] or "config"


def collection_name(model_name: str, size: int, overlap: int, full_corpus: bool) -> str:
    scope = "full" if full_corpus else "sample"
    return f"banking_exp_{scope}_{slug(model_name)}_{size}_{overlap}"


# --------------------------------------------------------------------------- #
# Embedders
# --------------------------------------------------------------------------- #
class STEmbedder:
    def __init__(self, model_id, query_prefix="", passage_prefix="", device="auto"):
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device
        self.model = SentenceTransformer(model_id, device=device)
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts, prefix, batch=64):
        texts = [prefix + t for t in texts]
        v = self.model.encode(texts, normalize_embeddings=True, batch_size=batch,
                              show_progress_bar=False, convert_to_numpy=True)
        return v.astype(np.float32)

    def passages(self, texts):
        return self.encode(texts, self.passage_prefix)

    def queries(self, texts):
        return self.encode(texts, self.query_prefix)


class OpenAIEmbedder:
    def __init__(self, model_id):
        import os
        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_id = model_id
        self.dim = None

    def _embed(self, texts, batch=256):
        out = []
        for i in range(0, len(texts), batch):
            chunk = [t.replace("\n", " ") for t in texts[i : i + batch]]
            for attempt in range(4):
                try:
                    resp = self.client.embeddings.create(model=self.model_id, input=chunk)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 * (attempt + 1))
            out.extend([d.embedding for d in resp.data])
        arr = np.asarray(out, dtype=np.float32)
        # L2-normalize so cosine == dot product.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.clip(norms, 1e-9, None)
        self.dim = arr.shape[1]
        return arr

    def passages(self, texts):
        return self._embed(texts)

    def queries(self, texts):
        return self._embed(texts)


def make_embedder(spec, device="auto"):
    if spec["kind"] == "st":
        return STEmbedder(spec["model_id"], spec.get("query_prefix", ""),
                          spec.get("passage_prefix", ""), device=device)
    return OpenAIEmbedder(spec["model_id"])


# --------------------------------------------------------------------------- #
# Retrieval + metrics
# --------------------------------------------------------------------------- #
def build_chunks(sections, size, overlap):
    chunk_ids, chunk_texts, chunk_sources, chunk_metadatas = [], [], [], []
    for sec in sections:
        chunks = chunk_text(sec["text"], size, overlap)
        for i, ch in enumerate(chunks):
            chunk_ids.append(f"{sec['id']}::c{i}")
            chunk_texts.append(ch)
            chunk_sources.append(sec["id"])
            chunk_metadatas.append(
                {
                    "source": sec["id"],
                    "title": sec.get("title", sec["id"]),
                    "url": sec.get("url", ""),
                    "category": sec.get("category", ""),
                    "part": str(sec.get("part", "")),
                    "section": str(sec.get("section", "")),
                    "chunk_index": i,
                    "chunk_size": size,
                    "chunk_overlap": overlap,
                }
            )
    return chunk_ids, chunk_texts, chunk_sources, chunk_metadatas


def persist_vectors(
    name,
    model_spec,
    chunking,
    ids,
    texts,
    vectors,
    metadatas,
    batch=1000,
):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    EXPERIMENT_CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(EXPERIMENT_CHROMA_PATH),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )
    try:
        client.delete_collection(name)
    except Exception:
        pass

    col = client.get_or_create_collection(
        name=name,
        metadata={
            "hnsw:space": "cosine",
            "model": model_spec["name"],
            "model_id": model_spec["model_id"],
            "kind": model_spec["kind"],
            "chunk_size": chunking["size"],
            "chunk_overlap": chunking["overlap"],
        },
    )
    vector_lists = vectors.astype(np.float32).tolist()
    for start in range(0, len(ids), batch):
        end = min(start + batch, len(ids))
        col.add(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=vector_lists[start:end],
            metadatas=metadatas[start:end],
        )
    return col.count()


def delete_collections(collections):
    if not collections:
        return
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.PersistentClient(
        path=str(EXPERIMENT_CHROMA_PATH),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )
    for name in collections:
        try:
            client.delete_collection(name)
        except Exception:
            pass


def rrf_fuse(dense_idx, sparse_idx):
    scores = defaultdict(float)
    for rank, idx in enumerate(dense_idx):
        scores[idx] += 1.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(sparse_idx):
        scores[idx] += 1.0 / (RRF_K + rank + 1)
    return [idx for idx, _ in sorted(scores.items(), key=lambda x: -x[1])]


def metrics_for(ranked_sources, relevant, k):
    top = ranked_sources[:k]
    found = [s for s in top if s in relevant]
    hit = 1.0 if found else 0.0
    recall = (len(set(found)) / len(relevant)) if relevant else 0.0
    precision = len(found) / len(top) if top else 0.0
    rr = 0.0
    for rank, s in enumerate(top, start=1):
        if s in relevant:
            rr = 1.0 / rank
            break
    dcg = sum(1.0 / np.log2(rank + 1) for rank, s in enumerate(top, start=1) if s in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg else 0.0
    return hit, rr, recall, precision, ndcg


def evaluate_config(matrix, embedder, chunk_texts, chunk_sources, queries, qa, k=K):
    qvecs = embedder.queries(queries)                  # (Q, d) normalized

    # BM25 over chunks (shared structure; depends only on chunking).
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([tokenize(t) for t in chunk_texts])

    agg = {"dense": defaultdict(float), "hybrid": defaultdict(float)}
    cat_hybrid_rr = defaultdict(float)
    cat_n = defaultdict(int)

    for qi, item in enumerate(qa):
        relevant = set(item["relevant_sources"])
        cat = item.get("category", "all")

        sims = matrix @ qvecs[qi]
        dense_idx = np.argsort(-sims)[:DENSE_K]
        bm_scores = bm25.get_scores(tokenize(queries[qi]))
        sparse_idx = np.argsort(-bm_scores)[:SPARSE_K]

        dense_sources = [chunk_sources[i] for i in dense_idx]
        hybrid_sources = [chunk_sources[i] for i in rrf_fuse(list(dense_idx), list(sparse_idx))]

        dh, drr, drecall, dprecision, dndcg = metrics_for(dense_sources, relevant, k)
        hh, hrr, hrecall, hprecision, hndcg = metrics_for(hybrid_sources, relevant, k)
        agg["dense"]["hit"] += dh
        agg["dense"]["rr"] += drr
        agg["dense"]["recall"] += drecall
        agg["dense"]["precision"] += dprecision
        agg["dense"]["ndcg"] += dndcg
        agg["hybrid"]["hit"] += hh
        agg["hybrid"]["rr"] += hrr
        agg["hybrid"]["recall"] += hrecall
        agg["hybrid"]["precision"] += hprecision
        agg["hybrid"]["ndcg"] += hndcg
        cat_hybrid_rr[cat] += hrr
        cat_n[cat] += 1

    q = len(qa)
    per_cat = {c: round(cat_hybrid_rr[c] / cat_n[c], 4) for c in cat_n}
    return {
        "dim": int(matrix.shape[1]),
        "num_chunks": int(matrix.shape[0]),
        "dense_mrr": round(agg["dense"]["rr"] / q, 4),
        "dense_hit": round(agg["dense"]["hit"] / q, 4),
        "dense_recall": round(agg["dense"]["recall"] / q, 4),
        "dense_precision": round(agg["dense"]["precision"] / q, 4),
        "dense_ndcg": round(agg["dense"]["ndcg"] / q, 4),
        "hybrid_mrr": round(agg["hybrid"]["rr"] / q, 4),
        "hybrid_hit": round(agg["hybrid"]["hit"] / q, 4),
        "hybrid_recall": round(agg["hybrid"]["recall"] / q, 4),
        "hybrid_precision": round(agg["hybrid"]["precision"] / q, 4),
        "hybrid_ndcg": round(agg["hybrid"]["ndcg"] / q, 4),
        "per_category_hybrid_mrr": per_cat,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def build_experiment_corpus(distractors, seed, full_corpus=False):
    import random

    random.seed(seed)
    sections = load_jsonl_corpus(SECTIONS)
    qa = load_jsonl_corpus(QA)
    if full_corpus:
        return sections, qa

    relevant_ids = set()
    for item in qa:
        relevant_ids.update(item["relevant_sources"])

    by_id = {s["id"]: s for s in sections}
    relevant_sections = [by_id[i] for i in relevant_ids if i in by_id]
    others = [s for s in sections if s["id"] not in relevant_ids]
    random.shuffle(others)
    corpus = relevant_sections + others[:distractors]
    return corpus, qa


def write_results(state):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main():
    import os
    from dotenv import load_dotenv
    from app.config import BACKEND_DIR

    load_dotenv(BACKEND_DIR / ".env")  # make OPENAI_API_KEY available to os.environ

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="subset by name")
    ap.add_argument("--distractors", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--full-corpus",
        action="store_true",
        help="index all banking sections for each config, so the winner is production-complete",
    )
    ap.add_argument(
        "--no-persist-vectors",
        action="store_true",
        help="score configs without saving their vectors to Chroma",
    )
    ap.add_argument(
        "--keep-only-winner",
        action="store_true",
        help="after a successful run, delete all persisted experiment collections except the best one",
    )
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="device for local sentence-transformer models",
    )
    args = ap.parse_args()

    settings = get_settings()
    corpus, qa = build_experiment_corpus(args.distractors, args.seed, args.full_corpus)
    queries = [item["question"] for item in qa]
    categories = sorted({item.get("category", "all") for item in qa})
    print(f"Experiment corpus: {len(corpus)} sections | eval: {len(qa)} questions")

    selected = MODELS
    if args.models:
        selected = [m for m in MODELS if m["name"] in args.models]
    # Skip OpenAI models if no key is configured.
    runnable = []
    skipped = []
    for m in selected:
        if m["kind"] == "openai" and not os.environ.get("OPENAI_API_KEY"):
            skipped.append(m["name"])
        else:
            runnable.append(m)

    total = len(runnable) * len(CHUNKINGS)
    state = {
        "status": "running",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "k": K,
        "eval_size": len(qa),
        "categories": categories,
        "corpus_sections": len(corpus),
        "distractors": args.distractors,
        "full_corpus": args.full_corpus,
        "persisted_vectors": not args.no_persist_vectors,
        "experiment_chroma_path": str(EXPERIMENT_CHROMA_PATH),
        "total_configs": total,
        "completed": 0,
        "skipped_models": skipped,
        "results": [],
    }
    write_results(state)
    print(f"{len(runnable)} models x {len(CHUNKINGS)} chunkings = {total} configs"
          + (f" | skipped (no key): {skipped}" if skipped else ""))

    for m in runnable:
        print(f"\n=== model: {m['name']} ===")
        try:
            embedder = make_embedder(m, device=args.device)
        except Exception as exc:  # noqa: BLE001
            print(f"  failed to load: {exc}")
            continue
        for ck in CHUNKINGS:
            t0 = time.time()
            chunk_ids, chunk_texts, chunk_sources, chunk_metadatas = build_chunks(
                corpus, ck["size"], ck["overlap"]
            )
            matrix = embedder.passages(chunk_texts)
            col_name = collection_name(m["name"], ck["size"], ck["overlap"], args.full_corpus)
            persisted_count = 0
            if not args.no_persist_vectors:
                persisted_count = persist_vectors(
                    col_name,
                    m,
                    ck,
                    chunk_ids,
                    chunk_texts,
                    matrix,
                    chunk_metadatas,
                )
            res = evaluate_config(matrix, embedder, chunk_texts, chunk_sources, queries, qa)
            elapsed = round(time.time() - t0, 1)
            row = {
                "model": m["name"],
                "kind": m["kind"],
                "model_id": m["model_id"],
                "chunk_size": ck["size"],
                "chunk_overlap": ck["overlap"],
                "collection_name": col_name if not args.no_persist_vectors else None,
                "persisted_chunks": persisted_count,
                "seconds": elapsed,
                **res,
            }
            state["results"].append(row)
            state["completed"] += 1
            state["generated_at"] = datetime.now(timezone.utc).isoformat()
            write_results(state)
            print(f"  size={ck['size']:>4} ov={ck['overlap']:>3} "
                  f"chunks={res['num_chunks']:>6} "
                  f"dense_MRR={res['dense_mrr']:.3f} hybrid_MRR={res['hybrid_mrr']:.3f} "
                  f"({elapsed}s)")
        del embedder

    state["status"] = "complete"
    state["generated_at"] = datetime.now(timezone.utc).isoformat()
    write_results(state)

    best = max(state["results"], key=lambda r: r["hybrid_mrr"], default=None)
    if best:
        winner = {
            "selected_at": datetime.now(timezone.utc).isoformat(),
            "metric": "hybrid_mrr",
            "experiment_chroma_path": str(EXPERIMENT_CHROMA_PATH),
            "full_corpus": args.full_corpus,
            "production_ready": args.full_corpus and not args.no_persist_vectors,
            **best,
        }
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        WINNER.write_text(json.dumps(winner, indent=2), encoding="utf-8")
        if args.keep_only_winner and not args.no_persist_vectors:
            losers = [
                r["collection_name"]
                for r in state["results"]
                if r.get("collection_name") and r["collection_name"] != best["collection_name"]
            ]
            delete_collections(losers)
            state["deleted_losing_collections"] = len(losers)
            write_results(state)
        print(f"\nBest hybrid MRR: {best['hybrid_mrr']:.3f} "
              f"({best['model']}, size={best['chunk_size']}, ov={best['chunk_overlap']})")
        if not winner["production_ready"]:
            print("Winner is for benchmarking only; rerun with --full-corpus to save a production-complete index.")
        print(f"Wrote winner metadata -> {WINNER}")
    print(f"Wrote {RESULTS}")


if __name__ == "__main__":
    main()
