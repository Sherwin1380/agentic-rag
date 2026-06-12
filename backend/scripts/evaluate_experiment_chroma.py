"""Re-evaluate persisted experiment Chroma collections without re-embedding docs.

Use this after copying/downloading `storage/experiment_chroma` from Colab. The
script opens each saved benchmark collection, embeds only the QA queries with the
matching model, rebuilds BM25 from stored chunk documents, and writes metrics.

Run from backend/:
    python scripts/evaluate_experiment_chroma.py
    python scripts/evaluate_experiment_chroma.py --chroma-path storage/experiment_chroma
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATA_DIR, STORAGE_DIR  # noqa: E402
from app.ingest import load_jsonl_corpus  # noqa: E402

QA = DATA_DIR / "banking" / "qa.jsonl"
OUT_DIR = DATA_DIR / "experiments"
OUT = OUT_DIR / "reeval_chroma_results.json"

K = 5
DENSE_K = 12
SPARSE_K = 12
RRF_K = 60

MODEL_SPECS = {
    "all-MiniLM-L6-v2": {
        "kind": "st",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "bge-small-en-v1.5": {
        "kind": "st",
        "model_id": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "gte-small": {"kind": "st", "model_id": "thenlper/gte-small"},
    "e5-small-v2": {
        "kind": "st",
        "model_id": "intfloat/e5-small-v2",
        "query_prefix": "query: ",
    },
    "text-embedding-3-small": {
        "kind": "openai",
        "model_id": "text-embedding-3-small",
    },
    "text-embedding-3-large": {
        "kind": "openai",
        "model_id": "text-embedding-3-large",
    },
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class STQueryEmbedder:
    def __init__(self, model_id: str, query_prefix: str = "", device: str = "auto"):
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.model = SentenceTransformer(model_id, device=device)
        self.query_prefix = query_prefix

    def queries(self, texts: list[str]) -> np.ndarray:
        texts = [self.query_prefix + t for t in texts]
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)


class OpenAIQueryEmbedder:
    def __init__(self, model_id: str):
        import os
        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_id = model_id

    def queries(self, texts: list[str], batch: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            chunk = [t.replace("\n", " ") for t in texts[i : i + batch]]
            for attempt in range(4):
                try:
                    resp = self.client.embeddings.create(
                        model=self.model_id, input=chunk
                    )
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 * (attempt + 1))
            out.extend([d.embedding for d in resp.data])
        arr = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-9, None)


def make_query_embedder(model_name: str, model_id: str, kind: str, device: str):
    spec = MODEL_SPECS.get(model_name, {})
    if kind == "openai":
        return OpenAIQueryEmbedder(model_id)
    return STQueryEmbedder(model_id, spec.get("query_prefix", ""), device=device)


def list_collections(path: Path):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.PersistentClient(
        path=str(path),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )
    return client, client.list_collections()


def get_all_collection_data(collection, batch: int = 5000):
    total = collection.count()
    ids, docs, metas = [], [], []
    for offset in range(0, total, batch):
        res = collection.get(
            limit=batch,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids.extend(res.get("ids") or [])
        docs.extend(res.get("documents") or [])
        metas.extend(res.get("metadatas") or [])
    return ids, docs, metas


def rrf_fuse(dense_ids: list[str], sparse_ids: list[str]) -> list[str]:
    scores = defaultdict(float)
    for rank, _id in enumerate(dense_ids):
        scores[_id] += 1.0 / (RRF_K + rank + 1)
    for rank, _id in enumerate(sparse_ids):
        scores[_id] += 1.0 / (RRF_K + rank + 1)
    return [_id for _id, _ in sorted(scores.items(), key=lambda x: -x[1])]


def metrics_for(ranked_sources: list[str], relevant: set[str], k: int):
    top = ranked_sources[:k]
    found = [s for s in top if s in relevant]
    hit = 1.0 if found else 0.0
    recall = (len(set(found)) / len(relevant)) if relevant else 0.0
    precision = len(found) / len(top) if top else 0.0
    rr = 0.0
    for rank, source in enumerate(top, start=1):
        if source in relevant:
            rr = 1.0 / rank
            break
    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, source in enumerate(top, start=1)
        if source in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg else 0.0
    return hit, rr, recall, precision, ndcg


def evaluate_collection(collection, qa, device: str, k: int):
    from rank_bm25 import BM25Okapi

    meta = collection.metadata or {}
    model_name = meta.get("model") or ""
    model_id = meta.get("model_id") or MODEL_SPECS.get(model_name, {}).get("model_id")
    kind = meta.get("kind") or MODEL_SPECS.get(model_name, {}).get("kind")
    if not model_id or not kind:
        raise RuntimeError(f"Cannot infer model for collection {collection.name}")

    ids, docs, metas = get_all_collection_data(collection)
    by_id = {
        ids[i]: {
            "source": (metas[i] or {}).get("source"),
            "metadata": metas[i],
        }
        for i in range(len(ids))
    }

    bm25 = BM25Okapi([tokenize(d) for d in docs])
    queries = [item["question"] for item in qa]
    qvecs = make_query_embedder(model_name, model_id, kind, device).queries(queries)

    agg = {"dense": defaultdict(float), "hybrid": defaultdict(float)}
    cat_hybrid_rr = defaultdict(float)
    cat_n = defaultdict(int)

    for qi, item in enumerate(qa):
        relevant = set(item["relevant_sources"])
        cat = item.get("category", "all")

        dense = collection.query(
            query_embeddings=[qvecs[qi].tolist()],
            n_results=min(DENSE_K, len(ids)),
            include=[],
        )
        dense_ids = (dense.get("ids") or [[]])[0]

        bm_scores = bm25.get_scores(tokenize(queries[qi]))
        sparse_idx = np.argsort(-bm_scores)[:SPARSE_K]
        sparse_ids = [ids[i] for i in sparse_idx if bm_scores[i] > 0]

        dense_sources = [by_id[_id]["source"] for _id in dense_ids]
        hybrid_sources = [by_id[_id]["source"] for _id in rrf_fuse(dense_ids, sparse_ids)]

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
        "collection_name": collection.name,
        "model": model_name,
        "kind": kind,
        "model_id": model_id,
        "chunk_size": int(meta.get("chunk_size", 0)),
        "chunk_overlap": int(meta.get("chunk_overlap", 0)),
        "num_chunks": len(ids),
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


def evaluate_collection_by_name(chroma_path: Path, name: str, qa, device: str, k: int):
    client, _ = list_collections(chroma_path)
    collection = client.get_collection(name)
    return evaluate_collection(collection, qa, device, k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chroma-path",
        default=str(STORAGE_DIR / "experiment_chroma"),
        help="path containing persisted experiment Chroma collections",
    )
    parser.add_argument("--qa", default=str(QA))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--collections", nargs="*", default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of collections to evaluate concurrently",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="overwrite the output file instead of merging with existing rows",
    )
    args = parser.parse_args()

    chroma_path = Path(args.chroma_path)
    qa = load_jsonl_corpus(Path(args.qa))
    client, collections = list_collections(chroma_path)
    if args.collections:
        wanted = set(args.collections)
        collections = [c for c in collections if c.name in wanted]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    existing_results = []
    if out.exists() and not args.fresh:
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
            existing_results = existing.get("results", [])
        except Exception:
            existing_results = []
    results_by_collection = {
        row.get("collection_name"): row
        for row in existing_results
        if row.get("collection_name")
    }

    state = {
        "status": "running",
        "qa": str(args.qa),
        "chroma_path": str(chroma_path),
        "k": args.k,
        "workers": args.workers,
        "fresh": args.fresh,
        "preserved_results": len(results_by_collection),
        "eval_size": len(qa),
        "total_configs": len(collections),
        "completed": 0,
        "results": sorted(results_by_collection.values(), key=lambda r: r["collection_name"]),
    }

    collection_names = [c.name for c in sorted(collections, key=lambda c: c.name)]

    def record_result(name: str, row: dict):
        results_by_collection[name] = row
        state["completed"] += 1
        state["results"] = sorted(
            results_by_collection.values(), key=lambda r: r["collection_name"]
        )
        out.write_text(json.dumps(state, indent=2), encoding="utf-8")
        if "error" in row:
            print(f"  ERROR {row['error']}")
        else:
            print(
                f"  {row['model']} {row['chunk_size']}/{row['chunk_overlap']} "
                f"chunks={row['num_chunks']} dense_MRR={row['dense_mrr']:.3f} "
                f"hybrid_MRR={row['hybrid_mrr']:.3f}"
            )
        print(f"  progress {state['completed']}/{state['total_configs']} ({name})")

    if args.workers <= 1:
        for name in collection_names:
            print(f"\n=== {name} ===")
            client, _ = list_collections(chroma_path)
            collection = client.get_collection(name)
            try:
                row = evaluate_collection(collection, qa, args.device, args.k)
            except Exception as exc:  # noqa: BLE001
                row = {"collection_name": name, "error": str(exc)}
            record_result(name, row)
    else:
        print(f"Evaluating {len(collection_names)} collections with {args.workers} workers")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    evaluate_collection_by_name,
                    chroma_path,
                    name,
                    qa,
                    args.device,
                    args.k,
                ): name
                for name in collection_names
            }
            for future in as_completed(futures):
                name = futures[future]
                print(f"\n=== {name} ===")
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    row = {"collection_name": name, "error": str(exc)}
                record_result(name, row)

    state["status"] = "complete"
    best = max(
        [r for r in state["results"] if "hybrid_mrr" in r],
        key=lambda r: r["hybrid_mrr"],
        default=None,
    )
    state["best"] = best
    out.write_text(json.dumps(state, indent=2), encoding="utf-8")
    if best:
        print(
            f"\nBest hybrid MRR: {best['hybrid_mrr']:.3f} "
            f"({best['model']}, {best['chunk_size']}/{best['chunk_overlap']})"
        )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
