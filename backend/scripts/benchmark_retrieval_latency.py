"""Measure retrieval latency for the configured Chroma/BM25 setup.

Run from backend/:
    python scripts/benchmark_retrieval_latency.py
    python scripts/benchmark_retrieval_latency.py --repeat 5 --no-sparse
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import embeddings, retriever, vectorstore  # noqa: E402
from app.config import get_settings  # noqa: E402

DEFAULT_QUERIES = [
    "What are the funds availability rules for next-day items under Regulation CC?",
    "What is the threshold for filing a suspicious activity report?",
    "What disclosures are required for electronic fund transfers?",
]


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000


def summarize(values: list[float]) -> str:
    if not values:
        return "n/a"
    return (
        f"avg={statistics.mean(values):.1f}ms "
        f"p50={statistics.median(values):.1f}ms "
        f"min={min(values):.1f}ms max={max(values):.1f}ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--no-sparse", action="store_true")
    parser.add_argument(
        "--chroma-path",
        default=None,
        help="Override CHROMA_PATH for this benchmark run.",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.chroma_path:
        settings.chroma_path = args.chroma_path
    elif settings.chroma_path.startswith("/app/") and not Path(settings.chroma_path).exists():
        local_chroma_path = Path(__file__).resolve().parent.parent / "storage" / "experiment_chroma"
        if local_chroma_path.exists():
            print(f"Using local Chroma path instead of Render path: {local_chroma_path}")
            settings.chroma_path = str(local_chroma_path)
    if args.no_sparse:
        settings.enable_sparse_bm25 = False

    queries = args.query or DEFAULT_QUERIES
    k = args.k or settings.top_k

    print("Retrieval latency benchmark")
    print(f"  chroma_path: {settings.chroma_path}")
    print(f"  collection:  {settings.collection_name}")
    print(f"  embedding:   {settings.embedding_model}")
    print(f"  top_k:       {k}")
    print(f"  dense_k:     {settings.dense_k}")
    print(f"  sparse_k:    {settings.sparse_k}")
    print(f"  bm25:        {settings.enable_sparse_bm25 and not args.no_sparse}")
    print(f"  indexed:     {vectorstore.count()} chunks")

    # Load model and Chroma before measured repeats so warm timings are clear.
    warm_query_vector, warm_embed_ms = timed(lambda: embeddings.embed_query("warmup query"))
    _, warm_chroma_ms = timed(lambda: vectorstore.query(warm_query_vector, 1))
    print(f"\nWarmup: embed={warm_embed_ms:.1f}ms chroma={warm_chroma_ms:.1f}ms")

    timings: dict[str, list[float]] = {
        "embed": [],
        "dense_query": [],
        "sparse_query": [],
        "full_retrieval": [],
    }

    for query in queries:
        print(f"\nQuery: {query}")
        for i in range(args.repeat):
            qv, embed_ms = timed(lambda: embeddings.embed_query(query))
            dense, dense_ms = timed(lambda: vectorstore.query(qv, settings.dense_k))
            if settings.enable_sparse_bm25 and not args.no_sparse:
                sparse, sparse_ms = timed(lambda: retriever._sparse_search(query, settings.sparse_k))
            else:
                sparse, sparse_ms = [], 0.0
            full, full_ms = timed(lambda: retriever.hybrid_search(query, top_k=k))

            timings["embed"].append(embed_ms)
            timings["dense_query"].append(dense_ms)
            if settings.enable_sparse_bm25 and not args.no_sparse:
                timings["sparse_query"].append(sparse_ms)
            timings["full_retrieval"].append(full_ms)

            print(
                f"  run {i + 1}: embed={embed_ms:.1f}ms "
                f"dense={dense_ms:.1f}ms/{len(dense)} "
                f"sparse={sparse_ms:.1f}ms/{len(sparse)} "
                f"full={full_ms:.1f}ms/{len(full)}"
            )

    print("\nSummary")
    for name, values in timings.items():
        print(f"  {name:14s} {summarize(values)}")


if __name__ == "__main__":
    main()
