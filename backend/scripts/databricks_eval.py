"""Databricks-style retrieval evaluation notebook.

Each section is a logical "cell" as would appear in a Databricks notebook.
Evaluates retrieval quality on the QA eval set and emits structured JSON
output matching the /experiments API endpoint format.

Metrics:
  Hit@1 / Hit@3 / Hit@5 — fraction of questions where a relevant source
                           appears in the top-K retrieved chunks
  MRR                    — mean reciprocal rank
  NDCG@K                 — normalized discounted cumulative gain
  Latency p50 / p95      — retrieval wall-clock time per query

Usage (from backend/):
    python scripts/databricks_eval.py
    python scripts/databricks_eval.py --top-k 5 --output data/experiments/eval_results.json
    python scripts/databricks_eval.py --qa data/banking/qa.jsonl
"""
# COMMAND ----------
# Cell 0: Setup and imports

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATA_DIR, EVAL_DIR, get_settings  # noqa: E402
from app import retriever  # noqa: E402


# COMMAND ----------
# Cell 1: Load QA evaluation dataset

def load_eval_set(path: Path) -> list[dict]:
    """Load the JSONL evaluation set. Each record: {question, relevant_sources, ...}."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# COMMAND ----------
# Cell 2: Metric computation helpers

def _relevant_sources(record: dict) -> set[str]:
    return set(record.get("relevant_sources", []))


def _retrieved_sources(results: list[dict]) -> list[str]:
    return [(r.get("metadata") or {}).get("source", "") for r in results]


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return float(any(s in relevant for s in retrieved[:k]))


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, s in enumerate(retrieved, start=1):
        if s in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return sum(
        1.0 / math.log2(rank + 1)
        for rank, s in enumerate(retrieved[:k], start=1)
        if s in relevant
    )


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    ideal = dcg_at_k(list(relevant)[:k], relevant, k)
    return dcg_at_k(retrieved, relevant, k) / ideal if ideal else 0.0


# COMMAND ----------
# Cell 3: Evaluation loop

def run_eval(eval_records: list[dict], top_k: int = 5, verbose: bool = True) -> dict:
    hits_1, hits_3, hits_5 = [], [], []
    mrr_scores, ndcg_scores, latencies = [], [], []

    for i, rec in enumerate(eval_records):
        question = rec.get("question", "")
        relevant = _relevant_sources(rec)

        t0 = time.perf_counter()
        results = retriever.hybrid_search(question, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved = _retrieved_sources(results)
        latencies.append(latency_ms)

        if relevant:
            hits_1.append(hit_at_k(retrieved, relevant, 1))
            hits_3.append(hit_at_k(retrieved, relevant, 3))
            hits_5.append(hit_at_k(retrieved, relevant, 5))
            mrr_scores.append(reciprocal_rank(retrieved, relevant))
            ndcg_scores.append(ndcg_at_k(retrieved, relevant, top_k))

        if verbose and (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(eval_records)} evaluated ...")

    def _mean(lst: list) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    latencies_sorted = sorted(latencies)
    p50_idx = len(latencies_sorted) // 2
    p95_idx = max(0, int(len(latencies_sorted) * 0.95) - 1)

    return {
        "num_questions": len(eval_records),
        "top_k": top_k,
        "hit_at_1": _mean(hits_1),
        "hit_at_3": _mean(hits_3),
        "hit_at_5": _mean(hits_5),
        "mrr": _mean(mrr_scores),
        "ndcg": _mean(ndcg_scores),
        "latency_p50_ms": round(latencies_sorted[p50_idx], 1) if latencies_sorted else 0.0,
        "latency_p95_ms": round(latencies_sorted[p95_idx], 1) if latencies_sorted else 0.0,
    }


# COMMAND ----------
# Cell 4: Display results

def print_results(metrics: dict, settings_snapshot: dict) -> None:
    print("\n=== Databricks Retrieval Evaluation Results ===")
    print(f"  Embedding model : {settings_snapshot['embedding_model']}")
    print(f"  Collection      : {settings_snapshot['collection_name']}")
    print(f"  Chunk size      : {settings_snapshot['chunk_size']}  overlap: {settings_snapshot['chunk_overlap']}")
    print(f"  Questions       : {metrics['num_questions']}  top_k: {metrics['top_k']}")
    print()
    print(f"  Hit@1   : {metrics['hit_at_1']:.4f}")
    print(f"  Hit@3   : {metrics['hit_at_3']:.4f}")
    print(f"  Hit@5   : {metrics['hit_at_5']:.4f}")
    print(f"  MRR     : {metrics['mrr']:.4f}")
    print(f"  NDCG@K  : {metrics['ndcg']:.4f}")
    print(f"  Lat p50 : {metrics['latency_p50_ms']:.1f} ms")
    print(f"  Lat p95 : {metrics['latency_p95_ms']:.1f} ms")


# COMMAND ----------
# Cell 5: Main entry point

def main() -> None:
    parser = argparse.ArgumentParser(description="Databricks-style retrieval eval notebook")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "experiments" / "eval_results.json"),
    )
    parser.add_argument(
        "--qa",
        default=str(DATA_DIR / "banking" / "qa.jsonl"),
        help="path to QA JSONL (falls back to legacy eval set if banking not built)",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="log metrics to MLflow (local ./mlruns or Databricks tracking server)",
    )
    parser.add_argument(
        "--experiment",
        default="agentic-rag-retrieval-eval",
        help="MLflow experiment name (default: agentic-rag-retrieval-eval)",
    )
    args = parser.parse_args()

    qa_path = Path(args.qa)
    if not qa_path.exists():
        qa_path = EVAL_DIR / "qa.jsonl"
    if not qa_path.exists():
        print(f"Eval file not found: {qa_path}", file=sys.stderr)
        sys.exit(1)

    settings = get_settings()
    settings_snapshot = {
        "embedding_model": settings.embedding_model,
        "collection_name": settings.collection_name,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }

    print("=== Databricks Retrieval Evaluation Notebook ===")
    print(f"Eval file : {qa_path}")

    eval_records = load_eval_set(qa_path)
    print(f"Loaded {len(eval_records)} eval records")

    print("\nRunning evaluation ...")
    metrics = run_eval(eval_records, top_k=args.top_k, verbose=True)
    print_results(metrics, settings_snapshot)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {"metrics": metrics, "settings": settings_snapshot}
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults written to {out_path}")

    if args.mlflow:
        from app.mlflow_tracker import log_eval_run, is_databricks

        run_id = log_eval_run(
            run_name=f"hybrid_eval_top{args.top_k}",
            params={**settings_snapshot, "top_k": args.top_k, "retrieval_mode": "hybrid_bm25_dense_rrf"},
            metrics={k: v for k, v in metrics.items() if isinstance(v, float)},
            artifact_path=str(out_path),
            experiment=args.experiment,
        )
        if run_id:
            target = "Databricks Experiments" if is_databricks() else f"mlflow ui  (http://localhost:5000, experiment: {args.experiment})"
            print(f"MLflow run logged: {run_id}  →  {target}")
        else:
            print("MLflow not installed — skipping. pip install mlflow to enable.")


if __name__ == "__main__":
    main()
