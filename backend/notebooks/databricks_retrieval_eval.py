# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Agentic RAG — Retrieval Evaluation Notebook
# MAGIC
# MAGIC **Purpose:** Evaluate hybrid retrieval quality (BM25 + dense + RRF) on the
# MAGIC U.S. banking regulations corpus (Title 12 CFR) and log every metric to MLflow.
# MAGIC
# MAGIC **Works on:** Databricks Free Edition (Serverless), Databricks Community Edition
# MAGIC (cluster), or locally as a plain Python script.
# MAGIC
# MAGIC **No setup required.** All data is fetched automatically from the public
# MAGIC GitHub repository. Just run all cells.

# COMMAND ----------

# MAGIC %pip install sentence-transformers rank-bm25

# COMMAND ----------

# MAGIC %md ## Cell 1 — Environment detection and configuration

# COMMAND ----------

from __future__ import annotations

import json
import math
import re
import time
import urllib.request
from pathlib import Path

# --- Detect Databricks vs local ---
try:
    dbutils  # noqa: F821 — injected by the Databricks runtime
    ON_DATABRICKS = True
except NameError:
    ON_DATABRICKS = False

# --- Repo raw-content base URL (public GitHub) ---
_GITHUB_RAW = (
    "https://raw.githubusercontent.com/Sherwin1380/agentic-rag/main/backend"
)

# Corpus: U.S. banking regulations — 5,002 Title 12 CFR sections (OCC, FRS,
# FDIC, NCUA, CFPB), the same data the production RAG agent searches over.
_BANKING_SECTIONS = "data/banking/sections.jsonl"

# Eval set: 88 labelled QA pairs drawn from the banking corpus.
_QA_FILE = "data/banking/qa.jsonl"

# Max sections to embed for this run.  Set to 0 to embed all 5,002 (≈5 min on
# Serverless CPU).  The default keeps a fast demo: all QA-relevant sections are
# always included; the rest are random distractors to fill SAMPLE_SIZE.
SAMPLE_SIZE = 500

# Tmp dir: always writable in both Serverless and cluster runtimes.
TMP_DIR = Path("/tmp/agentic-rag")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# --- Retrieval config ---
EMBEDDING_MODEL = "intfloat/e5-small-v2"   # 384-dim, ~24 MB, CPU-friendly
QUERY_PREFIX    = "query: "
PASSAGE_PREFIX  = "passage: "
TOP_K           = 5
DENSE_K         = 12
SPARSE_K        = 12
RRF_K           = 60

# --- MLflow experiment ---
EXPERIMENT_NAME = "agentic-rag-retrieval-eval"

print(f"ON_DATABRICKS : {ON_DATABRICKS}")
print(f"TMP_DIR       : {TMP_DIR}")
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")

# COMMAND ----------

# MAGIC %md ## Cell 2 — Fetch corpus and eval set from GitHub

# COMMAND ----------

def _fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode("utf-8")


def _load_jsonl_str(text: str) -> list[dict]:
    return [json.loads(l) for l in text.splitlines() if l.strip()]


# --- Download QA eval set first (needed to select QA-relevant sections) ---
print("Fetching eval set from GitHub ...")
qa_url     = f"{_GITHUB_RAW}/{_QA_FILE}"
qa_text    = _fetch_text(qa_url)
qa_records = _load_jsonl_str(qa_text)
print(f"  {len(qa_records)} eval questions loaded")

relevant_ids: set[str] = set()
for rec in qa_records:
    relevant_ids.update(rec.get("relevant_sources", []))

# --- Download all banking sections ---
print("\nFetching banking corpus from GitHub ...")
sections_url  = f"{_GITHUB_RAW}/{_BANKING_SECTIONS}"
sections_text = _fetch_text(sections_url)
all_sections  = _load_jsonl_str(sections_text)
print(f"  {len(all_sections):,} sections in full corpus")

# Build evaluation corpus: all QA-relevant sections + random distractors.
# This guarantees every eval question has its answer section in the corpus
# while keeping the embedding step fast for the Serverless demo.
import random as _rnd
_rnd.seed(42)
relevant_secs  = [s for s in all_sections if s["id"] in relevant_ids]
distractor_secs = [s for s in all_sections if s["id"] not in relevant_ids]
distractor_n   = max(0, (SAMPLE_SIZE or len(all_sections)) - len(relevant_secs))
sample         = relevant_secs + _rnd.sample(distractor_secs, min(distractor_n, len(distractor_secs)))

corpus_docs: list[dict] = [
    {"id": s["id"], "title": s["title"], "text": s["text"]}
    for s in sample
]
print(f"  {len(relevant_secs)} QA-relevant + {len(sample) - len(relevant_secs)} distractors "
      f"= {len(corpus_docs)} sections in eval corpus")

# COMMAND ----------

# MAGIC %md ## Cell 3 — Chunk corpus and build BM25 index

# COMMAND ----------

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _chunk_text(text: str, size: int = 1500, overlap: int = 255) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > size:
                start = 0
                while start < len(para):
                    chunks.append(para[start: start + size])
                    start += size - overlap
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


print("Chunking corpus ...")
chunk_ids:     list[str] = []
chunk_texts:   list[str] = []
chunk_sources: list[str] = []

for doc in corpus_docs:
    chunks = _chunk_text(doc["text"])
    for i, ch in enumerate(chunks):
        chunk_ids.append(f"{doc['id']}::c{i}")
        chunk_texts.append(ch)
        chunk_sources.append(doc["id"])

print(f"  {len(corpus_docs)} docs → {len(chunk_texts)} chunks")

print("\nBuilding BM25 index ...")
t0 = time.perf_counter()
from rank_bm25 import BM25Okapi
bm25 = BM25Okapi([_tokenize(t) for t in chunk_texts])
print(f"  Done in {(time.perf_counter() - t0) * 1000:.0f} ms")

# COMMAND ----------

# MAGIC %md ## Cell 4 — Build dense embedding index

# COMMAND ----------

import numpy as np
from sentence_transformers import SentenceTransformer

print(f"Loading {EMBEDDING_MODEL} ...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print(f"Embedding {len(chunk_texts)} chunks ...")
t0 = time.perf_counter()
passage_matrix = embedder.encode(
    [PASSAGE_PREFIX + t for t in chunk_texts],
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True,
).astype(np.float32)
print(f"  Shape: {passage_matrix.shape}  ({time.perf_counter() - t0:.1f}s)")

# COMMAND ----------

# MAGIC %md ## Cell 5 — Hybrid retrieval (BM25 + dense + RRF)

# COMMAND ----------

def _dense_top(query: str, k: int) -> list[int]:
    qv = embedder.encode(
        [QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)[0]
    return list(np.argsort(-(passage_matrix @ qv))[:k])


def _sparse_top(query: str, k: int) -> list[int]:
    scores = bm25.get_scores(_tokenize(query))
    ranked = np.argsort(-scores)[:k]
    return [int(i) for i in ranked if scores[i] > 0]


def _rrf(dense: list[int], sparse: list[int], k: int) -> list[int]:
    rrf: dict[int, float] = {}
    for rank, i in enumerate(dense):
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, i in enumerate(sparse):
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
    return [i for i, _ in sorted(rrf.items(), key=lambda x: -x[1])][:k]


def hybrid_search(query: str, top_k: int = TOP_K) -> list[str]:
    fused = _rrf(_dense_top(query, DENSE_K), _sparse_top(query, SPARSE_K), top_k)
    return [chunk_sources[i] for i in fused]

# COMMAND ----------

# MAGIC %md ## Cell 6 — Evaluation loop

# COMMAND ----------

def _hit(retrieved: list[str], relevant: set[str], k: int) -> float:
    return float(any(s in relevant for s in retrieved[:k]))


def _rr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, s in enumerate(retrieved, start=1):
        if s in relevant:
            return 1.0 / rank
    return 0.0


def _dcg(retrieved: list[str], relevant: set[str], k: int) -> float:
    return sum(
        1.0 / math.log2(r + 1)
        for r, s in enumerate(retrieved[:k], start=1)
        if s in relevant
    )


def _ndcg(retrieved: list[str], relevant: set[str], k: int) -> float:
    ideal = _dcg(list(relevant)[:k], relevant, k)
    return _dcg(retrieved, relevant, k) / ideal if ideal else 0.0


print(f"Evaluating {len(qa_records)} questions (top_k={TOP_K}) ...")
hits_1, hits_3, hits_5, mrr_scores, ndcg_scores, latencies = [], [], [], [], [], []

for i, rec in enumerate(qa_records):
    question = rec.get("question", "")
    relevant = set(rec.get("relevant_sources", []))

    t0 = time.perf_counter()
    retrieved = hybrid_search(question)
    latencies.append((time.perf_counter() - t0) * 1000)

    if relevant:
        hits_1.append(_hit(retrieved, relevant, 1))
        hits_3.append(_hit(retrieved, relevant, 3))
        hits_5.append(_hit(retrieved, relevant, 5))
        mrr_scores.append(_rr(retrieved, relevant))
        ndcg_scores.append(_ndcg(retrieved, relevant, TOP_K))

    flag = "OK " if hits_5 and hits_5[-1] == 1.0 else "MISS"
    print(f"  [{flag}] {question[:65]}")


def _mean(lst: list) -> float:
    return round(sum(lst) / len(lst), 4) if lst else 0.0


lat = sorted(latencies)
metrics = {
    "num_questions":  len(qa_records),
    "num_chunks":     len(chunk_texts),
    "num_docs":       len(corpus_docs),
    "top_k":          TOP_K,
    "hit_at_1":       _mean(hits_1),
    "hit_at_3":       _mean(hits_3),
    "hit_at_5":       _mean(hits_5),
    "mrr":            _mean(mrr_scores),
    "ndcg":           _mean(ndcg_scores),
    "latency_p50_ms": round(lat[len(lat) // 2], 1) if lat else 0.0,
    "latency_p95_ms": round(lat[max(0, int(len(lat) * 0.95) - 1)], 1) if lat else 0.0,
}

print("\n=== Results ===")
for k, v in metrics.items():
    print(f"  {k:22s}: {v}")

# COMMAND ----------

# MAGIC %md ## Cell 7 — Log to MLflow

# COMMAND ----------

run_id = "not-logged"

if ON_DATABRICKS:
    # Databricks Free Edition Serverless: the MLflow Python SDK routes through
    # Spark Connect to read spark.mlflow.modelRegistryUri, which is not exposed
    # in this tier. Bypass the SDK entirely and call the MLflow REST API directly
    # using the notebook-injected token — no Spark connection required.
    import urllib.request as _req

    _ctx   = dbutils.notebook.entry_point.getDbutils().notebook().getContext()  # noqa: F821
    _host  = _ctx.apiUrl().get()
    _token = _ctx.apiToken().get()
    _hdrs  = {"Authorization": f"Bearer {_token}", "Content-Type": "application/json"}

    def _mlflow_api(path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode()
        req  = _req.Request(f"{_host}/api/2.0/mlflow/{path}", data=body, headers=_hdrs)
        with _req.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    # Resolve the user's personal experiment folder path
    try:
        _user = _ctx.tags().apply("user")
        _exp_path = f"/Users/{_user}/{EXPERIMENT_NAME}"
    except Exception:
        _exp_path = f"/{EXPERIMENT_NAME}"

    # Get or create the experiment
    try:
        _exp = _mlflow_api("experiments/get-by-name", {"experiment_name": _exp_path})
        _exp_id = _exp["experiment"]["experiment_id"]
    except Exception:
        try:
            _exp = _mlflow_api("experiments/create", {"name": _exp_path})
            _exp_id = _exp["experiment_id"]
        except Exception:
            _exp_id = "0"  # default experiment

    # Create run
    _run = _mlflow_api("runs/create", {
        "experiment_id": _exp_id,
        "run_name": f"hybrid_eval_top{TOP_K}",
        "start_time": int(time.time() * 1000),
    })
    run_id = _run["run"]["info"]["run_id"]

    # Log params + metrics in one batch call
    _ts = int(time.time() * 1000)
    _mlflow_api("runs/log-batch", {
        "run_id": run_id,
        "params": [
            {"key": "embedding_model",  "value": EMBEDDING_MODEL},
            {"key": "query_prefix",     "value": QUERY_PREFIX},
            {"key": "passage_prefix",   "value": PASSAGE_PREFIX},
            {"key": "top_k",            "value": str(TOP_K)},
            {"key": "dense_k",          "value": str(DENSE_K)},
            {"key": "sparse_k",         "value": str(SPARSE_K)},
            {"key": "rrf_k",            "value": str(RRF_K)},
            {"key": "num_docs",         "value": str(len(corpus_docs))},
            {"key": "num_chunks",       "value": str(len(chunk_texts))},
            {"key": "retrieval_mode",   "value": "hybrid_bm25_dense_rrf"},
        ],
        "metrics": [
            {"key": k, "value": float(v), "timestamp": _ts, "step": 0}
            for k, v in metrics.items() if isinstance(v, (int, float))
        ],
    })

    # Mark run finished
    _mlflow_api("runs/update", {
        "run_id": run_id,
        "status": "FINISHED",
        "end_time": int(time.time() * 1000),
    })

    print(f"\nMLflow run logged: {run_id}")
    print(f"View: left sidebar → Experiments → {_exp_path}")

else:
    # Local: use the MLflow SDK normally (no Spark Connect involved)
    import mlflow
    try:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
        except Exception:
            pass
        with mlflow.start_run(run_name=f"hybrid_eval_top{TOP_K}"):
            mlflow.log_params({
                "embedding_model": EMBEDDING_MODEL,
                "query_prefix": QUERY_PREFIX,
                "passage_prefix": PASSAGE_PREFIX,
                "top_k": TOP_K, "dense_k": DENSE_K,
                "sparse_k": SPARSE_K, "rrf_k": RRF_K,
                "num_docs": len(corpus_docs), "num_chunks": len(chunk_texts),
                "retrieval_mode": "hybrid_bm25_dense_rrf",
            })
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
            results_file = str(TMP_DIR / "eval_results.json")
            Path(results_file).write_text(
                json.dumps({"metrics": metrics, "config": {
                    "embedding_model": EMBEDDING_MODEL,
                    "top_k": TOP_K, "dense_k": DENSE_K,
                    "sparse_k": SPARSE_K, "rrf_k": RRF_K,
                }}, indent=2),
                encoding="utf-8",
            )
            mlflow.log_artifact(results_file, artifact_path="results")
            run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run logged: {run_id}")
        print(f"View: mlflow ui  →  http://localhost:5000  (experiment: {EXPERIMENT_NAME})")
    except Exception as mlflow_exc:
        print(f"\nMLflow logging skipped ({type(mlflow_exc).__name__}: {mlflow_exc!s:.120})")
        print("Metrics (not logged to MLflow):")
        for k, v in metrics.items():
            print(f"  {k:22s}: {v}")

# COMMAND ----------

# MAGIC %md ## Cell 8 — Results table

# COMMAND ----------

_rows = "".join(
    f"<tr><td>{k}</td><td><b>{v}</b></td></tr>"
    for k, v in metrics.items()
)
_html = f"""
<h3>Retrieval Evaluation — {EMBEDDING_MODEL}</h3>
<table border='1' cellpadding='6' style='border-collapse:collapse;font-family:monospace'>
  <tr style='background:#333;color:#fff'><th>Metric</th><th>Value</th></tr>
  {_rows}
</table>
<p style='color:grey;font-size:12px'>
  MLflow run: <code>{run_id}</code> &nbsp;|&nbsp;
  Experiment: <i>{EXPERIMENT_NAME}</i>
</p>
"""

try:
    displayHTML(_html)  # noqa: F821 — Databricks built-in
except NameError:
    print("\n" + "=" * 50)
    for k, v in metrics.items():
        print(f"  {k:22s}: {v}")
    print("=" * 50)
