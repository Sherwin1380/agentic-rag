# Agentic RAG for U.S. Banking Regulations

A retrieval-augmented assistant for U.S. banking regulations, focused on Title
12 of the Code of Federal Regulations. The app combines a tool-calling agent,
hybrid retrieval, source citations, an embedding/chunking benchmark UI, and a
full LLMOps and governance layer.

Hosted:

- Frontend: <https://agentic-rag-pied.vercel.app/>
- Backend: <https://agentic-rag-api-43w8.onrender.com>

---

## What It Demonstrates

**Retrieval and reasoning**

- Agentic RAG: the assistant decides when to retrieve, calculate, search the web,
  or answer directly — it is not a naive embed-and-retrieve pipeline.
- Domain corpus: 5,002 eCFR Title 12 sections across OCC, Federal Reserve, FDIC,
  NCUA, and CFPB regulations.
- Hybrid retrieval: dense Chroma search plus BM25, fused with Reciprocal Rank
  Fusion (RRF).
- Citations: answers include inline `[n]` citations tied to retrieved regulation
  chunks.
- Retrieval benchmarking: compares embedding models and chunking strategies with
  MRR, Hit@k, Recall@k, Precision@k, NDCG@k, and latency p50/p95.
- Production index: `intfloat/e5-small-v2`, chunk size 1500, overlap 255.

**LLMOps**

- Prompt versioning: every system prompt is tagged with a semantic version
  (`PROMPT_VERSION`). Every response log entry and audit record carries the
  version that produced it, enabling rollback-aware operations.
- Cost tracking: token counts from the Groq API are mapped to provider pricing
  and logged per request.
- Explicit fallback model: when the primary model returns a `tool_use_failed`
  error, the agent automatically routes to a smaller fallback model
  (`llama3-8b-8192`) without retrying the same failure.
- Structured response log: every chat request is appended to
  `storage/response_log.jsonl` — including latency, model used, fallback
  activation, cost estimate, and grounding score.
- Deployment manifest: a version snapshot (`storage/deployment_manifest.json`)
  records the active prompt version, both models, embedding model, and
  collection config at deploy time.

**Governance and risk**

- Grounding score: the fraction of answer sentences that cite a retrieved source
  (0–1 scale). Surfaces answers that may be drawing on parametric knowledge.
- Hallucination risk classification: `low`, `medium`, or `high`, based on
  whether any sources were retrieved and the grounding score.
- Citation validation: every `[n]` in the answer is checked against the set of
  actually retrieved sources. Orphaned citations (no matching source) are
  flagged.
- Retrieval traceability: each response carries a full audit trail — every query
  issued, every chunk retrieved, dense and sparse scores.
- Append-only compliance audit log: `storage/audit_log.jsonl`. Never mutated
  after write.

**Data warehouse pipeline**

- Snowflake-style 3-tier architecture: RAW → STAGING → ENRICHED.
- Default backend: SQLite (no credentials required). Swap to the Snowflake
  adapter via `WAREHOUSE_BACKEND=snowflake`.
- Every ingest run writes corpus lineage through all three tiers.

---

## Stack

| Layer | Choice |
|-------|--------|
| LLM (primary) | Groq, `llama-3.3-70b-versatile` |
| LLM (fallback) | Groq, `llama3-8b-8192` |
| Embeddings | `intfloat/e5-small-v2` |
| Vector DB | Chroma, persisted under `backend/storage/experiment_chroma` |
| Sparse retrieval | `rank-bm25` |
| Backend | FastAPI + Uvicorn |
| Frontend | Next.js |
| Observability | Langfuse (optional) |
| LLMOps | Custom — prompt registry, cost tracker, model router, response log |
| Governance | Custom — grounding score, hallucination risk, citation validation, audit log |
| Data warehouse | SQLite (default) or Snowflake via `WarehouseAdapter` |

The runtime app needs `GROQ_API_KEY`. No OpenAI key is needed for the production
index.

---

## Production Retrieval Config

The shipped Chroma collection is:

```text
banking_exp_full_e5_small_v2_1500_255
```

Use these backend environment variables:

```text
CHROMA_PATH=/app/storage/experiment_chroma
COLLECTION_NAME=banking_exp_full_e5_small_v2_1500_255
EMBEDDING_MODEL=intfloat/e5-small-v2
EMBEDDING_QUERY_PREFIX=query:
EMBEDDING_PASSAGE_PREFIX=passage:
```

The e5 prefixes matter: documents were embedded with `passage: ` and user
queries must be embedded with `query: `.

---

## Run Locally

### Backend

```powershell
cd backend
.\.venv\Scripts\activate

$env:GROQ_API_KEY="your-groq-key"
$env:CHROMA_PATH="C:\projects\agentic-rag\backend\storage\experiment_chroma"
$env:COLLECTION_NAME="banking_exp_full_e5_small_v2_1500_255"
$env:EMBEDDING_MODEL="intfloat/e5-small-v2"
$env:EMBEDDING_QUERY_PREFIX="query: "
$env:EMBEDDING_PASSAGE_PREFIX="passage: "

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Check:

- <http://127.0.0.1:8000/health>
- <http://127.0.0.1:8000/search?q=Regulation%20CC%20funds%20availability>
- <http://127.0.0.1:8000/llmops/status>
- <http://127.0.0.1:8000/governance/audit>

### Frontend

```powershell
cd frontend
npm install
$env:NEXT_PUBLIC_API_URL="http://127.0.0.1:8000"
npm run dev
```

Open <http://127.0.0.1:3000>.

---

## Data and Scripts

Important files:

```text
backend/data/banking/sections.jsonl           fetched Title 12 CFR sections
backend/data/banking/qa.jsonl                 generated evaluation questions
backend/data/experiments/reeval_chroma_results.json
backend/storage/experiment_chroma/            production Chroma index
backend/storage/warehouse.db                  3-tier corpus lineage (SQLite)
backend/storage/response_log.jsonl            per-request LLMOps log
backend/storage/audit_log.jsonl               append-only governance audit log
backend/storage/deployment_manifest.json      rollback-ready version snapshot
```

Useful scripts:

```powershell
cd backend

# Fetch Title 12 data from eCFR
python scripts/fetch_ecfr.py

# Build the default banking index from sections.jsonl
# (also writes warehouse.db through RAW → STAGING → ENRICHED)
python scripts/ingest_banking.py

# Generate a labelled banking eval set with Groq
python scripts/generate_eval.py --n 100

# Evaluate retrieval and optionally full answers
python scripts/evaluate.py
python scripts/evaluate.py --answers

# Databricks-style evaluation notebook (Hit@K, MRR, NDCG, latency p50/p95)
python scripts/databricks_eval.py
python scripts/databricks_eval.py --top-k 5 --qa data/banking/qa.jsonl

# Benchmark embedding models and chunking strategies
python scripts/experiment.py --full-corpus

# Re-score persisted benchmark Chroma collections
python scripts/evaluate_experiment_chroma.py
```

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service status, indexed chunks, model config |
| `POST` | `/chat` | Chat with answer, sources, steps, grounding score, hallucination risk, latency |
| `GET` | `/search?q=...&k=5` | Raw hybrid retrieval results |
| `GET` | `/experiments` | Embedding/chunking benchmark results |
| `GET` | `/llmops/status` | Prompt version, deployment manifest, recent log summary |
| `GET` | `/governance/audit?limit=20` | Last N compliance audit entries |
| `GET` | `/docs` | FastAPI OpenAPI UI |

### `/chat` response shape

```json
{
  "answer": "string",
  "sources": [{ "n": 1, "title": "...", "source": "...", "snippet": "..." }],
  "steps": [{ "tool": "search_documentation", "arguments": {}, "summary": "..." }],
  "trace_id": "optional-langfuse-id",
  "model_used": "llama-3.3-70b-versatile",
  "fallback_used": false,
  "latency_ms": 1240.5,
  "grounding_score": 0.857,
  "hallucination_risk": "low"
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Primary LLM |
| `FALLBACK_MODEL` | `llama3-8b-8192` | Fallback LLM on tool_use_failed |
| `CHROMA_PATH` | `backend/storage/experiment_chroma` | ChromaDB path |
| `COLLECTION_NAME` | `banking_exp_full_e5_small_v2_1500_255` | Chroma collection |
| `EMBEDDING_MODEL` | `intfloat/e5-small-v2` | Sentence transformer model |
| `EMBEDDING_QUERY_PREFIX` | `query: ` | E5 query prefix |
| `EMBEDDING_PASSAGE_PREFIX` | `passage: ` | E5 passage prefix |
| `ENABLE_WEB_SEARCH` | `true` | Enable DuckDuckGo fallback tool |
| `WAREHOUSE_BACKEND` | `sqlite` | `sqlite` or `snowflake` |
| `SNOWFLAKE_ACCOUNT` | — | Required when `WAREHOUSE_BACKEND=snowflake` |
| `SNOWFLAKE_USER` | — | Required when `WAREHOUSE_BACKEND=snowflake` |
| `SNOWFLAKE_PASSWORD` | — | Required when `WAREHOUSE_BACKEND=snowflake` |
| `SNOWFLAKE_DATABASE` | — | Required when `WAREHOUSE_BACKEND=snowflake` |
| `SNOWFLAKE_WAREHOUSE` | — | Required when `WAREHOUSE_BACKEND=snowflake` |
| `LANGFUSE_PUBLIC_KEY` | — | Optional Langfuse observability |
| `LANGFUSE_SECRET_KEY` | — | Optional Langfuse observability |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |

---

## Deploy

### Backend: Render

1. Push the repo to GitHub.
2. In Render, create a Blueprint from `render.yaml`.
3. Set `GROQ_API_KEY`.
4. Confirm these values are present:

```text
CHROMA_PATH=/app/storage/experiment_chroma
COLLECTION_NAME=banking_exp_full_e5_small_v2_1500_255
EMBEDDING_MODEL=intfloat/e5-small-v2
EMBEDDING_QUERY_PREFIX=query:
EMBEDDING_PASSAGE_PREFIX=passage:
CORS_ORIGINS=*
ENABLE_WEB_SEARCH=true
```

The Docker build does not re-ingest or re-embed the corpus. It uses the
persisted Chroma index committed with the backend.

### Frontend: Vercel

1. Import this repo in Vercel.
2. Set the root directory to `frontend`.
3. Set `NEXT_PUBLIC_API_URL` to the Render backend URL.
4. Deploy.

---

## Architecture Overview

```
User query
    │
    ▼
FastAPI /chat
    │
    ▼
Agent loop (agent.py)
    ├── Langfuse trace (observability.py)
    ├── LLMOps: prompt version, token tracking, latency (llmops.py)
    │
    ├── Tool: search_documentation
    │       └── Hybrid retrieval (retriever.py)
    │               ├── Dense: Chroma vector search (vectorstore.py)
    │               └── Sparse: BM25 (rank_bm25)
    │               └── RRF fusion
    │
    ├── Tool: calculator (safe AST eval)
    ├── Tool: web_search (DuckDuckGo)
    │
    ├── Fallback model routing on tool_use_failed
    │
    └── Governance checks (governance.py)
            ├── Grounding score
            ├── Hallucination risk
            ├── Citation validation
            └── Audit log (append-only JSONL)

Ingest pipeline (ingest.py)
    ├── Chunk + embed corpus
    ├── Write to ChromaDB
    └── Warehouse pipeline (warehouse.py)
            ├── RAW tier
            ├── STAGING tier (validated)
            └── ENRICHED tier (chunk metadata)
```

---

## Notes

This is a technical demo and not legal advice. Retrieved regulation text should
be checked against the official eCFR source for compliance decisions.

---

## Databricks Free Edition Setup

Databricks Free Edition (Serverless) is free with no account expiry. The
evaluation notebook runs end-to-end on it without any cluster setup or file
uploads — it fetches all data directly from this public GitHub repo.

### Benchmark results (live run)

| Metric | Value |
|--------|-------|
| Hit@1 | **0.80** |
| Hit@3 | **1.00** |
| Hit@5 | **1.00** |
| MRR | **0.889** |
| Latency p50 | **190 ms** |
| Latency p95 | **234 ms** |

*Claude API documentation corpus · 15 questions · `intfloat/e5-small-v2` · BM25 + dense + RRF*

### 1. Sign up

Go to <https://databricks.com/try-databricks> and sign up for Free Edition
(choose the option without a cloud provider — this gives you the Serverless
workspace).

### 2. Import the notebook

In Databricks: **Workspace → Create → Import**

- Source: **File**
- Upload: `backend/notebooks/databricks_retrieval_eval.py`

Databricks recognises the `# Databricks notebook source` header and imports it
as a proper multi-cell notebook.

### 3. Run the notebook

No cluster or compute configuration needed — Free Edition uses Serverless
compute automatically.

- Open the imported notebook
- Click **Run all** (or Shift+Enter cell by cell)

Cell 0 installs `sentence-transformers` and `rank-bm25` via `%pip install`.
Cells 1–2 fetch corpus and eval data from GitHub over HTTPS (no file upload
required). Cell 7 logs all params and metrics to MLflow via the Databricks REST
API, bypassing the Spark Connect integration (which is not available in
Serverless).

### 4. View results in MLflow

In Databricks: **left sidebar → Experiments**

The run appears under `/Users/<your-email>/agentic-rag-retrieval-eval`.

Each run records:

| Category | Logged values |
|---|---|
| Params | embedding model, query/passage prefixes, top_k, dense_k, sparse_k, rrf_k, num_chunks |
| Metrics | Hit@1, Hit@3, Hit@5, MRR, NDCG, latency p50/p95 |

> **Note:** Free Edition Serverless does not expose `spark.mlflow.modelRegistryUri`
> via Spark Connect. The notebook's Cell 7 detects this and logs using the
> Databricks MLflow REST API directly (`/api/2.0/mlflow/runs/*`) with the
> notebook-injected bearer token — no manual config required.

### Run the same eval locally with MLflow

```powershell
cd backend
pip install mlflow
python scripts/databricks_eval.py --mlflow
mlflow ui  # open http://localhost:5000
```

---

See [CHANGES.md](CHANGES.md) for a detailed explanation of the LLMOps, governance,
and data warehouse changes.
