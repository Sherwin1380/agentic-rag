# Agentic RAG for U.S. Banking Regulations

A retrieval-augmented assistant for U.S. banking regulations, focused on Title
12 of the Code of Federal Regulations. The app combines a tool-calling agent,
hybrid retrieval, source citations, and an embedding/chunking benchmark UI.

Hosted:

- Frontend: <https://agentic-qp02t0qgo-sherwin13-projects.vercel.app/>
- Backend: <https://agentic-rag-api-43w8.onrender.com>

## What It Demonstrates

- Agentic RAG: the assistant decides when to retrieve, calculate, search the web,
  or answer directly.
- Domain corpus: 5,002 eCFR Title 12 sections across OCC, Federal Reserve, FDIC,
  NCUA, and CFPB regulations.
- Hybrid retrieval: dense Chroma search plus BM25, fused with Reciprocal Rank
  Fusion.
- Citations: answers include inline citations tied to retrieved regulation
  chunks.
- Benchmarking: compares embedding models and chunking strategies with MRR,
  Hit@k, Recall@k, Precision@k, and NDCG@k.
- Production index: uses the best open-source result, `intfloat/e5-small-v2`
  with chunk size `1500` and overlap `255`.

## Stack

| Layer | Choice |
| --- | --- |
| LLM | Groq, `llama-3.3-70b-versatile` |
| Embeddings | `intfloat/e5-small-v2` |
| Vector DB | Chroma, persisted under `backend/storage/experiment_chroma` |
| Sparse retrieval | `rank-bm25` |
| Backend | FastAPI + Uvicorn |
| Frontend | Next.js |
| Optional tracing | Langfuse |

The runtime app needs `GROQ_API_KEY`. No OpenAI key is needed for the selected
production index.

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

### Frontend

```powershell
cd frontend
npm install
$env:NEXT_PUBLIC_API_URL="http://127.0.0.1:8000"
npm run dev
```

Open <http://127.0.0.1:3000>.

## Data And Scripts

Important files:

```text
backend/data/banking/sections.jsonl      fetched Title 12 CFR sections
backend/data/banking/qa.jsonl            generated evaluation questions
backend/data/experiments/reeval_chroma_results.json
backend/storage/experiment_chroma/       selected production Chroma index
```

Useful scripts:

```powershell
cd backend

# Fetch Title 12 data from eCFR
python scripts/fetch_ecfr.py

# Build the default banking index from sections.jsonl
python scripts/ingest_banking.py

# Generate a labelled banking eval set with Groq
python scripts/generate_eval.py --n 100

# Evaluate retrieval and optionally full answers
python scripts/evaluate.py
python scripts/evaluate.py --answers

# Benchmark embedding models and chunking strategies
python scripts/experiment.py --full-corpus

# Re-score persisted benchmark Chroma collections
python scripts/evaluate_experiment_chroma.py
```

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Service status, indexed chunks, model config |
| `POST` | `/chat` | Chat with answer, sources, tool steps, trace id |
| `GET` | `/search?q=...&k=5` | Raw hybrid retrieval results |
| `GET` | `/experiments` | Embedding/chunking benchmark results |
| `GET` | `/docs` | FastAPI OpenAPI UI |

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

The Docker build does not re-ingest or re-embed the corpus. It uses the persisted
Chroma index committed with the backend.

### Frontend: Vercel

1. Import this repo in Vercel.
2. Set the root directory to `frontend`.
3. Set `NEXT_PUBLIC_API_URL` to the Render backend URL.
4. Deploy.

## Notes

This is a technical demo and not legal advice. Retrieved regulation text should
be checked against the official eCFR source for compliance decisions.
