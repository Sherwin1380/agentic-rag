# Agentic RAG over the Claude API docs

A retrieval-augmented assistant with a real **agent layer** on top — not naive
"embed and retrieve." An agent decides, per question, whether to run hybrid search
over a documentation corpus, call a calculator, search the web, or just answer
directly. Every answer carries **inline citations**, and the system ships with a
hand-labeled **eval harness** and optional **Langfuse** observability.

The bundled corpus is the **Anthropic Claude API documentation** (models, pricing,
tool use, streaming, prompt caching, thinking/effort). Swap in your own `.md` files
to point it at a different domain.

```
┌──────────────┐    /chat     ┌─────────────────────────────────────────────┐
│  Next.js UI  │ ───────────▶ │  FastAPI backend                            │
│  (Vercel)    │ ◀─────────── │                                             │
└──────────────┘  answer +    │   Agent loop (Groq · Llama 3.3 70B)         │
                  sources +   │     ├─ search_documentation ─┐              │
                  steps       │     ├─ calculator            │ hybrid       │
                              │     └─ web_search            │ retrieval    │
                              │                              ▼              │
                              │   Dense (Chroma + MiniLM) + BM25  →  RRF     │
                              │                              │              │
                              │   Langfuse traces (optional) ┘              │
                              └─────────────────────────────────────────────┘
```

## What it demonstrates

- **Chunking strategy** — paragraph-aware chunks (~900 chars) with sliding overlap.
- **Hybrid search** — dense vectors (Chroma + `all-MiniLM-L6-v2`) fused with sparse
  BM25 via **Reciprocal Rank Fusion**. See [`retriever.py`](backend/app/retriever.py).
- **Agentic layer** — a Groq tool-calling loop that chooses retrieve / calculate /
  web-search / answer. See [`agent.py`](backend/app/agent.py).
- **Source citations** — chunks get stable `[n]` numbers tracked across retrievals.
- **Eval harness** — retrieval Hit@k / Recall@k / Precision@k / MRR plus optional
  end-to-end answer grading. See [`scripts/evaluate.py`](backend/scripts/evaluate.py).
- **Observability** — every retrieval and LLM step logged to Langfuse when keys are set.

## Free stack

| Layer | Choice | Cost |
| --- | --- | --- |
| LLM | Groq — Llama 3.3 70B (`llama-3.3-70b-versatile`) | Free tier |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, local CPU | Free |
| Vector DB | Chroma (local, persisted on disk) | Free |
| Sparse | `rank-bm25` | Free |
| Web search | DuckDuckGo (`duckduckgo-search`) | Free, no key |
| Backend | FastAPI + Uvicorn (Render free tier) | Free |
| Frontend | Next.js (Vercel free tier) | Free |
| Observability | Langfuse cloud (optional) | Free tier |

The **only** key needed to run the agent is a free `GROQ_API_KEY`.

---

## Run it locally (Windows / macOS / Linux)

### 1. Backend

```bash
cd backend
python -m venv ..venv\Scripts\activate
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# CPU-only torch keeps the install small:
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
pip install -r requirements.txt

copy .env.example .env        # (Windows)  /  cp .env.example .env
# paste your free key from https://console.groq.com into GROQ_API_KEY

python scripts/ingest.py      # build the vector + BM25 index (downloads MiniLM once)
uvicorn app.main:app --reload # serves on http://localhost:8000
```

Check it: open <http://localhost:8000/health> and <http://localhost:8000/docs>.

### 2. Frontend

```bash
cd frontend
npm install
copy .env.local.example .env.local   # default points at http://localhost:8000
npm run dev                           # http://localhost:3000
```

Open <http://localhost:3000> and ask a question.

### 3. Evaluate

```bash
cd backend
python scripts/evaluate.py            # retrieval metrics (no LLM key needed)
python scripts/evaluate.py --answers  # also grade generated answers (needs GROQ_API_KEY)
```

---

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Index size, LLM/Langfuse status, model |
| `POST` | `/chat` | `{ message, history[] }` → `{ answer, sources[], steps[], trace_id }` |
| `GET` | `/search?q=...&k=5` | Raw hybrid-retrieval results (no LLM) — handy for debugging |
| `GET` | `/docs` | Interactive OpenAPI docs |

---

## Deploy (free tiers)

These steps need **your** accounts (signup is free). The configs are already written.

### Backend → Render

1. Push this repo to GitHub.
2. In Render: **New → Blueprint**, select the repo. It reads
   [`render.yaml`](render.yaml) and builds [`backend/Dockerfile`](backend/Dockerfile)
   (the index is built into the image, so it's ready on boot).
3. Set `GROQ_API_KEY` (and optionally the Langfuse keys) in the dashboard.
4. Note the service URL, e.g. `https://agentic-rag-api.onrender.com`.

> The Render free tier sleeps after inactivity; the first request after idle takes a
> few seconds to wake. The MiniLM model and Chroma index are baked into the image.

### Frontend → Vercel

1. In Vercel: **New Project → Import** this repo, set the root directory to `frontend`.
2. Add an env var `NEXT_PUBLIC_API_URL` = your Render backend URL.
3. Deploy. Set `CORS_ORIGINS` on the backend to your Vercel domain (or leave `*`).

### Observability → Langfuse (optional)

Sign up at <https://cloud.langfuse.com>, create a project, and set
`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` on the backend. Each chat then produces
a trace with nested spans for retrieval and every LLM decision.

---

## Project layout

```
agentic-rag/
├── backend/
│   ├── app/
│   │   ├── main.py          FastAPI app + routes
│   │   ├── agent.py         agentic tool-calling loop
│   │   ├── retriever.py     hybrid search (dense + BM25 + RRF)
│   │   ├── vectorstore.py   Chroma wrapper
│   │   ├── embeddings.py    local sentence-transformers
│   │   ├── tools.py         calculator + web search
│   │   ├── ingest.py        chunking + indexing
│   │   ├── llm.py           Groq client
│   │   ├── observability.py Langfuse (no-op without keys)
│   │   ├── models.py        Pydantic schemas
│   │   └── config.py        settings
│   ├── data/
│   │   ├── corpus/          the Claude API docs (.md with frontmatter)
│   │   └── eval/qa.jsonl    hand-labeled QA set
│   ├── scripts/
│   │   ├── ingest.py        build the index
│   │   └── evaluate.py      eval harness
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                Next.js chat UI
├── render.yaml              backend blueprint
└── README.md
```

## Use your own corpus

Drop Markdown files into `backend/data/corpus/` with this frontmatter, then re-run
`python scripts/ingest.py`:

```markdown
---
title: My document title
url: https://example.com/source
---

# Body content...
```

Update `backend/data/eval/qa.jsonl` with questions, `relevant_sources` (filenames),
and `expected_keywords` to keep the eval harness meaningful.

Hosted right now 
FE - https://agentic-qp02t0qgo-sherwin13-projects.vercel.app/
BE - https://agentic-rag-api-43w8.onrender.com