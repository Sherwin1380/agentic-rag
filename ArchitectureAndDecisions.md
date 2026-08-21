What this project is
This is a full-stack, agentic RAG system for answering questions about U.S. banking regulations in Title 12 of the CFR.
The repository contains:
- 5,002 regulatory sections from OCC, Federal Reserve, FDIC, NCUA, and CFPB.
- A persisted Chroma index containing 17,971 chunks.
- Dense vector retrieval using E5 embeddings.
- Optional BM25 keyword retrieval combined through Reciprocal Rank Fusion.
- A tool-calling LLM agent using Groq.
- Source citations, governance checks, evaluation, observability, and a Next.js interface.
A good 30-second interview description is:
“I built an agentic RAG assistant for U.S. banking regulations. I collected approximately 5,000 Title 12 CFR sections from the official eCFR API, chunked and embedded them using E5, and stored roughly 18,000 chunks in Chroma. At runtime, a Groq-powered agent decides whether it needs regulatory retrieval, a calculator, web search, or a direct response. Retrieval can combine semantic search and BM25 using Reciprocal Rank Fusion. The system returns traceable citations and records latency, model version, grounding signals, and audit information. I also built an evaluation pipeline for comparing embedding models and chunking strategies.”

Architecture
eCFR API
   │
   ▼
5,002 Title 12 sections
   │
   ├── RAW warehouse layer
   ├── STAGING validated layer
   │
   ▼
Paragraph-aware chunking
   │
   ▼
E5 passage embeddings
   │
   ├── ENRICHED warehouse metadata
   ▼
Chroma vector index
   │
   └──────────────┐
                  │
User → Next.js → FastAPI → Tool-calling agent
                              │
                 ┌────────────┼─────────────┐
                 ▼            ▼             ▼
             Retrieval    Calculator    Web search
                 │
          Dense + BM25
                 │
              RRF fusion
                 │
            Top 5 chunks
                 │
                 ▼
          Grounded LLM answer
                 │
        Citations + governance
                 │
       Langfuse + JSONL logs
Step-by-step execution
1. Data collection
The offline pipeline starts in [fetch_ecfr.py (line 69)](C:/projects/agentic-rag/backend/scripts/fetch_ecfr.py:69).
It:
1. Calls the public eCFR API.
2. Finds the current version of Title 12.
3. Downloads parts belonging to five major banking regulators.
4. Parses the eCFR XML.
5. Produces one JSONL record per CFR section.
Each record contains fields such as:
{
  "id": "12CFR-229.10",
  "title": "...",
  "category": "FRS",
  "part": "229",
  "section": "229.10",
  "url": "https://www.ecfr.gov/...",
  "text": "..."
}
Interview point:
“Using one CFR section as the source-level document gave me stable identifiers for citations and labelled retrieval evaluation.”

2. Chunking
Chunking is implemented in [ingest.py (line 43)](C:/projects/agentic-rag/backend/app/ingest.py:43).
The production configuration is:
- Chunk size: 1,500 characters.
- Overlap: 255 characters.
- Current index: 17,971 chunks.
The implementation first groups paragraphs until it reaches the target size. Oversized paragraphs are hard-split, and adjacent chunks receive character overlap.
Why use overlap?
“A regulation requirement can begin near the end of one chunk and finish in the next. Overlap reduces the probability of losing that context.”

One detail to state accurately: these sizes are measured in characters, not tokens.
3. Embedding
Embeddings are generated in [embeddings.py (line 56)](C:/projects/agentic-rag/backend/app/embeddings.py:56) using:
intfloat/e5-small-v2
The model runs locally on CPU and produces normalized vectors.
E5 requires asymmetric prefixes:
query: user question
passage: regulation text
This matters because E5 was trained to distinguish query and passage roles. Forgetting the prefixes can reduce retrieval quality.
The model is lazily loaded once using a lock, which avoids loading the Transformer multiple times during concurrent first requests.
4. Vector indexing
The vectors and metadata are stored in Chroma through [vectorstore.py (line 68)](C:/projects/agentic-rag/backend/app/vectorstore.py:68).
Chroma uses cosine distance. The wrapper converts it to a similarity-like score:
score = 1.0 - distance
Metadata includes the source section, title, URL, regulator, part, section, and chunk number. That metadata is later used for citations and evaluation.
The index is built separately from the Docker image. This prevents every deployment from re-embedding the complete corpus.
5. Warehouse and lineage
The ingestion pipeline also implements a three-layer data pattern in [warehouse.py (line 24)](C:/projects/agentic-rag/backend/app/warehouse.py:24):
- RAW: original eCFR records.
- STAGING: cleaned and validated records.
- ENRICHED: chunk-level indexing metadata.
SQLite is the functional default implementation.
Be precise about Snowflake in an interview: a Snowflake adapter exists, but only its RAW write is substantially implemented. STAGING and ENRICHED are currently placeholders. Present it as an extensibility boundary, not as a complete Snowflake production pipeline.
6. Receiving a chat request
The Next.js UI calls POST /chat using [api.ts (line 34)](C:/projects/agentic-rag/frontend/app/lib/api.ts:34). The main interface is in [page.tsx (line 26)](C:/projects/agentic-rag/frontend/app/page.tsx:26).
The FastAPI handler is in [main.py (line 88)](C:/projects/agentic-rag/backend/app/main.py:88). It:
1. Validates that the message is not empty.
2. Passes the message and conversation history to the agent.
3. Serializes the result through Pydantic models.
4. Returns the answer, sources, tool steps, and operational metadata.
The frontend sends the entire visible history, but the backend keeps only the last six messages. That constrains prompt size, although it is message-count-based rather than token-aware.
7. Agent decision
The core logic is in [agent.py (line 279)](C:/projects/agentic-rag/backend/app/agent.py:279).
The agent receives three possible tools:
- search_documentation
- calculator
- Optional web_search
The system prompt instructs it to:
- Retrieve for regulatory questions.
- Use the calculator for arithmetic.
- Use web search for current or out-of-domain questions.
- Answer greetings directly.
- Cite retrieved sources.
The loop allows at most three LLM decision steps. This limits cost and protects against repeated tool calls.
Why call it agentic?
“Retrieval is exposed as a tool. The LLM first classifies the intent implicitly and decides whether retrieval is necessary. A conventional RAG pipeline retrieves for every message, including greetings and unrelated questions.”

It is a constrained tool-calling agent, not a multi-agent system or an open-ended autonomous planner.
8. Hybrid retrieval
Retrieval is implemented in [retriever.py (line 84)](C:/projects/agentic-rag/backend/app/retriever.py:84).
Dense branch
1. Prefix the question with query:.
2. Generate an E5 query vector.
3. Retrieve the top 12 Chroma candidates.
Dense retrieval is useful for conceptual similarity. For example, it can connect “when can I access deposited funds?” with “funds availability.”
Sparse branch
1. Tokenize all chunks.
2. Build an in-memory BM25 index.
3. Retrieve the top 12 keyword candidates.
BM25 is useful for exact terminology, regulation names, section references, and thresholds.
Fusion
The two rankings are combined with Reciprocal Rank Fusion:
RRF(document) = Σ 1 / (60 + rank + 1)
The final top five chunks are returned.
Why RRF instead of adding the raw scores?
“Cosine similarity and BM25 scores have different, query-dependent scales. RRF combines their ranks without requiring score calibration.”

Important deployment nuance: local defaults enable BM25, but the Render blueprint sets ENABLE_SPARSE_BM25=false and WARM_BM25_ON_STARTUP=false. Therefore, the provided hosted configuration is dense-only to reduce startup time and memory consumption. Don’t claim the hosted path is hybrid unless that flag has been enabled.
9. Citation handling
Every retrieved chunk is registered using its stable Chroma ID.
The first new source becomes [1], the second becomes [2], and so on. If the agent performs another search and retrieves an existing chunk, its citation number remains stable.
The full chunk text is returned to the LLM, while the API exposes a shorter snippet and metadata to the UI.
10. Generating the answer
After tool execution:
1. The assistant tool-call message is appended to the conversation.
2. The tool result is appended as a tool message.
3. The LLM sees the retrieved regulation text.
4. It produces a final answer with inline citations.
The safe calculator is worth mentioning: it parses expressions into an AST and allows only approved numeric operators and functions. It does not run arbitrary Python eval.
11. Tool-call recovery
The project handles a real integration issue: some models may emit XML-like tool calls instead of valid JSON.
The recovery path:
1. Detects tool_use_failed.
2. Attempts to extract the intended function and arguments.
3. Executes the tool directly.
4. Feeds the result back to the model.
5. If parsing fails, retrieves using the original question and generates an answer with tools disabled.
This is a good challenge to discuss:
“The model’s intent was often correct, but the serialization violated the provider’s tool-call schema. I added a recovery layer so formatting failures did not immediately become user-facing errors.”

However, the configured primary and fallback models are currently both llama-3.1-8b-instant. This is fallback execution behavior, but not genuine model diversity.
12. Governance
The governance layer is in [governance.py (line 30)](C:/projects/agentic-rag/backend/app/governance.py:30).
It calculates:
- Grounding score: fraction of answer sentences containing a citation.
- Hallucination-risk classification.
- Citation validity: whether every [n] maps to an actual retrieved source.
- Retrieval trace: queries, source IDs, dense scores, and sparse scores.
- Append-only JSONL audit entries.
A critical distinction:
“The grounding score is a monitoring heuristic, not a semantic faithfulness metric. It detects missing citations, but it does not prove that the cited text entails the answer.”

For example, a sentence can cite an irrelevant source and still increase the score.
13. LLMOps and observability
The operational layer includes:
- Prompt version v1.3.0.
- Request identifiers.
- Latency measurement.
- Model and fallback tracking.
- Token tracking.
- Cost estimation.
- Response logging.
- Deployment manifest.
- Optional Langfuse traces.
Relevant implementations are [llmops.py (line 25)](C:/projects/agentic-rag/backend/app/llmops.py:25) and [observability.py (line 56)](C:/projects/agentic-rag/backend/app/observability.py:56).
Langfuse is optional. Without credentials, its wrapper becomes a no-op, so observability cannot break the main application.
14. Evaluation
The project uses 88 labelled evaluation questions. Each question contains one or more relevant CFR source IDs.
Retrieval metrics include:
- Hit@k: was any relevant source retrieved?
- Recall@k: what fraction of relevant sources was retrieved?
- Precision@k: what fraction of retrieved results was relevant?
- MRR: how highly ranked was the first relevant source?
- NDCG: ranking quality with position discounts.
- p50 and p95 latency.
The README reports:
Hit@1: 0.80
Hit@3: 1.00
Hit@5: 1.00
MRR:   0.889
p50:   190 ms
p95:   234 ms
Phrase this as “the recorded benchmark reported,” unless you personally reran it in a reproducible environment.
The benchmark code compares:
- Six embedding models.
- Chunk sizes 400, 900, and 1,500.
- Each size with and without approximately 17% overlap.
- Dense retrieval versus hybrid retrieval.
That is potentially 36 configurations when all credentials and models are available.