# Backend container for the Agentic RAG API.
# The vector index is not built during image build; build/promote it separately
# so deploys do not re-embed the banking corpus every time.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache \
    TRANSFORMERS_OFFLINE=0

WORKDIR /app

# System deps (sentence-transformers/torch need libgomp).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU-only torch first to avoid pulling CUDA wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
    && pip install -r requirements.txt

COPY . .

# Pre-download the embedding model only. Do not run ingest here, because that
# re-embeds the full banking corpus on every Render deploy.
RUN python -c "from app.embeddings import _get_model; _get_model()"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
