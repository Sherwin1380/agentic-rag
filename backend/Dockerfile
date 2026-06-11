# Backend container for the Agentic RAG API.
# Builds the vector index at image-build time so the service is ready on boot.
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

# Pre-download the embedding model and build the index at build time.
RUN python -c "from app.embeddings import _get_model; _get_model()" \
    && python scripts/ingest_banking.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
