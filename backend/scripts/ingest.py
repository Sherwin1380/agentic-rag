"""Build the vector + BM25 index from the corpus.

Run from the backend/ directory:
    python scripts/ingest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest import build_index  # noqa: E402

if __name__ == "__main__":
    print("Building index from corpus ...")
    n = build_index(verbose=True)
    print(f"\nDone. {n} chunks indexed.")
