"""Build the Chroma index from the eCFR banking-regulations corpus.

Run from backend/ after fetch_ecfr.py:
    python scripts/ingest_banking.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest import build_banking_index  # noqa: E402

if __name__ == "__main__":
    print("Building banking-regulations index ...")
    n = build_banking_index(verbose=True)
    print(f"\nDone. {n} chunks indexed.")
