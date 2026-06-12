"""Keep only the winning experiment Chroma collection.

This is intentionally separate from evaluation so deleting losing vector indexes
is an explicit step. It reads the best row from a results JSON file and deletes
all other collections from the experiment Chroma store.

Run from backend/:
    python scripts/keep_winner_collection.py
    python scripts/keep_winner_collection.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATA_DIR, STORAGE_DIR  # noqa: E402

DEFAULT_RESULTS = DATA_DIR / "experiments" / "reeval_chroma_results.json"
DEFAULT_CHROMA = STORAGE_DIR / "experiment_chroma"


def load_winner(path: Path, collection_name: str | None) -> str:
    if collection_name:
        return collection_name
    data = json.loads(path.read_text(encoding="utf-8"))
    best = data.get("best")
    if not best or not best.get("collection_name"):
        rows = [r for r in data.get("results", []) if "hybrid_mrr" in r]
        if not rows:
            raise RuntimeError(f"No scored rows found in {path}")
        best = max(rows, key=lambda r: r["hybrid_mrr"])
    return best["collection_name"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--chroma-path", default=str(DEFAULT_CHROMA))
    parser.add_argument("--collection-name", default=None, help="override winner")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import chromadb
    from chromadb.config import Settings as ChromaSettings

    chroma_path = Path(args.chroma_path)
    winner = load_winner(Path(args.results), args.collection_name)
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )
    collections = sorted(c.name for c in client.list_collections())
    if winner not in collections:
        raise RuntimeError(f"Winner collection not found in Chroma store: {winner}")

    losers = [c for c in collections if c != winner]
    print(f"Chroma path: {chroma_path}")
    print(f"Winner: {winner}")
    print(f"Collections: {len(collections)} total, {len(losers)} to delete")
    if args.dry_run:
        for name in losers:
            print(f"  would delete {name}")
        return

    for name in losers:
        print(f"  deleting {name}")
        client.delete_collection(name)
    print("\nDone.")
    print("Use these production env vars:")
    print(f"  CHROMA_PATH={chroma_path}")
    print(f"  COLLECTION_NAME={winner}")


if __name__ == "__main__":
    main()
