"""Remove orphan Chroma segment folders after deleting collections.

Chroma can leave HNSW segment directories on disk after collections are deleted
from the SQLite registry. This script compares UUID-like directories beside
`chroma.sqlite3` with live segment IDs in SQLite and removes only directories
that are no longer referenced.

Run from backend/:
    python scripts/prune_chroma_orphans.py --dry-run
    python scripts/prune_chroma_orphans.py
"""
from __future__ import annotations

import argparse
import re
import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import STORAGE_DIR

DEFAULT_CHROMA = STORAGE_DIR / "experiment_chroma"
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "select 1 from sqlite_master where type='table' and name=?", (name,)
    ).fetchone()
    return row is not None


def live_segment_ids(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        if not table_exists(conn, "segments"):
            raise RuntimeError("Could not find Chroma 'segments' table")
        rows = conn.execute("select id from segments").fetchall()
        return {str(row[0]) for row in rows}
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chroma-path", default=str(DEFAULT_CHROMA))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    chroma_path = Path(args.chroma_path)
    db_path = chroma_path / "chroma.sqlite3"
    if not db_path.exists():
        raise RuntimeError(f"Missing {db_path}")

    live_ids = live_segment_ids(db_path)
    segment_dirs = [
        p for p in chroma_path.iterdir() if p.is_dir() and UUID_RE.match(p.name)
    ]
    orphans = [p for p in segment_dirs if p.name not in live_ids]

    print(f"Chroma path: {chroma_path}")
    print(f"Live segment ids in SQLite: {len(live_ids)}")
    print(f"Segment directories on disk: {len(segment_dirs)}")
    print(f"Orphan directories: {len(orphans)}")

    for path in sorted(orphans):
        if args.dry_run:
            print(f"  would delete {path.name}")
        else:
            print(f"  deleting {path.name}")
            shutil.rmtree(path)

    if args.dry_run:
        print("\nDry run only; no files were deleted.")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
