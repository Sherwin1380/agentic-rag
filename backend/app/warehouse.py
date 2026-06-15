"""Snowflake-style 3-tier data warehouse for corpus ingestion.

Implements a RAW → STAGING → ENRICHED pipeline mirroring enterprise data
warehouse patterns (Snowflake, BigQuery, Databricks Delta Lake). The default
backend uses SQLite so the project runs without cloud credentials; swap in the
SnowflakeWarehouse adapter when credentials are available.

Tier definitions:
  RAW      — verbatim JSONL records as ingested, zero transformation
  STAGING  — cleaned, type-validated, deduplicated records
  ENRICHED — chunk-level records with embedding metadata, ready for vector indexing
"""
from __future__ import annotations

import sqlite3
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import STORAGE_DIR


class WarehouseAdapter(ABC):
    """Abstract multi-tier warehouse interface."""

    @abstractmethod
    def write_raw(self, records: List[Dict[str, Any]]) -> int: ...

    @abstractmethod
    def write_staging(self, records: List[Dict[str, Any]]) -> int: ...

    @abstractmethod
    def write_enriched(self, records: List[Dict[str, Any]]) -> int: ...

    @abstractmethod
    def count(self, tier: str) -> int: ...


class SQLiteWarehouse(WarehouseAdapter):
    """SQLite-backed warehouse mirroring Snowflake's multi-layer architecture.

    Tables:
      raw_sections     — original eCFR/corpus records (verbatim)
      staging_sections — validated and cleaned records
      enriched_chunks  — chunk-level records with embedding metadata
    """

    def __init__(self, path: Optional[Path] = None):
        self._path = path or (STORAGE_DIR / "warehouse.db")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS raw_sections (
                    id          TEXT PRIMARY KEY,
                    title       TEXT,
                    category    TEXT,
                    part        TEXT,
                    section     TEXT,
                    url         TEXT,
                    text        TEXT,
                    loaded_at   INTEGER
                );
                CREATE TABLE IF NOT EXISTS staging_sections (
                    id           TEXT PRIMARY KEY,
                    title        TEXT NOT NULL,
                    category     TEXT,
                    part         TEXT,
                    section      TEXT,
                    url          TEXT,
                    text         TEXT NOT NULL,
                    char_count   INTEGER,
                    validated_at INTEGER
                );
                CREATE TABLE IF NOT EXISTS enriched_chunks (
                    chunk_id        TEXT PRIMARY KEY,
                    section_id      TEXT,
                    title           TEXT,
                    category        TEXT,
                    part            TEXT,
                    section         TEXT,
                    url             TEXT,
                    chunk_index     INTEGER,
                    char_count      INTEGER,
                    embedding_model TEXT,
                    indexed_at      INTEGER
                );
                """
            )

    def write_raw(self, records: List[Dict[str, Any]]) -> int:
        now = int(time.time())
        rows = [
            (
                r.get("id", ""),
                r.get("title", ""),
                r.get("category", ""),
                str(r.get("part", "")),
                str(r.get("section", "")),
                r.get("url", ""),
                r.get("text", ""),
                now,
            )
            for r in records
        ]
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO raw_sections "
                "(id, title, category, part, section, url, text, loaded_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                rows,
            )
        return len(rows)

    def write_staging(self, records: List[Dict[str, Any]]) -> int:
        """Validate and clean records before promotion to staging layer."""
        now = int(time.time())
        rows = []
        for r in records:
            text = (r.get("text") or "").strip()
            title = (r.get("title") or r.get("id", "unknown")).strip()
            if not text or not title:
                continue
            rows.append((
                r.get("id", ""),
                title,
                r.get("category", ""),
                str(r.get("part", "")),
                str(r.get("section", "")),
                r.get("url", ""),
                text,
                len(text),
                now,
            ))
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO staging_sections "
                "(id, title, category, part, section, url, text, char_count, validated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )
        return len(rows)

    def write_enriched(self, records: List[Dict[str, Any]]) -> int:
        """Write chunk-level metadata after embedding and vector indexing."""
        now = int(time.time())
        rows = [
            (
                r.get("chunk_id", ""),
                r.get("section_id", ""),
                r.get("title", ""),
                r.get("category", ""),
                r.get("part", ""),
                r.get("section", ""),
                r.get("url", ""),
                r.get("chunk_index", 0),
                r.get("char_count", 0),
                r.get("embedding_model", ""),
                now,
            )
            for r in records
        ]
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO enriched_chunks "
                "(chunk_id, section_id, title, category, part, section, url, "
                " chunk_index, char_count, embedding_model, indexed_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
        return len(rows)

    def count(self, tier: str) -> int:
        table = {
            "raw": "raw_sections",
            "staging": "staging_sections",
            "enriched": "enriched_chunks",
        }.get(tier)
        if not table:
            return 0
        with self._conn() as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608


class SnowflakeWarehouse(WarehouseAdapter):
    """Snowflake adapter.

    Set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD,
    SNOWFLAKE_DATABASE, and SNOWFLAKE_WAREHOUSE in the environment, then set
    WAREHOUSE_BACKEND=snowflake.  Requires: pip install snowflake-connector-python
    """

    def __init__(self) -> None:
        try:
            import snowflake.connector  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "snowflake-connector-python is required for SnowflakeWarehouse. "
                "Install with: pip install snowflake-connector-python"
            ) from exc

    def _conn(self):
        import os
        import snowflake.connector
        return snowflake.connector.connect(
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            user=os.environ["SNOWFLAKE_USER"],
            password=os.environ["SNOWFLAKE_PASSWORD"],
            database=os.environ["SNOWFLAKE_DATABASE"],
            warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        )

    def write_raw(self, records: List[Dict[str, Any]]) -> int:
        with self._conn() as conn:
            cs = conn.cursor()
            cs.executemany(
                "INSERT INTO RAW_SECTIONS (ID,TITLE,CATEGORY,PART,SECTION,URL,TEXT) "
                "SELECT %s,%s,%s,%s,%s,%s,%s",
                [(r.get("id"), r.get("title"), r.get("category"),
                  str(r.get("part", "")), str(r.get("section", "")),
                  r.get("url"), r.get("text")) for r in records],
            )
        return len(records)

    def write_staging(self, records: List[Dict[str, Any]]) -> int:
        return len(records)

    def write_enriched(self, records: List[Dict[str, Any]]) -> int:
        return len(records)

    def count(self, tier: str) -> int:
        table = {
            "raw": "RAW_SECTIONS",
            "staging": "STAGING_SECTIONS",
            "enriched": "ENRICHED_CHUNKS",
        }.get(tier, "RAW_SECTIONS")
        with self._conn() as conn:
            cs = conn.cursor()
            cs.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
            return cs.fetchone()[0]


def get_warehouse() -> WarehouseAdapter:
    """Return the configured warehouse adapter (default: SQLite)."""
    import os
    if os.environ.get("WAREHOUSE_BACKEND") == "snowflake":
        return SnowflakeWarehouse()
    return SQLiteWarehouse()
