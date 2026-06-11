"""Fetch U.S. Title 12 CFR (Banks and Banking) from the official eCFR API.

Public domain, no API key. We pull the major federal banking-regulator chapters
and write one record per CFR *section* to data/banking/sections.jsonl. Each
record carries a `category` (the regulator/chapter) plus part/section metadata,
which the eval harness uses to measure retrieval quality per category.

Usage (from backend/):
    python scripts/fetch_ecfr.py                 # default chapters
    python scripts/fetch_ecfr.py --chapters II X # subset
    python scripts/fetch_ecfr.py --limit-parts 5 # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import requests

BASE = "https://www.ecfr.gov/api/versioner/v1"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "banking"

# Title 12 chapters -> regulator (these become the eval categories).
CHAPTER_AGENCY = {
    "I": "OCC (Office of the Comptroller of the Currency)",
    "II": "FRS (Federal Reserve System)",
    "III": "FDIC (Federal Deposit Insurance Corporation)",
    "VII": "NCUA (National Credit Union Administration)",
    "X": "CFPB (Consumer Financial Protection Bureau)",
}
DEFAULT_CHAPTERS = ["I", "II", "III", "VII", "X"]

_WS = re.compile(r"\s+")
session = requests.Session()
session.headers.update({"User-Agent": "agentic-rag-eval/1.0 (educational)"})


def clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def get_date() -> str:
    titles = session.get(f"{BASE}/titles.json", timeout=60).json()["titles"]
    t12 = next(t for t in titles if t["number"] == 12)
    return t12.get("up_to_date_as_of") or t12["latest_issue_date"]


def get_structure(date: str) -> dict:
    return session.get(f"{BASE}/structure/{date}/title-12.json", timeout=120).json()


def walk_parts(node: dict, chapter: str | None, acc: list[tuple[str, str]]) -> None:
    """Collect (chapter_id, part_id) for every part under the tree."""
    ntype = node.get("type")
    ident = node.get("identifier")
    if ntype == "chapter":
        chapter = ident
    if ntype == "part" and chapter is not None:
        acc.append((chapter, ident))
        return  # don't descend further; we fetch the whole part at once
    for child in node.get("children", []) or []:
        walk_parts(child, chapter, acc)


def fetch_part_xml(date: str, part: str, retries: int = 3) -> str | None:
    url = f"{BASE}/full/{date}/title-12.xml"
    for attempt in range(retries):
        try:
            r = session.get(url, params={"part": part}, timeout=240)
            if r.status_code == 200:
                return r.text
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def parse_sections(xml_text: str, chapter: str, part: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    agency = CHAPTER_AGENCY.get(chapter, f"Chapter {chapter}")
    docs: list[dict] = []
    for div in root.iter("DIV8"):
        if div.get("TYPE") != "SECTION":
            continue
        sec = div.get("N", "").strip()
        head_el = div.find("HEAD")
        heading = clean("".join(head_el.itertext())) if head_el is not None else sec
        body = clean("".join(div.itertext()))
        if len(body) < 80:
            continue
        docs.append(
            {
                "id": f"12CFR-{sec}",
                "title": heading,
                "category": agency,
                "chapter": chapter,
                "part": part,
                "section": sec,
                "url": f"https://www.ecfr.gov/current/title-12/section-{sec}",
                "text": body,
            }
        )
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", nargs="*", default=DEFAULT_CHAPTERS)
    ap.add_argument("--limit-parts", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    date = get_date()
    print(f"Title 12 as of {date}; chapters: {args.chapters}")

    structure = get_structure(date)
    parts: list[tuple[str, str]] = []
    walk_parts(structure, None, parts)
    parts = [(c, p) for (c, p) in parts if c in args.chapters]
    if args.limit_parts:
        parts = parts[: args.limit_parts]
    print(f"Found {len(parts)} parts to fetch.")

    out_path = OUT_DIR / "sections.jsonl"
    total_secs = 0
    total_chars = 0
    seen_ids: set[str] = set()
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (chapter, part) in enumerate(parts, 1):
            xml_text = fetch_part_xml(date, part)
            if not xml_text:
                print(f"  [{i}/{len(parts)}] ch {chapter} part {part}: (skip)")
                continue
            secs = parse_sections(xml_text, chapter, part)
            written = 0
            for d in secs:
                if d["id"] in seen_ids:
                    continue
                seen_ids.add(d["id"])
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                total_chars += len(d["text"])
                written += 1
            total_secs += written
            print(f"  [{i}/{len(parts)}] ch {chapter} part {part}: {written} sections")
            time.sleep(0.25)

    print(
        f"\nWrote {total_secs} sections ({total_chars/1e6:.1f}M chars) -> {out_path}"
    )


if __name__ == "__main__":
    main()
