"""Generate a labelled QA eval set from the banking corpus using Groq.

For each sampled CFR section we ask the LLM to write one realistic question that
is specifically answerable from that section. The section it was generated from
is the ground-truth `relevant_sources` label, so the retrieval harness can
compute MRR/Hit/Recall. Sampling is balanced across categories so the 100
questions cover every regulator.

Run from backend/ (needs GROQ_API_KEY):
    python scripts/generate_eval.py --n 100
"""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm  # noqa: E402
from app.config import DATA_DIR  # noqa: E402
from app.ingest import load_jsonl_corpus  # noqa: E402

SECTIONS = DATA_DIR / "banking" / "sections.jsonl"
OUT = DATA_DIR / "banking" / "qa.jsonl"

SYS = (
    "You write evaluation questions for a banking-regulations retrieval system. "
    "Given the text of one U.S. Code of Federal Regulations (Title 12) section, "
    "write ONE natural question that a banker, compliance officer, or consumer "
    "might ask, which is specifically and uniquely answerable from THIS section. "
    "Mention the concrete topic (e.g. the regulation subject, a threshold, a "
    "defined term, a requirement) so the question is self-contained. Do NOT "
    "mention the section number. Output ONLY the question text, nothing else."
)


def clean_question(text: str) -> str:
    text = text.strip().strip('"').strip()
    # Drop any leading "Question:" style preamble.
    text = re.sub(r"^(question\s*:\s*)", "", text, flags=re.IGNORECASE).strip()
    return text.split("\n")[0].strip()


def gen_question(section_text: str) -> str:
    messages = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": section_text[:4000]},
    ]
    completion = llm.chat(messages, tools=None, temperature=0.3)
    return clean_question(completion.choices[0].message.content or "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    docs = load_jsonl_corpus(SECTIONS)

    # Keep substantive sections only (skip reserved/very short).
    docs = [
        d
        for d in docs
        if len(d["text"]) > 600 and "[reserved]" not in d["text"].lower()[:120]
    ]

    by_cat = defaultdict(list)
    for d in docs:
        by_cat[d["category"]].append(d)
    cats = sorted(by_cat)
    print(f"{len(docs)} usable sections across {len(cats)} categories")

    # Balanced quota across categories (distribute remainder).
    base = args.n // len(cats)
    quota = {c: base for c in cats}
    for c in cats[: args.n - base * len(cats)]:
        quota[c] += 1

    # Sample sections per category.
    picked = []
    for c in cats:
        pool = by_cat[c]
        random.shuffle(pool)
        picked.extend(pool[: quota[c]])
    random.shuffle(picked)

    written = 0
    seen_q = set()
    with open(OUT, "w", encoding="utf-8") as f:
        for i, d in enumerate(picked, 1):
            try:
                q = gen_question(d["text"])
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}] ERROR {exc}")
                continue
            if not q or q.lower() in seen_q:
                continue
            seen_q.add(q.lower())
            rec = {
                "question": q,
                "relevant_sources": [d["id"]],
                "category": d["category"],
                "part": d["part"],
                "section": d["section"],
                "title": d["title"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if i % 10 == 0:
                print(f"  {i}/{len(picked)} generated")

    print(f"\nWrote {written} questions -> {OUT}")
    # Per-category tally.
    counts = defaultdict(int)
    for line in open(OUT, encoding="utf-8"):
        counts[json.loads(line)["category"]] += 1
    for c in sorted(counts):
        print(f"  {c}: {counts[c]}")


if __name__ == "__main__":
    main()
