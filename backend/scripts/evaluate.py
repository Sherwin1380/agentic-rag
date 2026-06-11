"""Evaluation harness.

Two layers of evaluation:

1. Retrieval quality (no LLM key needed): for each labelled question we check
   whether the hybrid retriever surfaces the documents we marked relevant.
   Reports Hit@k, Recall@k, Precision@k, and MRR.

2. End-to-end answer quality (needs GROQ_API_KEY, pass --answers): runs the full
   agent and checks that the answer contains the expected keywords and cites a
   source. A lightweight stand-in for RAGAS-style faithfulness/answer-relevancy.

Run from backend/:
    python scripts/evaluate.py            # retrieval metrics only
    python scripts/evaluate.py --answers  # also grade generated answers
    python scripts/evaluate.py --k 5
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATA_DIR, EVAL_DIR  # noqa: E402
from app import retriever  # noqa: E402

from collections import defaultdict  # noqa: E402


def load_qa(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def eval_retrieval(items, k, verbose=True):
    hits = 0
    recall_sum = 0.0
    precision_sum = 0.0
    rr_sum = 0.0
    # Per-category aggregation (for the banking eval).
    cat_rr = defaultdict(float)
    cat_hit = defaultdict(int)
    cat_n = defaultdict(int)

    print(f"\n=== Retrieval evaluation (k={k}) ===")
    for item in items:
        q = item["question"]
        relevant = set(item.get("relevant_sources", []))
        cat = item.get("category", "all")
        results = retriever.hybrid_search(q, top_k=k)
        retrieved = [(r.get("metadata") or {}).get("source") for r in results]
        retrieved_set = set(retrieved)

        found = relevant & retrieved_set
        hit = 1 if found else 0
        hits += hit
        recall_sum += (len(found) / len(relevant)) if relevant else 0.0
        precision_sum += (len(found) / len(retrieved)) if retrieved else 0.0

        rr = 0.0
        for rank, src in enumerate(retrieved, start=1):
            if src in relevant:
                rr = 1.0 / rank
                break
        rr_sum += rr

        cat_rr[cat] += rr
        cat_hit[cat] += hit
        cat_n[cat] += 1

        if verbose:
            flag = "OK " if hit else "MISS"
            print(f"  [{flag}] rr={rr:.2f}  {q[:62]}")

    n = len(items)
    print(f"\n  Questions:    {n}")
    print(f"  Hit@{k}:       {hits / n:.3f}")
    print(f"  Recall@{k}:    {recall_sum / n:.3f}")
    print(f"  Precision@{k}: {precision_sum / n:.3f}")
    print(f"  MRR:          {rr_sum / n:.3f}")

    if len(cat_n) > 1:
        print(f"\n  Per-category (Hit@{k} / MRR):")
        for c in sorted(cat_n):
            print(
                f"    {c[:48]:48s} n={cat_n[c]:>3}  "
                f"Hit={cat_hit[c]/cat_n[c]:.3f}  MRR={cat_rr[c]/cat_n[c]:.3f}"
            )


def eval_answers(items):
    from app import agent

    print("\n=== End-to-end answer evaluation ===")
    kw_pass = 0
    cite_pass = 0
    errors = 0
    for item in items:
        q = item["question"]
        expected = [w.lower() for w in item.get("expected_keywords", [])]
        try:
            result = agent.run_agent(q, [])
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            errors += 1
            print(f"  [ERR ] {q[:55]:55s} {exc}")
            continue
        answer = result["answer"].lower()
        has_all = all(w in answer for w in expected) if expected else True
        has_cite = "[" in result["answer"] and "]" in result["answer"]
        kw_pass += 1 if has_all else 0
        cite_pass += 1 if has_cite else 0
        flag = "OK " if has_all else "WEAK"
        print(f"  [{flag}] {q[:55]:55s} kw={has_all} cite={has_cite} "
              f"steps={[s.tool for s in result['steps']]}")

    n = len(items)
    print(f"\n  Keyword coverage: {kw_pass / n:.3f}")
    print(f"  Cited a source:   {cite_pass / n:.3f}")
    print(f"  Agent errors:     {errors}/{n}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--answers", action="store_true",
                        help="also run full agent (needs GROQ_API_KEY)")
    parser.add_argument("--qa", default=str(DATA_DIR / "banking" / "qa.jsonl"),
                        help="path to the QA jsonl (default: banking eval)")
    parser.add_argument("--quiet", action="store_true", help="hide per-question lines")
    args = parser.parse_args()

    qa_path = Path(args.qa)
    if not qa_path.exists():
        # Fall back to the original Claude-docs eval if banking isn't built yet.
        qa_path = EVAL_DIR / "qa.jsonl"
    print(f"Eval set: {qa_path}")

    items = load_qa(qa_path)
    eval_retrieval(items, args.k, verbose=not args.quiet)
    if args.answers:
        eval_answers(items)


if __name__ == "__main__":
    main()
