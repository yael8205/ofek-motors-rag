"""Asks Claude every golden question and checks the answers. Costs about one cent with claude-haiku-5-5.

  python eval_answers.py --chunker headings --mode hybrid --current-only
  python eval_answers.py --only 13          # one question
"""
import argparse
import json
from pathlib import Path

from eval_retrieval import build_index, needles
from rag.answer import ask, verify_citations
from rag.search import search

PRICE_IN, PRICE_OUT = 0.10, 0.50  # claude-haiku-5-5, dollars per million tokens


def check(q: dict, answer, bad_citations, chunks) -> tuple[bool, str]:
    if q["expect"] == "insufficient":
        ok = answer.insufficient_context and not answer.citations
        return ok, "" if ok else "should have said there is not enough information"
    if answer.insufficient_context:
        return False, "said there is not enough information"
    if bad_citations:
        return False, f"{len(bad_citations)} citation(s) not found in the chunks"
    quotes = "\n".join(c.quote for c in answer.citations)
    if not all(n in quotes for n in needles(q)):
        return False, "the quotes do not contain the answer"
    for word in q.get("forbidden_in_answer", []):
        if word in answer.answer:
            return False, f"the answer contains {word!r}"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunker", choices=["fixed", "headings"], default="fixed")
    ap.add_argument("--mode", choices=["vector", "hybrid"], default="vector")
    ap.add_argument("--current-only", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--only", type=int, help="run a single question id")
    args = ap.parse_args()

    golden = json.loads(Path("golden.json").read_text(encoding="utf-8"))
    if args.only:
        golden = [q for q in golden if q["id"] == args.only]
    index = build_index(args.chunker)
    passed, tokens_in, tokens_out = 0, 0, 0
    for q in golden:
        chunks = search(index, q["question"], args.mode, args.k, args.current_only)
        answer, usage = ask(q["question"], chunks)
        tokens_in += usage.input_tokens
        tokens_out += usage.output_tokens
        bad = verify_citations(answer, chunks)
        ok, why = check(q, answer, bad, chunks)
        passed += ok
        print(f"\n{q['id']:>2} {'✓' if ok else '✗'} [{q['type']}] {q['question']}")
        print(f"   {answer.answer}")
        for c in answer.citations:
            mark = "✗" if c in bad else "✓"
            print(f"   {mark} {c.chunk_id}: \"{c.quote}\"")
        if why:
            print(f"   → {why}")

    cost = tokens_in / 1e6 * PRICE_IN + tokens_out / 1e6 * PRICE_OUT
    print(f"\npassed {passed}/{len(golden)} · {tokens_in:,} input + {tokens_out:,} output tokens · ${cost:.4f}")


if __name__ == "__main__":
    main()
