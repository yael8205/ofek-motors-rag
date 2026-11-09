"""Measures retrieval on the golden questions. Free: no Claude call.
Prints, for each question, whether the answer is in the top-k chunks and in which place.

  python eval_retrieval.py                                     # the baseline (v0)
  python eval_retrieval.py --chunker headings --mode hybrid --current-only
  python eval_retrieval.py --show 7                            # print what question 7 retrieved
  python eval_retrieval.py --out results/v0.json               # also save the results
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from rag.chunking import chunk_by_headings, chunk_fixed
from rag.embed import Embedder
from rag.loader import load_docs
from rag.search import Index, search

CHUNKERS = {"fixed": chunk_fixed, "headings": chunk_by_headings}


def needles(q: dict) -> list[str]:
    need = q.get("answer_contains", [])
    return [need] if isinstance(need, str) else need


def is_hit(q: dict, chunks) -> bool:
    """The answer text is in the top-k chunks, and no outdated text came along with it."""
    text = "\n".join(c.text for c in chunks)
    if not all(n in text for n in needles(q)):
        return False
    bad = q.get("forbidden_contains")
    return not (bad and bad in text)


def reciprocal_rank(q: dict, chunks) -> float:
    """1 / the position of the first chunk that holds the answer (1 = it came first), or 0."""
    for rank, c in enumerate(chunks, start=1):
        if needles(q)[0] in c.text:
            return 1 / rank
    return 0.0


def build_index(chunker: str) -> Index:
    chunks = [c for doc in load_docs() for c in CHUNKERS[chunker](doc)]
    print(f"{len(chunks)} chunks ({chunker})")
    return Index(chunks, Embedder())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunker", choices=CHUNKERS, default="fixed")
    ap.add_argument("--mode", choices=["vector", "hybrid"], default="vector")
    ap.add_argument("--current-only", action="store_true")
    ap.add_argument("--k", type=int, default=3)  # small corpus, so a small k shows the differences
    ap.add_argument("--show", type=int, help="print the chunks retrieved for this question id")
    ap.add_argument("--out", help="save the results to this JSON file")
    args = ap.parse_args()

    golden = json.loads(Path("golden.json").read_text(encoding="utf-8"))
    index = build_index(args.chunker)
    by_type, rows = defaultdict(lambda: [0, 0, 0.0]), []  # type -> [hits, questions, sum of reciprocal ranks]
    for q in golden:
        found = search(index, q["question"], args.mode, args.k, args.current_only)
        if not needles(q):  # questions with no answer in the data are checked in eval_answers.py
            print(f"  {q['id']:>2}  —        [{q['type']}] {q['question']}")
            continue
        hit, rr = is_hit(q, found), reciprocal_rank(q, found)
        by_type[q["type"]][0] += hit
        by_type[q["type"]][1] += 1
        by_type[q["type"]][2] += rr
        rows.append({"id": q["id"], "type": q["type"], "hit": hit, "rr": round(rr, 3), "chunks": [c.id for c in found]})
        place = f"#{round(1 / rr)}" if rr else "-"
        print(f"  {q['id']:>2}  {'✓' if hit else '✗'}  {place:>3}   [{q['type']}] {q['question']}")
        if args.show == q["id"]:
            for c in found:
                print(f"\n      --- {c.id} ---\n      " + c.text.replace("\n", "\n      "))
            print()

    print(f"\n  {'type':<12} recall@{args.k}   MRR")
    for t, (h, n, rr) in by_type.items():
        print(f"  {t:<12} {h}/{n:<8} {rr / n:.2f}")
    total_h = sum(v[0] for v in by_type.values())
    total_n = sum(v[1] for v in by_type.values())
    total_rr = sum(v[2] for v in by_type.values())
    print(f"  {'total':<12} {total_h}/{total_n:<8} {total_rr / total_n:.2f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"config": vars(args), "results": rows}, ensure_ascii=False, indent=2),
                                  encoding="utf-8")


if __name__ == "__main__":
    main()
