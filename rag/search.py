"""Search over the chunks: by meaning (vectors), by words (BM25), and the two combined."""
import re

import numpy as np
from rank_bm25 import BM25Okapi

from rag.chunking import Chunk
from rag.embed import Embedder


def tokenize(text: str) -> list[str]:
    # Keeps codes like "of-450h" and "e-26" as single tokens.
    return re.findall(r"[\w\-]+", text.lower())


class Index:
    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.vectors = embedder.embed([c.text for c in chunks], "passage")
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def vector_search(self, query: str, k: int = 50) -> list[int]:
        """Indexes of the k chunks closest in meaning to the query, best first."""
        q = self.embedder.embed([query], "query")[0]
        scores = self.vectors @ q
        return [int(i) for i in np.argsort(-scores)[:k]]

    def bm25_search(self, query: str, k: int = 50) -> list[int]:
        """Indexes of the k chunks with the best word match, best first."""
        scores = self.bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(-scores)[:k] if scores[i] > 0]


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion.

    `rankings` is a list of ranked lists of chunk indexes, best first.
    A chunk gets 1 / (k + rank) from every list it appears in (rank starts at 1),
    and the scores are summed. Return all chunk indexes, sorted by score, best first.
    """
    # Ranks are 1-based as stated in the docstring: first position = rank 1.
    scores: dict[int, float] = {}
    for ranked_list in rankings:
        for rank, chunk_idx in enumerate(ranked_list, start=1):  # rank starts at 1
            scores[chunk_idx] = scores.get(chunk_idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.__getitem__, reverse=True)


def is_current(chunk: Chunk) -> bool:
    """False for chunks from documents that were replaced by a newer version.

    Uses the 'status' field in the document front-matter metadata.
    Only warranty-2023.md carries status: superseded; all other docs are current
    (including docs with no status field, which default to current).
    """
    return chunk.meta.get("status") != "superseded"


def search(index: Index, query: str, mode: str = "vector", top_k: int = 5,
           current_only: bool = False) -> list[Chunk]:
    if mode == "vector":
        ranked = index.vector_search(query)
    elif mode == "hybrid":
        ranked = rrf([index.vector_search(query), index.bm25_search(query)])
    else:
        raise ValueError(f"unknown mode: {mode}")
    if current_only:
        ranked = [i for i in ranked if is_current(index.chunks[i])]
    return [index.chunks[i] for i in ranked[:top_k]]


def search_multi(index: Index, queries: list[str], mode: str = "hybrid",
                 top_k_each: int = 3, current_only: bool = False) -> list[Chunk]:
    """Query decomposition: run one search per sub-query and merge results with RRF.

    Useful for multi-source questions whose answer is spread across several documents,
    e.g. "which EV has the longer range?" can be split into one query per model.
    Each sub-query retrieves top_k_each chunks independently; the ranked lists are
    then fused with RRF so chunks that rank well across multiple sub-queries rise to
    the top. Duplicates are removed (a chunk is returned at most once).

    Example:
        results = search_multi(index,
            ["טווח נסיעה אופק E5 חשמלי", "טווח נסיעה אופק E7 חשמלי"],
            top_k_each=3, current_only=True)
    """
    all_ranked: list[list[int]] = []
    for q in queries:
        if mode == "vector":
            ranked = index.vector_search(q, k=top_k_each * 2)
        elif mode == "hybrid":
            ranked = rrf([index.vector_search(q, k=50), index.bm25_search(q, k=50)])
        else:
            raise ValueError(f"unknown mode: {mode}")
        if current_only:
            ranked = [i for i in ranked if is_current(index.chunks[i])]
        all_ranked.append(ranked[:top_k_each * 2])

    fused = rrf(all_ranked)
    # Return top_k_each results per sub-query worth, deduplicated
    seen: set[int] = set()
    result: list[Chunk] = []
    for idx in fused:
        if idx not in seen:
            seen.add(idx)
            result.append(index.chunks[idx])
        if len(result) >= top_k_each * len(queries):
            break
    return result
