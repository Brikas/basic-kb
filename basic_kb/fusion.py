"""Reciprocal rank fusion: merge ranked lists by position, ignoring their raw scores.

Cosine similarity and BM25 live on unrelated scales (BM25 is unbounded and query-
dependent), and cosine scores of two different queries are not comparable either. Rank
is. A key's fused score is the sum, over every list it appears in, of `weight / (k + rank)`
with ranks counted from 1 (Cormack, Clarke & Büttcher 2009).

`k` sets how fast credit decays down a list. The paper's k=60 came from fusing dozens of
systems over queries with many relevant documents; with a few lists and one or two right
answers per query, a smaller k favours agreement near the top (Qdrant's tuning study found
2-5 best there). basic-kb defaults to 20 and the config can move it.

Weights multiply a list's contribution, as in Elasticsearch, qmd and LangChain; a weight
of 0 or less drops the list. Oracles: qdrant/qdrant-client `hybrid/fusion.py` and its
tests, tobi/qmd `reciprocalRankFusion` (see SPEC.md).
"""
from __future__ import annotations

from typing import Hashable, Optional, Sequence

DEFAULT_RRF_K = 20.0

# Which ranked lists a search builds: both retrievers, or one of them.
RETRIEVAL_MODES = ("hybrid", "vector", "keyword")
DEFAULT_RETRIEVAL = "hybrid"


def rrf(lists: Sequence[Sequence[Hashable]], k: float = DEFAULT_RRF_K,
        weights: Optional[Sequence[float]] = None) -> list[tuple[Hashable, float]]:
    """Fuse ranked lists of keys into [(key, score)], best first.

    A key repeated inside one list counts once, at its best rank. Ties break by the best
    rank the key reached in any list, then by first appearance, so the order is
    deterministic. An empty list contributes nothing.
    """
    if k < 0:
        raise ValueError(f"RRF k must be >= 0, got {k}")
    if weights is not None and len(weights) != len(lists):
        raise ValueError(f"RRF got {len(weights)} weights for {len(lists)} lists")
    scores: dict[Hashable, float] = {}
    best_rank: dict[Hashable, int] = {}
    first_seen: dict[Hashable, int] = {}
    for i, ranked in enumerate(lists):
        weight = 1.0 if weights is None else float(weights[i])
        if weight <= 0:
            continue
        seen: set = set()
        for rank, key in enumerate(ranked, start=1):
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            best_rank[key] = min(best_rank.get(key, rank), rank)
            first_seen.setdefault(key, len(first_seen))
    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], first_seen[kv[0]]))
