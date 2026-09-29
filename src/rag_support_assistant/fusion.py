"""Hybrid fusion functions for `QdrantVectorStore(hybrid_fusion_fn=...)`.

Both are passed explicitly, so no default decides how dense and BM25 results
combine.

- `convex_fusion`: alpha * minmax(dense) + (1 - alpha) * minmax(bm25), a
  convex combination of min-max normalized scores. It is LlamaIndex's
  `relative_score_fusion`, re-exported here so the grid names it. A chunk
  missing from one list scores 0 there. Bruch et al. (2023, ACM TOIS,
  arXiv 2210.11934) found a convex combination tuned on a small labeled set
  beats RRF in and out of domain. alpha is tuned on dev.
- `rrf_fusion`: reciprocal rank fusion, sum of 1 / (k + rank) with k = 60
  (Cormack, Clarke and Buettcher, SIGIR 2009). Rank-based, so it has no scale
  to tune and ignores alpha.
"""

from __future__ import annotations

from typing import Any

from llama_index.core.vector_stores.types import VectorStoreQueryResult
from llama_index.vector_stores.qdrant.utils import relative_score_fusion

RRF_K = 60

convex_fusion = relative_score_fusion


def rrf_fusion(
    dense_result: VectorStoreQueryResult,
    sparse_result: VectorStoreQueryResult,
    alpha: float = 0.5,  # accepted for the interface, ignored by RRF
    top_k: int = 2,
    k: int = RRF_K,
    **_: Any,
) -> VectorStoreQueryResult:
    """Fuse by reciprocal rank. Ties break on the node ID so the order is deterministic."""
    scores: dict[str, float] = {}
    nodes: dict[str, Any] = {}
    for result in (dense_result, sparse_result):
        if not result.nodes:
            continue
        sims = result.similarities or [0.0] * len(result.nodes)
        ranked = sorted(zip(sims, result.nodes, strict=True), key=lambda x: -x[0])
        for rank, (_, node) in enumerate(ranked, start=1):
            scores[node.node_id] = scores.get(node.node_id, 0.0) + 1.0 / (k + rank)
            nodes.setdefault(node.node_id, node)
    order = sorted(scores, key=lambda node_id: (-scores[node_id], node_id))[:top_k]
    return VectorStoreQueryResult(
        nodes=[nodes[i] for i in order],
        similarities=[scores[i] for i in order],
        ids=order,
    )
