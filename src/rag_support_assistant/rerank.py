"""Cross-encoder reranking as a LlamaIndex node postprocessor.

The reranker rescores the first `top_n` chunks (10, per the plan) and puts
them back in its order. Chunks below the cut keep their first-stage order
after them, so recall at depths past 10 is unchanged and only the head moves.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from pydantic import PrivateAttr

from rag_support_assistant.models import RerankerSpec

ScoreFn = Callable[[str, Sequence[str]], Sequence[float]]

RERANK_DEPTH = 10


class CrossEncoderRerank(BaseNodePostprocessor):
    """Rerank the head of a result list with a scoring function (query, passages) -> scores."""

    top_n: int = RERANK_DEPTH
    model_key: str = "custom"
    _score_fn: ScoreFn = PrivateAttr()

    def __init__(self, score_fn: ScoreFn, top_n: int = RERANK_DEPTH, model_key: str = "custom"):
        super().__init__(top_n=top_n, model_key=model_key)
        self._score_fn = score_fn

    @classmethod
    def class_name(cls) -> str:
        return "CrossEncoderRerank"

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None
    ) -> list[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("reranking needs the query")
        head, tail = nodes[: self.top_n], nodes[self.top_n :]
        if not head:
            return nodes
        passages = [n.node.get_content(metadata_mode=MetadataMode.EMBED) for n in head]
        scores = list(self._score_fn(query_bundle.query_str, passages))
        if len(scores) != len(head):
            raise ValueError(f"reranker returned {len(scores)} scores for {len(head)} passages")
        # Stable sort: equal scores keep first-stage order.
        order = sorted(range(len(head)), key=lambda i: -scores[i])
        reranked = [NodeWithScore(node=head[i].node, score=float(scores[i])) for i in order]
        return reranked + tail


def load_cross_encoder(spec: RerankerSpec, batch_size: int = 16) -> ScoreFn:
    """A pinned sentence-transformers CrossEncoder as a score function. Downloads on first use."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(
        spec.hf_id, revision=spec.revision, max_length=spec.max_length, device="cpu"
    )

    def score(query: str, passages: Sequence[str]) -> list[float]:
        pairs: list[Any] = [(query, p) for p in passages]
        return [float(s) for s in model.predict(pairs, batch_size=batch_size)]

    return score
