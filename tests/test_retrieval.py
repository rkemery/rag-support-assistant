from __future__ import annotations

import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryResult

from rag_support_assistant.fusion import convex_fusion, rrf_fusion
from rag_support_assistant.rerank import CrossEncoderRerank
from rag_support_assistant.retrieval import RetrievalConfig, rank_articles


def _result(ids: list[str], sims: list[float]) -> VectorStoreQueryResult:
    nodes = [TextNode(id_=i, text=i) for i in ids]
    return VectorStoreQueryResult(nodes=nodes, similarities=sims, ids=ids)


def test_rrf_sums_reciprocal_ranks():
    dense = _result(["a", "b", "c"], [0.9, 0.8, 0.1])
    sparse = _result(["c", "a"], [12.0, 3.0])
    fused = rrf_fusion(dense, sparse, top_k=3)
    expected = {"a": 1 / 61 + 1 / 62, "b": 1 / 62, "c": 1 / 63 + 1 / 61}
    assert fused.ids == sorted(expected, key=lambda k: -expected[k])
    assert fused.similarities == pytest.approx(sorted(expected.values(), reverse=True))


def test_convex_fusion_extremes_follow_one_side():
    dense = _result(["a", "b", "c"], [0.9, 0.5, 0.1])
    sparse = _result(["c", "b", "a"], [9.0, 5.0, 1.0])
    assert convex_fusion(dense, sparse, alpha=1.0, top_k=3).ids == ["a", "b", "c"]
    assert convex_fusion(dense, sparse, alpha=0.0, top_k=3).ids == ["c", "b", "a"]


def test_config_validation():
    with pytest.raises(ValueError, match="fusion"):
        RetrievalConfig("header", "bge-small", "hybrid")
    with pytest.raises(ValueError, match="alpha"):
        RetrievalConfig("header", "bge-small", "hybrid", fusion="rrf", alpha=0.5)
    cfg = RetrievalConfig("header", "bge-small", "hybrid", fusion="convex", alpha=0.7)
    assert cfg.slug == "header-hybrid-bge-small-convex0.7"


def test_rank_articles_keeps_first_appearance():
    def node(i: str, art: str) -> NodeWithScore:
        return NodeWithScore(node=TextNode(id_=i, text=i, metadata={"article_id": art}), score=1.0)

    assert rank_articles([node("1", "x"), node("2", "y"), node("3", "x")]) == ["x", "y"]


def test_reranker_reorders_only_the_head():
    nodes = [NodeWithScore(node=TextNode(id_=str(i), text=f"t{i}"), score=1.0) for i in range(5)]
    rerank = CrossEncoderRerank(
        lambda q, ps: [float(len(ps) - i) * -1 for i in range(len(ps))], top_n=3
    )
    out = rerank.postprocess_nodes(nodes, query_bundle=QueryBundle("q"))
    assert [n.node.node_id for n in out] == ["2", "1", "0", "3", "4"]


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid-convex", "hybrid-rrf"])
def test_every_mode_retrieves_from_qdrant(fake_ctx, questions, mode):
    kwargs: dict = {"mode": mode.split("-")[0]}
    if mode == "hybrid-convex":
        kwargs |= {"fusion": "convex", "alpha": 0.5}
    elif mode == "hybrid-rrf":
        kwargs |= {"fusion": "rrf"}
    config = RetrievalConfig("header", "bge-small", **kwargs)
    result = fake_ctx.retriever(config).retrieve("q", questions[0].question)
    assert len(result.chunks) == 50
    assert len(result.articles) == len(set(result.articles))
    assert result.latency_ms > 0


def test_bm25_mode_matches_the_encoder_scores(fake_ctx):
    config = RetrievalConfig("header", "bge-small", "bm25")
    retriever = fake_ctx.retriever(config)
    index = fake_ctx.index("header", "bge-small")
    query = "out-of-network ATM fee on Plus"
    result = retriever.retrieve("q", query)
    top = result.chunks[:5]
    from rag_support_assistant.chunking import embed_text

    for item in top:
        assert item.score == pytest.approx(index.bm25.score(query, embed_text(item.node)), rel=1e-5)
