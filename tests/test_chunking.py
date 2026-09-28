from __future__ import annotations

import re

from rag_support_assistant.chunking import add_contexts, chunk_articles, embed_text, to_chunk


def test_chunk_ids_are_stable_uuids(articles):
    a = chunk_articles(articles[:5], "header")
    b = chunk_articles(articles[:5], "header")
    assert [n.node_id for n in a] == [n.node_id for n in b]
    assert all(re.fullmatch(r"[0-9a-f-]{36}", n.node_id) for n in a)


def test_title_is_in_embedding_text_only_when_asked(articles):
    art = articles[0]
    fixed = chunk_articles([art], "fixed")
    titled = chunk_articles([art], "fixed-title")
    header = chunk_articles([art], "header")
    assert all(not embed_text(n).startswith("title:") for n in fixed)
    assert all(embed_text(n).startswith(f"title: {art.title}") for n in titled + header)
    # The text given to the answer model never carries metadata.
    assert all(to_chunk(n).text == n.get_content() for n in header)


def test_header_chunks_follow_sections(articles):
    art = next(a for a in articles if a.article_id == "atm-surcharges")
    nodes = chunk_articles([art], "header")
    assert len(nodes) == art.body.count("\n## ") + 1


def test_add_contexts_prepends_context_for_encoders_only(articles):
    nodes = chunk_articles(articles[:1], "fixed-title")
    contexts = {n.node_id: f"context {i}" for i, n in enumerate(nodes)}
    out = add_contexts(nodes, contexts)
    assert embed_text(out[0]).startswith("context: context 0\ntitle:")
    assert out[0].get_content() == nodes[0].get_content()
    assert "context" not in nodes[0].metadata  # originals untouched
