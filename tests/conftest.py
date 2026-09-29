"""Shared fixtures. Nothing here downloads a model or needs a key."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from rag_support_assistant.data import Article, Question, load_articles, load_questions
from rag_support_assistant.embeddings import HashingEmbedding
from rag_support_assistant.grid import GridContext


def overlap_reranker(key: str):
    """A stand-in cross-encoder: counts shared lowercase words."""

    def score(query: str, passages: Sequence[str]) -> list[float]:
        q = set(query.lower().split())
        return [float(len(q & set(p.lower().split()))) for p in passages]

    return score


def hashing_factory(key: str) -> HashingEmbedding:
    # Different dimensions per key so "different models" give different rankings.
    return HashingEmbedding(model_name=f"hash-{key}", dim=48 + 16 * (len(key) % 4))


@pytest.fixture(scope="session")
def articles() -> list[Article]:
    return load_articles()


@pytest.fixture(scope="session")
def questions() -> list[Question]:
    return load_questions()


@pytest.fixture
def fake_ctx(articles: list[Article]) -> GridContext:
    return GridContext(articles, embed_factory=hashing_factory, rerank_factory=overlap_reranker)
