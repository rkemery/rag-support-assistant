from __future__ import annotations

import math

import pytest

from rag_support_assistant.data import Question
from rag_support_assistant.scoring import per_query_scores


def _q(qid: str, gold: tuple[str, ...]) -> Question:
    return Question(qid, "test", "q", gold, "ref", True, None, None, "easy")


def test_article_metrics_match_hand_computation():
    qs = [_q("q1", ("a", "c")), _q("q2", ("z",)), _q("q3", ())]
    rankings = {"q1": ["b", "a", "d", "c"], "q2": ["x", "y"], "q3": []}
    scores = per_query_scores(qs, rankings)
    assert set(scores) == {"q1", "q2"}  # no gold, not scored
    q1 = scores["q1"]
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3)
    assert q1["ndcg@10"] == pytest.approx(dcg / idcg)
    assert q1["mrr@10"] == pytest.approx(0.5)
    assert q1["recall@5"] == pytest.approx(1.0)
    assert q1["hit_rate@5"] == pytest.approx(1.0)
    assert scores["q2"] == dict.fromkeys(scores["q2"], 0.0)


def test_missing_ranking_raises():
    with pytest.raises(ValueError, match="no ranking"):
        per_query_scores([_q("q1", ("a",))], {})
