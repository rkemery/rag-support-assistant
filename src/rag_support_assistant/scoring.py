"""Article-level retrieval metrics with ranx, one score per question.

Gold labels are articles, so each question's ranked chunks are collapsed to a
ranked article list before scoring. ranx needs scores, not ranks, so the
article at rank r gets score 1 / r, which keeps the order and has no ties.
Only questions with at least one gold article are scored (answerable and
false-premise questions).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

from numba.core.errors import NumbaTypeSafetyWarning
from ranx import Qrels, Run, evaluate

from rag_support_assistant.data import Question

METRICS: tuple[str, ...] = ("ndcg@10", "mrr@10", "recall@5", "recall@10", "hit_rate@5")
PRIMARY_METRIC = "ndcg@10"


def per_query_scores(
    questions: Sequence[Question],
    rankings: Mapping[str, Sequence[str]],
    metrics: Sequence[str] = METRICS,
) -> dict[str, dict[str, float]]:
    """{question_id: {metric: value}} for every question that has gold articles."""
    scored = [q for q in questions if q.has_gold]
    if not scored:
        raise ValueError("no questions with gold articles to score")
    missing = [q.question_id for q in scored if q.question_id not in rankings]
    if missing:
        raise ValueError(f"no ranking for {missing[:5]}")
    qrels = Qrels({q.question_id: dict.fromkeys(q.gold_article_ids, 1) for q in scored})
    run_dict = {}
    for q in scored:
        ranking = list(dict.fromkeys(rankings[q.question_id]))  # first occurrence wins
        if ranking:  # an empty ranking is added back by make_comparable and scores 0
            run_dict[q.question_id] = {aid: 1.0 / r for r, aid in enumerate(ranking, start=1)}
    run = Run(run_dict)
    with warnings.catch_warnings():
        # ranx's nDCG kernel casts uint64 to int64 internally. Harmless for binary relevance.
        warnings.simplefilter("ignore", NumbaTypeSafetyWarning)
        evaluate(qrels, run, list(metrics), make_comparable=True)
    return {
        q.question_id: {m: float(run.scores[m][q.question_id]) for m in metrics} for q in scored
    }
