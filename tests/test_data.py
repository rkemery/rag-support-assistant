from __future__ import annotations

import sys

from rag_support_assistant.data import REPO_ROOT, cluster_key, load_questions, retrieval_questions

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from sync_data import verify


def test_vendored_files_match_manifest():
    assert verify() == [
        "articles.jsonl",
        "policies.yaml",
        "questions_dev.jsonl",
        "questions_test.jsonl",
    ]


def test_split_sizes_and_unanswerable_counts(questions, articles):
    assert len(articles) == 151
    assert len(load_questions("dev")) == 50
    assert len(load_questions("test")) == 150
    assert sum(not q.answerable for q in questions) == 40
    behaviors = [q.expected_behavior for q in load_questions("test")]
    assert behaviors.count("answer") == 120
    assert behaviors.count("decline") == 20
    assert behaviors.count("correct_premise") == 10


def test_retrieval_questions_are_those_with_gold(questions):
    scored = retrieval_questions(questions)
    assert all(q.gold_article_ids for q in scored)
    # answerable plus false premise: 160 + 13
    assert len(scored) == 173


def test_cluster_is_first_gold_article_or_own_cluster(questions):
    for q in questions:
        key = cluster_key(q)
        if q.gold_article_ids:
            assert key == min(q.gold_article_ids)
        else:
            assert key == f"none:{q.question_id}"
