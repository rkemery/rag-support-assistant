"""Load the vendored Tallowbrook corpus, questions and facts.

Paths default to `data/tallowbrook/` in the repo. `RAG_DATA_DIR` overrides it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", REPO_ROOT / "data" / "tallowbrook"))

Split = Literal["dev", "test"]
SPLITS: tuple[Split, ...] = ("dev", "test")


@dataclass(frozen=True)
class Article:
    article_id: str
    title: str
    section: str
    plans: tuple[str, ...]
    effective_date: str
    version: int
    supersedes: str | None
    body: str


@dataclass(frozen=True)
class Question:
    question_id: str
    split: Split
    question: str
    gold_article_ids: tuple[str, ...]
    reference_answer: str
    answerable: bool
    unanswerable_type: str | None
    plan: str | None
    difficulty: str

    @property
    def expected_behavior(self) -> Literal["answer", "decline", "correct_premise"]:
        """What a good assistant does: answer, decline, or correct a false premise.

        False-premise questions count as unanswerable in the dataset, but the
        right move is an answer that corrects the premise, not a refusal.
        """
        if self.answerable:
            return "answer"
        if self.unanswerable_type == "false_premise":
            return "correct_premise"
        return "decline"

    @property
    def has_gold(self) -> bool:
        return bool(self.gold_article_ids)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
    return rows


def load_articles(data_dir: Path = DATA_DIR) -> list[Article]:
    rows = _read_jsonl(data_dir / "articles.jsonl")
    articles = [
        Article(
            article_id=r["article_id"],
            title=r["title"],
            section=r["section"],
            plans=tuple(r["plans"]),
            effective_date=r["effective_date"],
            version=int(r["version"]),
            supersedes=r["supersedes"],
            body=r["body"],
        )
        for r in rows
    ]
    ids = [a.article_id for a in articles]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate article_id in the corpus")
    return articles


def load_questions(split: Split | None = None, data_dir: Path = DATA_DIR) -> list[Question]:
    """Questions for one split, or dev then test when `split` is None."""
    splits = SPLITS if split is None else (split,)
    questions = []
    for name in splits:
        for r in _read_jsonl(data_dir / f"questions_{name}.jsonl"):
            if r["split"] != name:
                raise ValueError(
                    f"{r['question_id']} is in questions_{name}.jsonl but split={r['split']}"
                )
            questions.append(
                Question(
                    question_id=r["question_id"],
                    split=r["split"],
                    question=r["question"],
                    gold_article_ids=tuple(r["gold_article_ids"]),
                    reference_answer=r["reference_answer"],
                    answerable=bool(r["answerable"]),
                    unanswerable_type=r["unanswerable_type"],
                    plan=r["plan"],
                    difficulty=r["difficulty"],
                )
            )
    return questions


def load_facts(data_dir: Path = DATA_DIR) -> dict[str, Any]:
    with (data_dir / "policies.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def superseded_ids(articles: Iterable[Article]) -> set[str]:
    """Articles that a newer version points back to."""
    return {a.supersedes for a in articles if a.supersedes}


def cluster_key(question: Question) -> str:
    """Cluster for CIs: the question's first gold article (gold lists are sorted by ID).

    Questions that cite the same leading article share source material, so their
    outcomes are correlated. Questions with no gold article (out of scope, near
    miss) are their own cluster.
    """
    if question.gold_article_ids:
        return min(question.gold_article_ids)
    return f"none:{question.question_id}"


def retrieval_questions(questions: Sequence[Question]) -> list[Question]:
    """Questions that can be scored for retrieval: those with at least one gold article."""
    return [q for q in questions if q.has_gold]
