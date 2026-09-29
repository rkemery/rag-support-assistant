"""Stages of the evaluation that call models, plus the offline context export they need.

Every stage takes a `ClientStack` (live or replay) and writes harness JSONL
records under results/. A replay reads the committed cache/ and sends nothing.

Arms: the three retrieval configs chosen on dev for generation, plus
`full-context` (the whole corpus in the prompt, no retrieval).
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llm_eval_harness import EvalRecord, ModelClient, ModelRequest, read_records, write_records
from llm_eval_harness.client import DEFAULT_PRICES, cost_usd

from rag_support_assistant import generation as gen
from rag_support_assistant.clients import item_level_errors
from rag_support_assistant.data import (
    REPO_ROOT,
    Article,
    Question,
    Split,
    cluster_key,
    load_articles,
    load_questions,
)
from rag_support_assistant.judging import (
    JudgeInput,
    evidence_ids,
    judge_items,
    make_judge,
    passages_block,
    reference_block,
)
from rag_support_assistant.validation import load_perturbation_set, load_ragtruth_subset

RESULTS = REPO_ROOT / "results"
FULL_CONTEXT = "full-context"


def contexts_path(slug: str, results: Path = RESULTS) -> Path:
    return results / "contexts" / f"{slug}.jsonl"


def write_contexts(slugs: Sequence[str], results: Path = RESULTS) -> dict[str, int]:
    """Run each generation config's retrieval on every question and freeze the top chunks.

    Offline (CPU). Freezing the contexts means a replay sends byte-identical
    requests even on a machine where float rounding would reorder a tie.
    """
    from rag_support_assistant.grid import GridContext, load_selection
    from rag_support_assistant.retrieval import RetrievalConfig

    selection = load_selection(results / "retrieval")
    ctx = GridContext(load_articles(), cache_dir=REPO_ROOT / ".cache" / "embeddings")
    questions = load_questions()
    counts = {}
    for slug in slugs:
        config = RetrievalConfig(**selection["configs"][slug]["config"])
        retriever = ctx.retriever(config)
        rows = []
        for q in questions:
            result = retriever.retrieve(q.question_id, q.question)
            chunks = [
                asdict(
                    gen.ContextChunk(
                        chunk_id=item.node.node_id,
                        article_id=item.node.metadata["article_id"],
                        title=item.node.metadata["title"],
                        effective_date=item.node.metadata["effective_date"],
                        text=item.node.get_content(),
                    )
                )
                for item in result.chunks[: gen.CONTEXT_CHUNKS]
            ]
            rows.append(
                {
                    "question_id": q.question_id,
                    "retrieval_ms": round(result.latency_ms, 3),
                    "chunks": chunks,
                }
            )
        path = contexts_path(slug, results)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[slug] = len(rows)
    return counts


def load_contexts(slug: str, results: Path = RESULTS) -> dict[str, dict[str, Any]]:
    with contexts_path(slug, results).open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {row["question_id"]: row for row in rows}


def generation_arms(results: Path = RESULTS) -> list[str]:
    selection = json.loads((results / "retrieval" / "selection.json").read_text(encoding="utf-8"))
    return [*selection["generation_configs"], FULL_CONTEXT]


def generation_path(arm: str, split: Split, results: Path = RESULTS) -> Path:
    return results / "generation" / split / f"{arm}.jsonl"


def _answer_record(
    arm: str,
    split: Split,
    question: Question,
    request: ModelRequest,
    stack_client: ModelClient,
    context_ids: Sequence[str] | None,
    retrieval_ms: float,
) -> EvalRecord:
    start = time.perf_counter()
    try:
        response = stack_client.complete(request)
    except item_level_errors() as exc:
        return EvalRecord(
            run_id=f"generate/{split}/{arm}",
            item_id=question.question_id,
            config=arm,
            model=gen.ANSWER_MODEL,
            scores={},
            cluster=cluster_key(question),
            error=f"{type(exc).__name__}: {exc}",
            meta={"expected": question.expected_behavior},
        )
    wall_ms = (time.perf_counter() - start) * 1000.0
    price = DEFAULT_PRICES[gen.ANSWER_MODEL]
    base: dict[str, Any] = {
        "run_id": f"generate/{split}/{arm}",
        "item_id": question.question_id,
        "config": arm,
        "model": gen.ANSWER_MODEL,
        "cluster": cluster_key(question),
        "tokens_in": response.input_tokens,
        "tokens_out": response.output_tokens,
        "reasoning_tokens": response.reasoning_tokens,
        "cost_usd": cost_usd(price, response),
        "latency_ms": retrieval_ms + response.latency_ms,
    }
    meta: dict[str, Any] = {
        "expected": question.expected_behavior,
        "retrieval_ms": round(retrieval_ms, 3),
        "model_ms": round(response.latency_ms, 3),
        "cached_input_tokens": response.cached_input_tokens,
        "from_cache": response.from_cache,
        "wall_ms": round(wall_ms, 3),
        "finish_reason": response.finish_reason,
    }
    try:
        answer = gen.parse_answer(response.text)
    except gen.AnswerParseError as exc:
        meta["raw"] = response.text[:2000]
        return EvalRecord(**base, scores={}, error=f"AnswerParseError: {exc}", meta=meta)
    scores: dict[str, bool] = {"abstained": answer.abstain}
    if not answer.abstain:
        cited = set(answer.citations)
        if context_ids is not None:
            scores["citations_in_context"] = bool(cited) and cited <= set(context_ids)
        if question.gold_article_ids:
            scores["cites_gold"] = bool(cited & set(question.gold_article_ids))
    meta |= {"answer": answer.answer, "citations": list(answer.citations)}
    if context_ids is not None:
        meta["context_articles"] = list(dict.fromkeys(context_ids))
    return EvalRecord(**base, scores=scores, meta=meta)


def run_generation(
    stack_client: ModelClient, arm: str, split: Split, results: Path = RESULTS
) -> Path:
    questions = load_questions(split)
    records = []
    if arm == FULL_CONTEXT:
        corpus = gen.format_corpus(load_articles())
        for q in questions:
            request = gen.full_context_request(q.question, corpus)
            records.append(_answer_record(arm, split, q, request, stack_client, None, 0.0))
    else:
        contexts = load_contexts(arm, results)
        for q in questions:
            row = contexts[q.question_id]
            chunks = [gen.ContextChunk(**c) for c in row["chunks"]]
            request = gen.rag_request(q.question, chunks)
            ids = [c.article_id for c in chunks]
            records.append(
                _answer_record(arm, split, q, request, stack_client, ids, row["retrieval_ms"])
            )
    path = generation_path(arm, split, results)
    write_records(path, records)
    return path


def answer_judge_inputs(
    records: Sequence[EvalRecord], questions: dict[str, Question], articles: dict[str, Article]
) -> list[JudgeInput]:
    """Judge inputs for answered records (not abstained, not errored).

    The evidence is the gold articles plus the cited ones.
    """
    inputs = []
    for r in records:
        if r.error is not None or r.scores.get("abstained", True):
            continue
        q = questions[r.item_id]
        ids = evidence_ids(q.gold_article_ids, r.meta.get("citations", []), articles)
        inputs.append(
            JudgeInput(
                item_id=r.item_id,
                question=q.question,
                answer=r.meta["answer"],
                reference=reference_block(q.reference_answer, ids, articles),
                cluster=r.cluster,
                meta={"arm": r.config, "expected": q.expected_behavior, "evidence": ids},
            )
        )
    return inputs


def judge_path(judge_key: str, target: str, results: Path = RESULTS) -> Path:
    return results / "judge" / judge_key / f"{target}.jsonl"


def run_answer_judging(
    stack_client: ModelClient, judge_key: str, arm: str, split: Split, results: Path = RESULTS
) -> Path:
    records = read_records(generation_path(arm, split, results))
    questions = {q.question_id: q for q in load_questions(split)}
    articles = {a.article_id: a for a in load_articles()}
    judge = make_judge(stack_client, judge_key)
    inputs = answer_judge_inputs(records, questions, articles)
    judged = judge_items(
        judge, inputs, run_id=f"judge/{judge_key}/answers/{split}/{arm}", config=arm
    )
    path = judge_path(judge_key, f"answers/{split}/{arm}", results)
    write_records(path, judged)
    return path


def perturbation_inputs(split: Split) -> list[JudgeInput]:
    pset = load_perturbation_set()
    articles = {a.article_id: a for a in load_articles()}
    questions = {q.question_id: q for q in load_questions()}
    return [
        JudgeInput(
            item_id=item.item_id,
            question=item.question,
            answer=item.answer,
            reference=reference_block(item.reference_answer, list(item.gold_article_ids), articles),
            cluster=cluster_key(questions[item.question_id]),
            meta={"type": item.type, "question_id": item.question_id},
        )
        for item in pset.by_split(split)
    ]


def run_perturbation_judging(
    stack_client: ModelClient, judge_key: str, split: Split, results: Path = RESULTS
) -> Path:
    judge = make_judge(stack_client, judge_key)
    judged = judge_items(
        judge,
        perturbation_inputs(split),
        run_id=f"judge/{judge_key}/perturbations/{split}",
        config="perturbations",
        progress_every=50,
    )
    path = judge_path(judge_key, f"perturbations/{split}", results)
    write_records(path, judged)
    return path


def ragtruth_inputs() -> list[JudgeInput]:
    return [
        JudgeInput(
            item_id=item.item_id,
            question=item.question,
            answer=item.response,
            reference=passages_block(item.passages),
            cluster=None,
            meta={"source_model": item.model, "source_id": item.source_id},
        )
        for item in load_ragtruth_subset()
    ]


def run_ragtruth_judging(
    stack_client: ModelClient, judge_key: str, results: Path = RESULTS
) -> Path:
    judge = make_judge(stack_client, judge_key)
    judged = judge_items(
        judge,
        ragtruth_inputs(),
        run_id=f"judge/{judge_key}/ragtruth",
        config="ragtruth",
        progress_every=50,
    )
    path = judge_path(judge_key, "ragtruth", results)
    write_records(path, judged)
    return path
