"""The model-calling pipeline end to end, with fake models behind the real client stack."""

from __future__ import annotations

import json

import pytest
from llm_eval_harness import FakeClient, ModelRequest, read_records

from rag_support_assistant import analysis, pipeline, readme
from rag_support_assistant.clients import wrap_for_tests
from rag_support_assistant.data import load_questions

ARM = "fixed-title-dense-bge-small"


def fake_model(request: ModelRequest) -> str:
    """Answers for gpt-6-luna, verdicts for the judges. Deterministic per prompt."""
    text = json.dumps(request.input)
    if request.model == "gpt-6-luna":
        abstain = "bitcoin" in text.lower() or len(text) % 7 == 0
        return json.dumps(
            {
                "answer": "It costs $5.00.",
                "citations": [] if abstain else ["plan-plus"],
                "abstain": abstain,
            }
        )
    grounded = len(text) % 3 != 0
    return json.dumps(
        {
            "correct": {"pass": grounded, "reason": "r"},
            "grounded": {"pass": grounded, "reason": "r"},
        }
    )


@pytest.fixture
def results(tmp_path):
    (tmp_path / "retrieval").mkdir()
    selection = {
        "steps": [],
        "alpha_sweep": {},
        "configs": {ARM: {"name": "fixed-title / bge-small", "config": {}, "role": "grid"}},
        "generation_configs": [ARM],
        "compute": {},
    }
    (tmp_path / "retrieval" / "selection.json").write_text(json.dumps(selection))
    chunk = {
        "chunk_id": "c1",
        "article_id": "plan-plus",
        "title": "The Plus plan",
        "effective_date": "2026-06-01",
        "text": "Plus costs $5.00 a month.",
    }
    path = pipeline.contexts_path(ARM, tmp_path)
    path.parent.mkdir(parents=True)
    with path.open("w") as fh:
        for q in load_questions():
            fh.write(
                json.dumps({"question_id": q.question_id, "retrieval_ms": 3.0, "chunks": [chunk]})
                + "\n"
            )
    return tmp_path


def test_generation_and_judging_write_harness_records(results, tmp_path):
    stack = wrap_for_tests(FakeClient(fake_model), tmp_path / "cache", cap_usd=5.0)
    for arm in pipeline.generation_arms(results):
        pipeline.run_generation(stack.client, arm, "test", results)
        for key in ("llama", "gpt5mini"):
            pipeline.run_answer_judging(stack.client, key, arm, "test", results)
    gen = read_records(pipeline.generation_path(ARM, "test", results))
    assert len(gen) == 150
    assert all(r.model == "gpt-6-luna" and r.cost_usd > 0 and r.cluster for r in gen)
    answered = [r for r in gen if not r.scores["abstained"]]
    assert all(r.scores["citations_in_context"] for r in answered)
    judged = read_records(pipeline.judge_path("llama", f"answers/test/{ARM}", results))
    assert {r.item_id for r in judged} == {r.item_id for r in answered}
    full = read_records(pipeline.generation_path(pipeline.FULL_CONTEXT, "test", results))
    assert all("citations_in_context" not in r.scores for r in full)
    assert stack.cap.spent_usd > 0

    summary = analysis.summarize_generation(ARM, gen, judged)
    assert sum(sum(row.values()) for row in summary.table.values()) == 150
    assert {"false_refusal", "hallucination", "accuracy"} <= set(summary.metrics)

    names = {ARM: "fixed-title / bge-small", pipeline.FULL_CONTEXT: "Full context"}
    text = readme.generation_section(results, [ARM, pipeline.FULL_CONTEXT], names)
    assert readme.PENDING not in text.split("**Abstention")[0]

    # A second pass is served entirely from the cache and costs nothing more.
    spent = stack.cap.spent_usd
    pipeline.run_generation(stack.client, ARM, "test", results)
    assert stack.cap.spent_usd == spent


def test_generation_section_shows_pending_without_results(tmp_path):
    text = readme.generation_section(tmp_path, ["x", pipeline.FULL_CONTEXT], {})
    assert text.count(readme.PENDING) >= 14
