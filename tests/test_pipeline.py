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
    assert readme.PENDING not in text.split("Abstention table")[0]

    # A second pass is served entirely from the cache and costs nothing more.
    spent = stack.cap.spent_usd
    pipeline.run_generation(stack.client, ARM, "test", results)
    assert stack.cap.spent_usd == spent


def test_generation_section_shows_pending_without_results(tmp_path):
    text = readme.generation_section(tmp_path, ["x", pipeline.FULL_CONTEXT], {})
    assert text.count(readme.PENDING) >= 14


def test_a_refused_request_is_recorded_and_the_run_goes_on(results, tmp_path):
    import httpx
    import openai

    first = load_questions("test")[0].question_id

    def refusing(request: ModelRequest) -> str:
        if request.model == "gpt-6-luna" and first in json.dumps(request.input):
            raise AssertionError("question ids never reach the prompt")
        if request.model == "gpt-6-luna" and "Customer question" in json.dumps(request.input):
            text = json.dumps(request.input)
            if load_questions("test")[0].question in text:
                response = httpx.Response(
                    400, request=httpx.Request("POST", "https://example.test")
                )
                raise openai.BadRequestError("content filtered", response=response, body=None)
        return fake_model(request)

    stack = wrap_for_tests(FakeClient(refusing), tmp_path / "cache", cap_usd=5.0)
    pipeline.run_generation(stack.client, ARM, "test", results)
    records = {r.item_id: r for r in read_records(pipeline.generation_path(ARM, "test", results))}
    assert records[first].error.startswith("BadRequestError")
    assert sum(r.error is not None for r in records.values()) == 1
    summary = analysis.summarize_generation(ARM, list(records.values()), None)
    assert sum(row["error"] for row in summary.table.values()) == 1


def test_flag_counts_and_judge_disagreement_render_once_judged(results, tmp_path):
    from llm_eval_harness import EvalRecord, write_records

    from rag_support_assistant.judging import make_judge
    from rag_support_assistant.validation import load_perturbation_set

    stack = wrap_for_tests(FakeClient(fake_model), tmp_path / "cache", cap_usd=5.0)
    pipeline.run_generation(stack.client, ARM, "test", results)
    for key in ("llama", "gpt5mini"):
        pipeline.run_answer_judging(stack.client, key, ARM, "test", results)
    # A calibration run from the same judge that is right on 9 of every 10 items.
    pset = load_perturbation_set()
    fingerprint = make_judge(None, "llama").fingerprint
    calibration = []
    for n, item in enumerate(pset.by_split("test")):
        verdict = item.labels["grounded"] if n % 10 else not item.labels["grounded"]
        calibration.append(
            EvalRecord(
                run_id="judge/llama/perturbations/test",
                item_id=item.item_id,
                config="perturbations",
                model="Llama-3.3-70B-Instruct",
                scores={"grounded": verdict},
                meta={"judge_fingerprint": fingerprint},
            )
        )
    write_records(pipeline.judge_path("llama", "perturbations/test", results), calibration)
    text = readme._judge_rows(results, [ARM], {ARM: "arm"})
    row = next(line for line in text if line.startswith("| arm |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1] != readme.PENDING  # Llama flags
    assert cells[3] != readme.PENDING  # disagreement between the two judges was computed
    assert " / " in cells[3]
