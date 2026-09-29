from __future__ import annotations

import json

import pytest
from llm_eval_harness import FakeClient

from rag_support_assistant import judging


def _verdict(correct: bool, grounded: bool) -> str:
    return json.dumps(
        {"correct": {"pass": correct, "reason": "r"}, "grounded": {"pass": grounded, "reason": "r"}}
    )


def test_reference_block_marks_superseded_articles(articles):
    by_id = {a.article_id: a for a in articles}
    old = next(a for a in articles if any(b.supersedes == a.article_id for b in articles))
    block = judging.reference_block("Ref.", [old.article_id], by_id)
    assert block.startswith("Reference answer:\nRef.")
    assert "SUPERSEDED on" in block


def test_evidence_is_gold_then_valid_citations_capped():
    ids = judging.evidence_ids(["g1", "g2"], ["c1", "g1", "bogus"], ["g1", "g2", "c1"])
    assert ids == ["g1", "g2", "c1"]
    many = judging.evidence_ids([f"g{i}" for i in range(12)], [], [])
    assert len(many) == judging.MAX_EVIDENCE_ARTICLES


def test_judges_use_the_repo_template_and_settings():
    llama = judging.make_judge(FakeClient([]), "llama")
    mini = judging.make_judge(FakeClient([]), "gpt5mini")
    assert llama.temperature == 0.0
    assert llama.reasoning_effort is None
    assert mini.temperature is None
    assert mini.reasoning_effort == "minimal"
    req = llama.build_request("q", "a", "ref")
    assert "current source articles" in req.input
    assert llama.fingerprint != mini.fingerprint


def test_judge_items_turns_parse_failures_into_score_errors():
    judge = judging.make_judge(FakeClient([_verdict(True, False), "not json"]), "llama")
    inputs = [judging.JudgeInput(f"i{i}", "q", "a", "ref", "c", {}) for i in range(2)]
    records = judging.judge_items(judge, inputs, run_id="r", config="c")
    assert records[0].scores == {"correct": True, "grounded": False}
    assert records[0].meta["judge_fingerprint"] == judge.fingerprint
    assert records[1].scores == {}
    assert records[1].score_error.startswith("JudgeParseError")


def test_freeze_gate(tmp_path, monkeypatch):
    with pytest.raises(judging.JudgeNotFrozen, match="does not exist"):
        judging.check_frozen(tmp_path)
    judging.freeze(tmp_path, "test")
    judging.check_frozen(tmp_path)
    monkeypatch.setattr(
        judging,
        "load_template",
        lambda: "changed $question $reference $answer $checklist $item_ids $example",
    )
    with pytest.raises(judging.JudgeNotFrozen, match="changed"):
        judging.check_frozen(tmp_path)


# One of the 13 malformed Llama dev replies, verbatim.
LLAMA_STRAY_PAREN = (
    '{"correct": {"pass": true, "reason": "The answer matches the reference answer on all '
    'points."), "grounded": {"pass": true, "reason": "Every claim in the answer is supported '
    'by the reference answer or source articles."}}'
)


def test_stray_paren_reply_is_repaired_and_flagged():
    judge = judging.make_judge(FakeClient([LLAMA_STRAY_PAREN]), "llama")
    inputs = [judging.JudgeInput("i0", "q", "a", "ref", "c", {})]
    records = judging.judge_items(judge, inputs, run_id="r", config="c")
    assert records[0].scores == {"correct": True, "grounded": True}
    assert records[0].score_error is None
    assert records[0].meta["reply_repaired"] is True


def test_valid_reply_with_paren_inside_a_reason_is_left_alone():
    text = json.dumps(
        {
            "correct": {"pass": False, "reason": 'It says "fee (waived")'},
            "grounded": {"pass": True, "reason": "r"},
        }
    )
    judge = judging.make_judge(FakeClient([text]), "llama")
    records = judging.judge_items(
        judge, [judging.JudgeInput("i0", "q", "a", "ref", "c", {})], run_id="r", config="c"
    )
    assert records[0].scores == {"correct": False, "grounded": True}
    assert "reply_repaired" not in records[0].meta


def test_unrepairable_reply_stays_a_score_error():
    judge = judging.make_judge(FakeClient(['{"correct": {"pass": tru']), "llama")
    records = judging.judge_items(
        judge, [judging.JudgeInput("i0", "q", "a", "ref", "c", {})], run_id="r", config="c"
    )
    assert records[0].scores == {}
    assert records[0].score_error.startswith("JudgeParseError")


def test_repair_rule_is_part_of_the_fingerprint():
    from llm_eval_harness import ChecklistJudge

    judge = judging.make_judge(FakeClient([]), "llama")
    plain = ChecklistJudge(
        judge.client,
        judge.model,
        judge.checklist,
        template=judge.template,
        max_output_tokens=judge.max_output_tokens,
        temperature=judge.temperature,
    )
    assert judge.fingerprint != plain.fingerprint
