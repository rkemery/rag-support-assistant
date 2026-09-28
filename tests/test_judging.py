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
