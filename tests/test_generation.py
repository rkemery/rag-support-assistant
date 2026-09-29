from __future__ import annotations

import pytest

from rag_support_assistant import generation as gen


def test_rag_request_uses_structured_output_and_effort_none():
    chunks = [
        gen.ContextChunk("c1", "plan-plus", "The Plus plan", "2026-06-01", "Plus costs $5.00.")
    ]
    req = gen.rag_request("How much is Plus?", chunks)
    assert req.model == "gpt-6-luna"
    assert req.reasoning_effort == "none"
    assert req.temperature is None
    assert req.max_output_tokens == gen.ANSWER_MAX_TOKENS
    fmt = req.extra["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["strict"] is True
    assert set(fmt["schema"]["required"]) == {"answer", "citations", "abstain"}
    assert "[plan-plus] The Plus plan (effective 2026-06-01)" in req.input[0]["content"]


def test_full_context_prefix_is_identical_and_question_last(articles):
    corpus = gen.format_corpus(articles)
    a = gen.full_context_request("first question", corpus)
    b = gen.full_context_request("second question", corpus)
    assert a.input[0] == b.input[0]
    assert a.input[-1]["role"] == "user"
    assert a.input[-1]["content"].endswith("first question")
    assert all(f"[{art.article_id}]" in a.input[0]["content"] for art in articles)


def test_parse_answer_strict():
    ok = gen.parse_answer('{"answer": " Yes. ", "citations": ["a"], "abstain": false}')
    assert ok == gen.Answer("Yes.", ("a",), False)
    for bad in ("not json", '{"answer": "x", "citations": [], "abstain": "no"}', '{"answer": "x"}'):
        with pytest.raises(gen.AnswerParseError):
            gen.parse_answer(bad)
