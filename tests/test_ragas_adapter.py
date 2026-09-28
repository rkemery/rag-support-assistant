"""Ragas behind the harness client, with a fake model that answers Ragas' prompts."""

from __future__ import annotations

import json

import pytest
from llm_eval_harness import FakeClient, ModelRequest

pytest.importorskip("ragas")

from rag_support_assistant.clients import wrap_for_tests
from rag_support_assistant.ragas_adapter import RagasScorer


def fake_ragas_model(request: ModelRequest) -> str:
    prompt = request.input
    assert isinstance(prompt, str)
    if "Break down each sentence" in prompt:
        return json.dumps({"statements": ["Plus costs $5.00 a month.", "Plus includes a lounge."]})
    if "judge the faithfulness" in prompt:
        return json.dumps(
            {
                "statements": [
                    {"statement": "Plus costs $5.00 a month.", "reason": "stated", "verdict": 1},
                    {"statement": "Plus includes a lounge.", "reason": "not stated", "verdict": 0},
                ]
            }
        )
    if "can be attributed to the given context" in prompt:
        return json.dumps(
            {
                "classifications": [
                    {"statement": "Plus costs $5.00.", "reason": "ok", "attributed": 1}
                ]
            }
        )
    raise AssertionError(f"unexpected prompt: {prompt[:200]}")


def test_ragas_metrics_run_through_the_capped_cached_client(tmp_path):
    fake = FakeClient(fake_ragas_model)
    stack = wrap_for_tests(fake, tmp_path, cap_usd=1.0)
    scorer = RagasScorer(stack.client)
    scores = scorer.score(
        "How much is Plus?",
        "Plus costs $5.00 a month. It includes a lounge.",
        ["The Plus plan costs $5.00 a month."],
        "Plus costs $5.00 a month.",
    )
    assert scores == {"faithfulness": pytest.approx(0.5), "context_recall": pytest.approx(1.0)}
    assert stack.cap.calls >= 3
    assert all(r.model == "gpt-6-luna" and r.temperature is None for r in fake.calls)
