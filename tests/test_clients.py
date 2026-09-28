from __future__ import annotations

import pytest
from llm_eval_harness import BudgetExceeded, CacheMiss, FakeClient, ModelRequest

from rag_support_assistant.clients import (
    RateLimitedClient,
    build_stack,
    estimated_tokens,
    wrap_for_tests,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_rate_limiter_waits_for_the_window():
    clock = FakeClock()
    req = ModelRequest(model="m", input="hi", max_output_tokens=400)
    per_call = estimated_tokens(req)
    limiter = RateLimitedClient(
        FakeClient(lambda r: "ok"),
        {"m": 2 * per_call},
        headroom=1.0,
        clock=clock,
        sleep=clock.sleep,
    )
    limiter.complete(req)
    limiter.complete(req)
    assert clock.slept == []
    limiter.complete(req)
    assert clock.slept
    assert clock.now >= 60.0
    assert limiter.waited_s == pytest.approx(sum(clock.slept))


def test_estimate_counts_max_output_tokens():
    small = ModelRequest(model="m", input="x" * 350, max_output_tokens=10)
    big = ModelRequest(model="m", input="x" * 350, max_output_tokens=1000)
    assert estimated_tokens(big) - estimated_tokens(small) == 990


def test_replay_never_reaches_a_model(tmp_path):
    stack = build_stack("replay", tmp_path)
    with pytest.raises(CacheMiss):
        stack.client.complete(ModelRequest(model="gpt-6-luna", input="x", max_output_tokens=5))


def test_live_without_endpoint_fails_before_any_call(tmp_path, monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="AZURE_OPENAI_BASE_URL"):
        build_stack("live", tmp_path)


def test_stack_caps_spend_and_caches(tmp_path):
    stack = wrap_for_tests(FakeClient(lambda r: "reply"), tmp_path, cap_usd=1.0)
    req = ModelRequest(model="gpt-6-luna", input="question", max_output_tokens=50)
    stack.client.complete(req)
    stack.client.complete(req)
    assert stack.cache.hits == 1
    assert stack.cap.calls == 1
    tiny = wrap_for_tests(FakeClient(lambda r: "reply"), tmp_path / "t", cap_usd=1e-9)
    with pytest.raises(BudgetExceeded):
        tiny.client.complete(req)
