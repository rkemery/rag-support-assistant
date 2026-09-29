"""Ragas 0.4.3 behind a harness-style interface. Optional extra: `uv sync --extra ragas`.

Ragas brings its own LLM plumbing. Here its LLM calls go through a harness
`ModelClient` instead, so they hit the same disk cache and the same DollarCap
as every other live call, and replay offline like the rest.

Two metrics, both from Ragas' single-turn API: `faithfulness` (share of the
answer's statements the retrieved contexts support) and `context_recall`
(share of the reference answer's claims the contexts cover). The judge model
is gpt-6-luna, which only accepts the default temperature, so the temperature
Ragas asks for is not sent.

Ragas sends usage analytics unless RAGAS_DO_NOT_TRACK is "true". This module
sets it before importing Ragas. With it set, every send path in
ragas/_analytics.py (0.4.3) returns before any request. Ragas still writes a
random user id to a local file in its user data directory.
"""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Sequence
from typing import Any

from llm_eval_harness import EvalRecord, ModelClient, ModelRequest

os.environ["RAGAS_DO_NOT_TRACK"] = "true"

from langchain_core.outputs import Generation, LLMResult
from ragas.dataset_schema import SingleTurnSample
from ragas.llms.base import BaseRagasLLM
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._faithfulness import Faithfulness

RAGAS_VERSION = "0.4.3"
RAGAS_MODEL = "gpt-6-luna"
RAGAS_MAX_TOKENS = 1500


class HarnessRagasLLM(BaseRagasLLM):
    """A Ragas LLM that sends every prompt through a harness ModelClient."""

    def __init__(
        self,
        client: ModelClient,
        model: str = RAGAS_MODEL,
        max_output_tokens: int = RAGAS_MAX_TOKENS,
    ) -> None:
        super().__init__()
        self.client = client
        self.model = model
        self.max_output_tokens = max_output_tokens

    def _request(self, text: str, trial: int) -> ModelRequest:
        return ModelRequest(
            model=self.model,
            input=text,
            max_output_tokens=self.max_output_tokens,
            reasoning_effort="none",
            trial=trial,
        )

    def generate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        text = prompt.to_string()
        generations = []
        for trial in range(n):  # n samples need n distinct cache entries
            response = self.client.complete(self._request(text, trial))
            generations.append(
                Generation(
                    text=response.text, generation_info={"finish_reason": response.finish_reason}
                )
            )
        return LLMResult(generations=[generations])

    async def agenerate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        return self.generate_text(prompt, n=n, temperature=temperature, stop=stop)

    def is_finished(self, response: LLMResult) -> bool:
        return all(
            (g.generation_info or {}).get("finish_reason") == "stop"
            for gens in response.generations
            for g in gens
        )


class RagasScorer:
    """score(question, answer, contexts, reference) -> {"faithfulness": x, "context_recall": y}."""

    def __init__(self, client: ModelClient, model: str = RAGAS_MODEL) -> None:
        self.llm = HarnessRagasLLM(client, model)
        self.metrics = [Faithfulness(llm=self.llm), LLMContextRecall(llm=self.llm)]

    def score(
        self, question: str, answer: str, contexts: Sequence[str], reference: str
    ) -> dict[str, float]:
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=list(contexts),
            reference=reference,
        )
        return {m.name: float(asyncio.run(m.single_turn_ascore(sample))) for m in self.metrics}

    def score_records(
        self, items: Sequence[dict[str, Any]], run_id: str, config: str
    ) -> list[EvalRecord]:
        """items: dicts with item_id, question, answer, contexts, reference, cluster."""
        records = []
        for item in items:
            scores = self.score(
                item["question"], item["answer"], item["contexts"], item["reference"]
            )
            finite = {k: v for k, v in scores.items() if not math.isnan(v)}
            missing = sorted(set(scores) - set(finite))
            records.append(
                EvalRecord(
                    run_id=run_id,
                    item_id=item["item_id"],
                    config=config,
                    model=self.llm.model,
                    scores=finite,
                    cluster=item.get("cluster"),
                    score_error=f"ragas returned NaN for {missing}" if missing else None,
                    meta={"ragas_version": RAGAS_VERSION},
                )
            )
        return records
