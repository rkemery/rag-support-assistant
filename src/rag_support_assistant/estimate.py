"""Cost of a full live run, estimated before making it.

Builds the requests the live run will send (where they do not depend on
earlier answers) and prices them at list prices:

- expected: input at about 4 bytes per token and a typical output length per
  call type, with the full-context prefix served from the prompt cache after
  the first call.
- worst case: what DollarCap reserves per call (one token per input byte plus
  `max_output_tokens`), the bound the cap enforces.

Answer-judging requests depend on the answers, so they use the reference
answer as a stand-in answer. Ragas is priced from its call count only.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from llm_eval_harness import ModelRequest
from llm_eval_harness.client import DEFAULT_PRICES, input_token_bound, max_cost_usd

from rag_support_assistant import contextual, generation
from rag_support_assistant.chunking import chunk_articles
from rag_support_assistant.clients import DEPLOYMENT_TPM, TPM_HEADROOM, estimated_tokens
from rag_support_assistant.data import load_articles, load_questions
from rag_support_assistant.judging import (
    JUDGES,
    JudgeInput,
    evidence_ids,
    make_judge,
    reference_block,
)
from rag_support_assistant.pipeline import (
    FULL_CONTEXT,
    generation_arms,
    load_contexts,
    perturbation_inputs,
    ragtruth_inputs,
)

BYTES_PER_TOKEN = 4.0
# Typical billed output tokens per call (reasoning included).
TYPICAL_OUTPUT = {"answer": 120, "context": 60, "llama": 120, "gpt5mini": 350}
RAGAS_CALLS_PER_ANSWER = 3


@dataclass
class Line:
    stage: str
    calls: int = 0
    expected_usd: float = 0.0
    worst_usd: float = 0.0
    input_tokens: int = 0
    model: str = ""
    # What Azure counts against TPM per call: prompt estimate plus max_output_tokens.
    quota_tokens: list[int] = field(default_factory=list)

    def minutes_at_quota(self, tpm: dict[str, int]) -> float:
        """Least wall time the rate limiter allows. A call larger than the per-minute
        budget goes alone and takes a whole minute, as `RateLimitedClient` sends it."""
        budget = tpm.get(self.model, 0) * TPM_HEADROOM
        if not budget:
            return 0.0
        return sum(min(t, budget) for t in self.quota_tokens) / budget


@dataclass
class Estimate:
    lines: list[Line] = field(default_factory=list)

    @property
    def expected_usd(self) -> float:
        return sum(line.expected_usd for line in self.lines)

    @property
    def worst_usd(self) -> float:
        return sum(line.worst_usd for line in self.lines)


def _price(
    requests: Iterable[ModelRequest], stage: str, output: int, cached_prefix: int = 0
) -> Line:
    line = Line(stage)
    for i, req in enumerate(requests):
        price = DEFAULT_PRICES[req.model]
        tokens = int(input_token_bound(req) / BYTES_PER_TOKEN)
        cached = cached_prefix if i > 0 else 0  # the first call writes the prompt cache
        cached_rate = (
            price.cached_input_per_m if price.cached_input_per_m is not None else price.input_per_m
        )
        line.expected_usd += (
            (tokens - cached) * price.input_per_m
            + cached * cached_rate
            + output * price.output_per_m
        ) / 1e6
        line.worst_usd += max_cost_usd(price, req)
        line.input_tokens += tokens
        line.quota_tokens.append(estimated_tokens(req))
        line.model = req.model
        line.calls += 1
    return line


def _judge_requests(key: str, inputs: list[JudgeInput]) -> list[ModelRequest]:
    judge = make_judge(None, key)  # type: ignore[arg-type]  # build_request never calls the client
    return [judge.build_request(i.question, i.answer, i.reference) for i in inputs]


def estimate(results: Path) -> Estimate:
    est = Estimate()
    questions = load_questions("test")
    articles = {a.article_id: a for a in load_articles()}
    arms = generation_arms(results)
    for split in ("dev", "test"):
        inputs = perturbation_inputs(split)
        for key in JUDGES:
            est.lines.append(
                _price(
                    _judge_requests(key, inputs),
                    f"judge perturbations {split} ({key})",
                    TYPICAL_OUTPUT[key],
                )
            )
    rt = ragtruth_inputs()
    for key in JUDGES:
        est.lines.append(
            _price(_judge_requests(key, rt), f"judge RAGTruth ({key})", TYPICAL_OUTPUT[key])
        )
    corpus = generation.format_corpus(list(articles.values()))
    for arm in arms:
        if arm == FULL_CONTEXT:
            reqs = [generation.full_context_request(q.question, corpus) for q in questions]
            prefix = int(len(corpus.encode("utf-8")) / BYTES_PER_TOKEN)
            est.lines.append(_price(reqs, f"answers {arm}", TYPICAL_OUTPUT["answer"], prefix))
        else:
            ctx = load_contexts(arm, results)
            reqs = [
                generation.rag_request(
                    q.question, [generation.ContextChunk(**c) for c in ctx[q.question_id]["chunks"]]
                )
                for q in questions
            ]
            est.lines.append(_price(reqs, f"answers {arm}", TYPICAL_OUTPUT["answer"]))
    stand_in = [
        JudgeInput(
            q.question_id,
            q.question,
            q.reference_answer,
            reference_block(
                q.reference_answer, evidence_ids(q.gold_article_ids, [], articles), articles
            ),
            None,
            {},
        )
        for q in questions
    ]
    for key in JUDGES:
        line = _price(
            _judge_requests(key, stand_in) * len(arms),
            f"judge answers, {len(arms)} arms ({key})",
            TYPICAL_OUTPUT[key],
        )
        est.lines.append(line)
    config = contextual.chosen_config(results)
    nodes = chunk_articles(list(articles.values()), config.chunking)
    ctx_reqs = [
        contextual.context_request(
            contextual.document_text(articles[n.metadata["article_id"]]), n.get_content()
        )
        for n in nodes
    ]
    est.lines.append(_price(ctx_reqs, "contextual retrieval contexts", TYPICAL_OUTPUT["context"]))
    ragas = Line("Ragas (2 configs, luna)")
    ragas.calls = 2 * len(questions) * RAGAS_CALLS_PER_ANSWER
    luna = DEFAULT_PRICES["gpt-6-luna"]
    ragas.expected_usd = ragas.calls * (1500 * luna.input_per_m + 300 * luna.output_per_m) / 1e6
    ragas.worst_usd = ragas.calls * (8000 * luna.input_per_m + 1000 * luna.output_per_m) / 1e6
    est.lines.append(ragas)
    return est


def as_table(est: Estimate, tpm: dict[str, int] | None = None) -> str:
    """Cost per stage, plus the least wall time each stage needs at the deployment's TPM quota."""
    tpm = DEPLOYMENT_TPM if tpm is None else tpm
    rows = [
        "| Stage | Calls | Expected $ | DollarCap worst case $ | Minutes at default quota |",
        "|---|---|---|---|---|",
    ]
    for line in est.lines:
        minutes = line.minutes_at_quota(tpm)
        rows.append(
            f"| {line.stage} | {line.calls} | {line.expected_usd:.3f} | {line.worst_usd:.3f} | "
            f"{f'{minutes:.0f}' if minutes else 'n/a'} |"
        )
    total_calls = sum(line.calls for line in est.lines)
    total_minutes = sum(line.minutes_at_quota(tpm) for line in est.lines)
    rows.append(
        f"| Total | {total_calls} | {est.expected_usd:.2f} | {est.worst_usd:.2f} | "
        f"{total_minutes:.0f} |"
    )
    return "\n".join(rows)
