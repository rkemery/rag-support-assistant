"""The answer judge: the harness `ChecklistJudge` with this repo's template and evidence.

Two checks, both binary:

- `correct`: agrees with the reference answer on what the question asks.
- `grounded`: every factual claim is supported by the reference answer or the
  current source articles. An answer that is not grounded has at least one
  unsupported claim, which is how the hallucination rate is defined.

The harness template takes a single `reference` slot, so the reference answer
and the source articles go in together (`reference_block`). The template and
checklist are part of the judge fingerprint, which calibration uses to refuse
verdicts from a judge other than the frozen one.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from llm_eval_harness import ChecklistJudge, EvalRecord, ModelClient
from llm_eval_harness.client import DEFAULT_PRICES, cost_usd
from llm_eval_harness.judge import (
    Checklist,
    ChecklistItem,
    ChecklistVerdict,
    JudgeOutcome,
    JudgeParseError,
    parse_checklist_reply,
)

from rag_support_assistant.clients import item_level_errors
from rag_support_assistant.data import Article

MAX_EVIDENCE_ARTICLES = 8


@dataclass(frozen=True)
class JudgeSpec:
    key: str
    model: str
    temperature: float | None
    reasoning_effort: str | None
    max_output_tokens: int


JUDGES: dict[str, JudgeSpec] = {
    # Cross-family judge: non-reasoning, accepts temperature 0. A two-check verdict is
    # about 100 tokens. Azure counts max_output_tokens against the TPM quota on arrival,
    # so a loose cap costs throughput.
    "llama": JudgeSpec("llama", "Llama-3.3-70B-Instruct", 0.0, None, 300),
    # Second judge, same vendor as the answer model. Reasoning tokens bill as output,
    # so the cap leaves room for a few hundred of them at effort "minimal".
    "gpt5mini": JudgeSpec("gpt5mini", "gpt-5-mini", None, "minimal", 600),
}
PRIMARY_JUDGE = "llama"


def _resource(name: str) -> str:
    return (resources.files("rag_support_assistant") / "prompts" / name).read_text(encoding="utf-8")


def load_checklist() -> Checklist:
    data = json.loads(_resource("checklist.json"))
    items = tuple(ChecklistItem(id=i["id"], question=i["question"]) for i in data["items"])
    return Checklist(name=data["name"], items=items)


def load_template() -> str:
    return _resource("judge_template.txt")


# Llama 3.3 70B sometimes closes a checklist item with `")` instead of `"}` (13 of 160
# dev replies, always followed by a comma). Foundry ignored JSON mode and a strict JSON
# schema for this deployment, so the reply is repaired instead, by this one rule only.
_STRAY_PAREN = re.compile(r'"\)(?=\s*(?:,|\}|$))')
REPLY_REPAIR = (
    "stray-paren-v1: a reply that fails to parse gets '\")' before ',', '}' or the end "
    "replaced by '\"}', then one reparse"
)


def repair_stray_paren(text: str) -> str:
    return _STRAY_PAREN.sub('"}', text)


class RepairingChecklistJudge(ChecklistJudge):
    """`ChecklistJudge` that retries a failed parse once after `repair_stray_paren`.

    Replies that parse are never touched. `last_repaired` says whether the last
    verdict needed the repair, so records can carry it. The repair rule is part of
    the fingerprint, so the frozen judge is the model plus this exact rule.
    """

    last_repaired: bool = False

    @property
    def fingerprint(self) -> str:
        blob = json.dumps({"base": super().fingerprint, "reply_repair": REPLY_REPAIR})
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def judge(self, question: str, answer: str, reference: str) -> ChecklistVerdict:
        self.last_repaired = False
        response = self.client.complete(self.build_request(question, answer, reference))
        try:
            results = parse_checklist_reply(response.text, self.checklist.ids)
        except JudgeParseError as exc:
            original = JudgeParseError(str(exc), raw=exc.raw, response=response)
            fixed = repair_stray_paren(response.text)
            if fixed == response.text:
                raise original from exc
            try:
                results = parse_checklist_reply(fixed, self.checklist.ids)
            except JudgeParseError:
                raise original from exc
            self.last_repaired = True
        return ChecklistVerdict(results=results, response=response)


def make_judge(client: ModelClient, key: str) -> ChecklistJudge:
    spec = JUDGES[key]
    return RepairingChecklistJudge(
        client,
        spec.model,
        load_checklist(),
        template=load_template(),
        max_output_tokens=spec.max_output_tokens,
        temperature=spec.temperature,
        reasoning_effort=spec.reasoning_effort,
    )


def format_article(article: Article, superseded_by: Mapping[str, Article]) -> str:
    header = f"[{article.article_id}] {article.title} (effective {article.effective_date})"
    newer = superseded_by.get(article.article_id)
    if newer is not None:
        header += (
            f" SUPERSEDED on {newer.effective_date} by [{newer.article_id}]. "
            "Old version, shown for context only."
        )
    return f"{header}\n{article.body.strip()}"


def evidence_ids(gold: Sequence[str], cited: Iterable[str], known: Iterable[str]) -> list[str]:
    """Gold articles first, then cited articles that exist, capped at MAX_EVIDENCE_ARTICLES."""
    known_set = set(known)
    ordered = list(dict.fromkeys([*gold, *(c for c in cited if c in known_set)]))
    return ordered[:MAX_EVIDENCE_ARTICLES]


def reference_block(
    reference_answer: str | None,
    article_ids: Sequence[str],
    articles: Mapping[str, Article],
) -> str:
    superseded_by = {a.supersedes: a for a in articles.values() if a.supersedes}
    ref = reference_answer if reference_answer else "(none for this item)"
    parts = [f"Reference answer:\n{ref}"]
    if article_ids:
        body = "\n\n".join(format_article(articles[i], superseded_by) for i in article_ids)
        parts.append(f"Source articles:\n{body}")
    else:
        parts.append("Source articles: none.")
    return "\n\n".join(parts)


def passages_block(passages: str) -> str:
    """Reference slot for RAGTruth items, which have passages but no reference answer."""
    return f"Reference answer:\n(none for this item)\n\nSource articles:\n{passages.strip()}"


@dataclass(frozen=True)
class JudgeInput:
    item_id: str
    question: str
    answer: str
    reference: str  # the full reference block
    cluster: str | None
    meta: dict[str, Any]


def judge_items(
    judge: ChecklistJudge,
    items: Sequence[JudgeInput],
    *,
    run_id: str,
    config: str,
    progress_every: int = 0,
) -> list[EvalRecord]:
    """Judge each item once. Parse failures become `score_error`, never a pass."""
    price = DEFAULT_PRICES[judge.model]
    records = []
    for n, item in enumerate(items, start=1):
        start = time.perf_counter()
        try:
            outcome = judge.score(item.question, item.answer, item.reference)
        except item_level_errors() as exc:
            outcome = JudgeOutcome(
                scores={}, reasons={}, error=f"{type(exc).__name__}: {exc}", response=None
            )
        wall_ms = (time.perf_counter() - start) * 1000.0
        response = outcome.response
        meta = {
            **item.meta,
            "judge_fingerprint": judge.fingerprint,
            "judge_reasons": outcome.reasons,
        }
        if response is not None:
            meta["from_cache"] = response.from_cache
        if outcome.error is None and getattr(judge, "last_repaired", False):
            meta["reply_repaired"] = True
        records.append(
            EvalRecord(
                run_id=run_id,
                item_id=item.item_id,
                config=config,
                model=judge.model,
                scores=dict(outcome.scores),
                cluster=item.cluster,
                tokens_in=response.input_tokens if response else 0,
                tokens_out=response.output_tokens if response else 0,
                reasoning_tokens=response.reasoning_tokens if response else 0,
                cost_usd=cost_usd(price, response) if response else 0.0,
                latency_ms=response.latency_ms if response else wall_ms,
                score_error=outcome.error,
                meta=meta,
            )
        )
        if progress_every and n % progress_every == 0:
            print(f"  judged {n}/{len(items)} for {run_id}", flush=True)
    return records


def frozen_path(root: Path) -> Path:
    return root / "judge_frozen.json"


def fingerprints() -> dict[str, str]:
    """Fingerprint of every judge as currently defined (model, template, checklist, settings).

    A fingerprint does not depend on the client, so none is needed here.
    """
    return {key: make_judge(None, key).fingerprint for key in JUDGES}  # type: ignore[arg-type]


class JudgeNotFrozen(RuntimeError):
    """Test-set judging was requested before the judge was frozen, or after it changed."""


def check_frozen(root: Path) -> None:
    path = frozen_path(root)
    if not path.exists():
        raise JudgeNotFrozen(
            f"{path} does not exist. Run the dev perturbation set first and freeze the judge "
            "(`make judge-freeze`) before scoring anything on test."
        )
    frozen = json.loads(path.read_text(encoding="utf-8"))["fingerprints"]
    current = fingerprints()
    changed = sorted(k for k in current if frozen.get(k) != current[k])
    if changed:
        raise JudgeNotFrozen(
            f"judge(s) {changed} changed since they were frozen in {path}. Test results from a "
            "judge tuned after freezing would be optimistic. Revert the prompt or re-run dev "
            "and freeze again, knowingly."
        )


def freeze(root: Path, note: str) -> dict[str, Any]:
    payload = {"fingerprints": fingerprints(), "note": note}
    frozen_path(root).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
