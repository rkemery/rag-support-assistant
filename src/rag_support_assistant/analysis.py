"""Turn result records into the numbers the README reports, using the harness stats.

- Retrieval: per-config means with percentile cluster-bootstrap CIs
  (`stats.bootstrap_means`, clusters = first gold article), and paired
  comparisons along the dev-chosen path (`analysis.compare_runs` with
  clusters: a clustered paired t-test with an MDE).
- Generation: abstention table, false refusal, answered-unanswerable,
  accuracy, hallucination rate, each a clustered Wilson interval
  (`analysis.summarize_metric`).
- Judges: TPR, TNR and kappa against constructed labels
  (`calibration.pair_labels` and `judge_agreement`), and a hallucination rate
  corrected for judge error (`calibration.corrected_pass_rate`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from llm_eval_harness import EvalRecord
from llm_eval_harness.analysis import MetricSummary, RunComparison, compare_runs, summarize_metric
from llm_eval_harness.calibration import (
    CorrectedPassRate,
    JudgeAgreement,
    check_same_judge,
    corrected_pass_rate,
    judge_agreement,
    pair_labels,
)
from llm_eval_harness.labeling import LabelRecord
from llm_eval_harness.stats import Interval, bootstrap_means, n_clusters, percentile_interval

N_BOOT = 10_000
SEED = 0


@dataclass(frozen=True)
class MetricCI:
    metric: str
    interval: Interval
    n_clusters: int


def cluster_bootstrap(records: Sequence[EvalRecord], metric: str) -> MetricCI:
    """Mean of a per-question metric with a 95% percentile bootstrap over clusters."""
    values = np.array([float(r.scores[metric]) for r in records])
    clusters = [r.cluster for r in records]
    reps = bootstrap_means(values, clusters, n_boot=N_BOOT, seed=SEED)
    g = n_clusters(clusters, len(values))
    interval = percentile_interval(
        reps,
        float(values.mean()),
        len(values),
        0.95,
        f"percentile cluster bootstrap ({g} clusters)",
    )
    return MetricCI(metric, interval, g)


def latency_percentiles(records: Sequence[EvalRecord]) -> tuple[float, float]:
    values = np.array([r.latency_ms for r in records if r.error is None])
    return float(np.percentile(values, 50)), float(np.percentile(values, 95))


def paired(
    baseline: Sequence[EvalRecord], candidate: Sequence[EvalRecord], metric: str
) -> RunComparison:
    return compare_runs(baseline, candidate, metric, use_clusters=True, n_boot=N_BOOT, seed=SEED)


# ---------------------------------------------------------------- generation


def _subset(
    records: Sequence[EvalRecord], metric: str, keep: dict[str, bool | None]
) -> list[EvalRecord]:
    """Records re-scored with only `metric`, for the items in `keep` that have a value."""
    out = []
    for r in records:
        value = keep.get(r.item_id)
        if value is None:
            continue
        out.append(replace(r, scores={metric: value}, error=None, score_error=None))
    return out


@dataclass(frozen=True)
class GenerationSummary:
    arm: str
    n: int
    table: dict[
        str, dict[str, int]
    ]  # expected behavior -> {"answered": n, "abstained": n, "error": n}
    metrics: dict[str, MetricSummary]
    cost_usd: float
    tokens_in: int
    tokens_out: int
    cached_input_tokens: int
    latency_p50_ms: float
    latency_p95_ms: float


def generation_outcomes(
    gen_records: Sequence[EvalRecord], judge_records: Sequence[EvalRecord] | None
) -> dict[str, dict[str, bool | None]]:
    """Per item: which derived metrics apply and their value. None means not applicable."""
    verdicts = {r.item_id: r for r in judge_records or []}
    out: dict[str, dict[str, bool | None]] = {
        "false_refusal": {},
        "answered_unanswerable": {},
        "accuracy": {},
        "premise_corrected": {},
        "hallucination": {},
        "hallucinated_any": {},
    }
    for r in gen_records:
        expected = r.meta["expected"]
        if r.error is not None:
            continue
        abstained = bool(r.scores["abstained"])
        verdict = verdicts.get(r.item_id)
        judged = verdict is not None and verdict.score_error is None and verdict.scores
        grounded = bool(verdict.scores["grounded"]) if judged else None
        correct = bool(verdict.scores["correct"]) if judged else None
        if expected == "answer":
            out["false_refusal"][r.item_id] = abstained
            out["accuracy"][r.item_id] = False if abstained else correct
        elif expected == "decline":
            out["answered_unanswerable"][r.item_id] = not abstained
        else:
            out["premise_corrected"][r.item_id] = False if abstained else correct
        if not abstained:
            out["hallucination"][r.item_id] = None if grounded is None else not grounded
        out["hallucinated_any"][r.item_id] = (
            False if abstained else (None if grounded is None else not grounded)
        )
    return out


def summarize_generation(
    arm: str, gen_records: Sequence[EvalRecord], judge_records: Sequence[EvalRecord] | None
) -> GenerationSummary:
    table: dict[str, dict[str, int]] = {}
    for r in gen_records:
        row = table.setdefault(r.meta["expected"], {"answered": 0, "abstained": 0, "error": 0})
        if r.error is not None:
            row["error"] += 1
        else:
            row["abstained" if r.scores["abstained"] else "answered"] += 1
    metrics = {}
    for metric, keep in generation_outcomes(gen_records, judge_records).items():
        subset = _subset(gen_records, metric, keep)
        if len(subset) >= 2:
            metrics[metric] = summarize_metric(subset, metric, use_clusters=True)
    ok = [r for r in gen_records if r.error is None]
    p50, p95 = latency_percentiles(ok) if ok else (float("nan"), float("nan"))
    return GenerationSummary(
        arm=arm,
        n=len(gen_records),
        table=table,
        metrics=metrics,
        cost_usd=sum(r.cost_usd for r in gen_records),
        tokens_in=sum(r.tokens_in for r in gen_records),
        tokens_out=sum(r.tokens_out for r in gen_records),
        cached_input_tokens=sum(int(r.meta.get("cached_input_tokens", 0)) for r in gen_records),
        latency_p50_ms=p50,
        latency_p95_ms=p95,
    )


# ---------------------------------------------------------------- judges


def agreement(
    judge_records: Sequence[EvalRecord],
    labels: Sequence[LabelRecord],
    check: str,
    items: Sequence[str],
) -> JudgeAgreement:
    """Judge vs constructed labels on `items` that carry a label for `check`."""
    labeled = {lab.item_id for lab in labels if check in lab.labels}
    wanted = [i for i in items if i in labeled]
    pairs = pair_labels(
        judge_records,
        [lab for lab in labels if lab.item_id in labeled],
        check,
        only_items=wanted,
        on_error="exclude",
    )
    return judge_agreement(pairs.judge, pairs.human, n_boot=N_BOOT, seed=SEED)


def detection_by_type(
    judge_records: Sequence[EvalRecord], types: dict[str, str], check: str = "grounded"
) -> dict[str, Interval]:
    """Share of items of each perturbation type the judge failed on `check`.

    For error types this is the catch rate (TNR within the type). For
    original and paraphrase it is the false alarm rate.
    """
    from llm_eval_harness.stats import wilson_interval

    by_type: dict[str, list[bool]] = {}
    for r in judge_records:
        if check in r.scores:
            by_type.setdefault(types[r.item_id], []).append(not bool(r.scores[check]))
    return {t: wilson_interval(sum(v), len(v)) for t, v in by_type.items()}


def corrected_hallucination_rate(
    answer_judge_records: Sequence[EvalRecord],
    calibration_records: Sequence[EvalRecord],
    labels: Sequence[LabelRecord],
    test_items: Sequence[str],
) -> CorrectedPassRate:
    """Rogan-Gladen corrected grounded rate of real answers, calibrated on perturbation test.

    Refuses (CalibrationError) when the answers were judged by a different
    judge fingerprint than the calibration run.
    """
    check_same_judge(calibration_records, answer_judge_records)
    labeled = {lab.item_id for lab in labels if "grounded" in lab.labels}
    pairs = pair_labels(
        calibration_records,
        [lab for lab in labels if lab.item_id in labeled],
        "grounded",
        only_items=[i for i in test_items if i in labeled],
        on_error="exclude",
    )
    judged = [r for r in answer_judge_records if "grounded" in r.scores]
    verdicts = [bool(r.scores["grounded"]) for r in judged]
    clusters = [r.cluster for r in judged]
    return corrected_pass_rate(
        verdicts,
        pairs.judge,
        pairs.human,
        test_clusters=clusters,
        n_boot=N_BOOT,
        seed=SEED,
    )
