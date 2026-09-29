"""Render the README results section from the committed result files.

Rows whose results do not exist yet (anything that needs a live model run)
print "pending live run" instead of a number. Nothing here calls a model.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llm_eval_harness import EvalRecord, read_records
from llm_eval_harness.analysis import RunComparison
from llm_eval_harness.report import write_section
from llm_eval_harness.stats import Interval

from rag_support_assistant import analysis
from rag_support_assistant.data import REPO_ROOT, load_questions
from rag_support_assistant.judging import JUDGES
from rag_support_assistant.pipeline import FULL_CONTEXT, generation_path, judge_path
from rag_support_assistant.scoring import PRIMARY_METRIC

PENDING = "pending live run"
RETRIEVAL_COLUMNS = (
    ("ndcg@10", "nDCG@10"),
    ("mrr@10", "MRR@10"),
    ("recall@5", "Recall@5"),
    ("recall@10", "Recall@10"),
)


def _ci(interval: Interval, digits: int = 3) -> str:
    return (
        f"{interval.estimate:.{digits}f} ({interval.low:.{digits}f} to {interval.high:.{digits}f})"
    )


def _pct(interval: Interval) -> str:
    return (
        f"{interval.estimate * 100:.1f}% ({interval.low * 100:.1f}% to {interval.high * 100:.1f}%)"
    )


def _diff(result: RunComparison, digits: int = 3) -> str:
    c = result.comparison
    return f"{c.diff:+.{digits}f} ({c.low:+.{digits}f} to {c.high:+.{digits}f})"


def _p(result: RunComparison) -> str:
    p = result.comparison.pvalue
    return "n/a" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def _mde(result: RunComparison, digits: int = 3) -> str:
    return "n/a" if result.mde is None else f"{result.mde:.{digits}f}"


def _load(path: Path) -> list[EvalRecord] | None:
    return read_records(path) if path.exists() else None


# ---------------------------------------------------------------- retrieval


def retrieval_section(results: Path) -> str:
    selection_path = results / "retrieval" / "selection.json"
    if not selection_path.exists():
        return "Retrieval results are missing. Run `make retrieval`.\n"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    test = {
        slug: read_records(results / "retrieval" / "test" / f"{slug}.jsonl")
        for slug in selection["configs"]
    }
    n = len(next(iter(test.values())))
    chosen = {step["chosen"] for step in selection["steps"]}
    gen = set(selection["generation_configs"])
    out = [
        f"**Retrieval, test split** ({n} questions with a gold article, article-level, no LLM). "
        "Mean with a 95% percentile bootstrap CI over clusters (questions grouped by their first "
        "gold article). Latency is per query on CPU, on a 4-vCPU container shared with other "
        "jobs, so it is rough (see Limitations).",
        "",
        "| Config | Role | nDCG@10 | MRR@10 | Recall@5 | Recall@10 | p50 / p95 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    clusters = 0
    for slug, info in selection["configs"].items():
        records = test[slug]
        cells = []
        for metric, _ in RETRIEVAL_COLUMNS:
            ci = analysis.cluster_bootstrap(records, metric)
            clusters = ci.n_clusters
            cells.append(_ci(ci.interval))
        p50, p95 = analysis.latency_percentiles(records)
        role = "strong arm (offline only)" if info["role"] == "strong_arm" else "grid"
        tags = [
            t for t, on in (("chosen on dev", slug in chosen), ("generation", slug in gen)) if on
        ]
        if tags:
            role += ", " + ", ".join(tags)
        out.append(f"| {info['name']} | {role} | {' | '.join(cells)} | {p50:.0f} / {p95:.0f} |")
    out += [
        "",
        f'{clusters} clusters. "chosen on dev" marks the winner of each step on dev nDCG@10, '
        '"generation" the three configs picked on dev for the answer runs.',
        "",
    ]
    out += _retrieval_comparisons(selection, test)
    out += _dev_path(selection)
    return "\n".join(out) + "\n"


def _comparison_pairs(selection: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(step, baseline slug, candidate slug): each candidate against its step's incumbent."""
    pairs = []
    for step in selection["steps"]:
        slugs = list(step["dev"])
        for candidate in slugs[1:]:
            pairs.append((step["step"], slugs[0], candidate))
    strong = [s for s, info in selection["configs"].items() if info["role"] == "strong_arm"]
    by_model = {s: selection["configs"][s]["config"] for s in selection["configs"]}
    final = selection["steps"][-1]["chosen"]
    embed_winner = selection["steps"][1]["chosen"]
    for slug in strong:
        base = final if by_model[slug]["reranker"] else embed_winner
        pairs.append(("strong arm", base, slug))
    return pairs


def _retrieval_comparisons(
    selection: dict[str, Any], test: dict[str, list[EvalRecord]]
) -> list[str]:
    names = {s: info["name"] for s, info in selection["configs"].items()}
    out = [
        "**Paired comparisons on test**, same questions, clustered paired t-test (harness "
        "`compare_runs`), with the minimum detectable effect at 80% power. nDCG@10 is the "
        "primary metric. Each step's candidates were fixed on dev before test ran, so the "
        "nDCG@10 tests on the grid rows are the confirmatory ones. The recall@5 columns and the "
        "strong-arm rows are exploratory.",
        "",
        "| Step | Baseline -> candidate | nDCG@10 change (95% CI) | p | MDE | "
        "Recall@5 change (95% CI) | p |",
        "|---|---|---|---|---|---|---|",
    ]
    for step, base, cand in _comparison_pairs(selection):
        ndcg = analysis.paired(test[base], test[cand], "ndcg@10")
        rec = analysis.paired(test[base], test[cand], "recall@5")
        out.append(
            f"| {step} | {names[base]} -> {names[cand]} | {_diff(ndcg)} | {_p(ndcg)} | "
            f"{_mde(ndcg)} | {_diff(rec)} | {_p(rec)} |"
        )
    out.append("")
    return out


def _dev_path(selection: dict[str, Any]) -> list[str]:
    names = {s: info["name"] for s, info in selection["configs"].items()}
    out = [
        "**How the path was chosen (dev split, mean nDCG@10).** Test was run once, after these "
        "choices were fixed.",
        "",
        "| Step | Candidates (dev nDCG@10) | Chosen |",
        "|---|---|---|",
    ]
    for step in selection["steps"]:
        cands = ", ".join(f"{names.get(s, s)} {v:.3f}" for s, v in step["dev"].items())
        out.append(f"| {step['step']} | {cands} | {names.get(step['chosen'], step['chosen'])} |")
    sweep = ", ".join(f"{a}: {v:.3f}" for a, v in selection["alpha_sweep"].items())
    out += ["", f"Convex fusion weight on dense (alpha) swept on dev: {sweep}.", ""]
    return out


# ---------------------------------------------------------------- generation


def generation_section(results: Path, arms: Sequence[str], names: dict[str, str]) -> str:
    n_test = len(load_questions("test"))
    out = [
        f"**Answers, test split** ({n_test} questions: 120 answerable, 20 that should be "
        "declined, 10 false premises). gpt-6-luna, reasoning effort none, top 8 chunks. "
        f"Judge: {JUDGES['llama'].model}. Clustered Wilson 95% CIs.",
        "",
        "| Arm | Accuracy (answerable) | False refusal | Answered unanswerable | "
        "Hallucination rate | Premise corrected | $ per 1k answers | p50 ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    summaries = {}
    for arm in arms:
        gen = _load(generation_path(arm, "test", results))
        judged = _load(judge_path("llama", f"answers/test/{arm}", results))
        label = names.get(arm, arm)
        if gen is None:
            out.append(f"| {label} | {' | '.join([PENDING] * 7)} |")
            continue
        s = analysis.summarize_generation(arm, gen, judged)
        summaries[arm] = s

        def cell(metric: str, s: analysis.GenerationSummary = s) -> str:
            m = s.metrics.get(metric)
            return _pct(m.interval) + f", n={m.interval.n}" if m else PENDING

        per_1k = s.cost_usd / max(s.n, 1) * 1000
        out.append(
            f"| {label} | {cell('accuracy')} | {cell('false_refusal')} | "
            f"{cell('answered_unanswerable')} | {cell('hallucination')} | "
            f"{cell('premise_corrected')} | {per_1k:.3f} | {s.latency_p50_ms:.0f} |"
        )
    out.append("")
    if summaries:
        out += _generation_notes(results, summaries, arms, names)
    out += _abstention_tables(summaries, arms, names)
    if summaries:
        out += _decline_split(results, [a for a in arms if a in summaries])
    out += _judge_rows(results, arms, names)
    return "\n".join(out) + "\n"


# Two Llama replies in the rrf arm use bare `no` and `yes` instead of JSON booleans, so the
# frozen judge records them as parse errors. These are the verdicts as written in the cached
# replies (cache/40/4051f9f3... and cache/c0/c0877ec5...), used only for the footnote.
UNPARSED_AS_WRITTEN: dict[tuple[str, str], dict[str, bool]] = {
    ("fixed-title-hybrid-bge-small-rrf", "q-test-007"): {"correct": False, "grounded": True},
    ("fixed-title-hybrid-bge-small-rrf", "q-test-068"): {"correct": False, "grounded": False},
}


def _share(outcomes: dict[str, bool | None]) -> tuple[int, int]:
    known = [v for v in outcomes.values() if v is not None]
    return sum(known), len(known)


def _generation_notes(
    results: Path,
    summaries: dict[str, analysis.GenerationSummary],
    arms: Sequence[str],
    names: dict[str, str],
) -> list[str]:
    """Hallucination denominators per arm, and a footnote for replies the judge couldn't parse."""
    from dataclasses import replace

    present = [a for a in arms if a in summaries]
    parts = []
    for arm in present:
        t = summaries[arm].table
        answered = [
            t.get(k, {}).get("answered", 0) for k in ("answer", "decline", "correct_premise")
        ]
        parts.append(f"{sum(answered)} ({' + '.join(str(x) for x in answered)})")
    out = [
        "The hallucination rate counts answered questions only (answerable + should-decline + "
        "false premise), so its denominator differs by arm: "
        + ", ".join(parts[:-1])
        + (" and " if len(parts) > 1 else "")
        + parts[-1]
        + ", in table order. The table's n is smaller where the judge couldn't parse a reply "
        "(see below).",
        "",
    ]
    for arm in present:
        gen = _load(generation_path(arm, "test", results)) or []
        judged = _load(judge_path("llama", f"answers/test/{arm}", results)) or []
        failed = [r for r in judged if r.score_error is not None]
        if not failed:
            continue
        ids = ", ".join(r.item_id for r in failed)
        note = (
            f"{names.get(arm, arm)}: Llama's reply couldn't be parsed for {len(failed)} answers "
            f"({ids}), so they're left out of its accuracy and hallucination rate."
        )
        if all((arm, r.item_id) in UNPARSED_AS_WRITTEN for r in failed):
            patched = [
                replace(r, scores=UNPARSED_AS_WRITTEN[(arm, r.item_id)], score_error=None)
                if r.score_error is not None
                else r
                for r in judged
            ]
            outcomes = analysis.generation_outcomes(gen, patched)
            acc, acc_n = _share(outcomes["accuracy"])
            hal, hal_n = _share(outcomes["hallucination"])
            note += (
                " The replies use bare `no` and `yes` instead of JSON booleans. Read as written, "
                f"accuracy would be {acc}/{acc_n} = {acc / acc_n:.1%} and the hallucination rate "
                f"{hal}/{hal_n} = {hal / hal_n:.1%} (see Limitations)."
            )
        out += [note, ""]
    return out


def _judge_rows(results: Path, arms: Sequence[str], names: dict[str, str]) -> list[str]:
    """Answers flagged as not grounded by each judge, and how often the LLM judges disagree."""
    from rag_support_assistant.validation import load_perturbation_set, load_ragtruth_subset

    out = [
        "**Hallucination flags on the answers, by judge** (flagged as not grounded / judged). "
        "Disagree counts the answers both LLM judges read where their `grounded` verdicts differ.",
        "",
        "| Arm | Llama | gpt-5-mini | Llama vs gpt-5-mini disagree | HHEM |",
        "|---|---|---|---|---|",
    ]
    flags: dict[str, list[EvalRecord]] = {}
    mini_flags: dict[str, list[EvalRecord]] = {}
    for arm in arms:
        label = names.get(arm, arm)
        llama = _load(judge_path("llama", f"answers/test/{arm}", results))
        mini = _load(judge_path("gpt5mini", f"answers/test/{arm}", results))
        hhem = _load(results / "hhem" / "answers" / f"{arm}.jsonl")
        if llama is None:
            out.append(f"| {label} | {PENDING} | {PENDING} | {PENDING} | {PENDING} |")
            continue
        flags[arm] = llama
        if mini is not None:
            mini_flags[arm] = mini
        disagree = PENDING
        if mini is not None:
            a = {r.item_id: r.scores["grounded"] for r in llama if "grounded" in r.scores}
            b = {r.item_id: r.scores["grounded"] for r in mini if "grounded" in r.scores}
            both = sorted(set(a) & set(b))
            disagree = f"{sum(a[i] != b[i] for i in both)} / {len(both)}"
        out.append(
            f"| {label} | {_flagged(llama)} | {_flagged(mini)} | {disagree} | {_flagged(hhem)} |"
        )
    out.append("")
    calibration = _load(judge_path("llama", "perturbations/test", results))
    ragtruth = _load(judge_path("llama", "ragtruth", results))
    hhem_pert = _load(results / "hhem" / "perturbations_test.jsonl")
    if not flags or calibration is None or ragtruth is None:
        return out
    pset = load_perturbation_set()
    test_ids = list(pset.split.test)
    pert = analysis.agreement(calibration, pset.labels, "grounded", test_ids)
    rt_labels = _ragtruth_labels(load_ragtruth_subset())
    rt = analysis.agreement(ragtruth, rt_labels, "grounded", [lab.item_id for lab in rt_labels])
    corrected = [
        analysis.corrected_hallucination_rate(llama, calibration, pset.labels, test_ids)
        for llama in flags.values()
    ]
    unchanged = all(math.isclose(c.corrected.estimate, c.observed.estimate) for c in corrected)
    effect = (
        "returns the judged rate unchanged for every arm"
        if unchanged
        else "gives " + ", ".join(f"{(1 - c.corrected.estimate) * 100:.1f}%" for c in corrected)
    )
    # Rogan-Gladen with the RAGTruth rates: theta = (p + TNR - 1) / (TPR + TNR - 1).
    tpr, tnr = rt.tpr.estimate, rt.tnr.estimate
    rg = [1 - (c.observed.estimate + tnr - 1) / (tpr + tnr - 1) for c in corrected]
    rg_text = (
        f"a negative rate for every arm ({min(rg) * 100:.0f}% to {max(rg) * 100:.0f}%), which "
        "clips to 0"
        if max(rg) < 0
        else ", ".join(f"{h * 100:.1f}%" for h in rg)
    )
    rates_text = (
        f"both at {pert.tpr.estimate:.2f}"
        if f"{pert.tpr.estimate:.2f}" == f"{pert.tnr.estimate:.2f}"
        else f"at {pert.tpr.estimate:.2f} and {pert.tnr.estimate:.2f}"
    )
    out += [
        "A Rogan-Gladen correction for judge error was run and can't be identified here. With "
        f"the Llama judge's perturbation TPR and TNR {rates_text} it {effect}, the bootstrap "
        "over answers collapses when an arm has 0 or 1 flagged answers, and plugging in the "
        f"RAGTruth rates (TPR {tpr:.0%}, TNR {tnr:.0%}) gives {rg_text}. The true "
        "hallucination rate is unknown, bracketed by a judge that catches "
        f"{pert.tnr.estimate:.0%} of synthetic errors and {tnr:.0%} of RAGTruth's, neither of "
        "which matches this task. RAGTruth was also judged without a reference answer, while "
        "the answers here are judged with one.",
        "",
        f"The arms are at most {_spread(flags)} flagged answers apart by Llama"
        + (f" and {_spread(mini_flags)} by gpt-5-mini" if mini_flags else "")
        + ", and the two LLM judges rank the retrieval arms differently, so the arms can't be "
        "told apart on hallucination.",
        "",
    ]
    if hhem_pert is not None:
        types = {item.item_id: item.type for item in pset.items}
        wanted = set(test_ids)
        rates = analysis.detection_by_type([r for r in hhem_pert if r.item_id in wanted], types)
        out.append(
            "HHEM flags far more answers than either LLM judge, which fits its "
            f"{rates['paraphrase'].estimate:.0%} false-alarm rate on faithful paraphrases."
        )
    out.append("")
    return out


def _spread(by_arm: dict[str, list[EvalRecord]]) -> int:
    counts = [
        sum(not r.scores["grounded"] for r in rs if "grounded" in r.scores)
        for rs in by_arm.values()
    ]
    return max(counts) - min(counts)


def _flagged(records: list[EvalRecord] | None) -> str:
    if records is None:
        return PENDING
    judged = [r for r in records if "grounded" in r.scores]
    return f"{sum(not r.scores['grounded'] for r in judged)} / {len(judged)}"


def _abstention_tables(
    summaries: dict[str, analysis.GenerationSummary], arms: Sequence[str], names: dict[str, str]
) -> list[str]:
    out = [
        "**Abstention table, test split** (answered / abstained). The first two rows are the "
        "2x2. False-premise questions count as unanswerable in the dataset, but the right move "
        "is to answer and correct the premise, so they get their own row.",
        "",
        "| Expected | " + " | ".join(names.get(a, a) for a in arms) + " |",
        "|---|" + "---|" * len(arms),
    ]
    for expected, label in (
        ("answer", "Answerable (120)"),
        ("decline", "Should decline (20)"),
        ("correct_premise", "False premise (10)"),
    ):
        cells = []
        for arm in arms:
            s = summaries.get(arm)
            if s is None:
                cells.append(PENDING)
            else:
                row = s.table.get(expected, {"answered": 0, "abstained": 0, "error": 0})
                err = f", {row['error']} errors" if row["error"] else ""
                cells.append(f"{row['answered']} / {row['abstained']}{err}")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def _decline_split(results: Path, arms: Sequence[str]) -> list[str]:
    """Answered should-decline questions by unanswerable type, and what the judge made of them."""
    questions = {q.question_id: q for q in load_questions("test")}
    declines = [q for q in questions.values() if q.expected_behavior == "decline"]
    types = sorted({str(q.unanswerable_type) for q in declines})
    cells, passed, answered_all = [], 0, 0
    for arm in arms:
        gen = _load(generation_path(arm, "test", results)) or []
        judged = _load(judge_path("llama", f"answers/test/{arm}", results)) or []
        verdicts = {r.item_id: r.scores for r in judged if r.score_error is None}
        answered = [
            r.item_id
            for r in gen
            if r.meta["expected"] == "decline" and r.error is None and not r.scores["abstained"]
        ]
        answered_all += len(answered)
        passed += sum(
            1
            for i in answered
            if verdicts.get(i, {}).get("correct") and verdicts.get(i, {}).get("grounded")
        )
        counts = [sum(1 for i in answered if questions[i].unanswerable_type == t) for t in types]
        cells.append(" / ".join(str(c) for c in counts))
    kinds = " / ".join(t.replace("_", "-") for t in types)
    split = " and ".join(
        f"{sum(1 for q in declines if q.unanswerable_type == t)} {t.replace('_', '-')}"
        for t in types
    )
    return [
        f"The {len(declines)} should-decline questions are {split}. Answered ones by type "
        f"({kinds}), in table order: {', '.join(cells)}. A near-miss question is on a covered "
        "topic but asks for a detail the help center doesn't give, and its reference answer is a "
        'partial answer ("the help center doesn\'t list the stores. It says..."). The answer '
        "prompt says to abstain when the excerpts don't contain the answer, yet Llama passed "
        f"{'all' if passed == answered_all else f'{passed} of the'} {answered_all} answered "
        "replies as correct and grounded. So "
        '"answered unanswerable" mostly counts replies that say the help center doesn\'t cover '
        "the detail but leave the abstain flag off, and the reference answers and the prompt "
        "disagree on whether that's right. The metric stays as defined before the run.",
        "",
    ]


# ---------------------------------------------------------------- judge validation


def judge_section(results: Path) -> str:
    from rag_support_assistant.validation import load_perturbation_set, load_ragtruth_subset

    pset = load_perturbation_set()
    test_ids = list(pset.split.test)
    types = {item.item_id: item.type for item in pset.items}
    n_neg = sum(1 for i in pset.items if i.split == "test" and not i.labels["grounded"])
    rt = load_ragtruth_subset()
    out = [
        "**Judge validation without human labels.** Perturbation test split: "
        f"{len(test_ids)} items built from the facts file ({len(test_ids) - n_neg} faithful, "
        f"{n_neg} with one injected error), labels known by construction. The judge prompt is "
        f"tuned on the {len(pset.split.dev)} dev items only and frozen, by fingerprint, before "
        f"test is judged. RAGTruth: {len(rt)} human-annotated QA "
        "responses (half with a hallucination). TPR is the share of good answers passed, TNR the "
        "share of flawed answers caught. Wilson 95% CIs, kappa with a bootstrap CI.",
        "",
        "| Judge | Set | Check | n | TPR | TNR | Cohen's kappa |",
        "|---|---|---|---|---|---|---|",
    ]
    rows: list[tuple[str, str, str, list[EvalRecord] | None, Any]] = []
    for key, spec in JUDGES.items():
        pert = _load(judge_path(key, "perturbations/test", results))
        rag = _load(judge_path(key, "ragtruth", results))
        rows.append((spec.model, "perturbations", "grounded", pert, test_ids))
        rows.append((spec.model, "perturbations", "correct", pert, test_ids))
        rows.append((spec.model, "RAGTruth", "grounded", rag, None))
    hhem_pert = _load(results / "hhem" / "perturbations_test.jsonl")
    hhem_rag = _load(results / "hhem" / "ragtruth.jsonl")
    rows.append(("HHEM-2.1-Open (threshold 0.5)", "perturbations", "grounded", hhem_pert, test_ids))
    rows.append(("HHEM-2.1-Open (threshold 0.5)", "RAGTruth", "grounded", hhem_rag, None))
    rag_labels = _ragtruth_labels(rt)
    for model, dataset, check, records, items in rows:
        if records is None:
            out.append(f"| {model} | {dataset} | {check} | {PENDING} | | | |")
            continue
        labels = pset.labels if dataset == "perturbations" else rag_labels
        ids = items if items is not None else [lab.item_id for lab in rag_labels]
        a = analysis.agreement(records, labels, check, ids)
        out.append(
            f"| {model} | {dataset} | {check} | {a.n} | {_pct(a.tpr)} | {_pct(a.tnr)} | "
            f"{a.kappa.estimate:.2f} ({a.kappa.low:.2f} to {a.kappa.high:.2f}) |"
        )
    out.append("")
    out += _by_type(results, types, test_ids, hhem_pert)
    return "\n".join(out) + "\n"


def _ragtruth_labels(items: Sequence[Any]) -> list[Any]:
    from llm_eval_harness.labeling import LabelRecord, items_fingerprint

    ids = [i.item_id for i in items]
    fp = items_fingerprint(ids)
    return [
        LabelRecord(
            item_id=i.item_id,
            labels={"grounded": i.grounded},
            labeler="ragtruth-annotators",
            sampling="uniform",
            seed=0,
            items_sha256=fp,
            n_target=len(ids),
            created_at="2024-01-01T00:00:00+00:00",
        )
        for i in items
    ]


def _by_type(
    results: Path, types: dict[str, str], test_ids: Sequence[str], hhem: list[EvalRecord] | None
) -> list[str]:
    from rag_support_assistant.perturb import TYPES

    wanted = set(test_ids)
    counts = {t: sum(1 for i in wanted if types[i] == t) for t in TYPES}
    out = [
        "**Flagged as not grounded, by perturbation type (test).** For the four error types "
        "this is the catch rate, for original and paraphrase the false alarm rate.",
        "",
        "| Judge | " + " | ".join(f"{t} (n={counts[t]})" for t in TYPES) + " |",
        "|---|" + "---|" * len(TYPES),
    ]
    sources: list[tuple[str, list[EvalRecord] | None]] = [
        (spec.model, _load(judge_path(key, "perturbations/test", results)))
        for key, spec in JUDGES.items()
    ]
    sources.append(("HHEM-2.1-Open", hhem))
    for name, records in sources:
        if records is None:
            out.append(f"| {name} | " + " | ".join([PENDING] * len(TYPES)) + " |")
            continue
        rates = analysis.detection_by_type([r for r in records if r.item_id in wanted], types)
        out.append(
            f"| {name} | "
            + " | ".join(_pct(rates[t]) if t in rates else "n/a" for t in TYPES)
            + " |"
        )
    out.append("")
    return out


# ---------------------------------------------------------------- live-only cells


def extras_section(results: Path) -> str:
    out = ["**Live-only cells.**", ""]
    ctx = results / "contextual" / "selection.json"
    if ctx.exists():
        info = json.loads(ctx.read_text(encoding="utf-8"))
        out.append(f"- Contextual retrieval ({info['name']}), test nDCG@10: {info['summary']}.")
    else:
        out.append(f"- Contextual retrieval (one cell, LLM-written context per chunk): {PENDING}.")
    ragas_dir = results / "ragas"
    if ragas_dir.exists() and any(ragas_dir.glob("*.jsonl")):
        for path in sorted(ragas_dir.glob("*.jsonl")):
            records = read_records(path)
            parts = []
            for metric in sorted({m for r in records for m in r.scores}):
                vals = [float(r.scores[metric]) for r in records if metric in r.scores]
                vals = [v for v in vals if not math.isnan(v)]
                if vals:
                    parts.append(f"{metric} {sum(vals) / len(vals):.3f} (n={len(vals)})")
            out.append(f"- Ragas on {path.stem}: {', '.join(parts)}.")
    else:
        out.append(f"- Ragas 0.4.3 faithfulness and context recall on two configs: {PENDING}.")
    live = results / "live_run.json"
    if live.exists():
        info = json.loads(live.read_text(encoding="utf-8"))
        out.append(
            f"- Live run spend: ${info['spent_usd']:.2f} of a ${info['cap_usd']:.2f} cap "
            f"({info['calls']} calls)."
        )
    out.append("")
    return "\n".join(out)


def render(results: Path = REPO_ROOT / "results") -> str:
    selection_path = results / "retrieval" / "selection.json"
    arms: list[str] = [FULL_CONTEXT]
    names = {FULL_CONTEXT: "Full context (no retrieval)"}
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        arms = [*selection["generation_configs"], FULL_CONTEXT]
        names |= {s: info["name"] for s, info in selection["configs"].items()}
    body = [
        retrieval_section(results),
        generation_section(results, arms, names),
        judge_section(results),
        extras_section(results),
    ]
    note = (
        "> Every number shown was produced offline by `make demo` from committed results, "
        "including the replies of the live model runs."
    )
    if any(PENDING in part for part in body):
        note = (
            f'> Rows marked "{PENDING}" need Azure model calls, which have not been made '
            "yet. Every number shown was produced offline by `make demo` from committed results."
        )
    return "\n".join([note, "", *body])


def render_cost(results: Path = REPO_ROOT / "results") -> str:
    """The run estimate from `eval estimate`, the cap, and actual spend once a live run exists."""
    from rag_support_assistant.clients import DEFAULT_CAP_USD, DEPLOYMENT_TPM
    from rag_support_assistant.estimate import as_table, estimate

    est = estimate(results)
    total_minutes = sum(line.minutes_at_quota(DEPLOYMENT_TPM) for line in est.lines)
    used = {line.model for line in est.lines}
    lines = [
        "Estimated before the live run by `eval estimate`, which builds the requests the run "
        'sends and prices them at list prices. "Expected" assumes about 4 bytes per token, '
        "typical output lengths and the full-context prefix served from the prompt cache after "
        'the first call. "Worst case" is what `DollarCap` reserves per call (one token per '
        "input byte plus `max_output_tokens`), the bound it enforces. Answer judging uses the "
        "reference answer as a stand-in answer, since the estimate runs before any answers exist.",
        "",
        as_table(est),
        "",
        '"Minutes at default quota" is the least wall time the rate limiter allows at the day-1 '
        "capacities ("
        + ", ".join(f"{m} {t // 1000}K" for m, t in DEPLOYMENT_TPM.items() if m in used)
        + " tokens per minute), using 80% of each. Stages run one after another, so at those "
        f"quotas the run would take about {total_minutes / 60:.0f} hours, and luna's default "
        "quota can't fit the full-context prompt at all. The live run used raised capacities "
        "(see Limitations).",
        "",
        f"`make eval-live` runs with a hard cap of ${DEFAULT_CAP_USD:.2f} (`make eval-live "
        "CAP=...` to change it), a bit more than twice the expected spend. A refused call stops "
        "the run, and cached calls cost nothing when it is started again.",
        "",
    ]
    runs = results / "live_runs.jsonl"
    if runs.exists():
        rows = [json.loads(line) for line in runs.read_text(encoding="utf-8").splitlines() if line]
        spent = sum(r.get("spent_usd", 0.0) for r in rows)
        errors = sum(r.get("charged_for_errors_usd", 0.0) for r in rows)
        failed = sum(r.get("failed_calls", 0) for r in rows)
        calls = sum(r.get("calls", 0) for r in rows)
        lines.append(
            f"Actual spend: ${spent - errors:.2f} token-priced over {calls} calls in {len(rows)} "
            f"runs. DollarCap's accounting shows ${spent:.2f}, which includes ${errors:.2f} "
            f"reserved for {failed} failed, retried calls that Azure doesn't bill."
        )
        lines += _cache_line(results)
    else:
        lines.append(f"Actual spend: {PENDING}.")
    return "\n".join(lines) + "\n"


def _cache_line(results: Path) -> list[str]:
    """Prompt caching on the full-context arm, from its generation records."""
    gen = _load(generation_path(FULL_CONTEXT, "test", results))
    if not gen:
        return []
    cached = sum(int(r.meta.get("cached_input_tokens", 0)) for r in gen)
    total = sum(r.tokens_in for r in gen)
    cold = [r for r in gen if not r.meta.get("cached_input_tokens")]
    warm = sum(r.cost_usd for r in gen) / len(gen) * 1000
    line = (
        f"Prompt caching served {cached / 1e6:.2f}M of the full-context arm's {total / 1e6:.2f}M "
        f"input tokens: {len(gen) - len(cold)} of {len(gen)} calls hit the cache"
    )
    if cold:
        cold_cost = sum(r.cost_usd for r in cold) / len(cold)
        line += (
            f", and a cold call cost ${cold_cost:.4f}, about ${cold_cost * 1000:.1f} per 1,000 "
            f"answers against ${warm:.2f} warm"
        )
    return ["", line + "."]


def update_readme(
    readme: Path = REPO_ROOT / "README.md", results: Path = REPO_ROOT / "results"
) -> bool:
    changed = write_section(readme, "results", render(results))
    return write_section(readme, "cost", render_cost(results)) or changed


__all__ = ["PRIMARY_METRIC", "render", "render_cost", "update_readme"]
