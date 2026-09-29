"""`eval`: the command line for every offline and live step.

Offline (CPU, no keys): retrieval, contexts, build-validation, hhem,
export-snapshot, estimate, demo, verify-data.
Model calls: judge-dev, judge-freeze, run. Each takes --live or --replay.
--live needs AZURE_OPENAI_BASE_URL and a key or Entra ID, and every call goes
through CachedClient, RetryingClient, DollarCap (--cap) and a TPM limiter.
--replay reads the committed cache/ and never touches the network.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from rag_support_assistant.data import REPO_ROOT

RESULTS = REPO_ROOT / "results"
CACHE = REPO_ROOT / "cache"
EMBED_CACHE = REPO_ROOT / ".cache" / "embeddings"
STAGES = (
    "judge-dev",
    "freeze",
    "judge-test",
    "ragtruth",
    "generate",
    "judge-answers",
    "contextual",
    "ragas",
)


def _mode_args(p: argparse.ArgumentParser) -> None:
    group = p.add_mutually_exclusive_group()
    group.add_argument("--live", action="store_true", help="call Azure (needs env config)")
    group.add_argument("--replay", action="store_true", help="replay cache/ only (default)")
    p.add_argument("--cap", type=float, default=None, help="dollar cap for --live (default 8.00)")


def _stack(args: argparse.Namespace):
    from rag_support_assistant.clients import DEFAULT_CAP_USD, build_stack

    mode = "live" if args.live else "replay"
    return build_stack(mode, CACHE, cap_usd=args.cap or DEFAULT_CAP_USD)


def _log_run(stack, stages: list[str], started: str) -> None:
    if stack.mode != "live":
        return
    path = RESULTS / "live_runs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {"started": started, "stages": stages, **stack.summary()}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")


# ---------------------------------------------------------------- offline commands


def cmd_retrieval(args: argparse.Namespace) -> int:
    from rag_support_assistant.grid import run_grid, set_torch_threads

    set_torch_threads(args.threads)
    run = run_grid(RESULTS / "retrieval", EMBED_CACHE, strong=not args.no_strong_arm)
    for step in run.steps:
        print(f"{step['step']}: chose {step['chosen']} ({step['dev']})")
    print(f"generation configs: {run.generation_configs}")
    print(f"compute: {run.compute['wall_s']} s wall, {run.compute['cpu_s']} s CPU")
    return 0


def cmd_contexts(args: argparse.Namespace) -> int:
    from rag_support_assistant.grid import set_torch_threads
    from rag_support_assistant.pipeline import generation_arms, write_contexts

    set_torch_threads(args.threads)
    arms = [a for a in generation_arms() if a != "full-context"]
    for slug, n in write_contexts(arms).items():
        print(f"contexts for {slug}: {n} questions")
    return 0


def cmd_build_validation(args: argparse.Namespace) -> int:
    from rag_support_assistant.validation import build_perturbation_set, build_ragtruth_subset

    pset = build_perturbation_set()
    print(
        f"perturbations: {len(pset.items)} items "
        f"({len(pset.split.dev)} dev, {len(pset.split.test)} test)"
    )
    if not args.skip_ragtruth:
        manifest = build_ragtruth_subset(args.ragtruth_raw, download=not args.no_download)
        print(f"RAGTruth subset: {manifest['n_items']} items, sha256 {manifest['subset_sha256']}")
    return 0


def cmd_hhem(args: argparse.Namespace) -> int:
    from llm_eval_harness import read_records, write_records

    from rag_support_assistant.data import cluster_key, load_articles, load_questions
    from rag_support_assistant.grid import set_torch_threads
    from rag_support_assistant.hhem import HHEM, HHEMInput, score_items
    from rag_support_assistant.judging import evidence_ids
    from rag_support_assistant.pipeline import generation_arms, generation_path
    from rag_support_assistant.validation import load_perturbation_set, load_ragtruth_subset

    set_torch_threads(args.threads)
    model = HHEM()
    articles = {a.article_id: a for a in load_articles()}
    questions = {q.question_id: q for q in load_questions()}
    out = RESULTS / "hhem"
    start = time.process_time()

    def premise(reference: str | None, ids: list[str]) -> str:
        parts = [reference] if reference else []
        parts += [f"{articles[i].title}\n{articles[i].body}" for i in ids]
        return "\n\n".join(parts)

    if not args.answers_only:
        pset = load_perturbation_set()
        items = [
            HHEMInput(
                item.item_id,
                premise(item.reference_answer, list(item.gold_article_ids)),
                item.answer,
                cluster_key(questions[item.question_id]),
            )
            for item in pset.by_split("test")
        ]
        write_records(
            out / "perturbations_test.jsonl", score_items(model, items, "hhem/perturbations/test")
        )
        print(f"HHEM on {len(items)} perturbation test items")
        rt = [HHEMInput(i.item_id, i.passages, i.response, None) for i in load_ragtruth_subset()]
        write_records(out / "ragtruth.jsonl", score_items(model, rt, "hhem/ragtruth"))
        print(f"HHEM on {len(rt)} RAGTruth items")
    for arm in generation_arms():
        path = generation_path(arm, "test")
        if not path.exists():
            continue
        items = []
        for r in read_records(path):
            if r.error is None and not r.scores.get("abstained", True):
                q = questions[r.item_id]
                ids = evidence_ids(q.gold_article_ids, r.meta.get("citations", []), articles)
                items.append(
                    HHEMInput(
                        r.item_id, premise(q.reference_answer, ids), r.meta["answer"], r.cluster
                    )
                )
        write_records(
            out / "answers" / f"{arm}.jsonl", score_items(model, items, f"hhem/answers/{arm}")
        )
        print(f"HHEM on {len(items)} answers from {arm}")
    print(f"HHEM CPU time: {time.process_time() - start:.0f} s")
    return 0


def cmd_export_snapshot(args: argparse.Namespace) -> int:
    from rag_support_assistant.snapshot import export

    files = export(REPO_ROOT / "snapshot", RESULTS, EMBED_CACHE)
    for name, digest in files.items():
        print(f"snapshot/{name} {digest}")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    from rag_support_assistant.estimate import as_table, estimate

    est = estimate(RESULTS)
    print(as_table(est))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from rag_support_assistant.readme import update_readme

    changed = update_readme()
    print("README.md results section " + ("updated" if changed else "already up to date"))
    return 0


def cmd_verify_data(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from sync_data import SyncError, verify

    from rag_support_assistant.snapshot import verify as verify_snapshot
    from rag_support_assistant.validation import load_perturbation_set, load_ragtruth_subset

    try:
        names = verify()
    except SyncError as exc:
        print(f"data: {exc}", file=sys.stderr)
        return 1
    load_ragtruth_subset()
    pset = load_perturbation_set()
    if (REPO_ROOT / "snapshot" / "MANIFEST.json").exists():
        verify_snapshot()
    print(
        f"data OK: {len(names)} Tallowbrook files, {len(pset.items)} perturbations, "
        "RAGTruth subset, snapshot"
    )
    return 0


# ---------------------------------------------------------------- model-calling commands


def _print_agreement(judge_key: str, split: str) -> None:
    from llm_eval_harness import read_records

    from rag_support_assistant import analysis
    from rag_support_assistant.pipeline import judge_path
    from rag_support_assistant.validation import load_perturbation_set

    pset = load_perturbation_set()
    records = read_records(judge_path(judge_key, f"perturbations/{split}"))
    ids = list(pset.split.dev if split == "dev" else pset.split.test)
    for check in ("grounded", "correct"):
        a = analysis.agreement(records, pset.labels, check, ids)
        print(
            f"  {judge_key} {split} {check}: n={a.n} TPR={a.tpr.estimate:.3f} "
            f"TNR={a.tnr.estimate:.3f} kappa={a.kappa.estimate:.2f}"
        )


def cmd_judge_dev(args: argparse.Namespace) -> int:
    from rag_support_assistant.judging import JUDGES
    from rag_support_assistant.pipeline import run_perturbation_judging

    stack = _stack(args)
    started = datetime.now(UTC).isoformat()
    try:
        for key in JUDGES:
            run_perturbation_judging(stack.client, key, "dev")
            _print_agreement(key, "dev")
    finally:
        _log_run(stack, ["judge-dev"], started)
        print(json.dumps(stack.summary()))
    return 0


def cmd_judge_freeze(args: argparse.Namespace) -> int:
    from rag_support_assistant.judging import freeze

    payload = freeze(RESULTS, args.note)
    print(f"froze judges: {payload['fingerprints']}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from rag_support_assistant import contextual
    from rag_support_assistant.judging import JUDGES, check_frozen, freeze, frozen_path
    from rag_support_assistant.pipeline import (
        FULL_CONTEXT,
        generation_arms,
        run_answer_judging,
        run_generation,
        run_perturbation_judging,
        run_ragtruth_judging,
    )

    stages = args.stages or list(STAGES)
    unknown = sorted(set(stages) - set(STAGES))
    if unknown:
        print(f"unknown stages {unknown}, expected some of {STAGES}", file=sys.stderr)
        return 2
    stack = _stack(args)
    client = stack.client
    started = datetime.now(UTC).isoformat()
    arms = generation_arms()
    try:
        if "judge-dev" in stages:
            for key in JUDGES:
                run_perturbation_judging(client, key, "dev")
                _print_agreement(key, "dev")
        if "freeze" in stages and not frozen_path(RESULTS).exists():
            freeze(RESULTS, "Frozen by `eval run` right after the first dev run.")
            print("froze the judges after the dev run")
        if set(stages) & {"judge-test", "ragtruth", "judge-answers"}:
            check_frozen(RESULTS)
        if "judge-test" in stages:
            for key in JUDGES:
                run_perturbation_judging(client, key, "test")
                _print_agreement(key, "test")
        if "ragtruth" in stages:
            for key in JUDGES:
                run_ragtruth_judging(client, key)
        if "generate" in stages:
            for arm in arms:
                print(f"generating answers: {arm}")
                run_generation(client, arm, "test")
        if "judge-answers" in stages:
            for key in JUDGES:
                for arm in arms:
                    print(f"judging answers: {key} on {arm}")
                    run_answer_judging(client, key, arm, "test")
        if "contextual" in stages:
            contextual.contextualize(client, RESULTS)
            info = contextual.run_cell(RESULTS)
            print(f"contextual cell: {info['summary']}")
        if "ragas" in stages:
            _run_ragas(client, [a for a in arms if a != FULL_CONTEXT][:2])
    finally:
        _log_run(stack, stages, started)
        print(json.dumps(stack.summary()))
    return 0


def _run_ragas(client, arms: list[str]) -> None:
    try:
        from rag_support_assistant.ragas_adapter import RagasScorer
    except ImportError as exc:
        print(f"skipping Ragas, the optional extra is not installed ({exc})")
        return
    from llm_eval_harness import read_records, write_records

    from rag_support_assistant.data import load_questions
    from rag_support_assistant.pipeline import generation_path, load_contexts

    questions = {q.question_id: q for q in load_questions("test")}
    scorer = RagasScorer(client)
    for arm in arms:
        contexts = load_contexts(arm)
        items = [
            {
                "item_id": r.item_id,
                "question": questions[r.item_id].question,
                "answer": r.meta["answer"],
                "contexts": [c["text"] for c in contexts[r.item_id]["chunks"]],
                "reference": questions[r.item_id].reference_answer,
                "cluster": r.cluster,
            }
            for r in read_records(generation_path(arm, "test"))
            if r.error is None and not r.scores["abstained"] and questions[r.item_id].answerable
        ]
        records = scorer.score_records(items, run_id=f"ragas/{arm}", config=arm)
        write_records(RESULTS / "ragas" / f"{arm}.jsonl", records)
        print(f"Ragas on {len(items)} answers from {arm}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("retrieval", help="run the retrieval grid (offline, CPU)")
    p.add_argument("--no-strong-arm", action="store_true", help="skip the Qwen3 rows")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads (default 2)")
    p.set_defaults(func=cmd_retrieval)

    p = sub.add_parser("contexts", help="freeze the top chunks of each generation config")
    p.add_argument("--threads", type=int, default=None)
    p.set_defaults(func=cmd_contexts)

    p = sub.add_parser("build-validation", help="build the perturbation set and RAGTruth subset")
    p.add_argument("--skip-ragtruth", action="store_true")
    p.add_argument("--no-download", action="store_true", help="use files already in --ragtruth-raw")
    p.add_argument("--ragtruth-raw", type=Path, default=REPO_ROOT / ".cache" / "ragtruth_raw")
    p.set_defaults(func=cmd_build_validation)

    p = sub.add_parser("hhem", help="HHEM on the validation sets and any answers (offline)")
    p.add_argument("--answers-only", action="store_true")
    p.add_argument("--threads", type=int, default=None)
    p.set_defaults(func=cmd_hhem)

    p = sub.add_parser("export-snapshot", help="write snapshot/ for the agents repo")
    p.set_defaults(func=cmd_export_snapshot)

    p = sub.add_parser("estimate", help="estimate the cost of a full live run")
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("demo", help="regenerate the README results section (offline)")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("verify-data", help="check every committed dataset against its hashes")
    p.set_defaults(func=cmd_verify_data)

    p = sub.add_parser("judge-dev", help="judge the dev perturbations (prompt tuning loop)")
    _mode_args(p)
    p.set_defaults(func=cmd_judge_dev)

    p = sub.add_parser("judge-freeze", help="freeze the judges before anything is judged on test")
    p.add_argument("--note", default="Frozen by hand after tuning on the dev perturbations.")
    p.set_defaults(func=cmd_judge_freeze)

    p = sub.add_parser("run", help="the full model-calling pipeline on test")
    _mode_args(p)
    p.add_argument("--stages", nargs="+", default=None, help=f"subset of {', '.join(STAGES)}")
    p.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    from llm_eval_harness import BudgetExceeded, CacheMiss

    from rag_support_assistant.judging import JudgeNotFrozen

    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except CacheMiss as exc:
        print(f"eval: replay stopped at a request the cache has not seen. {exc}", file=sys.stderr)
    except BudgetExceeded as exc:
        print(f"eval: dollar cap reached, nothing past it was sent. {exc}", file=sys.stderr)
    except JudgeNotFrozen as exc:
        print(f"eval: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
