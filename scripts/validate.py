"""Validate the dataset against facts/policies.yaml and its own invariants.

Run: uv run python scripts/validate.py
Exits non-zero and lists every problem if any check fails.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bank_sim import PolicyError, apply_diff, diff, run  # noqa: E402
from facts import ROOT, load_facts, typed_values  # noqa: E402

PLANS = {"basic", "plus", "premium"}
UNANSWERABLE_TYPES = {"out_of_scope", "false_premise", "near_miss"}
DIFFICULTIES = {"easy", "medium", "hard"}
MIN_WORDS, MAX_WORDS = 150, 500

MONEY = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)")
PCT = re.compile(r"(\d+(?:\.\d+)?)\s?%")
DAY_RANGE = re.compile(r"(\d+)\s*(?:to|-|–)\s*(\d+)\s+(business\s+|calendar\s+)?days?\b", re.I)
DAYS = re.compile(r"(\d+)[\s-]+(business[\s-]+|calendar[\s-]+)?days?\b", re.I)
HOURS = re.compile(r"(\d+)[\s-]+hours?\b", re.I)
TIME = re.compile(r"\b(\d{1,2}:\d{2}\s?[AP]M)\b", re.I)


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def numbers_not_in_facts(text: str, tv: dict[str, set]) -> list[str]:
    """Every money amount, percentage, day count, hour count and clock time in
    `text` must appear in the facts file with the same unit."""
    bad = []
    for m in MONEY.finditer(text):
        if round(float(m.group(1).replace(",", "")), 2) not in tv["usd"]:
            bad.append(m.group(0))
    for m in PCT.finditer(text):
        if round(float(m.group(1)), 2) not in tv["pct"]:
            bad.append(m.group(0))
    for m in DAY_RANGE.finditer(text):
        kind = "business_days" if m.group(3) and "business" in m.group(3).lower() else "days"
        for g in (1, 2):
            if int(m.group(g)) not in tv[kind]:
                bad.append(m.group(0))
    for m in DAYS.finditer(text):
        kind = "business_days" if m.group(2) and "business" in m.group(2).lower() else "days"
        if int(m.group(1)) not in tv[kind]:
            bad.append(m.group(0))
    for m in HOURS.finditer(text):
        if int(m.group(1)) not in tv["hours"]:
            bad.append(m.group(0))
    for m in TIME.finditer(text):
        if re.sub(r"\s+", " ", m.group(1).upper()) not in tv["time_et"]:
            bad.append(m.group(0))
    return bad


def norm_q(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.passed: list[str] = []

    def check(self, name: str, problems: list[str]) -> None:
        if problems:
            self.errors += [f"{name}: {p}" for p in problems]
        else:
            self.passed.append(name)


def check_articles(r: Report, facts: dict, tv: dict) -> dict[str, dict]:
    arts = read_jsonl(ROOT / "corpus" / "articles.jsonl")
    by_id = {a["article_id"]: a for a in arts}
    as_of = facts["meta"]["as_of_date"]
    fields = ["article_id", "title", "section", "plans", "effective_date", "version", "supersedes", "body"]

    r.check("articles: unique ids", [a for a in by_id if sum(x["article_id"] == a for x in arts) > 1])
    r.check("articles: required fields", [
        f"{a.get('article_id')} missing {f}" for a in arts for f in fields if f not in a])
    probs = []
    for a in arts:
        if not a["plans"] or not set(a["plans"]) <= PLANS:
            probs.append(f"{a['article_id']} plans {a['plans']}")
        if dt.date.fromisoformat(a["effective_date"]) > as_of:
            probs.append(f"{a['article_id']} effective after as_of_date")
        if not isinstance(a["version"], int) or a["version"] < 1:
            probs.append(f"{a['article_id']} version {a['version']}")
    r.check("articles: field values", probs)
    probs = []
    for a in arts:
        s = a["supersedes"]
        if s is None:
            continue
        if s not in by_id:
            probs.append(f"{a['article_id']} supersedes missing {s}")
            continue
        old = by_id[s]
        if old["effective_date"] >= a["effective_date"]:
            probs.append(f"{a['article_id']} is not newer than {s}")
        if old["version"] >= a["version"]:
            probs.append(f"{a['article_id']} version not above {s}")
    superseded = [a["supersedes"] for a in arts if a["supersedes"]]
    probs += [f"{s} superseded twice" for s in set(superseded) if superseded.count(s) > 1]
    r.check("articles: supersedes points to a real, older article", probs)
    r.check("articles: every number matches facts (money, %, days, hours, times)", [
        f"{a['article_id']}: {x}" for a in arts for x in numbers_not_in_facts(a["body"], tv)])
    r.check(f"articles: body length {MIN_WORDS} to {MAX_WORDS} words", [
        f"{a['article_id']} has {len(a['body'].split())} words" for a in arts
        if not MIN_WORDS <= len(a["body"].split()) <= MAX_WORDS])
    r.check("articles: count about 150", [] if 140 <= len(arts) <= 160 else [f"{len(arts)} articles"])
    return by_id


def check_questions(r: Report, tv: dict, articles: dict[str, dict]) -> None:
    qs = read_jsonl(ROOT / "rag" / "questions.jsonl")
    superseded = {a["supersedes"] for a in articles.values() if a["supersedes"]}
    fields = ["question_id", "split", "question", "gold_article_ids", "reference_answer", "answerable",
              "unanswerable_type", "plan", "difficulty"]
    r.check("questions: required fields", [
        f"{q.get('question_id')} missing {f}" for q in qs for f in fields if f not in q])
    ids = [q["question_id"] for q in qs]
    r.check("questions: unique ids", sorted({i for i in ids if ids.count(i) > 1}))
    dev = [q for q in qs if q["split"] == "dev"]
    test = [q for q in qs if q["split"] == "test"]
    r.check("questions: 200 total, 50 dev, 150 test", [] if (len(qs), len(dev), len(test)) == (200, 50, 150)
            else [f"total {len(qs)}, dev {len(dev)}, test {len(test)}"])
    una = [q for q in qs if not q["answerable"]]
    probs = [] if len(una) == 40 else [f"{len(una)} unanswerable"]
    for split, rows in (("dev", dev), ("test", test)):
        types = {q["unanswerable_type"] for q in rows if not q["answerable"]}
        if types != UNANSWERABLE_TYPES:
            probs.append(f"{split} unanswerable types {sorted(t for t in types if t)}")
    r.check("questions: exactly 40 unanswerable, all three types in both splits", probs)
    probs = []
    for q in qs:
        if q["answerable"] and q["unanswerable_type"] is not None:
            probs.append(f"{q['question_id']} answerable with a type")
        if not q["answerable"] and q["unanswerable_type"] not in UNANSWERABLE_TYPES:
            probs.append(f"{q['question_id']} bad unanswerable_type")
        if q["plan"] is not None and q["plan"] not in PLANS:
            probs.append(f"{q['question_id']} plan {q['plan']}")
        if q["difficulty"] not in DIFFICULTIES:
            probs.append(f"{q['question_id']} difficulty {q['difficulty']}")
        if q["split"] not in ("dev", "test"):
            probs.append(f"{q['question_id']} split {q['split']}")
    r.check("questions: field values", probs)
    probs = []
    for q in qs:
        for g in q["gold_article_ids"]:
            if g not in articles:
                probs.append(f"{q['question_id']} gold {g} not in corpus")
        if q["answerable"] and not q["gold_article_ids"]:
            probs.append(f"{q['question_id']} answerable with no gold")
        if q["unanswerable_type"] in ("out_of_scope", "near_miss") and q["gold_article_ids"]:
            probs.append(f"{q['question_id']} {q['unanswerable_type']} should have no gold")
        if q["unanswerable_type"] == "false_premise" and not q["gold_article_ids"]:
            probs.append(f"{q['question_id']} false premise needs a refuting article")
        if q["unanswerable_type"] == "false_premise" and set(q["gold_article_ids"]) & superseded:
            probs.append(f"{q['question_id']} false premise refuted by a superseded article")
    r.check("questions: every gold_article_id exists and matches answerability", probs)
    seen: dict[str, str] = {}
    probs = []
    for q in qs:
        k = norm_q(q["question"])
        if k in seen:
            probs.append(f"{q['question_id']} duplicates {seen[k]}")
        seen[k] = q["question_id"]
    r.check("questions: no question text duplicated within or across splits", probs)
    r.check("questions: numbers in reference answers match facts", [
        f"{q['question_id']}: {x}" for q in qs for x in numbers_not_in_facts(q["reference_answer"], tv)])
    probs = []
    for split in ("dev", "test"):
        path = ROOT / "rag" / f"questions_{split}.jsonl"
        rows = read_jsonl(path) if path.exists() else None
        if rows != [q for q in qs if q["split"] == split]:
            probs.append(f"{path.name} doesn't match questions.jsonl")
    r.check("questions: per-split files match questions.jsonl", probs)


def check_agents(r: Report, facts: dict, articles: dict[str, dict]) -> None:
    seed = json.loads((ROOT / "agents" / "bank_seed.json").read_text())
    tasks = read_jsonl(ROOT / "agents" / "tasks.jsonl")
    cust_ids = [c["customer_id"] for c in seed["customers"]]
    r.check("seed: unique customer ids", sorted({c for c in cust_ids if cust_ids.count(c) > 1}))
    r.check("seed: about 30 customers", [] if 25 <= len(cust_ids) <= 35 else [f"{len(cust_ids)} customers"])
    probs = []
    for c in seed["cards"]:
        pan = c["test_pan"]
        digits = [int(x) for x in pan][::-1]
        total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits))
        if total % 10 or not pan.startswith("411111") or c["last4"] != pan[-4:]:
            probs.append(c["card_id"])
    for cu in seed["customers"]:
        if not cu["email"].endswith("@example.com") or not re.fullmatch(r"\+1 555 01\d\d", cu["phone"]):
            probs.append(cu["customer_id"])
    r.check("seed: identifiers are fake (411111 Luhn test PANs, example.com, 555-01xx)", probs)
    try:
        apply_diff(seed, {"updated": [], "inserted": []})
        r.check("seed: referential integrity", [])
    except ValueError as e:
        r.check("seed: referential integrity", [str(e)])

    ids = [t["task_id"] for t in tasks]
    r.check("tasks: 50 tasks with unique ids", ([] if len(tasks) == 50 else [f"{len(tasks)} tasks"])
            + sorted({i for i in ids if ids.count(i) > 1}))
    r.check("tasks: every customer_id exists", [
        f"{t['task_id']} {t['customer_id']}" for t in tasks if t["customer_id"] not in cust_ids])
    r.check("tasks: every policy_ref is a current article", [
        f"{t['task_id']} {p}" for t in tasks for p in t["policy_refs"]
        if p not in articles or any(a["supersedes"] == p for a in articles.values())])
    share = sum(t["needs_clarification"] for t in tasks) / max(len(tasks), 1)
    r.check("tasks: about 20% need clarification", [] if 0.15 <= share <= 0.25 else [f"{share:.0%}"])
    r.check("tasks: should_escalate matches an escalate_to_human action", [
        t["task_id"] for t in tasks
        if t["should_escalate"] != any(a["action"] == "escalate_to_human" for a in t["gold_actions"])])
    r.check("tasks: some tasks are do-nothing or refuse", [] if any(not t["gold_actions"] for t in tasks) else ["none"])
    probs_apply, probs_sim = [], []
    for t in tasks:
        try:
            apply_diff(seed, t["gold_final_state"])
        except ValueError as e:
            probs_apply.append(f"{t['task_id']}: {e}")
        try:
            state = run(seed, facts, t["customer_id"], t["gold_actions"])
            if diff(seed, state) != t["gold_final_state"]:
                probs_sim.append(f"{t['task_id']}: gold_actions don't produce gold_final_state")
        except PolicyError as e:
            probs_sim.append(f"{t['task_id']}: gold actions break policy: {e}")
    r.check("tasks: gold_final_state applies cleanly to bank_seed.json", probs_apply)
    r.check("tasks: gold_actions follow policy and reproduce gold_final_state", probs_sim)


def validate() -> Report:
    facts = load_facts()
    tv = typed_values(facts)
    r = Report()
    articles = check_articles(r, facts, tv)
    check_questions(r, tv, articles)
    check_agents(r, facts, articles)
    return r


def main() -> int:
    r = validate()
    for name in r.passed:
        print(f"ok    {name}")
    for e in r.errors:
        print(f"FAIL  {e}")
    print(f"\n{len(r.passed)} checks passed, {len(r.errors)} problems")
    return 1 if r.errors else 0


if __name__ == "__main__":
    sys.exit(main())
