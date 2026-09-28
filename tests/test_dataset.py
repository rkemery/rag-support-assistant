"""Tests for the dataset. The first test runs the full validator. The rest
check that the validator and the policy model catch the errors they claim to."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import validate  # noqa: E402
from bank_sim import PolicyError, apply_diff, run  # noqa: E402
from facts import load_facts, typed_values  # noqa: E402

FACTS = load_facts()
TV = typed_values(FACTS)
SEED = json.loads((ROOT / "agents" / "bank_seed.json").read_text())
TASKS = validate.read_jsonl(ROOT / "agents" / "tasks.jsonl")


def test_validator_passes():
    report = validate.validate()
    assert report.errors == []


@pytest.mark.parametrize("text", ["costs $4.50 a month", "a 2.2% fee", "within 17 days", "in 6 business days",
                                  "3 to 11 business days", "before 5:30 PM ET", "for 48 hours"])
def test_number_check_flags_values_not_in_facts(text):
    assert validate.numbers_not_in_facts(text, TV)


@pytest.mark.parametrize("text", ["costs $5.00 a month", "a 1% fee", "within 120 days", "in 10 business days",
                                  "7 to 10 business days", "before 4:00 PM ET", "for 24 hours", "$2,500 a day"])
def test_number_check_accepts_values_in_facts(text):
    assert validate.numbers_not_in_facts(text, TV) == []


def test_diff_with_wrong_from_value_does_not_apply():
    task = next(t for t in TASKS if t["gold_final_state"]["updated"])
    bad = copy.deepcopy(task["gold_final_state"])
    field = next(iter(bad["updated"][0]["changes"]))
    bad["updated"][0]["changes"][field]["from"] = "not-the-seed-value"
    with pytest.raises(ValueError):
        apply_diff(SEED, bad)


def test_diff_with_dangling_foreign_key_does_not_apply():
    bad = {"updated": [], "inserted": [{"table": "disputes", "match": {"customer_id": "cus_001", "txn_id": "txn_missing"}}]}
    with pytest.raises(ValueError):
        apply_diff(SEED, bad)


# Wrong moves an agent could make on the do-nothing and escalate tasks. Each
# must be rejected by the policy model, otherwise those gold labels are weak.
WRONG_MOVES = [
    ("cus_009", [("freeze_card", {"card_id": "card_009p"}), ("open_dispute", {"txn_id": "txn_009_volta", "reason": "unauthorized"})]),
    ("cus_012", [("open_dispute", {"txn_id": "txn_012_orbital", "reason": "merchant_not_as_described"})]),
    ("cus_013", [("freeze_card", {"card_id": "card_013p"}), ("open_dispute", {"txn_id": "txn_013_aurora", "reason": "unauthorized"}),
                 ("open_dispute", {"txn_id": "txn_013_skyline", "reason": "unauthorized"})]),
    ("cus_014", [("open_dispute", {"txn_id": "txn_014_ember", "reason": "merchant_not_as_described"})]),
    ("cus_015", [("open_dispute", {"txn_id": "txn_015_bright", "reason": "merchant_not_received"})]),
    ("cus_015", [("close_account", {})]),
    ("cus_020", [("refund_fee", {"txn_id": "txn_020_fxfee", "basis": "goodwill"})]),
    ("cus_023", [("refund_fee", {"txn_id": "txn_023_wirefee", "basis": "goodwill"})]),
    ("cus_024", [("close_account", {})]),
    ("cus_007", [("unfreeze_card", {"card_id": "card_007p"})]),
    ("cus_029", [("freeze_card", {"card_id": "card_029p"})]),
    ("cus_008", [("open_dispute", {"txn_id": "txn_008_vntx", "reason": "unauthorized"})]),
    ("cus_019", [("refund_fee", {"txn_id": "txn_019_atmfee", "basis": "error"})]),
    ("cus_019", [("open_dispute", {"txn_id": "txn_019_cof2", "reason": "duplicate"})]),
    ("cus_018", [("open_dispute", {"txn_id": "txn_018_sep", "reason": "cancelled_recurring"})]),
    ("cus_022", [("change_plan", {"new_plan": "plus", "effective": "immediate"})]),
    ("cus_001", [("freeze_card", {"card_id": "card_002p"})]),
]


@pytest.mark.parametrize("customer,moves", WRONG_MOVES)
def test_policy_model_rejects_wrong_moves(customer, moves):
    actions = [{"action": n, "args": a} for n, a in moves]
    with pytest.raises(PolicyError):
        run(SEED, FACTS, customer, actions)


def test_build_is_reproducible(tmp_path, monkeypatch):
    """Rebuilding from facts and scripts gives byte-identical data files."""
    import importlib

    tracked = ["corpus/articles.jsonl", "rag/questions.jsonl", "rag/questions_dev.jsonl",
               "rag/questions_test.jsonl", "agents/bank_seed.json", "agents/tasks.jsonl"]
    before = {p: (ROOT / p).read_bytes() for p in tracked}
    for mod in ("build_corpus", "build_questions", "build_agents"):
        sys.modules.pop(mod, None)
        importlib.import_module(mod).main()
    after = {p: (ROOT / p).read_bytes() for p in tracked}
    changed = [p for p in tracked if before[p] != after[p]]
    for p in changed:
        (ROOT / p).write_bytes(before[p])
    assert changed == []
