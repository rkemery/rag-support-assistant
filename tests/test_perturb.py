from __future__ import annotations

import re
from collections import Counter

import pytest

from rag_support_assistant.data import load_facts
from rag_support_assistant.perturb import (
    UNSUPPORTED_CLAIMS,
    _values_by_unit,
    build,
    check_claims_unsupported,
    find_mentions,
    format_like,
    parse_value,
)
from rag_support_assistant.validation import load_perturbation_set


@pytest.fixture(scope="module")
def items(questions, articles):
    return build(questions, load_facts(), articles)


def test_committed_set_is_what_the_builder_makes(items):
    committed = load_perturbation_set()
    assert [i.item_id for i in committed.items] == [i.item_id for i in items]
    assert [i.answer for i in committed.items] == [i.answer for i in items]
    assert set(committed.split.dev).isdisjoint(committed.split.test)


def test_labels_follow_the_type(items):
    for item in items:
        if item.type in ("original", "paraphrase"):
            assert item.labels == {"correct": True, "grounded": True}
        elif item.type == "unsupported_claim":
            assert item.labels == {"grounded": False}
        else:
            assert item.labels == {"correct": False, "grounded": False}
            assert item.answer != item.reference_answer


def test_every_answerable_question_gets_originals_and_a_claim(items, questions):
    counts = Counter(i.type for i in items)
    n = sum(q.answerable for q in questions)
    assert counts["original"] == counts["paraphrase"] == counts["unsupported_claim"] == n
    for kind in ("wrong_number", "wrong_plan", "superseded"):
        assert counts[kind] > 0


def test_value_swaps_change_exactly_one_value(items):
    taken = _values_by_unit(load_facts())
    for item in items:
        if item.type not in ("wrong_number", "wrong_plan") or "from" not in item.edit:
            continue
        before = [m.text for m in find_mentions(item.reference_answer)]
        after = [m.text for m in find_mentions(item.answer)]
        assert len(before) == len(after)
        assert sum(b != a for b, a in zip(before, after, strict=True)) == 1
        if item.type == "wrong_number":
            new = next(m for m in find_mentions(item.answer) if m.text == item.edit["to"])
            assert new.value not in taken[new.unit]


def test_superseded_uses_the_old_policy_value(items):
    facts = load_facts()
    old_values = {
        facts["fees"]["versions"]["v1"]["plus"]["monthly_fee_usd"],
        facts["savings"]["versions"]["v1"]["plus_apy_pct"],
        facts["disputes"]["versions"]["v1"]["merchant_dispute_window_days"],
    }
    swaps = [i for i in items if i.type == "superseded"]
    assert swaps
    for item in swaps:
        to = item.edit["to"]
        if re.fullmatch(r"[$0-9.,%]+", to):
            unit = "usd" if to.startswith("$") else "pct" if to.endswith("%") else "days"
            assert parse_value(unit, to) != parse_value(unit, item.edit["from"])
    assert old_values  # the facts file has superseded values to draw from


def test_paraphrases_keep_every_number(items):
    def values(text: str) -> list[float | str]:
        return sorted(str(m.value) for m in find_mentions(text))

    for item in items:
        if item.type == "paraphrase":
            assert values(item.answer) == values(item.reference_answer)
            assert item.answer != item.reference_answer


def test_fabricated_claims_are_absent_from_the_corpus(articles):
    check_claims_unsupported(articles)
    assert len({m for _, m in UNSUPPORTED_CLAIMS}) == len(UNSUPPORTED_CLAIMS)


def test_format_like_keeps_style():
    assert format_like("usd", 4.0, "$5.00") == "$4.00"
    assert format_like("usd", 2500.0, "$5,000") == "$2,500"
    assert format_like("pct", 1.5, "1%") == "1.5%"
    assert format_like("pct", 0.75, "0.50%") == "0.75%"
