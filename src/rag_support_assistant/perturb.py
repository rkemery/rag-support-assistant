"""Judge-validation set built by code from the facts file and the reference answers.

For every answerable question the set holds:

- `original`: the reference answer as written. Labels: correct, grounded.
- `paraphrase`: the reference reworded by meaning-preserving rules (contractions
  expanded, "$5.00" written "$5", a neutral lead-in). Labels: correct, grounded.
- one copy per applicable error type, each with exactly one injected error:
  - `wrong_number`: one amount, rate, day count or time replaced by a value
    that appears nowhere in the facts file for that unit.
  - `wrong_plan`: a plan-specific value replaced by another plan's value for
    the same fact, while the answer still names the original plan.
  - `superseded`: a current value replaced by the value the superseded
    version of that policy had (old fees, APYs, dispute window, ACH cutoff).
  - `unsupported_claim`: one fabricated sentence appended. Every such
    sentence carries a marker phrase that is checked to be absent from the
    whole corpus.
  Labels: not grounded. The three value swaps are also not correct. For
  `unsupported_claim` the `correct` label is left unset, since the reference
  facts are all still there.

So every label is known by construction. Items inherit the dataset's split
(dev questions make dev items), the judge prompt is tuned on dev only, and
TPR and TNR are reported on test.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rag_support_assistant.data import Article, Question

PLANS = ("basic", "plus", "premium")
PLAN_NAMES = {"basic": "Basic", "plus": "Plus", "premium": "Premium"}
SEED = 20260928

ERROR_TYPES = ("wrong_number", "wrong_plan", "superseded", "unsupported_claim")
TYPES = ("original", "paraphrase", *ERROR_TYPES)

# Numbers the perturbations may touch, with their unit.
_MENTION = re.compile(
    r"(?P<usd>\$[0-9][0-9,]*(?:\.[0-9]{2})?)"
    r"|(?P<pct>[0-9]+(?:\.[0-9]+)?%)"
    r"|(?P<time>\b[0-9]{1,2}:[0-9]{2} (?:AM|PM)\b)"
    r"|(?P<days>\b[0-9]+(?= (?:business |calendar )?days?\b))"
)


@dataclass(frozen=True)
class Mention:
    unit: str  # usd, pct, time, days
    text: str
    start: int
    end: int
    sentence: str

    @property
    def value(self) -> float | str:
        return parse_value(self.unit, self.text)


def parse_value(unit: str, text: str) -> float | str:
    if unit == "usd":
        return float(text.lstrip("$").replace(",", ""))
    if unit == "pct":
        return float(text.rstrip("%"))
    if unit == "days":
        return float(text)
    return text  # time, compared as text


def format_like(unit: str, value: float | str, original: str) -> str:
    """Write `value` in the same style as `original` ($5.00 vs $5, 0.50% vs 1%)."""
    if unit == "time":
        return str(value)
    assert isinstance(value, float)
    if unit == "usd":
        decimals = 2 if "." in original else 0
        if decimals == 0 and not value.is_integer():
            decimals = 2
        return f"${value:,.{decimals}f}"
    if unit == "pct":
        body = original.rstrip("%")
        decimals = len(body.split(".")[1]) if "." in body else 0
        if decimals == 0 and not value.is_integer():
            decimals = len(f"{value:g}".split(".")[1])
        return f"{value:.{decimals}f}%"
    return f"{int(value)}" if value.is_integer() else f"{value:g}"


def find_mentions(text: str) -> list[Mention]:
    mentions = []
    for match in _MENTION.finditer(text):
        unit = match.lastgroup
        assert unit is not None
        mentions.append(
            Mention(unit, match.group(), match.start(), match.end(), _sentence(text, match.start()))
        )
    return mentions


def _sentence(text: str, pos: int) -> str:
    start = max(text.rfind(". ", 0, pos), text.rfind("? ", 0, pos), -1) + 1
    end = text.find(". ", pos)
    return text[start : len(text) if end == -1 else end + 1].strip()


def _get(facts: dict[str, Any], path: str) -> Any:
    node: Any = facts
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


@dataclass(frozen=True)
class Slot:
    """A plan-indexed fact that answers state, with the words that show it is being stated."""

    name: str
    unit: str
    current: str  # path with {plan}
    context: str  # regex that must match the mention's sentence
    old: str | None = None  # superseded version's path with {plan}

    def value(self, facts: dict[str, Any], plan: str, *, old: bool = False) -> float | str | None:
        path = self.old if old else self.current
        if path is None:
            return None
        raw = _get(facts, path.format(plan=plan))
        if raw is None or isinstance(raw, bool):
            return None
        if self.unit == "time":
            return str(raw)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None  # "unlimited" and other words


FEES_V2, FEES_V1 = "fees.versions.v2.{plan}", "fees.versions.v1.{plan}"
SLOTS: tuple[Slot, ...] = (
    Slot(
        "monthly_fee",
        "usd",
        f"{FEES_V2}.monthly_fee_usd",
        r"month|costs?",
        f"{FEES_V1}.monthly_fee_usd",
    ),
    Slot(
        "out_of_network_atm_fee",
        "usd",
        f"{FEES_V2}.out_of_network_atm_fee_usd",
        r"ATM|out-of-network|withdrawal",
        f"{FEES_V1}.out_of_network_atm_fee_usd",
    ),
    Slot(
        "fx_fee",
        "pct",
        f"{FEES_V2}.fx_fee_pct",
        r"(?i)foreign|exchange|abroad|currenc",
        f"{FEES_V1}.fx_fee_pct",
    ),
    Slot(
        "international_atm_fee",
        "usd",
        f"{FEES_V2}.international_atm_fee_usd",
        r"(?i)international|abroad",
    ),
    Slot("card_replacement_fee", "usd", f"{FEES_V2}.card_replacement_fee_usd", r"(?i)replac"),
    Slot(
        "expedited_shipping_fee",
        "usd",
        f"{FEES_V2}.expedited_shipping_fee_usd",
        r"(?i)expedited|shipping",
    ),
    Slot("domestic_wire_fee", "usd", f"{FEES_V2}.domestic_wire_out_fee_usd", r"(?i)wire"),
    Slot("international_wire_fee", "usd", f"{FEES_V2}.international_wire_out_fee_usd", r"(?i)wire"),
    Slot(
        "savings_apy",
        "pct",
        "savings.versions.v2.{plan}_apy_pct",
        r"APY",
        "savings.versions.v1.{plan}_apy_pct",
    ),
    Slot("cushion_limit", "usd", "overdraft.cushion.{plan}_limit_usd", r"(?i)cushion|cover"),
    Slot(
        "daily_card_limit", "usd", "limits.{plan}.daily_card_spend_limit_usd", r"(?i)spend|purchase"
    ),
    Slot(
        "daily_atm_limit",
        "usd",
        "limits.{plan}.daily_atm_withdrawal_limit_usd",
        r"(?i)ATM|withdraw|cash",
    ),
    Slot(
        "daily_ach_limit",
        "usd",
        "limits.{plan}.daily_ach_out_limit_usd",
        r"(?i)ACH|external|transfer",
    ),
    Slot(
        "monthly_ach_limit",
        "usd",
        "limits.{plan}.monthly_ach_out_limit_usd",
        r"(?i)ACH|external|transfer",
    ),
    Slot("daily_brookpay_limit", "usd", "limits.{plan}.daily_brookpay_limit_usd", r"BrookPay"),
    Slot("daily_wire_limit", "usd", "limits.{plan}.daily_domestic_wire_limit_usd", r"(?i)wire"),
)

# Policies that are not per plan but have a superseded numeric version.
GLOBAL_SUPERSEDED: tuple[tuple[str, str, str, str], ...] = (
    # unit, current path, old path, context regex
    ("days", "disputes.versions.v2.merchant_dispute_window_days",
     "disputes.versions.v1.merchant_dispute_window_days", r"(?i)dispute|transaction date"),
    ("time", "transfers.ach.versions.v2.cutoff_time_et",
     "transfers.ach.versions.v1.cutoff_time_et", r"(?i)ACH|transfer|cutoff"),
)  # fmt: skip

# Superseded rules that are words, not numbers: (pattern, the old version's wording, slot).
TEXT_SUPERSEDED: tuple[tuple[str, str, str], ...] = (
    (r"Travel notices were retired\.",
     "Set a travel notice in the app under Travel mode at least 2 days before you leave.",
     "travel_notice"),
    (r"You don't need a travel notice\.",
     "Set a travel notice in the app under Travel mode at least 2 days before you leave.",
     "travel_notice"),
    (r"report it lost or stolen in the app",
     "call the card services line at +1 555 0142 to report it lost or stolen",
     "lost_stolen_channel"),
)  # fmt: skip

# Fabricated sentences. The marker phrase of each is asserted absent from the corpus.
UNSUPPORTED_CLAIMS: tuple[tuple[str, str], ...] = (
    ("You'll also earn 1% cash back on every debit card purchase.", "cash back"),
    ("Premium members also get free airport lounge access.", "lounge"),
    ("You can also sort this out in person at any Tallowbrook branch.", "branch"),
    (
        "We'll also mail you a paper confirmation letter within 3 business days.",
        "confirmation letter",
    ),
    ("Members who have been with us for more than five years get this for free.", "five years"),
    ("A one-time $10.00 processing fee also applies.", "processing fee"),
    ("You'll also get double reward points on purchases that month.", "reward points"),
    ("Our partner credit union can also help with this.", "credit union"),
)

_PARAPHRASE_RULES: tuple[tuple[str, str], ...] = (
    (r"\bdoesn't\b", "does not"),
    (r"\bdon't\b", "do not"),
    (r"\bcan't\b", "cannot"),
    (r"\bisn't\b", "is not"),
    (r"\baren't\b", "are not"),
    (r"\bwon't\b", "will not"),
    (r"\bwasn't\b", "was not"),
    (r"\bhasn't\b", "has not"),
    (r"\bIt's\b", "It is"),
    (r"\bit's\b", "it is"),
    (r"\bYou're\b", "You are"),
    (r"\byou're\b", "you are"),
    (r"\bYou'll\b", "You will"),
    (r"\byou'll\b", "you will"),
    (r"\bThere's\b", "There is"),
    (r"\bthere's\b", "there is"),
    (r"\bthat's\b", "that is"),
    (r"\byou've\b", "you have"),
    (r"\bright away\b", "immediately"),
    (r"\ba month\b", "per month"),
    (r"\ba day\b", "per day"),
    (r"\busually\b", "typically"),
    (r"(\$[0-9][0-9,]*)\.00\b", r"\1"),
)
_LEAD_INS = (
    "According to the Tallowbrook help center: ",
    "Here is what the help center says. ",
    "Short answer: ",
)


@dataclass(frozen=True)
class Perturbation:
    item_id: str
    question_id: str
    split: str
    type: str
    question: str
    answer: str
    reference_answer: str
    gold_article_ids: tuple[str, ...]
    labels: dict[str, bool]  # grounded always, correct unless the type leaves it unset
    edit: dict[str, str] = field(default_factory=dict)


def paraphrase(text: str, rng: random.Random) -> str:
    out = text
    for pattern, repl in _PARAPHRASE_RULES:
        out = re.sub(pattern, repl, out)
    changed = out != text
    if not changed or rng.random() < 0.5:
        out = rng.choice(_LEAD_INS) + out
    return out


def _replace(text: str, mention: Mention, new: str) -> str:
    return text[: mention.start] + new + text[mention.end :]


def _values_by_unit(facts: dict[str, Any]) -> dict[str, set[float | str]]:
    """Every value in the facts file, grouped by the unit its key suffix names."""
    out: dict[str, set[float | str]] = {"usd": set(), "pct": set(), "days": set(), "time": set()}

    def walk(node: Any, key: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, str(k))
        elif isinstance(node, list):
            for v in node:
                walk(v, key)
        elif key.endswith("_usd") and isinstance(node, int | float):
            out["usd"].add(float(node))
        elif key.endswith("_pct") and isinstance(node, int | float):
            out["pct"].add(float(node))
        elif key.endswith("_days") and isinstance(node, int | float):
            out["days"].add(float(node))
        elif key.endswith("_time_et") and isinstance(node, str):
            out["time"].add(node)

    walk(facts)
    return out


def _fabricated_value(mention: Mention, taken: set[float | str]) -> float | str | None:
    """A nearby value that appears nowhere in the facts file for this unit."""
    value = mention.value
    if mention.unit == "time":
        hour, rest = str(value).split(":")
        suffix = rest.split(" ")[1]
        for h in (int(hour) + 2, int(hour) + 1, int(hour) + 3):
            candidate = f"{(h - 1) % 12 + 1}:00 {suffix}"
            if candidate not in taken:
                return candidate
        return None
    assert isinstance(value, float)
    steps = {
        "usd": (value * 2 if value else 7.0, value + 7, value + 3, value * 3),
        "pct": (value + 0.25, value * 2 + 0.5, value + 1.75),
        "days": (value + 15, value * 2 + 1, value + 4, value + 9),
    }[mention.unit]
    for candidate in steps:
        candidate = round(float(candidate), 2)
        if candidate != value and candidate not in taken and candidate > 0:
            return candidate
    return None


def _plans_in(sentence: str, question: Question) -> list[str]:
    named = [p for p in PLANS if re.search(rf"\b{PLAN_NAMES[p]}\b", sentence)]
    if question.plan and question.plan not in named:
        named.append(question.plan)
    return named


def _slot_matches(
    mention: Mention, question: Question, facts: dict[str, Any]
) -> list[tuple[Slot, str]]:
    """(slot, plan) pairs whose current value equals this mention, in context, for a named plan."""
    hits = []
    context = f"{mention.sentence} {question.question}"
    for slot in SLOTS:
        if slot.unit != mention.unit or not re.search(slot.context, context):
            continue
        for plan in _plans_in(mention.sentence, question):
            if slot.value(facts, plan) == mention.value:
                hits.append((slot, plan))
    return hits


def _wrong_plan(
    answer: str, question: Question, facts: dict[str, Any], rng: random.Random
) -> tuple[str, dict[str, str]] | None:
    options = []
    for mention in find_mentions(answer):
        for slot, plan in _slot_matches(mention, question, facts):
            for other in PLANS:
                value = slot.value(facts, other)
                if other != plan and value is not None and value != mention.value:
                    options.append((mention, slot, plan, other, value))
    if not options:
        return None
    mention, slot, plan, other, value = rng.choice(options)
    new = format_like(mention.unit, value, mention.text)
    edit = {"from": mention.text, "to": new, "slot": slot.name, "plan": plan, "value_of": other}
    return _replace(answer, mention, new), edit


def _superseded(
    answer: str, question: Question, facts: dict[str, Any], rng: random.Random
) -> tuple[str, dict[str, str]] | None:
    options = []
    mentions = find_mentions(answer)
    present = {m.value for m in mentions}
    for mention in mentions:
        for slot, plan in _slot_matches(mention, question, facts):
            old = slot.value(facts, plan, old=True)
            if old is not None and old != mention.value and old not in present:
                options.append((mention, f"{slot.name}:{plan}", old))
        for unit, current, old_path, context in GLOBAL_SUPERSEDED:
            if unit != mention.unit or not re.search(
                context, f"{mention.sentence} {question.question}"
            ):
                continue
            now, old = _get(facts, current), _get(facts, old_path)
            now_value = str(now) if unit == "time" else float(now)
            old_value = str(old) if unit == "time" else float(old)
            if now_value == mention.value and old_value not in present:
                options.append((mention, current.split(".")[-1], old_value))
    for pattern, old_wording, slot_name in TEXT_SUPERSEDED:
        match = re.search(pattern, answer)
        if match:
            options.append((match, slot_name, old_wording))
    if not options:
        return None
    target, slot_name, old = rng.choice(options)
    if isinstance(target, Mention):
        new = format_like(target.unit, old, target.text)
        return _replace(answer, target, new), {"from": target.text, "to": new, "slot": slot_name}
    new_text = answer[: target.start()] + str(old) + answer[target.end() :]
    return new_text, {"from": target.group(), "to": str(old), "slot": slot_name}


def _wrong_number(
    answer: str, taken: dict[str, set[float | str]], rng: random.Random
) -> tuple[str, dict[str, str]] | None:
    mentions = find_mentions(answer)
    rng.shuffle(mentions)
    for mention in mentions:
        value = _fabricated_value(mention, taken[mention.unit])
        if value is not None:
            new = format_like(mention.unit, value, mention.text)
            return _replace(answer, mention, new), {"from": mention.text, "to": new}
    return None


def check_claims_unsupported(articles: Iterable[Article]) -> None:
    """Raise if any fabricated claim's marker phrase occurs in the corpus."""
    corpus = "\n".join(f"{a.title}\n{a.body}" for a in articles).lower()
    found = [marker for _, marker in UNSUPPORTED_CLAIMS if marker in corpus]
    if found:
        raise ValueError(f"fabricated-claim markers found in the corpus: {found}")


def _item(
    question: Question,
    kind: str,
    answer: str,
    labels: dict[str, bool],
    edit: dict[str, str] | None = None,
) -> Perturbation:
    return Perturbation(
        item_id=f"{question.question_id}:{kind}",
        question_id=question.question_id,
        split=question.split,
        type=kind,
        question=question.question,
        answer=answer,
        reference_answer=question.reference_answer,
        gold_article_ids=question.gold_article_ids,
        labels=labels,
        edit=edit or {},
    )


def _for_question(
    question: Question, facts: dict[str, Any], taken: dict[str, set[float | str]], seed: int
) -> list[Perturbation]:
    # One RNG per question, so adding a question never changes another's items.
    digest = hashlib.sha256(f"{seed}/{question.question_id}".encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    ref = question.reference_answer
    good = {"correct": True, "grounded": True}
    bad = {"correct": False, "grounded": False}
    items = [
        _item(question, "original", ref, good),
        _item(question, "paraphrase", paraphrase(ref, rng), good),
    ]
    # The order of these calls fixes how each one draws from rng. Keep it.
    swaps = (
        ("wrong_number", _wrong_number(ref, taken, rng)),
        ("wrong_plan", _wrong_plan(ref, question, facts, rng)),
        ("superseded", _superseded(ref, question, facts, rng)),
    )
    for kind, made in swaps:
        if made is not None and made[0] != ref:
            items.append(_item(question, kind, made[0], bad, made[1]))
    claim, marker = rng.choice(UNSUPPORTED_CLAIMS)
    items.append(
        _item(
            question,
            "unsupported_claim",
            f"{ref} {claim}",
            {"grounded": False},
            {"appended": claim, "marker": marker},
        )
    )
    return items


def build(
    questions: Sequence[Question],
    facts: dict[str, Any],
    articles: Sequence[Article],
    seed: int = SEED,
) -> list[Perturbation]:
    """Every perturbation for every answerable question, deterministic for a given seed."""
    check_claims_unsupported(articles)
    taken = _values_by_unit(facts)
    items: list[Perturbation] = []
    for question in questions:
        if question.answerable:
            items.extend(_for_question(question, facts, taken, seed))
    return items


def write_jsonl(path: Path, items: Sequence[Perturbation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            row = asdict(item)
            row["gold_article_ids"] = list(item.gold_article_ids)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Perturbation]:
    items = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                row["gold_article_ids"] = tuple(row["gold_article_ids"])
                items.append(Perturbation(**row))
    return items
