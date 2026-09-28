"""Reference semantics for the support tools used in agents/tasks.jsonl.

This is a small, deterministic model of what each write action does to
agents/bank_seed.json, plus a policy checker that encodes the support rules in
facts/policies.yaml. The dataset builder uses it to compute each task's
gold_final_state from its gold_actions, and scripts/validate.py uses it to
re-check them. Consumers (the triage-agent repo) are free to implement the
tools however they like, as long as the resulting state matches.
"""

from __future__ import annotations

import copy
import datetime as dt
from calendar import monthrange
from typing import Any

PK = {
    "customers": "customer_id",
    "accounts": "account_id",
    "cards": "card_id",
    "transactions": "txn_id",
    "disputes": "dispute_id",
    "escalations": "escalation_id",
}
# fields that the tool implementation is free to choose, so they are not
# part of an insert's match
UNMATCHED_FIELDS = {"date", "opened_date", "created_date", "summary", "description", "test_pan", "last4", "exp"}
FK_FIELDS = {
    "customer_id": "customers",
    "account_id": "accounts",
    "card_id": "cards",
    "txn_id": "transactions",
    "related_txn_id": "transactions",
    "replaces_card_id": "cards",
}
ACTIONS = {
    "freeze_card": ["card_id"],
    "unfreeze_card": ["card_id"],
    "report_card_lost_stolen": ["card_id", "reason"],
    "order_replacement_card": ["card_id", "reason", "shipping"],
    "open_dispute": ["txn_id", "reason"],
    "refund_fee": ["txn_id", "basis"],
    "change_plan": ["new_plan", "effective"],
    "close_account": [],
    "escalate_to_human": ["queue", "summary"],
}
QUEUES = ["fraud", "disputes", "verification", "account_services", "complaints"]
PLAN_ORDER = ["basic", "plus", "premium"]
FEE_TYPE_TO_KEY = {
    "out_of_network_atm": "out_of_network_atm_fee_usd",
    "international_atm": "international_atm_fee_usd",
    "card_replacement": "card_replacement_fee_usd",
    "expedited_shipping": "expedited_shipping_fee_usd",
    "domestic_wire_out": "domestic_wire_out_fee_usd",
    "international_wire_out": "international_wire_out_fee_usd",
    "retail_cash_deposit": "retail_cash_deposit_fee_usd",
    "plan_fee": "monthly_fee_usd",
}


class PolicyError(Exception):
    pass


def _d(s: str | dt.date) -> dt.date:
    return s if isinstance(s, dt.date) else dt.date.fromisoformat(s)


def statement_date(txn_date: dt.date) -> dt.date:
    return txn_date.replace(day=monthrange(txn_date.year, txn_date.month)[1])


def add_month(d: dt.date) -> dt.date:
    y, m = (d.year + (d.month // 12), d.month % 12 + 1)
    return d.replace(year=y, month=m, day=min(d.day, monthrange(y, m)[1]))


class Bank:
    """Mutable bank state plus the tool semantics."""

    def __init__(self, seed: dict, facts: dict, customer_id: str):
        self.state = copy.deepcopy(seed)
        self.facts = facts
        self.customer_id = customer_id
        self.as_of = _d(seed["as_of"])
        self._n = 0
        self.disputed_total = 0.0

    # ------------------------------------------------------------ lookup
    def get(self, table: str, rid: str) -> dict:
        for r in self.state[table]:
            if r[PK[table]] == rid:
                return r
        raise PolicyError(f"{table} {rid} not found")

    def customer(self) -> dict:
        return self.get("customers", self.customer_id)

    def main_account(self) -> dict:
        for a in self.state["accounts"]:
            if a["customer_id"] == self.customer_id and a["type"] == "main":
                return a
        raise PolicyError("no main account")

    def plan_fees(self, plan: str) -> dict:
        ver = self.facts["fees"]["current_version"]
        return self.facts["fees"]["versions"][f"v{ver}"][plan]

    def _new_id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}_new{self._n:02d}"

    def _own(self, rec: dict) -> None:
        if rec.get("customer_id") != self.customer_id:
            rid = next((rec[k] for k in ("card_id", "txn_id", "account_id") if k in rec), "record")
            raise PolicyError(f"{rid} does not belong to {self.customer_id}")

    def _charge(self, amount: float, fee_type: str, description: str) -> None:
        if amount <= 0:
            return
        acc = self.main_account()
        self.state["transactions"].append(dict(
            txn_id=self._new_id("txn"), account_id=acc["account_id"], customer_id=self.customer_id,
            date=self.as_of.isoformat(), status="posted", type="fee", fee_type=fee_type,
            amount=-round(amount, 2), description=description,
        ))
        acc["balance"] = round(acc["balance"] - amount, 2)

    def _require_active_customer(self) -> None:
        if self.customer()["status"] != "active":
            raise PolicyError("customer is not active, support can't act, escalate instead")

    # ------------------------------------------------------------ actions
    def freeze_card(self, card_id: str) -> None:
        self._require_active_customer()
        card = self.get("cards", card_id)
        self._own(card)
        if card["status"] != "active":
            raise PolicyError(f"can only freeze an active card, {card_id} is {card['status']}")
        card["status"] = "frozen"

    def unfreeze_card(self, card_id: str) -> None:
        self._require_active_customer()
        card = self.get("cards", card_id)
        self._own(card)
        if card["status"] != "frozen":
            raise PolicyError(f"can only unfreeze a frozen card, {card_id} is {card['status']}")
        card["status"] = "active"

    def report_card_lost_stolen(self, card_id: str, reason: str) -> None:
        self._require_active_customer()
        if reason not in ("lost", "stolen"):
            raise PolicyError("reason must be lost or stolen")
        card = self.get("cards", card_id)
        self._own(card)
        if card["status"] not in ("active", "frozen"):
            raise PolicyError(f"can't report a card that is {card['status']}")
        card["status"] = reason

    def order_replacement_card(self, card_id: str, reason: str, shipping: str) -> None:
        self._require_active_customer()
        card = self.get("cards", card_id)
        self._own(card)
        if card["kind"] != "physical":
            raise PolicyError("only physical cards are replaced by mail")
        if reason in ("lost", "stolen") and card["status"] != reason:
            raise PolicyError("report the card lost or stolen before ordering a replacement")
        if reason == "damaged" and card["status"] not in ("active", "frozen"):
            raise PolicyError("damaged replacement needs a working card")
        if shipping not in ("standard", "expedited"):
            raise PolicyError("shipping must be standard or expedited")
        plan = self.customer()["plan"]
        fees = self.plan_fees(plan)
        self.state["cards"].append(dict(
            card_id=self._new_id("card"), customer_id=self.customer_id, account_id=card["account_id"],
            kind="physical", material="metal" if plan == "premium" else "plastic",
            status="shipping", replaces_card_id=card_id, shipping=shipping,
        ))
        self._charge(float(fees["card_replacement_fee_usd"]), "card_replacement", "Card replacement fee")
        if shipping == "expedited":
            self._charge(float(fees["expedited_shipping_fee_usd"]), "expedited_shipping", "Expedited shipping fee")

    def open_dispute(self, txn_id: str, reason: str) -> None:
        self._require_active_customer()
        d = self.facts["disputes"]
        if reason not in d["reasons"]:
            raise PolicyError(f"unknown dispute reason {reason}")
        txn = self.get("transactions", txn_id)
        self._own(txn)
        if txn["status"] != "posted":
            raise PolicyError("pending transactions can't be disputed")
        if any(x["txn_id"] == txn_id and x["status"] == "open" for x in self.state["disputes"]):
            raise PolicyError("transaction already has an open dispute")
        tdate = _d(txn["date"])
        if reason == "unauthorized":
            window = d["ach_unauthorized_window_days"] if txn["type"].startswith("ach") else d["unauthorized_window_days"]
            if (self.as_of - statement_date(tdate)).days > window:
                raise PolicyError("outside the unauthorized dispute window")
            if txn.get("card_id"):
                if self.get("cards", txn["card_id"])["status"] not in ("frozen", "lost", "stolen", "fraud_lock"):
                    raise PolicyError("freeze or report the card before an unauthorized dispute")
        elif reason == "atm_cash_not_dispensed":
            if (self.as_of - tdate).days > self.facts["atm"]["cash_not_dispensed_dispute_window_days"]:
                raise PolicyError("outside the ATM dispute window")
        else:
            ver = d["current_version"]
            if (self.as_of - tdate).days > d["versions"][f"v{ver}"]["merchant_dispute_window_days"]:
                raise PolicyError("outside the merchant dispute window")
        amount = abs(txn["amount"])
        self.disputed_total += amount
        if self.disputed_total > d["support_max_dispute_usd"]:
            raise PolicyError("above the support dispute limit, escalate to disputes")
        self.state["disputes"].append(dict(
            dispute_id=self._new_id("dsp"), customer_id=self.customer_id, txn_id=txn_id,
            reason=reason, amount=round(amount, 2), status="open", opened_date=self.as_of.isoformat(),
        ))

    def _fee_should_be_zero(self, txn: dict, plan: str) -> bool:
        key = FEE_TYPE_TO_KEY.get(txn["fee_type"])
        if key is None:
            return False
        val = self.plan_fees(plan).get(key)
        return val is not None and float(val) == 0.0

    def _is_duplicate_fee(self, txn: dict) -> bool:
        return any(
            t is not txn and t.get("type") == "fee" and t.get("fee_type") == txn["fee_type"]
            and t.get("related_txn_id") and t.get("related_txn_id") == txn.get("related_txn_id")
            for t in self.state["transactions"]
        )

    def refund_fee(self, txn_id: str, basis: str) -> None:
        self._require_active_customer()
        r = self.facts["refunds"]
        txn = self.get("transactions", txn_id)
        self._own(txn)
        if txn["type"] != "fee":
            raise PolicyError("not a fee")
        if any(t.get("type") == "fee_refund" and t.get("related_txn_id") == txn_id for t in self.state["transactions"]):
            raise PolicyError("fee already refunded")
        cust = self.customer()
        amount = abs(txn["amount"])
        if basis == "error":
            if not (self._fee_should_be_zero(txn, cust["plan"]) or self._is_duplicate_fee(txn)):
                raise PolicyError("fee was not charged in error")
        elif basis == "goodwill":
            if txn["fee_type"] not in r["goodwill_eligible_fees"]:
                raise PolicyError("fee type not eligible for goodwill")
            if amount > r["goodwill_max_fee_usd"]:
                raise PolicyError("fee above the goodwill maximum")
            for t in self.state["transactions"]:
                if (t.get("type") == "fee_refund" and t.get("basis") == "goodwill" and t["customer_id"] == self.customer_id
                        and (self.as_of - _d(t["date"])).days < r["goodwill_per_rolling_days"]):
                    raise PolicyError("goodwill refund already used in the last 12 months")
        elif basis == "cooling_off":
            if txn["fee_type"] != "plan_fee":
                raise PolicyError("cooling-off only applies to plan fees")
            first = cust.get("first_paid_plan_charge_date")
            if first != txn["date"]:
                raise PolicyError("not the first paid plan charge")
            if (self.as_of - _d(txn["date"])).days > self.facts["plan_billing"]["cooling_off_days"]:
                raise PolicyError("outside the cooling-off period")
        else:
            raise PolicyError(f"unknown refund basis {basis}")
        acc = self.get("accounts", txn["account_id"])
        self.state["transactions"].append(dict(
            txn_id=self._new_id("txn"), account_id=acc["account_id"], customer_id=self.customer_id,
            date=self.as_of.isoformat(), status="posted", type="fee_refund", amount=round(amount, 2),
            related_txn_id=txn_id, basis=basis, description="Fee refund",
        ))
        acc["balance"] = round(acc["balance"] + amount, 2)

    def change_plan(self, new_plan: str, effective: str) -> None:
        self._require_active_customer()
        cust = self.customer()
        old = cust["plan"]
        if new_plan not in PLAN_ORDER or new_plan == old:
            raise PolicyError("invalid plan change")
        up = PLAN_ORDER.index(new_plan) > PLAN_ORDER.index(old)
        if up:
            if effective != "immediate":
                raise PolicyError("upgrades take effect immediately")
            fee = float(self.plan_fees(new_plan)["monthly_fee_usd"])
            if self.main_account()["balance"] < fee:
                raise PolicyError("not enough money for the new plan fee")
            cust["plan"] = new_plan
            cust["scheduled_plan_change"] = None
            cust["next_billing_date"] = add_month(self.as_of).isoformat()
            if cust.get("first_paid_plan_charge_date") is None:
                cust["first_paid_plan_charge_date"] = self.as_of.isoformat()
            self._charge(fee, "plan_fee", f"Plan fee: {new_plan.capitalize()}")
        else:
            if effective == "next_billing_date":
                cust["scheduled_plan_change"] = {"plan": new_plan, "effective_date": cust["next_billing_date"]}
            elif effective == "immediate":
                refunded = any(
                    t.get("type") == "fee_refund" and t.get("basis") == "cooling_off" and t["customer_id"] == self.customer_id
                    for t in self.state["transactions"]
                )
                if not refunded or new_plan != "basic":
                    raise PolicyError("immediate downgrades only happen with a cooling-off refund, to Basic")
                cust["plan"] = new_plan
                cust["next_billing_date"] = None
                cust["scheduled_plan_change"] = None
            else:
                raise PolicyError("effective must be immediate or next_billing_date")

    def close_account(self) -> None:
        self._require_active_customer()
        cid = self.customer_id
        accts = [a for a in self.state["accounts"] if a["customer_id"] == cid]
        if any(abs(a["balance"]) > 0.004 for a in accts):
            raise PolicyError("support can only close an account with a $0.00 balance")
        if any(t["customer_id"] == cid and t["status"] == "pending" for t in self.state["transactions"]):
            raise PolicyError("pending transactions")
        if any(x["customer_id"] == cid and x["status"] == "open" for x in self.state["disputes"]):
            raise PolicyError("open dispute")
        self.customer()["status"] = "closed"
        for a in accts:
            a["status"] = "closed"
        for c in self.state["cards"]:
            if c["customer_id"] == cid and c["status"] in ("active", "frozen", "shipping", "inactive"):
                c["status"] = "cancelled"

    def escalate_to_human(self, queue: str, summary: str = "") -> None:
        if queue not in QUEUES:
            raise PolicyError(f"unknown queue {queue}")
        self.state["escalations"].append(dict(
            escalation_id=self._new_id("esc"), customer_id=self.customer_id, queue=queue,
            status="open", created_date=self.as_of.isoformat(), summary=summary,
        ))

    # ------------------------------------------------------------ driver
    def apply(self, action: dict) -> None:
        name = action["action"]
        if name not in ACTIONS:
            raise PolicyError(f"unknown action {name}")
        args = action.get("args", {})
        missing = [a for a in ACTIONS[name] if a not in args and a != "summary"]
        if missing:
            raise PolicyError(f"{name} missing args {missing}")
        getattr(self, name)(**args)


def run(seed: dict, facts: dict, customer_id: str, actions: list[dict]) -> dict:
    bank = Bank(seed, facts, customer_id)
    for a in actions:
        bank.apply(a)
    return bank.state


def diff(seed: dict, state: dict) -> dict:
    updated, inserted = [], []
    for table, pk in PK.items():
        before = {r[pk]: r for r in seed[table]}
        for rec in state[table]:
            rid = rec[pk]
            if rid in before:
                changes = {}
                for k in sorted(set(before[rid]) | set(rec)):
                    if before[rid].get(k) != rec.get(k):
                        changes[k] = {"from": before[rid].get(k), "to": rec.get(k)}
                if changes:
                    updated.append({"table": table, "id": rid, "changes": changes})
            else:
                match = {k: v for k, v in rec.items() if k != pk and k not in UNMATCHED_FIELDS}
                inserted.append({"table": table, "match": match})
        if len({r[pk] for r in state[table]}) != len(state[table]):
            raise ValueError(f"duplicate ids in {table}")
    return {"updated": updated, "inserted": inserted}


def apply_diff(seed: dict, d: dict) -> dict:
    """Apply a gold_final_state diff to the seed. Raises ValueError if it doesn't apply cleanly."""
    state = copy.deepcopy(seed)
    for u in d["updated"]:
        table, rid = u["table"], u["id"]
        if table not in PK:
            raise ValueError(f"unknown table {table}")
        recs = [r for r in state[table] if r[PK[table]] == rid]
        if len(recs) != 1:
            raise ValueError(f"{table} {rid} not found in seed")
        rec = recs[0]
        for k, ch in u["changes"].items():
            if rec.get(k) != ch["from"]:
                raise ValueError(f"{table} {rid}.{k} is {rec.get(k)!r}, diff expects {ch['from']!r}")
            rec[k] = ch["to"]
    for i, ins in enumerate(d["inserted"]):
        table = ins["table"]
        if table not in PK:
            raise ValueError(f"unknown table {table}")
        rec = dict(ins["match"])
        rec[PK[table]] = f"__inserted_{i}"
        state[table].append(rec)
    # referential integrity over the resulting state
    ids = {t: {r[PK[t]] for r in state[t]} for t in PK}
    for table in PK:
        for rec in state[table]:
            for fk, target in FK_FIELDS.items():
                if fk == PK[table]:
                    continue
                if rec.get(fk) is not None and rec[fk] not in ids[target]:
                    raise ValueError(f"{table} {rec[PK[table]]}: {fk}={rec[fk]} does not exist")
    return state
