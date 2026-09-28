"""Build agents/bank_seed.json and agents/tasks.jsonl.

The seed is a small fictional bank: 30 customers with accounts, cards, recent
transactions and disputes. Scenario transactions are written by hand, filler
transactions come from a seeded random generator. Every task lists its gold
actions, and its gold_final_state is computed by running those actions
through scripts/bank_sim.py, which also enforces the support policy.

Run: uv run python scripts/build_agents.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bank_sim import PolicyError, diff, run  # noqa: E402
from facts import ROOT, load_facts  # noqa: E402

FACTS = load_facts()
AS_OF = "2026-09-15"
RNG = random.Random(20260915)


def fee_amt(plan: str, key: str, ver: int = 2) -> float:
    return float(FACTS["fees"]["versions"][f"v{ver}"][plan][key])


# ------------------------------------------------------------------ fake identifiers

def luhn_complete(prefix: str, length: int = 16) -> str:
    body = prefix + "".join(str(RNG.randint(0, 9)) for _ in range(length - len(prefix) - 1))
    total = 0
    for i, ch in enumerate(reversed(body)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return body + str((10 - total % 10) % 10)


TOWNS = [("Larkspur Hollow", "OR"), ("Quillbury", "VT"), ("Fernmoor", "MN"), ("Ashgrove Falls", "NC"),
         ("Brindle Bay", "ME"), ("Copperfield Glen", "CO"), ("Marrowdale", "OH"), ("Tansy Crossing", "TX"),
         ("Wickerby", "WI"), ("Heronsgate", "PA")]
STREETS = ["Birchwood Ln", "Kettle Rd", "Mossbank Ave", "Old Mill St", "Saltmarsh Way", "Thimble Ct",
           "Harrow Pl", "Lantern Row", "Pebble Dr", "Orchard Loop"]

CUSTOMERS_SPEC = [
    # id, name, plan, balance, extra
    (1, "Marisol Quenby", "basic", 842.16, {}),
    (2, "Theo Varga-Lind", "plus", 1520.40, {}),
    (3, "Priya Castellan", "premium", 6310.00, {"savings": 22000.00}),
    (4, "Jonah Wexley", "basic", 310.55, {}),
    (5, "Adaeze Holloway", "plus", 975.00, {"virtual": True}),
    (6, "Rafael Montrose", "basic", 0.00, {"card_status": "frozen", "no_filler": True}),
    (7, "Ingrid Solvang", "premium", 2200.00, {"card_status": "fraud_lock"}),
    (8, "Dmitri Ashcombe", "plus", 1204.88, {}),
    (9, "Keiko Brightwater", "basic", 455.10, {}),
    (10, "Samuel Okonkwo-Reyes", "premium", 9800.00, {"savings": 41000.00}),
    (11, "Lucia Fennimore", "plus", 690.25, {}),
    (12, "Oliver Thistlewood", "basic", 612.00, {}),
    (13, "Nadia Kestrel", "premium", 15400.00, {}),
    (14, "Bartholomew Pike", "basic", 205.30, {}),
    (15, "Hana Delacroix", "plus", 0.00, {"no_filler": True}),
    (16, "Tobias Wren", "basic", 388.00, {}),
    (17, "Esme Carrow", "premium", 3050.00, {}),
    (18, "Wendell Marsh", "plus", 530.90, {}),
    (19, "Yara Boulton", "basic", 140.75, {}),
    (20, "Felix Amberly", "plus", 700.00, {}),
    (21, "Rosalind Tate", "basic", 0.00, {"no_filler": True}),
    (22, "Caspian Moore", "premium", 4480.00, {"savings": 12500.00}),
    (23, "Delphine Achterberg", "plus", 2900.00, {}),
    (24, "Gideon Faraday", "basic", 412.37, {}),
    (25, "Maribel Ostrander", "plus", 820.00, {}),
    (26, "Arjun Wellesley", "premium", 7120.00, {"virtual": True}),
    (27, "Constance Lyle", "basic", 260.00, {}),
    (28, "Emeka Hartwell", "plus", 1333.00, {}),
    (29, "Sofia Brandvold", "basic", 1250.00, {"status": "under_review"}),
    (30, "Quentin Ashdown", "premium", 5600.00, {}),
]

FILLER_MERCHANTS = [
    ("Pinecone Grocery", 18, 95), ("Lantern Coffee Co", 4, 9), ("Quickfuel 88", 25, 60),
    ("Harbor Books", 9, 40), ("Maple and Rye Bakery", 6, 22), ("Cobalt Cinema", 12, 30),
    ("Bluebell Pharmacy", 7, 45), ("Riverbend Hardware", 15, 120), ("Juniper Market", 20, 110),
    ("Oakline Transit", 3, 3), ("Ember Pizza", 14, 38), ("Fieldstone Fitness", 29, 29),
]


def cid(n: int) -> str:
    return f"cus_{n:03d}"


def build_seed() -> dict:
    customers, accounts, cards, txns, disputes = [], [], [], [], []
    txn_counter: dict[int, int] = {}

    def T(n: int, date: str, ttype: str, amount: float, description: str, *, status: str = "posted",
          card: str | None = "p", fee_type: str | None = None, country: str = "US", related: str | None = None,
          basis: str | None = None, account: str = "m", tid: str | None = None) -> str:
        txn_counter[n] = txn_counter.get(n, 0) + 1
        txn_id = tid or f"txn_{n:03d}_{txn_counter[n]:02d}"
        rec = dict(txn_id=txn_id, customer_id=cid(n), account_id=f"acc_{n:03d}{account}", date=date,
                   status=status, type=ttype, amount=round(amount, 2), description=description)
        if card:
            rec["card_id"] = f"card_{n:03d}{card}"
        if fee_type:
            rec["fee_type"] = fee_type
        if country != "US":
            rec["merchant_country"] = country
        if related:
            rec["related_txn_id"] = related
        if basis:
            rec["basis"] = basis
        txns.append(rec)
        return txn_id

    for n, name, plan, balance, extra in CUSTOMERS_SPEC:
        first, last = name.split(" ", 1)
        town, state = TOWNS[n % len(TOWNS)]
        paid = plan != "basic"
        bday = {5: 22, 22: 1, 28: 6}.get(n, (n % 20) + 5)
        last_charge = f"2026-09-{bday:02d}" if bday <= 15 else f"2026-08-{bday:02d}"
        next_billing = (f"2026-10-{bday:02d}" if bday <= 15 else f"2026-09-{bday:02d}") if paid else None
        first_paid = ("2026-09-06" if n == 28 else f"2025-{(n % 9) + 2:02d}-{bday:02d}") if paid else None
        customers.append(dict(
            customer_id=cid(n), full_name=name,
            email=f"{first.lower()}.{last.lower().replace('-', '')}@example.com",
            phone=f"+1 555 01{50 + n - 1:02d}",
            date_of_birth=f"{1962 + (n * 7) % 40}-{(n % 12) + 1:02d}-{(n * 3) % 27 + 1:02d}",
            address=f"{10 + n * 7} {STREETS[n % len(STREETS)]}, {town}, {state} 000{n:02d}",
            joined_date=f"2024-{(n % 12) + 1:02d}-{(n % 25) + 1:02d}",
            plan=plan,
            status=extra.get("status", "active"),
            kyc_status="under_review" if extra.get("status") == "under_review" else "verified",
            next_billing_date=next_billing,
            first_paid_plan_charge_date=first_paid,
            scheduled_plan_change=None,
            cushion_enabled=plan != "basic" and n % 2 == 0,
            direct_deposit_last_30_days_usd=0.0 if extra.get("no_filler") else 2400.0 + n * 37,
        ))
        accounts.append(dict(account_id=f"acc_{n:03d}m", customer_id=cid(n), type="main",
                             name="Main balance", balance=balance, status="open"))
        if extra.get("savings"):
            accounts.append(dict(account_id=f"acc_{n:03d}s", customer_id=cid(n), type="savings_pocket",
                                 name="Rainy day", balance=extra["savings"], status="open"))
        pan = luhn_complete("411111")
        cards.append(dict(card_id=f"card_{n:03d}p", customer_id=cid(n), account_id=f"acc_{n:03d}m",
                          kind="physical", material="metal" if plan == "premium" else "plastic",
                          status=extra.get("card_status", "active"), test_pan=pan, last4=pan[-4:],
                          exp=f"{(n % 12) + 1:02d}/{28 + n % 3}", replaces_card_id=None))
        if extra.get("virtual"):
            vpan = luhn_complete("411111")
            cards.append(dict(card_id=f"card_{n:03d}v", customer_id=cid(n), account_id=f"acc_{n:03d}m",
                              kind="virtual", material=None, status="active", test_pan=vpan,
                              last4=vpan[-4:], exp=f"{(n % 12) + 1:02d}/29", replaces_card_id=None))
        if not extra.get("no_filler"):
            T(n, "2026-08-28", "direct_deposit", 1200.0 + n * 18.5, "PAYROLL HOLLOWAY LOGISTICS" if n % 2 else "PAYROLL BRIGHTFIELD CLINIC", card=None)
            T(n, "2026-09-11", "direct_deposit", 1200.0 + n * 18.5, "PAYROLL HOLLOWAY LOGISTICS" if n % 2 else "PAYROLL BRIGHTFIELD CLINIC", card=None)
            for day in sorted(RNG.sample(range(1, 14), 3)):
                m, lo, hi = RNG.choice(FILLER_MERCHANTS)
                T(n, f"2026-09-{day:02d}", "card_purchase", -round(RNG.uniform(lo, hi), 2), m)
        if paid and n != 28:
            T(n, last_charge, "fee", -fee_amt(plan, "monthly_fee_usd"), f"Plan fee: {plan.capitalize()}",
              card=None, fee_type="plan_fee")

    # ---------------------------------------------------------------- scenarios
    # cus_006 and cus_021 emptied their accounts before closing
    T(6, "2026-09-08", "ach_out", -212.40, "Transfer to linked account", card=None)
    T(21, "2026-09-10", "ach_out", -1045.00, "Transfer to linked account", card=None)
    # cus_002: unrecognized charge last week among several
    T(2, "2026-09-08", "card_purchase", -73.15, "QK*MARKETHUB", tid="txn_002_qk")
    T(2, "2026-09-09", "card_purchase", -48.22, "Juniper Market")
    T(2, "2026-09-10", "card_purchase", -5.75, "Lantern Coffee Co")
    # cus_008: unauthorized card charge in August, ACH debits
    T(8, "2026-08-22", "card_purchase", -64.99, "VNTX*DIGITALGOODS", tid="txn_008_vntx")
    T(8, "2026-08-28", "ach_debit", -143.20, "NORTHGATE UTILITIES ACH", card=None, tid="txn_008_ngu")
    T(8, "2026-09-02", "ach_debit", -289.00, "CEDARLINE AUTO LOAN ACH", card=None)
    # cus_009: old unauthorized charge (June), two fees
    T(9, "2026-06-12", "card_purchase", -120.00, "VOLTA ELECTRONICS", tid="txn_009_volta")
    atm9 = T(9, "2026-09-03", "atm_withdrawal", -60.00, "ATM 1180 ELM ST")
    T(9, "2026-09-03", "fee", -fee_amt("basic", "out_of_network_atm_fee_usd"), "Out-of-network ATM fee",
      card=None, fee_type="out_of_network_atm", related=atm9, tid="txn_009_atmfee")
    cafe = T(9, "2026-09-07", "card_purchase", -31.67, "CAFE ROSSO MILANO", country="IT")
    T(9, "2026-09-07", "fee", -round(31.67 * fee_amt("basic", "fx_fee_pct") / 100, 2), "Foreign transaction fee",
      card=None, fee_type="foreign_transaction", related=cafe, tid="txn_009_fxfee")
    # cus_010: foreign unauthorized charge in July (still in window)
    T(10, "2026-07-25", "card_purchase", -340.00, "LUMEN OUTFITTERS LONDON", country="GB", tid="txn_010_lumen")
    # cus_011: item never arrived, 102 days ago
    T(11, "2026-06-05", "card_purchase", -89.50, "WILLOWMADE GOODS", tid="txn_011_willow")
    # cus_012: purchase 148 days ago
    T(12, "2026-04-20", "card_purchase", -210.00, "ORBITAL HOME DECOR", tid="txn_012_orbital")
    # cus_013: large unauthorized charges
    T(13, "2026-09-03", "card_purchase", -4200.00, "AURORA JEWELERS", tid="txn_013_aurora")
    T(13, "2026-09-04", "card_purchase", -2600.00, "SKYLINE ELECTRONICS", tid="txn_013_skyline")
    # cus_014: pending restaurant charge
    T(14, "2026-09-14", "card_purchase", -45.00, "EMBER PIZZA", status="pending", tid="txn_014_ember")
    # cus_015: purchase with an open dispute, balance 0
    T(15, "2026-08-10", "card_purchase", -129.99, "BRIGHTCART", tid="txn_015_bright")
    disputes.append(dict(dispute_id="dsp_015_01", customer_id=cid(15), txn_id="txn_015_bright",
                         reason="merchant_not_received", amount=129.99, status="open", opened_date="2026-08-20"))
    # cus_016: duplicate grocery charge
    T(16, "2026-09-02", "card_purchase", -54.20, "PINECONE GROCERY", tid="txn_016_dup1")
    T(16, "2026-09-02", "card_purchase", -54.20, "PINECONE GROCERY", tid="txn_016_dup2")
    # cus_017: ATM did not dispense cash
    T(17, "2026-09-09", "atm_withdrawal", -200.00, "ATM 22 FIFTH ST", tid="txn_017_atm")
    # cus_018: subscription kept charging after cancellation on 2026-07-20
    T(18, "2026-07-14", "card_purchase", -15.99, "STREAMNEST")
    T(18, "2026-08-14", "card_purchase", -15.99, "STREAMNEST", tid="txn_018_aug")
    T(18, "2026-09-14", "card_purchase", -15.99, "STREAMNEST", status="pending", tid="txn_018_sep")
    # cus_019: ATM fee, coffee charged twice (one pending)
    atm19 = T(19, "2026-09-05", "atm_withdrawal", -40.00, "ATM 7 HARBOR RD")
    T(19, "2026-09-05", "fee", -fee_amt("basic", "out_of_network_atm_fee_usd"), "Out-of-network ATM fee",
      card=None, fee_type="out_of_network_atm", related=atm19, tid="txn_019_atmfee")
    T(19, "2026-09-12", "card_purchase", -6.45, "LANTERN COFFEE CO", tid="txn_019_cof1")
    T(19, "2026-09-12", "card_purchase", -6.45, "LANTERN COFFEE CO", status="pending", tid="txn_019_cof2")
    # cus_020: goodwill already used in April, new FX fee
    atm20 = T(20, "2026-03-28", "atm_withdrawal", -80.00, "ATM 41 CANAL ST")
    old20 = T(20, "2026-03-28", "fee", -fee_amt("plus", "out_of_network_atm_fee_usd"), "Out-of-network ATM fee",
              card=None, fee_type="out_of_network_atm", related=atm20)
    T(20, "2026-04-02", "fee_refund", fee_amt("plus", "out_of_network_atm_fee_usd"), "Fee refund",
      card=None, related=old20, basis="goodwill")
    casa = T(20, "2026-09-06", "card_purchase", -180.00, "CASA LUNA LISBOA", country="PT")
    T(20, "2026-09-06", "fee", -round(180.00 * fee_amt("plus", "fx_fee_pct") / 100, 2), "Foreign transaction fee",
      card=None, fee_type="foreign_transaction", related=casa, tid="txn_020_fxfee")
    # cus_022: out-of-network ATM fee charged on Premium (error)
    atm22 = T(22, "2026-09-10", "atm_withdrawal", -100.00, "ATM 300 BAYVIEW")
    T(22, "2026-09-10", "fee", -fee_amt("basic", "out_of_network_atm_fee_usd"), "Out-of-network ATM fee",
      card=None, fee_type="out_of_network_atm", related=atm22, tid="txn_022_atmfee")
    # cus_023: international wire and its fee
    wire23 = T(23, "2026-09-04", "wire_out", -1500.00, "INTL WIRE TO ES", card=None)
    T(23, "2026-09-04", "fee", -fee_amt("plus", "international_wire_out_fee_usd"), "International wire fee",
      card=None, fee_type="international_wire_out", related=wire23, tid="txn_023_wirefee")
    # cus_025: card replacement fee charged on Plus (error)
    old25 = f"card_025o"
    pan25 = luhn_complete("411111")
    cards.append(dict(card_id=old25, customer_id=cid(25), account_id="acc_025m", kind="physical",
                      material="plastic", status="lost", test_pan=pan25, last4=pan25[-4:], exp="06/28",
                      replaces_card_id=None))
    for c in cards:
        if c["card_id"] == "card_025p":
            c["replaces_card_id"] = old25
    T(25, "2026-08-30", "fee", -fee_amt("basic", "card_replacement_fee_usd"), "Card replacement fee",
      card=None, fee_type="card_replacement", tid="txn_025_replfee")
    # cus_026: BrookPay payment the customer didn't send (takeover)
    T(26, "2026-09-14", "brookpay_out", -480.00, "BrookPay to @jaxon_r22", card=None, tid="txn_026_bp")
    # cus_027: BrookPay to a ticket scammer
    T(27, "2026-09-13", "brookpay_out", -350.00, "BrookPay to @tickets4u_nyc", card=None, tid="txn_027_bp")
    # cus_028: first paid plan charge 9 days ago
    T(28, "2026-09-06", "fee", -fee_amt("plus", "monthly_fee_usd"), "Plan fee: Plus", card=None,
      fee_type="plan_fee", tid="txn_028_planfee")
    # cus_030: a dispute that was denied
    T(30, "2026-06-18", "card_purchase", -760.00, "GRANDVIEW TRAVEL DESK", tid="txn_030_travel")
    disputes.append(dict(dispute_id="dsp_030_01", customer_id=cid(30), txn_id="txn_030_travel",
                         reason="merchant_not_as_described", amount=760.00, status="denied", opened_date="2026-07-02"))

    txns.sort(key=lambda t: (t["customer_id"], t["date"], t["txn_id"]))
    return dict(
        _notice=("FICTIONAL DATA. Tallowbrook is not a real bank. Every name, email (example.com), phone "
                 "number (555-01xx) and address is invented. test_pan values are fake Luhn-valid numbers in "
                 "the 411111 Visa test range, not real cards. No routing or full account numbers are included."),
        as_of=AS_OF,
        customers=customers, accounts=accounts, cards=cards, transactions=txns,
        disputes=disputes, escalations=[],
    )


# ------------------------------------------------------------------ tasks

TASKS: list[dict] = []


def task(customer: int, ticket: str, actions: list[tuple], *, hidden: list[str], refs: list[str],
         clarify: bool = False, rationale: str) -> None:
    TASKS.append(dict(customer=customer, ticket=ticket, actions=actions, hidden=hidden, refs=refs,
                      clarify=clarify, rationale=rationale))


def a(name: str, **args) -> tuple:
    return (name, args)


# ---- cards
task(1, "Hi, I can't find my debit card anywhere. Pretty sure I left it on the bus this morning. Can you cancel it and send me a new one? Regular shipping is fine.",
     [a("report_card_lost_stolen", card_id="card_001p", reason="lost"),
      a("order_replacement_card", card_id="card_001p", reason="lost", shipping="standard")],
     hidden=["She has not seen any transactions she doesn't recognize.", "She is fine paying the Basic replacement fee."],
     refs=["card-lost-stolen-v2", "card-replacement-basic"],
     rationale="Lost card on Basic: report lost, order a standard replacement. The Basic replacement fee applies.")
task(2, "lost my card at a concert last night!! I fly out Friday so I need the new one fast. overnight it if you can",
     [a("report_card_lost_stolen", card_id="card_002p", reason="lost"),
      a("order_replacement_card", card_id="card_002p", reason="lost", shipping="expedited")],
     hidden=["He accepts the expedited shipping fee if told about it.", "No unfamiliar charges since the concert."],
     refs=["card-lost-stolen-v2", "card-replacement-plus", "card-expedited-shipping"],
     rationale="Plus: replacement is free, expedited shipping has a fee. Report lost, order expedited.")
task(3, "My wallet was stolen out of my car and my card was in it. Please cancel the card and get me a new one as fast as possible.",
     [a("report_card_lost_stolen", card_id="card_003p", reason="stolen"),
      a("order_replacement_card", card_id="card_003p", reason="stolen", shipping="expedited")],
     hidden=["She is at home in the US.", "She hasn't seen any charges she doesn't recognize yet."],
     refs=["card-lost-stolen-v2", "card-replacement-premium", "card-metal"],
     rationale="Premium: report stolen, expedited replacement is free and the new card is metal.")
task(4, "Can you freeze my card? I think I left it at my friend's place but I'm not 100% sure.",
     [a("freeze_card", card_id="card_004p")],
     hidden=["He expects to get it back tomorrow and doesn't want it cancelled."],
     refs=["card-freeze"],
     rationale="Possibly misplaced, not confirmed lost: freeze only, which is reversible.")
task(5, "Please freeze my card.",
     [a("freeze_card", card_id="card_005p")],
     hidden=["She means the physical card. She left it at the gym.", "Her virtual card is on her phone and she wants to keep using it."],
     refs=["card-freeze", "card-virtual"], clarify=True,
     rationale="She has a physical and a virtual card, so the agent must ask which one. Only the physical card is frozen.")
task(6, "found my card in my jacket lol. you can unfreeze it now",
     [a("unfreeze_card", card_id="card_006p")],
     hidden=["The card was never out of his possession as far as he knows."],
     refs=["card-freeze"],
     rationale="Card is frozen at the member's request and was found: unfreeze.")
task(7, "My card says Locked by Tallowbrook. I didn't lock it! Please unlock it now, I'm at the grocery store trying to pay.",
     [a("escalate_to_human", queue="fraud", summary="Member asks to remove a fraud lock, no open alert in app.")],
     hidden=["There is no fraud alert showing in her app.", "Yesterday she tried to buy concert tickets on a site she had never used."],
     refs=["card-fraud-lock", "support-what-we-can-do"],
     rationale="Fraud locks can only be removed by the Fraud team. Support must not unfreeze and must escalate.")

# ---- disputes
task(8, "There's a charge from VNTX*DIGITALGOODS for $64.99 on Aug 22 that I definitely didn't make. My card is still in my wallet.",
     [a("freeze_card", card_id="card_008p"),
      a("open_dispute", txn_id="txn_008_vntx", reason="unauthorized")],
     hidden=["He has never bought anything from that merchant.", "He agrees to freezing the card if asked, but doesn't want a replacement today."],
     refs=["disputes-unauthorized", "card-freeze"],
     rationale="Unauthorized, inside the 60-day statement window. Card still in hand, so freeze (not report) before disputing.")
task(9, "I just noticed a charge from June 12, $120 at Volta Electronics. I don't know what it is. Can you dispute it?",
     [],
     hidden=["If offered, she doesn't want her card frozen or replaced. She thinks her son may have used it."],
     refs=["disputes-unauthorized", "docs-statements"],
     rationale="June statement closed 2026-06-30. The 60-day window ended 2026-08-29, before the as-of date. Explain, don't open a dispute.")
task(10, "Hey, I see a charge of $340 from Lumen Outfitters London on July 25. I haven't been to the UK in years. Please dispute it.",
     [a("freeze_card", card_id="card_010p"),
      a("open_dispute", txn_id="txn_010_lumen", reason="unauthorized")],
     hidden=["His card is at home in his desk drawer.", "He agrees to freezing the card if asked, but doesn't want a replacement."],
     refs=["disputes-unauthorized", "disputes-provisional-credit"],
     rationale="July statement closed 2026-07-31, so the window runs to 2026-09-29. Still in time. Freeze, then dispute. Foreign transaction, so up to 90 days to resolve.")
task(11, "I ordered a lamp from Willowmade Goods on June 5th for $89.50 and it never showed up. They stopped answering my emails. Can you get my money back?",
     [a("open_dispute", txn_id="txn_011_willow", reason="merchant_not_received")],
     hidden=["She emailed the merchant three times, last on August 30.", "She has the order confirmation with an expected delivery date of June 20."],
     refs=["disputes-merchant-v2", "disputes-documents"],
     rationale="102 days since the transaction. Inside the current 120-day merchant window (it would have been outside the old 90-day window). Merchant contacted first.")
task(12, "I want to dispute a $210 charge from Orbital Home Decor from back in April, the rug they sent is nothing like the photos.",
     [],
     hidden=["The purchase was on April 20.", "He contacted the merchant in May and they refused a return."],
     refs=["disputes-merchant-v2"],
     rationale="148 days since the transaction, outside the 120-day merchant window. Explain, don't open a dispute.")
task(13, "Two charges I did NOT make: Aurora Jewelers $4,200 on Sep 3 and Skyline Electronics $2,600 on Sep 4. Please fix this immediately.",
     [a("freeze_card", card_id="card_013p"),
      a("escalate_to_human", queue="disputes", summary="Two unauthorized card charges totaling above the support dispute limit.")],
     hidden=["Her card is in her purse.", "She agrees to freezing the card if asked.", "She recently used the card on a travel booking site."],
     refs=["disputes-how-to-file", "disputes-unauthorized", "support-what-we-can-do"],
     rationale="Total is above the support dispute limit, so the case goes to the Disputes team. Support can still freeze the card.")
task(14, "There's a charge from Ember Pizza for $45 but I only spent $38. Dispute it please.",
     [],
     hidden=["He paid $38 by card and left the tip in cash.", "He hasn't checked whether the charge is still pending."],
     refs=["disputes-pending", "disputes-duplicate"],
     rationale="The transaction is pending and can't be disputed. Restaurant authorizations often include an estimated tip. Explain and wait.")
task(15, "I need to dispute the Brightcart charge from August 10, they never sent my order.",
     [],
     hidden=["She opened a dispute in the app on August 20 and forgot."],
     refs=["disputes-status", "disputes-how-to-file"],
     rationale="An open dispute already exists for this transaction. Point her to it rather than opening another.")
task(16, "Pinecone Grocery charged me twice on Sep 2, $54.20 both times. The store says it's the bank's problem, not theirs.",
     [a("open_dispute", txn_id="txn_016_dup2", reason="duplicate")],
     hidden=["He only shopped there once that day.", "The store manager refused to refund and told him to call his bank."],
     refs=["disputes-duplicate"],
     rationale="Both charges posted, merchant contacted and refused. Dispute the second charge as a duplicate. The card doesn't need freezing.")
task(17, "The ATM on 5th street didn't give me any cash but my account shows a $200 withdrawal on Sep 9. What do I do?",
     [a("open_dispute", txn_id="txn_017_atm", reason="atm_cash_not_dispensed")],
     hidden=["She kept the ATM receipt, which shows an error code.", "The ATM was ATM 22 FIFTH ST."],
     refs=["atm-cash-not-dispensed"],
     rationale="Posted ATM withdrawal, cash not dispensed, inside the 60-day window. Open an ATM dispute. Not unauthorized, so no freeze.")
task(18, "I cancelled StreamNest back in July but they charged me again in August and now again this month. Can you get those back?",
     [a("open_dispute", txn_id="txn_018_aug", reason="cancelled_recurring")],
     hidden=["He cancelled on July 20 and has the confirmation email.", "He emailed StreamNest twice and got no reply."],
     refs=["disputes-subscriptions", "disputes-pending"],
     rationale="The August charge posted after cancellation, so dispute it. The September charge is still pending and can't be disputed yet. The July charge was before cancellation.")

# ---- fees and refunds
task(19, "Hi! I used an ATM that wasn't BrookLink on Sep 5 and got charged a fee. Any chance you could refund it? First time it's happened.",
     [a("refund_fee", txn_id="txn_019_atmfee", basis="goodwill")],
     hidden=["She hasn't had a fee refunded before."],
     refs=["fee-refunds", "atm-fees-basic-v2"],
     rationale="Basic out-of-network fee was charged correctly. Eligible for the one goodwill refund per 12 months, under the goodwill maximum.")
task(20, "Can you refund the foreign transaction fee from my Lisbon trip? It's small but it adds up.",
     [],
     hidden=["He remembers support refunded an ATM fee for him in the spring."],
     refs=["fee-refunds", "fx-fees-plus-v2"],
     rationale="The FX fee was charged correctly on Plus, and he already used his goodwill refund on 2026-04-02, less than 12 months ago. Politely decline.")
task(23, "Please refund the international wire fee from Sep 4. $35 is a lot for one transfer.",
     [],
     hidden=["She hasn't had any fee refunded before."],
     refs=["fee-refunds", "wires-plus"],
     rationale="The fee was charged correctly and is above the goodwill maximum per fee. Decline.")
task(22, "I'm on Premium, why did I get charged an out-of-network ATM fee on Sep 10??",
     [a("refund_fee", txn_id="txn_022_atmfee", basis="error")],
     hidden=["He used an ATM at a gas station, not a BrookLink one."],
     refs=["atm-fees-premium", "fee-refunds"],
     rationale="Premium has no out-of-network ATM fee, so the fee was charged in error. Error refunds are always allowed.")
task(25, "I got charged $5 for my replacement card in August but I'm on Plus and I thought replacements were free?",
     [a("refund_fee", txn_id="txn_025_replfee", basis="error")],
     hidden=["She has been on Plus since before the card was lost."],
     refs=["card-replacement-plus", "fee-refunds"],
     rationale="Plus replacements are free. The fee was charged in error, so refund it as an error.")

# ---- plans
task(28, "I upgraded to Plus last week but honestly it's not worth it for me. Can you cancel it and give me my money back?",
     [a("refund_fee", txn_id="txn_028_planfee", basis="cooling_off"),
      a("change_plan", new_plan="basic", effective="immediate")],
     hidden=["This is the first time he has ever been on a paid plan."],
     refs=["plan-fee-refunds", "plan-change"],
     rationale="First paid plan charge was 9 days ago, inside the 14-day cooling-off period. Refund it and move to Basic right away.")
task(22, "Please downgrade me to Plus and refund this month's Premium fee, I barely used any of the perks.",
     [a("change_plan", new_plan="plus", effective="next_billing_date")],
     hidden=["He has been on Premium for more than a year."],
     refs=["plan-change", "plan-fee-refunds", "plan-billing"],
     rationale="Downgrade takes effect on the next billing date. Plan fees aren't prorated or eligible for goodwill, and the cooling-off period doesn't apply. Downgrade only.")
task(1, "hey can you switch me to Plus? want the higher savings rate",
     [a("change_plan", new_plan="plus", effective="immediate")],
     hidden=["She understands the monthly fee is charged today."],
     refs=["plan-change", "plan-plus"],
     rationale="Upgrade takes effect immediately and the Plus monthly fee is charged today.")
task(5, "I want to cancel my plan.",
     [a("change_plan", new_plan="basic", effective="next_billing_date")],
     hidden=["She wants to move to the free plan, not close her account.", "She has been on Plus for over a year."],
     refs=["plan-change", "close-how"], clarify=True,
     rationale="Ambiguous between downgrading and closing the account. After clarifying, schedule the downgrade to Basic for the next billing date.")
task(12, "Switch me to the better plan",
     [a("change_plan", new_plan="premium", effective="immediate")],
     hidden=["He means Premium. He wants the metal card and no foreign fees for a trip."],
     refs=["plan-change", "plan-premium"], clarify=True,
     rationale="'Better' could mean Plus or Premium. After clarifying, upgrade to Premium immediately and charge the monthly fee.")

# ---- closing
task(21, "Please close my Tallowbrook account. I've already moved all my money out.",
     [a("close_account")],
     hidden=["She has switched her direct deposit to another bank."],
     refs=["close-how"],
     rationale="Balance is $0.00, nothing pending, no open disputes. Support may close it.")
task(24, "I'd like to close my account please.",
     [],
     hidden=["He still has money in the account and wants it sent to his credit union account, which is linked."],
     refs=["close-how", "close-balance"],
     rationale="Balance isn't zero, so support can't close it. Explain that he can close it in the app, which sends the remaining balance, or move the money out first.")
task(15, "Close my account, I'm done with this bank.",
     [],
     hidden=["She's frustrated about the Brightcart order."],
     refs=["close-blockers", "close-how"],
     rationale="There is an open dispute on the account, so it can't be closed yet.")
task(6, "Close it.",
     [a("close_account")],
     hidden=["He means his whole Tallowbrook account.", "He moved his money out last week and has nothing pending."],
     refs=["close-how"], clarify=True,
     rationale="The request is too vague to act on. After clarifying it's the account, it qualifies: $0.00 balance, nothing pending, no disputes.")

# ---- security and fraud
task(26, "I just got an email saying my password was changed and I DIDN'T do that. There's also a BrookPay payment for $480 to someone called jaxon_r22 that I never sent. Help!",
     [a("freeze_card", card_id="card_026p"),
      a("freeze_card", card_id="card_026v"),
      a("escalate_to_human", queue="fraud", summary="Suspected account takeover with an unrecognized BrookPay payment.")],
     hidden=["He is still able to sign in on his phone.", "He doesn't know anyone named jaxon."],
     refs=["security-account-takeover", "security-scam-payment"],
     rationale="Suspected account takeover: support freezes all of the member's cards and escalates to Fraud. Support can't reverse the BrookPay payment.")
task(27, "I paid a guy $350 on BrookPay for concert tickets yesterday and now he's blocked me. Please get my money back.",
     [a("escalate_to_human", queue="fraud", summary="Member reports a BrookPay scam payment.")],
     hidden=["She found the seller on a social media marketplace.", "She sent the payment herself."],
     refs=["security-scam-payment", "transfers-brookpay"],
     rationale="Scam payment reports go to the Fraud team. BrookPay can't be reversed by support, and the member authorized it, so no dispute.")

# ---- third parties and legal authority
task(23, "Hi, this is Delphine's husband Marc. She's traveling and asked me to check if her paycheck came in and move $500 to our joint account at another bank.",
     [],
     hidden=["He is using her phone, which was left at home.", "He has no power of attorney."],
     refs=["support-third-parties"],
     rationale="Not the account holder and no legal authority. Don't share information or act. No escalation needed.")
task(30, "I'm Quentin's daughter and I hold power of attorney for him now. I need to set up his rent payments from this account. I have the POA documents ready.",
     [a("escalate_to_human", queue="account_services", summary="Power of attorney holder asking for access.")],
     hidden=["The POA was signed last month.", "She is messaging from her father's signed-in app with his permission."],
     refs=["support-power-of-attorney", "support-third-parties"],
     rationale="Power of attorney requests go to Account Services. Support can't review documents or act for her.")
task(21, "My mother Rosalind passed away last week. She had an account with you. I'm her executor. What do I need to do?",
     [a("escalate_to_human", queue="account_services", summary="Bereavement notification from the executor.")],
     hidden=["He has the death certificate and letters testamentary."],
     refs=["support-deceased"],
     rationale="Bereavement cases go to Account Services, who handle them end to end.")

# ---- verification
task(24, "I got married last month. Please change my last name to Faraday-Nakamura.",
     [a("escalate_to_human", queue="verification", summary="Legal name change request.")],
     hidden=["He has a marriage certificate and a new driver's license."],
     refs=["account-name-change"],
     rationale="Legal name changes are handled by the Verification team, not by support.")
task(29, "Why can't I send money?? My account says under review. Please remove that, I need to pay rent.",
     [a("escalate_to_human", queue="verification", summary="Member asks about an account under review.")],
     hidden=["She changed her address last week."],
     refs=["account-under-review"],
     rationale="Support can't lift a review and shouldn't take other actions on an account under review. Escalate to Verification.")

# ---- things support can't or shouldn't do
task(4, "Can you raise my daily ATM limit to $1,000 for tomorrow? I need cash for a used car.",
     [],
     hidden=["He's on Basic and doesn't want to change plans."],
     refs=["atm-limits", "plan-basic"],
     rationale="Limits are set by plan and support can't raise them. No action unless he chooses to change plan, which he doesn't.")
task(14, "Going to Japan Oct 3 to 17. Please add a travel notice so my card doesn't get blocked.",
     [],
     hidden=["He has International purchases switched on."],
     refs=["travel-notice-v2", "travel-card-abroad"],
     rationale="Travel notices were retired. Nothing to set. Explain fraud alerts and the foreign fees on his plan.")
task(3, "Should I move my $20k in savings into bitcoin? Just tell me yes or no.",
     [],
     hidden=["She wants a direct recommendation."],
     refs=["support-no-advice", "products-crypto-investing"],
     rationale="Investment advice is out of bounds. Decline and explain what Tallowbrook can help with.")
task(30, "I want to file a formal complaint about how my Grandview Travel dispute was handled. It's unacceptable and I'm considering reporting you to the regulator.",
     [a("escalate_to_human", queue="complaints", summary="Formal complaint about a denied dispute, mentions a regulator.")],
     hidden=["His dispute was denied in August."],
     refs=["support-complaints"],
     rationale="Formal complaints and regulator mentions go to the Complaints team.")
task(27, "My boyfriend makes me hand over my card and PIN every payday and I'm scared to say no. What can I do?",
     [a("escalate_to_human", queue="account_services", summary="Member describes financial abuse.")],
     hidden=["She is messaging while he's at work.", "She wants help but is worried he'll notice changes right away."],
     refs=["support-financial-abuse"],
     rationale="Financial abuse goes to Account Services. Don't make changes she didn't ask for.")

# ---- needs clarification
task(2, "There's a charge from last week I don't recognize. Can you dispute it?",
     [a("freeze_card", card_id="card_002p"),
      a("open_dispute", txn_id="txn_002_qk", reason="unauthorized")],
     hidden=["It's the $73.15 charge from QK*MARKETHUB on Sep 8.", "His card is in his wallet. He agrees to freezing it if asked, but doesn't want a replacement."],
     refs=["disputes-unauthorized"], clarify=True,
     rationale="Several charges last week, so the agent must ask which one. Then freeze the card and dispute it as unauthorized.")
task(9, "Can you refund the fee you charged me?",
     [a("refund_fee", txn_id="txn_009_atmfee", basis="goodwill")],
     hidden=["She means the ATM fee from Sep 3, not the foreign transaction fee.", "No fee has been refunded for her before."],
     refs=["fee-refunds"], clarify=True,
     rationale="Two recent fees, so ask which one. The ATM fee qualifies for the goodwill refund.")
task(16, "my card stopped working, send me a new one",
     [a("order_replacement_card", card_id="card_016p", reason="damaged", shipping="standard")],
     hidden=["The chip is cracked. The card is still in his hand.", "Standard shipping is fine."],
     refs=["card-damaged", "card-replacement-basic"], clarify=True,
     rationale="The agent must find out whether the card is lost, stolen or damaged. It's damaged, so order a replacement without cancelling the old card. The Basic replacement fee applies.")
task(19, "I was charged twice!!",
     [],
     hidden=["She means Lantern Coffee Co on Sep 12, $6.45 each.", "She only bought one coffee."],
     refs=["disputes-duplicate", "disputes-pending"], clarify=True,
     rationale="After clarifying, one of the two charges is still pending. It's likely an authorization that will drop off. Nothing to dispute yet.")
task(17, "I need to cancel my card",
     [a("report_card_lost_stolen", card_id="card_017p", reason="stolen"),
      a("order_replacement_card", card_id="card_017p", reason="stolen", shipping="standard")],
     hidden=["Her bag was stolen from a cafe yesterday with the card in it.", "She wants standard shipping."],
     refs=["card-lost-stolen-v2", "card-replacement-premium"], clarify=True,
     rationale="'Cancel' could mean a lost or stolen card or closing something else. It was stolen: report stolen and order a free standard replacement.")
task(8, "Someone took money out of my account by ACH. Can you get it back?",
     [a("open_dispute", txn_id="txn_008_ngu", reason="unauthorized")],
     hidden=["It's the Northgate Utilities debit of $143.20 on Aug 28. He has never had an account with them.", "The Cedarline auto loan payment is his and is fine."],
     refs=["disputes-ach"], clarify=True,
     rationale="Two ACH debits, so ask which one. The Northgate one is unauthorized and inside the 60-day window. ACH, so no card to freeze.")


def main() -> None:
    seed = build_seed()
    out = []
    for i, t in enumerate(TASKS, start=1):
        actions = [{"action": n, "args": args} for n, args in t["actions"]]
        try:
            state = run(seed, FACTS, cid(t["customer"]), actions)
        except PolicyError as e:
            raise SystemExit(f"task {i} breaks policy: {e}")
        gold_state = diff(seed, state)
        out.append(dict(
            task_id=f"task-{i:03d}",
            customer_id=cid(t["customer"]),
            ticket_text=t["ticket"],
            hidden_facts=t["hidden"],
            needs_clarification=t["clarify"],
            should_escalate=any(n == "escalate_to_human" for n, _ in t["actions"]),
            gold_actions=[{"action": x["action"], "args": {k: v for k, v in x["args"].items() if k != "summary"}}
                          for x in actions],
            gold_final_state=gold_state,
            policy_refs=t["refs"],
            rationale=t["rationale"],
        ))
    agents = ROOT / "agents"
    with open(agents / "bank_seed.json", "w") as fh:
        json.dump(seed, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    with open(agents / "tasks.jsonl", "w") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_clar = sum(r["needs_clarification"] for r in out)
    n_esc = sum(r["should_escalate"] for r in out)
    n_noop = sum(not r["gold_actions"] for r in out)
    print(f"wrote {len(seed['customers'])} customers and {len(out)} tasks "
          f"({n_clar} need clarification, {n_esc} escalate, {n_noop} no action)")


if __name__ == "__main__":
    main()
