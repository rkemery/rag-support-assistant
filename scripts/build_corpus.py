"""Build corpus/articles.jsonl from facts/policies.yaml.

Article text is hand-written prose with every policy number pulled from the
facts file through `F`. Per-plan variants and superseded versions come from
the same templates with different plan or version arguments, which is what
makes them realistic near-duplicates.

Run: uv run python scripts/build_corpus.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from facts import PLAN_NAME, PLANS, ROOT, Facts, get_path, money  # noqa: E402

F = Facts()
ALL = list(PLANS)

S_ACC = "Account opening and verification"
S_PLANS = "Plans and fees"
S_CARDS = "Cards"
S_ATM = "ATMs and cash"
S_TRAVEL = "Travel and foreign transactions"
S_TRANSFERS = "Transfers and deposits"
S_DISPUTES = "Disputes"
S_SECURITY = "Security and fraud"
S_CUSHION = "Overdraft and Cushion"
S_SAVINGS = "Savings"
S_CLOSE = "Closing your account"
S_SUPPORT = "Help and support"
S_DOCS = "Statements and documents"
S_PRODUCTS = "What we offer"

FOOTERS = [
    "Need more help? Message us in the app any time, day or night.",
    "Still stuck? Chat with us in the app. Someone is there around the clock.",
    "Questions? Our support team is available in the app at any hour.",
]

REGISTRY: list[dict] = []


def art(
    article_id: str,
    title: str,
    section: str,
    body: Callable[[], str],
    plans: list[str] | None = None,
    eff: str = "2025-01-01",
    version: int = 1,
    supersedes: str | None = None,
    tags: tuple[str, ...] = (),
) -> None:
    REGISTRY.append(
        dict(
            article_id=article_id,
            title=title,
            section=section,
            plans=list(plans or ALL),
            effective_date=eff,
            version=version,
            supersedes=supersedes,
            body_fn=body,
            tags=tuple(tags),
        )
    )


def eff_of(path: str) -> str:
    """Effective date of a versioned policy, read without recording."""
    return get_path(F.data, path).isoformat()


def pn(plan: str) -> str:
    return PLAN_NAME[plan]


def t(*tags: str) -> str:
    return F.mark(*[f"tag:{x}" for x in tags])


# =====================================================================
# Plans and fees
# =====================================================================

def plan_overview(plan: str) -> Callable[[], str]:
    def body() -> str:
        fee = F.fee_path
        name = pn(plan)
        monthly = F.fee_raw(plan, "monthly_fee_usd")
        intro = {
            "basic": f"Basic is Tallowbrook's free plan. The monthly fee is {money(monthly)}, there's no minimum balance, and you get the same app, debit card and support as every other member.",
            "plus": f"Plus costs {money(monthly)} a month. It's built for people who use their card a lot away from BrookLink ATMs or spend some time abroad, and it adds Cushion and a higher savings rate.",
            "premium": f"Premium costs {money(monthly)} a month. It removes nearly every fee we charge, comes with a metal card, and has the highest limits and savings rate we offer.",
        }[plan]
        lines = [intro, "", "## What's included", ""]
        lines.append(f"- **Monthly fee:** {F.fee(plan, 'monthly_fee_usd')}")
        oon = F.fee_raw(plan, "out_of_network_atm_fee_usd")
        free_n = F.raw(fee(plan, "free_out_of_network_withdrawals_per_month"))
        if plan == "premium":
            lines.append(f"- **ATMs:** no Tallowbrook fee at any ATM in the US, and we reimburse ATM owner surcharges up to {F.fee(plan, 'atm_surcharge_reimbursement_monthly_cap_usd')} a month.")
        elif free_n:
            lines.append(f"- **ATMs:** free at BrookLink ATMs, plus {free_n} free out-of-network withdrawals a month. After that it's {money(oon)} per withdrawal.")
        else:
            lines.append(f"- **ATMs:** free at BrookLink ATMs. Out-of-network withdrawals cost {money(oon)} each.")
        lines.append(f"- **Foreign transaction fee:** {F.fee(plan, 'fx_fee_pct')} of the purchase amount")
        lines.append(f"- **Card replacement:** {F.fee_or_free(plan, 'card_replacement_fee_usd')}")
        lines.append(f"- **Expedited card shipping:** {F.fee_or_free(plan, 'expedited_shipping_fee_usd')}")
        lines.append(f"- **Savings Pockets:** {F.v(f'savings.versions.v2.{plan}_apy_pct', decimals=2)} APY")
        if plan == "basic":
            t("cushion_basic_not_eligible")
            lines.append("- **Cushion:** not available on Basic")
        else:
            lines.append(f"- **Cushion:** fee-free coverage up to {F.v(f'overdraft.cushion.{plan}_limit_usd')} on debit card purchases")
        lines.append(f"- **Daily card spending limit:** {F.lim(plan, 'daily_card_spend_limit_usd')}")
        lines.append(f"- **Daily ATM limit:** {F.lim(plan, 'daily_atm_withdrawal_limit_usd')}")
        lines.append("")
        lines.append("## Good to know")
        lines.append("")
        if plan == "basic":
            t("intl_wire_not_on_basic")
            lines.append("International wires aren't available on Basic. You can still receive them. If you travel or send money abroad often, Plus or Premium will usually cost less over a year.")
        elif plan == "plus":
            lines.append(f"Plus includes outgoing international wires for {F.fee(plan, 'international_wire_out_fee_usd')} each. Your card is plastic, like Basic. Upgrade to Premium if you want the metal card.")
        else:
            t("metal_card", "emergency_card_abroad")
            lines.append(f"Premium members can get an emergency replacement card delivered abroad within {F.v('cards.premium_emergency_card_abroad_business_days')} at no cost. Your card is metal, and replacements are metal too.")
        lines.append("")
        lines.append("You can change plans any time in the app under Profile, then Plan. Upgrades start right away. Downgrades start on your next billing date.")
        t(f"plan_overview_{plan}", "plan_change_timing")
        return "\n".join(lines)

    return body


for _p in PLANS:
    art(f"plan-{_p}", f"The {pn(_p)} plan", S_PLANS, plan_overview(_p), plans=[_p],
        eff=eff_of("savings.versions.v2.effective_date"))


def plans_compare() -> str:
    t("plans_compare", "metal_card", "emergency_card_abroad", "intl_wire_not_on_basic", "cushion_basic_not_eligible")
    rows = [
        "| Feature | Basic | Plus | Premium |",
        "|---|---|---|---|",
        f"| Card | plastic | plastic | metal |",
        f"| Savings APY | {F.v('savings.versions.v2.basic_apy_pct', decimals=2)} | {F.v('savings.versions.v2.plus_apy_pct', decimals=2)} | {F.v('savings.versions.v2.premium_apy_pct', decimals=2)} |",
        f"| Cushion | not available | up to {F.v('overdraft.cushion.plus_limit_usd')} | up to {F.v('overdraft.cushion.premium_limit_usd')} |",
        f"| Daily card spending | {F.lim('basic', 'daily_card_spend_limit_usd')} | {F.lim('plus', 'daily_card_spend_limit_usd')} | {F.lim('premium', 'daily_card_spend_limit_usd')} |",
        f"| Daily ATM withdrawals | {F.lim('basic', 'daily_atm_withdrawal_limit_usd')} | {F.lim('plus', 'daily_atm_withdrawal_limit_usd')} | {F.lim('premium', 'daily_atm_withdrawal_limit_usd')} |",
        f"| Daily ACH transfers out | {F.lim('basic', 'daily_ach_out_limit_usd')} | {F.lim('plus', 'daily_ach_out_limit_usd')} | {F.lim('premium', 'daily_ach_out_limit_usd')} |",
        "| Outgoing international wires | not available | yes | yes |",
        "| Emergency card abroad | no | no | yes |",
    ]
    return "\n".join([
        "All three plans come with a Visa debit card, the Tallowbrook app, BrookPay transfers to other members, Savings Pockets and support around the clock. They differ in fees, limits and a few features.",
        "",
        *rows,
        "",
        "Fees are listed in the article \"Fees for all plans\". This page covers features and limits only.",
        "",
        "## Which plan fits",
        "",
        "Basic suits people who mostly use BrookLink ATMs and don't travel. Plus suits people who want Cushion and a better savings rate for a small monthly fee. Premium suits frequent travelers and people with larger balances, since the savings rate and the missing foreign fees can more than cover the monthly fee.",
        "",
        "You can switch plans in the app at any time.",
    ])


art("plans-compare", "Compare plans: features and limits", S_PLANS, plans_compare,
    eff=eff_of("savings.versions.v2.effective_date"))


FEE_ROWS = [
    ("Monthly fee", "monthly_fee_usd"),
    ("Out-of-network ATM withdrawal", "out_of_network_atm_fee_usd"),
    ("International ATM withdrawal", "international_atm_fee_usd"),
    ("Foreign transaction fee", "fx_fee_pct"),
    ("Card replacement", "card_replacement_fee_usd"),
    ("Expedited card shipping", "expedited_shipping_fee_usd"),
    ("Outgoing domestic wire", "domestic_wire_out_fee_usd"),
    ("Outgoing international wire", "international_wire_out_fee_usd"),
    ("Incoming international wire", "incoming_international_wire_fee_usd"),
    ("Cash deposit at a retail store", "retail_cash_deposit_fee_usd"),
]


def fee_cell(plan: str, key: str, ver: int) -> str:
    val = F.fee_raw(plan, key, ver)
    if val is None:
        return "not available"
    return F.fee(plan, key, ver)


def fees_all(ver: int) -> Callable[[], str]:
    def body() -> str:
        t("fee_schedule")
        lines = [
            f"This is the full fee schedule for all three plans, effective {F.date(f'fees.versions.v{ver}.effective_date')}.",
            "",
            "| Fee | Basic | Plus | Premium |",
            "|---|---|---|---|",
        ]
        for label, key in FEE_ROWS:
            lines.append(f"| {label} | " + " | ".join(fee_cell(p, key, ver) for p in PLANS) + " |")
        free_plus = F.raw(f"fees.versions.v{ver}.plus.free_out_of_network_withdrawals_per_month")
        lines += [
            "",
            "## Notes",
            "",
            f"- Plus includes {free_plus} free out-of-network ATM withdrawals each calendar month. The fee in the table applies after those.",
            f"- Premium reimburses ATM owner surcharges up to {F.fee('premium', 'atm_surcharge_reimbursement_monthly_cap_usd', ver)} a month.",
            f"- There are no overdraft fees ({F.v('fees.all_plans.overdraft_fee_usd')}), no fees for returned transfers and no inactivity fees on any plan.",
            f"- Incoming domestic wires are free ({F.fee('basic', 'incoming_domestic_wire_fee_usd', ver)}) on every plan.",
            "- ATM owners can add their own surcharge. That is separate from our fee.",
        ]
        return "\n".join(lines)

    return body


art("fees-all-v1", "Fees for all plans", S_PLANS, fees_all(1),
    eff=eff_of("fees.versions.v1.effective_date"), version=1)
art("fees-all-v2", "Fees for all plans", S_PLANS, fees_all(2),
    eff=eff_of("fees.versions.v2.effective_date"), version=2, supersedes="fees-all-v1")


def fees_plan(plan: str, ver: int) -> Callable[[], str]:
    def body() -> str:
        t("fee_schedule")
        name = pn(plan)
        lines = [
            f"Here is every fee that can apply to a {name} account, effective {F.date(f'fees.versions.v{ver}.effective_date')}. Fees are taken from your main balance when they're charged and show up in your activity feed with the word \"Fee\".",
            "",
            "| Fee | Amount |",
            "|---|---|",
        ]
        for label, key in FEE_ROWS:
            lines.append(f"| {label} | {fee_cell(plan, key, ver)} |")
        free_n = F.raw(f"fees.versions.v{ver}.{plan}.free_out_of_network_withdrawals_per_month")
        lines += ["", "## Details", ""]
        if plan == "plus":
            lines.append(f"Your first {free_n} out-of-network ATM withdrawals each calendar month are free. The fee applies from the next one.")
        elif plan == "premium":
            lines.append(f"Premium has no ATM fees from us, and we pay back ATM owner surcharges up to {F.fee(plan, 'atm_surcharge_reimbursement_monthly_cap_usd', ver)} each month.")
        else:
            lines.append("BrookLink ATMs are always free. The out-of-network fee applies to every withdrawal at any other ATM in the US.")
        if F.fee_raw(plan, "international_wire_out_fee_usd", ver) is None:
            t("intl_wire_not_on_basic")
            lines.append("")
            lines.append("Outgoing international wires aren't available on this plan.")
        lines += [
            "",
            f"No plan has overdraft fees, returned transfer fees or inactivity fees. The monthly fee is charged on your billing date and isn't prorated.",
        ]
        return "\n".join(lines)

    return body


for _p in PLANS:
    art(f"fees-schedule-{_p}-v1", f"{pn(_p)} plan fee schedule", S_PLANS, fees_plan(_p, 1), plans=[_p],
        eff=eff_of("fees.versions.v1.effective_date"), version=1)
    art(f"fees-schedule-{_p}-v2", f"{pn(_p)} plan fee schedule", S_PLANS, fees_plan(_p, 2), plans=[_p],
        eff=eff_of("fees.versions.v2.effective_date"), version=2, supersedes=f"fees-schedule-{_p}-v1")


def fee_changes_2026() -> str:
    t("fee_update_2026")
    return "\n".join([
        f"We updated our fees on {F.date('fees.versions.v2.effective_date')}. Here's everything that changed and what stayed the same.",
        "",
        "## What changed",
        "",
        f"- **Plus monthly fee:** from {F.fee('plus', 'monthly_fee_usd', 1)} to {F.fee('plus', 'monthly_fee_usd', 2)}.",
        f"- **Plus foreign transaction fee:** from {F.fee('plus', 'fx_fee_pct', 1)} to {F.fee('plus', 'fx_fee_pct', 2)}.",
        f"- **Out-of-network ATM fee on Basic and Plus:** from {F.fee('basic', 'out_of_network_atm_fee_usd', 1)} to {F.fee('basic', 'out_of_network_atm_fee_usd', 2)} per withdrawal.",
        f"- **Plus free out-of-network withdrawals:** from {F.raw('fees.versions.v1.plus.free_out_of_network_withdrawals_per_month')} to {F.raw('fees.versions.v2.plus.free_out_of_network_withdrawals_per_month')} a month.",
        "",
        "## What stayed the same",
        "",
        f"Basic is still free. Premium is still {F.fee('premium', 'monthly_fee_usd', 2)} a month with no ATM or foreign transaction fees. Card replacement, wire and cash deposit fees didn't change on any plan.",
        "",
        "## Why we made these changes",
        "",
        "Out-of-network ATM operators raised what they charge us, and we moved part of that cost into the fee. We lowered the Plus foreign transaction fee at the same time, so Plus members who travel pay less overall.",
        "",
        "If the new Plus price doesn't work for you, you can move to Basic in the app. The change takes effect on your next billing date.",
    ])


art("fee-changes-2026", "What changed in the March 2026 fee update", S_PLANS, fee_changes_2026,
    eff=eff_of("fees.versions.v2.effective_date"))


def plan_change() -> str:
    t("plan_change_timing", "plan_upgrade", "plan_downgrade")
    return "\n".join([
        "You can move between Basic, Plus and Premium in the app under Profile, then Plan. You don't need to contact support, though support can make the change for you if you ask.",
        "",
        "## Upgrading",
        "",
        "An upgrade takes effect immediately. We charge the new plan's full monthly fee that day, and that day becomes your new billing date. Your new limits, fees and savings rate apply from that moment.",
        "",
        "## Downgrading",
        "",
        "A downgrade takes effect on your next billing date. You keep your current plan's features until then. We don't refund part of a month.",
        "",
        "## Things to check before you downgrade",
        "",
        "- If you use Cushion and move to Basic, Cushion switches off. Any negative balance still has to be repaid.",
        "- Your Savings Pockets start earning the new plan's rate on the day the downgrade takes effect.",
        "- A Premium metal card keeps working until it expires. Replacements after that are plastic.",
        f"- If this is your first paid plan and you were charged less than {F.v('plan_billing.cooling_off_days')} ago, leaving the paid plan gets you a full refund of that charge. See \"Refunds of plan fees\".",
        "",
        "There's no limit on how often you can change plans.",
    ])


art("plan-change", "Changing your plan", S_PLANS, plan_change, eff="2025-01-01")


def plan_billing() -> str:
    t("plan_billing", "plan_fee_failed_payment", "plan_fee_not_goodwill")
    return "\n".join([
        "Paid plan fees are charged once a month on your billing date, from your main balance. Your billing date is the day you first joined a paid plan, or the day of your last upgrade.",
        "",
        "## If the fee can't be collected",
        "",
        f"If your balance is too low on the billing date, we try again after {F.v('plan_billing.failed_payment_retry_days')}. If the second try also fails, your plan moves to Basic. Cushion never covers plan fees. You can move back to a paid plan once you've added money.",
        "",
        "## Where to see it",
        "",
        "The charge shows in your activity feed as \"Plan fee\" with the plan name. Your next billing date is shown in the app under Profile, then Plan.",
        "",
        "## Refunds",
        "",
        f"Plan fees aren't prorated and aren't eligible for goodwill fee refunds. The only exception is the cooling-off rule for your first paid plan charge, which gives a full refund if you leave the paid plan within {F.v('plan_billing.cooling_off_days')} of that charge.",
    ])


art("plan-billing", "When and how plan fees are charged", S_PLANS, plan_billing, eff="2025-01-01")


def plan_fee_refunds() -> str:
    t("plan_fee_cooling_off", "plan_fee_not_goodwill", "closure_no_plan_refund")
    return "\n".join([
        f"We refund the first paid plan charge on your account in full if you leave the paid plan within {F.v('plan_billing.cooling_off_days')} of that charge. We call this the cooling-off period.",
        "",
        "## How it works",
        "",
        "- It applies only to the first paid plan charge your account has ever had. Later charges, even after switching plans, aren't covered.",
        "- When you ask for the refund, your plan returns to Basic right away rather than on your next billing date.",
        "- The refund goes back to your main balance, usually within minutes.",
        "",
        "## What isn't refunded",
        "",
        "Outside the cooling-off period, plan fees aren't refunded or prorated. That includes the month in which you downgrade and the month in which you close your account.",
        "",
        "## How to ask",
        "",
        "Message support in the app and say you'd like to cancel your paid plan under the cooling-off period. Support can process it for you straight away.",
    ])


art("plan-fee-refunds", "Refunds of plan fees", S_PLANS, plan_fee_refunds, eff="2025-01-01")


def fee_refunds() -> str:
    t("fee_refund_goodwill", "fee_refund_error", "plan_fee_not_goodwill")
    return "\n".join([
        "Support can refund some fees. There are two kinds of refund.",
        "",
        "## Fees charged in error",
        "",
        "If we charged a fee your plan doesn't have, or charged the same fee twice, we refund it. There's no limit on these refunds and they don't count toward anything else.",
        "",
        "## Goodwill refunds",
        "",
        f"Once in any rolling 12-month period, support can refund one fee of up to {F.v('refunds.goodwill_max_fee_usd')} as a courtesy, even if it was charged correctly. That covers ATM fees, foreign transaction fees, card replacement and expedited shipping fees, wire fees and cash deposit fees.",
        "",
        "Plan fees aren't eligible for goodwill refunds. They have their own cooling-off rule instead.",
        "",
        "## What support can't do",
        "",
        f"Support can't refund a second fee as goodwill within 12 months, can't refund a single fee above {F.v('refunds.goodwill_max_fee_usd')} as goodwill, and can't make exceptions to these rules. If you think a fee was charged in error, tell us why and we'll check.",
        "",
        "## How to ask",
        "",
        "Open the fee in your activity feed, tap Get help, and choose Question about a fee. Refunds show up in your balance right away.",
    ])


art("fee-refunds", "Asking for a fee refund", S_PLANS, fee_refunds, eff="2025-01-01")


# =====================================================================
# Cards
# =====================================================================

def card_activate() -> str:
    t("card_activate")
    return "\n".join([
        "Your physical card arrives inactive. You need to activate it in the app before you can use it for purchases or at ATMs.",
        "",
        "## How to activate",
        "",
        "1. Open the app and tap Cards.",
        "2. Tap Activate card.",
        "3. Scan the card with your camera, or type the last 4 digits and the expiry date.",
        "4. Set your PIN when the app asks.",
        "",
        "Activation is instant. Your card works right after.",
        "",
        "## Before your card arrives",
        "",
        "You can add a virtual card in the app and use it online or in a mobile wallet while you wait. It has its own number, so you don't need to update it when the physical card shows up.",
        "",
        "## If activation fails",
        "",
        f"Check that you're activating the newest card. An old or replaced card can't be activated. If your card hasn't arrived within {F.v('cards.standard_delivery_max_business_days')} of ordering it, message support so we can cancel it and send another one.",
        "",
        "Replacement cards need activating too. For a damaged card, your old card keeps working until you activate the new one.",
    ])


art("card-activate", "Activating your card", S_CARDS, card_activate)


def card_pin() -> str:
    t("card_pin", "never_ask")
    return "\n".join([
        "You set your PIN when you activate your card, and you can change it any time in the app.",
        "",
        "## Change your PIN",
        "",
        "1. Tap Cards, then Card settings.",
        "2. Tap Change PIN.",
        "3. Confirm it's you with your passcode or face or fingerprint unlock.",
        "4. Enter the new PIN twice.",
        "",
        "The new PIN works right away at card terminals. Some ATMs outside the BrookLink network only pick up the new PIN after your first chip-and-PIN purchase.",
        "",
        "## Forgot your PIN",
        "",
        "Use Change PIN in the app. You don't need the old one, because the app already knows it's you.",
        "",
        "## Locked PIN",
        "",
        "After several wrong tries, ATMs and terminals will refuse the PIN. Changing the PIN in the app unlocks it.",
        "",
        "## Keep it private",
        "",
        "Tallowbrook staff will never ask for your PIN. Support can't see it either. Anyone who asks for it, by phone, text, email or chat, isn't us.",
    ])


art("card-pin", "Setting or changing your PIN", S_CARDS, card_pin)


def card_freeze() -> str:
    t("card_freeze")
    return "\n".join([
        "Freezing your card blocks new purchases and ATM withdrawals straight away. It's the fastest thing to do if you've misplaced your card and think you might find it again.",
        "",
        "## How to freeze or unfreeze",
        "",
        "Tap Cards, choose the card, and switch Freeze card on. Switch it off to unfreeze. Both happen instantly, and you can do it as often as you like.",
        "",
        "## What still works while frozen",
        "",
        "- Recurring payments to merchants you've already authorized, such as subscriptions",
        "- Refunds from merchants",
        "- Deposits and incoming transfers",
        "- Your virtual card, which has its own freeze switch",
        "",
        "## Freeze or report lost?",
        "",
        "If you're sure the card is gone or stolen, report it lost or stolen instead. That cancels the card permanently and lets you order a replacement. A freeze is temporary and reversible.",
        "",
        "## Freeze versus fraud lock",
        "",
        "If the app says your card is locked by Tallowbrook, that's a fraud lock, not a freeze. You can't remove a fraud lock with the freeze switch. See \"Why is my card locked by Tallowbrook?\".",
        "",
        "Support can also freeze or unfreeze your card for you if you ask in the app.",
    ])


art("card-freeze", "Freezing and unfreezing your card", S_CARDS, card_freeze)


def lost_stolen(ver: int) -> Callable[[], str]:
    def body() -> str:
        t("card_lost_stolen")
        if ver == 1:
            F.mark("lost_stolen.versions.v1.steps")
            return "\n".join([
                "If your card is lost or stolen, call our card services line straight away so we can cancel it.",
                "",
                "## What to do",
                "",
                f"1. Call card services at {F.raw('meta.support_phone')}. The line is open {F.v('lost_stolen.versions.v1.phone_line_hours')} a day.",
                "2. The agent confirms your identity, cancels the card and orders a replacement.",
                "3. Check your recent transactions in the app and tell the agent about any you don't recognize.",
                "",
                "## Your replacement",
                "",
                f"The replacement card arrives in {F.n('cards.standard_delivery_min_business_days')} to {F.n('cards.standard_delivery_max_business_days')} business days. Basic members pay {F.fee('basic', 'card_replacement_fee_usd', 1)} for a replacement. Plus and Premium replacements are free.",
                "",
                "Until the new card arrives, you can use BrookPay and transfers from the app. Virtual cards aren't available.",
                "",
                "## Unauthorized transactions",
                "",
                f"You're not responsible for unauthorized card transactions reported within {F.v('disputes.unauthorized_window_days')} of the statement they first appear on.",
                "",
                "## Found your card?",
                "",
                "Once a card is reported, it's cancelled for good, even if you find it later. Destroy it and wait for the replacement.",
            ])
        return "\n".join([
            "If your card is lost or stolen, you can deal with it in the app in a couple of minutes. You don't need to call us.",
            "",
            "## What to do",
            "",
            "1. **Freeze the card.** Tap Cards and switch Freeze card on. This stops new purchases right away.",
            "2. **Report it.** If you can't find it, tap Report lost or stolen. This cancels the card permanently. It can't be undone.",
            "3. **Add a virtual card.** You can create one in the app and use it online and in mobile wallets right away while the replacement ships.",
            "4. **Order a replacement.** Pick standard or expedited shipping. Replacement and shipping fees depend on your plan.",
            "5. **Check your transactions.** Dispute anything you don't recognize.",
            "",
            "## Delivery",
            "",
            f"Standard shipping takes {F.range_bd('cards.standard_delivery_min_business_days', 'cards.standard_delivery_max_business_days')}. Expedited shipping takes {F.range_bd('cards.expedited_delivery_min_business_days', 'cards.expedited_delivery_max_business_days')}.",
            "",
            "## Unauthorized transactions",
            "",
            f"You're not responsible for unauthorized card transactions you report within {F.v('disputes.unauthorized_window_days')} of the statement they first appear on. If you found your card after freezing it, just unfreeze it. Nothing else changes.",
        ])

    return body


art("card-lost-stolen-v1", "Lost or stolen card", S_CARDS, lost_stolen(1),
    eff=eff_of("lost_stolen.versions.v1.effective_date"), version=1)
art("card-lost-stolen-v2", "Lost or stolen card", S_CARDS, lost_stolen(2),
    eff=eff_of("lost_stolen.versions.v2.effective_date"), version=2, supersedes="card-lost-stolen-v1")


def card_damaged() -> str:
    t("card_damaged")
    return "\n".join([
        "If your card is cracked, bent, or the chip or contactless stops working, order a replacement in the app.",
        "",
        "## How to order",
        "",
        "Tap Cards, choose the card, then Replace card, then Damaged. Confirm your address and pick a shipping speed.",
        "",
        "## Your old card keeps working",
        "",
        "When you report a card as damaged, we don't cancel it. You can keep using it (if it still works) until you activate the replacement. Activating the new card switches the old one off.",
        "",
        "The new card has the same card number, a new expiry date and a new security code. Merchants that store your card may need the new expiry date.",
        "",
        "## Cost and shipping",
        "",
        f"The replacement fee is the same as for a lost card: {F.fee('basic', 'card_replacement_fee_usd')} on Basic and free on Plus and Premium. Standard shipping takes {F.range_bd('cards.standard_delivery_min_business_days', 'cards.standard_delivery_max_business_days')}. Expedited shipping takes {F.range_bd('cards.expedited_delivery_min_business_days', 'cards.expedited_delivery_max_business_days')} and costs {F.fee('basic', 'expedited_shipping_fee_usd')} on Basic and Plus. It's free on Premium.",
        "",
        "If you're not sure whether the card is damaged or lost, freeze it first. You can decide afterward.",
    ])


art("card-damaged", "Replacing a damaged card", S_CARDS, card_damaged)


def card_delivery() -> str:
    t("card_delivery")
    return "\n".join([
        "We mail physical cards to the residential address on your profile. Cards can't be sent to a PO box or to a different address than the one on file.",
        "",
        "## How long it takes",
        "",
        f"- Standard shipping: {F.range_bd('cards.standard_delivery_min_business_days', 'cards.standard_delivery_max_business_days')}",
        f"- Expedited shipping: {F.range_bd('cards.expedited_delivery_min_business_days', 'cards.expedited_delivery_max_business_days')}, with tracking",
        "",
        "Business days don't include weekends or bank holidays.",
        "",
        "## Tracking",
        "",
        "Expedited cards come with a tracking link in the app under Cards. Standard cards don't have tracking, but the app shows the date we mailed it.",
        "",
        "## If it doesn't arrive",
        "",
        f"If a standard card hasn't arrived {F.v('cards.standard_delivery_max_business_days')} after we mailed it, message support. We'll cancel the card in transit and send a new one. There's no fee for a card that got lost in the mail.",
        "",
        "## Moving?",
        "",
        "Update your address before ordering a card. An address change may need proof of address. See \"Updating your address\".",
    ])


art("card-delivery", "Card delivery times and tracking", S_CARDS, card_delivery)


def card_expedited() -> str:
    t("card_expedited")
    return "\n".join([
        f"Expedited shipping gets a new or replacement card to you in {F.range_bd('cards.expedited_delivery_min_business_days', 'cards.expedited_delivery_max_business_days')} instead of the usual {F.range_bd('cards.standard_delivery_min_business_days', 'cards.standard_delivery_max_business_days')}.",
        "",
        "## Cost by plan",
        "",
        "| Plan | Expedited shipping |",
        "|---|---|",
        *[f"| {pn(p)} | {F.fee_or_free(p, 'expedited_shipping_fee_usd')} |" for p in PLANS],
        "",
        "The shipping fee is separate from any card replacement fee. On Basic, a lost card sent expedited costs both fees.",
        "",
        "## How to choose it",
        "",
        "When you order a card in the app, pick Expedited on the shipping screen. You can't upgrade the shipping speed after the card is mailed.",
        "",
        "## Tracking",
        "",
        "Expedited cards ship with tracking. The link appears in the app under Cards once the card leaves our facility.",
        "",
        "Expedited shipping is only available to US addresses. Premium members who are abroad can ask for an emergency card instead.",
    ])


art("card-expedited-shipping", "Expedited card shipping", S_CARDS, card_expedited)


def card_virtual() -> str:
    t("card_virtual")
    return "\n".join([
        F.mark("cards.virtual_card") + "Every verified member can create one virtual card in the app. It works right away for online purchases and in mobile wallets, so you can keep spending while a physical card is on its way.",
        "",
        "## Create a virtual card",
        "",
        "Tap Cards, then Add virtual card. The card number, expiry date and security code are shown in the app after you confirm it's you.",
        "",
        "## How it's different from your physical card",
        "",
        "- It has its own card number.",
        "- It has its own freeze switch. Freezing your physical card doesn't freeze the virtual one.",
        "- It can't be used at ATMs.",
        "- Reporting your physical card lost or stolen doesn't cancel the virtual card.",
        "",
        "## Replacing a virtual card",
        "",
        "If you think the virtual card number has been exposed, delete it in the app and create a new one. There's no fee. Remember to update any merchants that stored the old number.",
        "",
        "Limits and fees are the same as for your physical card, and virtual card spending counts toward the same daily card spending limit.",
    ])


art("card-virtual", "Virtual cards", S_CARDS, card_virtual)


def card_declined() -> str:
    t("card_declined")
    return "\n".join([
        "If a purchase was declined, the app usually tells you why in the activity feed. These are the most common reasons.",
        "",
        "## Not enough money",
        "",
        "We don't let your balance go negative on most transactions, so a purchase larger than your balance is declined. There's no fee for that. Plus and Premium members with Cushion may be covered for small debit card purchases.",
        "",
        "## Your daily limit",
        "",
        f"Daily card spending limits are {F.lim('basic', 'daily_card_spend_limit_usd')} on Basic, {F.lim('plus', 'daily_card_spend_limit_usd')} on Plus and {F.lim('premium', 'daily_card_spend_limit_usd')} on Premium. Limits can't be raised except by changing plan.",
        "",
        "## Card settings",
        "",
        "Check that the card is activated and isn't frozen, and that the right controls are on. If international purchases are switched off, foreign merchants will decline, even for online orders.",
        "",
        "## Fraud checks",
        "",
        "If our systems think a purchase looks unusual, we decline it and send a fraud alert. Confirm the alert in the app and try again. If you don't answer, the card may stay locked.",
        "",
        "## Where you are",
        "",
        "Cards don't work in comprehensively sanctioned countries and regions.",
        "",
        "If none of these explain it, message support with the merchant name and time of the purchase.",
    ])


art("card-declined", "Why was my card declined?", S_CARDS, card_declined)


def card_wallets() -> str:
    t("card_mobile_wallets")
    return "\n".join([
        "You can add your Tallowbrook physical or virtual card to the mobile wallet on your phone or watch and pay by tapping at checkout.",
        "",
        "## Add a card",
        "",
        "In the Tallowbrook app, tap Cards, choose the card, and tap Add to wallet. The app sends the card details straight to your wallet, so you don't have to type them.",
        "",
        "## Things to know",
        "",
        "- Your wallet uses a device-specific number instead of your card number. Merchants never see your real card number.",
        "- Freezing a card in the app also blocks wallet payments on that card.",
        "- If you report a card lost or stolen, remove it from your wallet and add the replacement once it's activated. A virtual card can be added right away.",
        "- Limits, fees and Cushion work the same way for wallet payments as for card payments.",
        "",
        "## If adding fails",
        "",
        "Make sure the card is activated and not frozen, and that your phone has a screen lock set. Wallets won't add a card on a device without a screen lock.",
        "",
        "If you've lost your phone, freeze your cards from another device by signing in to the app, or message support.",
    ])


art("card-mobile-wallets", "Using mobile wallets", S_CARDS, card_wallets)


def card_metal() -> str:
    t("metal_card")
    return "\n".join([
        "Premium members get a metal debit card. It works exactly like the plastic card, with the same number, limits and controls.",
        "",
        "## Getting your metal card",
        "",
        "When you upgrade to Premium, the app asks whether you want a metal card now or at your next renewal. If you order it now, your current card keeps working until you activate the metal one.",
        "",
        "## Replacing it",
        "",
        f"Replacements for a lost, stolen or damaged metal card are metal and free on Premium. Expedited shipping is free on Premium too, and takes {F.range_bd('cards.expedited_delivery_min_business_days', 'cards.expedited_delivery_max_business_days')}.",
        "",
        "## If you leave Premium",
        "",
        "Your metal card keeps working until it expires. After that, or if you need a replacement, you'll get a plastic card.",
        "",
        "## Disposal",
        "",
        "Metal cards can't be cut with scissors. When you replace one, you can mail it back to us in the prepaid envelope that comes with the new card, and we'll recycle it.",
        "",
        "Contactless works on metal cards the same way it does on plastic.",
    ])


art("card-metal", "The Premium metal card", S_CARDS, card_metal, plans=["premium"])


def card_renewal() -> str:
    t("card_renewal")
    return "\n".join([
        f"We mail a new card automatically about {F.v('cards.renewal_mailed_before_expiry_days')} before your current card expires. You don't need to ask for it.",
        "",
        "## What changes",
        "",
        "The renewed card has the same card number, a new expiry date and a new security code. Most merchants that store your card get the new expiry date automatically, but some don't, so check your subscriptions.",
        "",
        "## What to do when it arrives",
        "",
        "Activate the new card in the app. Your old card keeps working until you activate the new one or until it expires, whichever comes first.",
        "",
        "## If it doesn't arrive",
        "",
        "Check that your address in the app is current. If your old card expires and the new one still hasn't arrived, message support and we'll send another one at no cost.",
        "",
        "## No renewal?",
        "",
        "If your account is restricted or under review, we can't send a renewal card until that's resolved.",
    ])


art("card-renewal", "Card expiry and renewal", S_CARDS, card_renewal)


def card_controls() -> str:
    t("card_controls")
    return "\n".join([
        "Card controls let you decide where your card can be used. Changes take effect straight away.",
        "",
        "## Controls you can set",
        "",
        "In the app, tap Cards, choose the card, then Controls. You can switch each of these on or off:",
        "",
        "- Online purchases",
        "- International purchases (in stores and online with foreign merchants)",
        "- ATM withdrawals",
        "- Contactless payments",
        "",
        "## Common mix-ups",
        "",
        "- A declined online order from a foreign website is often the International purchases control, not the Online purchases one.",
        "- Switching off ATM withdrawals on your physical card doesn't affect your virtual card, which can't be used at ATMs anyway.",
        "- Controls don't change your limits. Limits are set by your plan.",
        "",
        "## Controls versus freezing",
        "",
        "Controls block certain kinds of transactions. Freezing blocks almost everything. If you think your card is at risk, freeze it.",
        "",
        "Support can explain your current settings, but you change controls yourself in the app.",
    ])


art("card-controls", "Card controls", S_CARDS, card_controls)


def card_replacement(plan: str) -> Callable[[], str]:
    def body() -> str:
        t("card_replacement_fee")
        name = pn(plan)
        repl = F.fee_raw(plan, "card_replacement_fee_usd")
        exp = F.fee_raw(plan, "expedited_shipping_fee_usd")
        lines = [f"This is what a replacement card costs on the {name} plan, whether it's lost, stolen or damaged.", ""]
        lines += ["| | Cost |", "|---|---|",
                  f"| Replacement card | {F.fee_or_free(plan, 'card_replacement_fee_usd')} |",
                  f"| Standard shipping ({F.range_bd('cards.standard_delivery_min_business_days', 'cards.standard_delivery_max_business_days')}) | free |",
                  f"| Expedited shipping ({F.range_bd('cards.expedited_delivery_min_business_days', 'cards.expedited_delivery_max_business_days')}) | {F.fee_or_free(plan, 'expedited_shipping_fee_usd')} |",
                  ""]
        if repl and exp:
            lines.append(f"On {name}, if you choose expedited shipping you pay both the replacement fee and the shipping fee. Standard shipping only costs the replacement fee.")
        elif exp:
            lines.append(f"On {name} the card itself is free. You only pay if you choose expedited shipping.")
        else:
            lines.append(f"On {name} both the card and expedited shipping are free.")
            if plan == "premium":
                lines.append("")
                lines.append("Your replacement is a metal card.")
        lines += [
            "",
            "## When we don't charge",
            "",
            "There's no fee when a card is lost in the mail, when we replace a card because of a problem on our side, or for your automatic renewal card.",
            "",
            "## How to order",
            "",
            "Tap Cards, choose the card, then Replace card. Pick lost, stolen or damaged, and choose a shipping speed. Fees are charged to your main balance when you place the order.",
            "",
            "Support can also order a replacement for you in the app chat. The same fees apply either way.",
        ]
        return "\n".join(lines)

    return body


for _p in PLANS:
    art(f"card-replacement-{_p}", f"Card replacement fees on {pn(_p)}", S_CARDS, card_replacement(_p), plans=[_p])


def card_fraud_lock() -> str:
    t("fraud_lock", "fraud_alerts")
    return "\n".join([
        "If the app says your card is \"Locked by Tallowbrook\", our fraud systems have placed a fraud lock on it. This is different from a freeze you set yourself.",
        "",
        "## Why it happens",
        "",
        f"We lock a card when we see activity that looks like fraud, or when a fraud alert goes unanswered for {F.v('security.fraud_alert_response_hours')}.",
        "",
        "## How to get it removed",
        "",
        "- If there's an open fraud alert in the app, answer it. Tapping \"Yes, that was me\" usually removes the lock straight away.",
        "- If there's no alert, message us in the app. Support will pass your case to the Fraud team. Support can't remove a fraud lock, and neither can the freeze switch.",
        "",
        f"The Fraud team replies within {F.v('support.escalation_response_business_days')}, often much sooner.",
        "",
        "## Meanwhile",
        "",
        "Deposits and incoming transfers still arrive while your card is locked. You can also still use BrookPay and transfers, unless the Fraud team has restricted those too.",
        "",
        "We will never ask you to confirm a fraud alert by giving us your full card number, PIN or a one-time code.",
    ])


art("card-fraud-lock", "Why is my card locked by Tallowbrook?", S_CARDS, card_fraud_lock)


# =====================================================================
# ATMs and cash
# =====================================================================

def atm_network() -> str:
    t("atm_network")
    return "\n".join([
        f"BrookLink is our ATM network. Withdrawals at BrookLink ATMs are free on every plan ({F.v('atm.in_network_fee_usd')} from us, and the ATM owner doesn't add a surcharge).",
        "",
        "## Find a BrookLink ATM",
        "",
        "Open the app, tap Cash, then ATM map. The map shows BrookLink ATMs near you with opening hours, since some are inside stores.",
        "",
        "## Using other ATMs",
        "",
        "You can use any ATM that accepts Visa. Outside BrookLink, you may pay an out-of-network fee depending on your plan, and the ATM owner may add a surcharge. The ATM screen should show the owner's surcharge before you confirm.",
        "",
        "## Daily limits",
        "",
        "ATM withdrawals count toward a daily ATM limit that depends on your plan. See \"ATM withdrawal limits\".",
        "",
        "## Depositing cash",
        "",
        "BrookLink ATMs don't take cash deposits. To add cash, use a participating retail store instead. See \"Depositing cash at a store\".",
    ])


art("atm-network", "Finding a BrookLink ATM", S_ATM, atm_network)


def atm_limits() -> str:
    t("atm_limits", "limits_fixed")
    return "\n".join([
        "How much cash you can take out each day depends on your plan. The limit covers all ATMs combined, in the US and abroad.",
        "",
        "| Plan | Daily ATM limit |",
        "|---|---|",
        *[f"| {pn(p)} | {F.lim(p, 'daily_atm_withdrawal_limit_usd')} |" for p in PLANS],
        "",
        "The daily limit resets at midnight Eastern Time.",
        "",
        "## Can I raise my limit?",
        "",
        F.mark("limits.limit_changes") + "No. Limits are set by plan, and support can't raise them for a day or make exceptions. The only way to get a higher limit is to change plan.",
        "",
        "## ATM limits versus the ATM's own limit",
        "",
        "Many ATMs have their own maximum per withdrawal that's lower than ours. If you need more, you can make more than one withdrawal, as long as the total stays within your daily limit. Each withdrawal outside BrookLink may carry a fee.",
        "",
        "Cash withdrawals are never covered by Cushion.",
    ])


art("atm-limits", "ATM withdrawal limits", S_ATM, atm_limits)


def atm_abroad() -> str:
    t("atm_abroad")
    return "\n".join([
        "Your Tallowbrook card works at ATMs abroad that accept Visa.",
        "",
        "## What it costs",
        "",
        "| Plan | International ATM fee |",
        "|---|---|",
        *[f"| {pn(p)} | {F.fee_or_free(p, 'international_atm_fee_usd')} |" for p in PLANS],
        "",
        "The international ATM fee replaces the out-of-network fee for withdrawals outside the US. The foreign transaction fee doesn't apply to ATM withdrawals. It only applies to purchases.",
        "",
        "The ATM owner may add its own surcharge. Premium surcharge reimbursement covers ATMs abroad too, up to the same monthly cap.",
        "",
        "## Pick the local currency",
        "",
        "If the ATM offers to convert the withdrawal into US dollars for you, decline and choose the local currency. The ATM's rate is usually worse than the network rate.",
        "",
        "## Limits",
        "",
        "Withdrawals abroad count toward the same daily ATM limit as withdrawals at home.",
        "",
        "## If the ATM keeps your card",
        "",
        "Freeze the card in the app straight away and report it lost. Premium members can ask for an emergency card abroad. Everyone else can use a virtual card in a mobile wallet until they're home.",
    ])


art("atm-abroad", "Using ATMs abroad", S_ATM, atm_abroad)


def atm_no_cash() -> str:
    t("atm_cash_not_dispensed", "dispute_how_to_file")
    return "\n".join([
        "If an ATM took money from your account but didn't give you the cash, or gave you less than you asked for, you can dispute it.",
        "",
        "## What to do",
        "",
        "1. Note the ATM location, date and time, and keep any receipt.",
        "2. Check the app. Sometimes the ATM reverses the withdrawal on its own within a few days, and the transaction disappears or shows a matching credit.",
        f"3. If it's still there once the withdrawal has posted, open a dispute in the app within {F.v('atm.cash_not_dispensed_dispute_window_days')}. Tap the transaction, then Get help, then Cash not dispensed.",
        "",
        "## What happens next",
        "",
        f"We ask the ATM operator for its records. Most ATM cases are resolved within {F.v('atm.cash_not_dispensed_resolution_business_days')}. If the records show the cash wasn't dispensed, we credit the full amount, including any ATM fee we charged for that withdrawal.",
        "",
        "## Pending withdrawals",
        "",
        "You can't dispute a withdrawal while it's still pending. Wait until it posts.",
    ])


art("atm-cash-not-dispensed", "ATM didn't give me my cash", S_ATM, atm_no_cash)


def cash_deposit() -> str:
    t("cash_deposit_retail")
    return "\n".join([
        F.mark("atm.retail_cash_deposit_note") + "You can add cash to your Tallowbrook account at participating retail stores. Open the app, tap Cash, then Deposit cash, and show the barcode to the cashier with your cash.",
        "",
        "## Fees",
        "",
        "| Plan | Fee per deposit |",
        "|---|---|",
        *[f"| {pn(p)} | {F.fee_or_free(p, 'retail_cash_deposit_fee_usd')} |" for p in PLANS],
        "",
        "## Limits",
        "",
        "| Plan | Daily cash deposit limit |",
        "|---|---|",
        *[f"| {pn(p)} | {F.lim(p, 'daily_retail_cash_deposit_limit_usd')} |" for p in PLANS],
        "",
        "## When the money arrives",
        "",
        "Funds usually appear within minutes. Keep the store receipt until you see the deposit in the app.",
        "",
        "## Things to know",
        "",
        "- BrookLink ATMs don't accept cash deposits.",
        "- The barcode is valid for one deposit only.",
        "- Stores set their own maximum per visit, which can be lower than your daily limit.",
    ])


art("cash-deposit-retail", "Depositing cash at a store", S_ATM, cash_deposit)


def atm_surcharges() -> str:
    t("atm_surcharge")
    return "\n".join([
        F.mark("atm.operator_surcharge_note") + "When you use an ATM outside BrookLink, up to two different charges can apply. They're easy to mix up.",
        "",
        "## The ATM owner's surcharge",
        "",
        "This is set by whoever owns the ATM, not by us. The ATM shows it on screen before you confirm, and it's added to the withdrawal amount. For example, if you withdraw cash and the owner charges a surcharge, one transaction for the total appears in your activity.",
        "",
        "## Tallowbrook's out-of-network fee",
        "",
        f"This is our fee, charged as a separate line called \"Out-of-network ATM fee\". It's {F.fee('basic', 'out_of_network_atm_fee_usd')} on Basic and on Plus after your free withdrawals, and free on Premium.",
        "",
        "## Premium surcharge reimbursement",
        "",
        f"Premium pays back ATM owner surcharges up to {F.fee('premium', 'atm_surcharge_reimbursement_monthly_cap_usd')} per calendar month. The refund is credited automatically on the first business day of the next month.",
        "",
        "## Avoiding both",
        "",
        "Use a BrookLink ATM. They never charge a surcharge, and we never charge a fee for them.",
    ])


art("atm-surcharges", "ATM owner surcharges and our fees", S_ATM, atm_surcharges)


def atm_fees(plan: str, ver: int) -> Callable[[], str]:
    def body() -> str:
        t("atm_fees")
        name = pn(plan)
        oon = F.fee(plan, "out_of_network_atm_fee_usd", ver)
        intl = F.fee_or_free(plan, "international_atm_fee_usd", ver)
        free_n = F.raw(f"fees.versions.v{ver}.{plan}.free_out_of_network_withdrawals_per_month")
        lines = [f"Here's what ATM withdrawals cost on the {name} plan.", ""]
        lines += ["| ATM | Tallowbrook fee |", "|---|---|", f"| BrookLink, in the US | {F.v('atm.in_network_fee_usd')} |"]
        if plan == "premium":
            lines.append("| Any other ATM in the US | no fee |")
        elif free_n:
            lines.append(f"| Any other ATM in the US | first {free_n} each month free, then {oon} |")
        else:
            lines.append(f"| Any other ATM in the US | {oon} |")
        lines.append(f"| ATMs outside the US | {intl} |")
        lines.append("")
        if plan == "premium":
            lines.append(f"Premium also reimburses ATM owner surcharges up to {F.fee(plan, 'atm_surcharge_reimbursement_monthly_cap_usd', ver)} each calendar month, in the US and abroad.")
        elif free_n:
            lines.append(f"The {free_n} free withdrawals reset on the first day of each calendar month. Unused ones don't carry over.")
        else:
            lines.append("Every out-of-network withdrawal is charged the fee. To avoid it, use a BrookLink ATM.")
        lines += [
            "",
            "## Surcharges",
            "",
            "ATM owners outside BrookLink can add their own surcharge on top of our fee. The ATM shows it on screen before you confirm.",
            "",
            "## Daily limit",
            "",
            f"You can withdraw up to {F.lim(plan, 'daily_atm_withdrawal_limit_usd')} a day on {name}.",
            "",
            "## Where fees show up",
            "",
            "Any Tallowbrook ATM fee appears as its own line in your activity feed, right after the withdrawal, so you can always see what you paid us and what the ATM owner charged. The app's ATM map shows the nearest BrookLink ATMs.",
        ]
        return "\n".join(lines)

    return body


art("atm-fees-basic-v1", "ATM fees on Basic", S_ATM, atm_fees("basic", 1), plans=["basic"],
    eff=eff_of("fees.versions.v1.effective_date"), version=1)
art("atm-fees-basic-v2", "ATM fees on Basic", S_ATM, atm_fees("basic", 2), plans=["basic"],
    eff=eff_of("fees.versions.v2.effective_date"), version=2, supersedes="atm-fees-basic-v1")
art("atm-fees-plus-v1", "ATM fees on Plus", S_ATM, atm_fees("plus", 1), plans=["plus"],
    eff=eff_of("fees.versions.v1.effective_date"), version=1)
art("atm-fees-plus-v2", "ATM fees on Plus", S_ATM, atm_fees("plus", 2), plans=["plus"],
    eff=eff_of("fees.versions.v2.effective_date"), version=2, supersedes="atm-fees-plus-v1")
art("atm-fees-premium", "ATM fees on Premium", S_ATM, atm_fees("premium", 2), plans=["premium"],
    eff=eff_of("fees.versions.v1.effective_date"))


# =====================================================================
# Travel and foreign transactions
# =====================================================================

def travel_abroad() -> str:
    t("card_abroad")
    return "\n".join([
        "Your Tallowbrook card works in most countries, anywhere Visa is accepted.",
        "",
        "## Before you go",
        "",
        "- Keep notifications on for the app so you can answer fraud alerts quickly.",
        "- Make sure International purchases is switched on under Cards, then Controls.",
        "- Check your plan's foreign transaction and international ATM fees.",
        "- Add a virtual card to your phone's wallet as a backup.",
        "",
        "## What it costs",
        "",
        f"Purchases in a foreign currency carry a foreign transaction fee of {F.fee('basic', 'fx_fee_pct')} on Basic, {F.fee('plus', 'fx_fee_pct')} on Plus and {F.fee('premium', 'fx_fee_pct')} on Premium. ATM withdrawals abroad have their own international ATM fee instead.",
        "",
        "## Paying in the local currency",
        "",
        "When a card terminal asks whether to pay in US dollars or the local currency, choose the local currency.",
        "",
        "## If your card is lost abroad",
        "",
        "Freeze it in the app and report it. Standard and expedited shipping only go to US addresses. Premium members can get an emergency card delivered abroad. Everyone else can keep spending with a virtual card in a mobile wallet.",
        "",
        "Do you need to tell us you're traveling? No. See \"Do I need a travel notice?\".",
    ])


art("travel-card-abroad", "Using your card abroad", S_TRAVEL, travel_abroad)


def travel_notice(ver: int) -> Callable[[], str]:
    def body() -> str:
        t("travel_notice")
        if ver == 1:
            F.mark("travel.versions.v1.rule", "travel.versions.v1.travel_notice_required")
            return "\n".join([
                "Yes. Set a travel notice before you leave the US so your card isn't declined abroad.",
                "",
                "## How to set a travel notice",
                "",
                "1. Open the app and tap Profile, then Travel mode.",
                "2. Add each country you'll visit.",
                "3. Enter your departure and return dates.",
                "",
                f"Set the notice at least {F.v('travel.versions.v1.notice_min_days_before_departure')} before you leave, so it reaches our fraud systems in time. One notice can cover a trip of up to {F.v('travel.versions.v1.notice_max_trip_days')}. For longer trips, add a second notice before the first one ends.",
                "",
                "## If you forget",
                "",
                "Purchases abroad without a travel notice are likely to be declined, and your card may be locked. If that happens, message support in the app and we'll help you unlock it.",
                "",
                "## Changing your plans",
                "",
                "You can edit or delete a travel notice in Travel mode at any time. If you come home early, delete the notice so any later foreign charges are checked as usual.",
            ])
        F.mark("travel.versions.v2.rule", "travel.versions.v2.travel_notice_required")
        return "\n".join([
            "No. You don't need to tell us before you travel. We retired travel notices and the Travel mode setting, and our fraud checks now use your location and spending patterns instead.",
            "",
            "## What to do instead",
            "",
            "- Keep app notifications on. If we see an unusual purchase, we'll send a fraud alert that you can confirm with one tap.",
            f"- Answer fraud alerts quickly. If an alert goes unanswered for {F.v('security.fraud_alert_response_hours')}, the card stays locked until you contact us.",
            "- Check that International purchases is switched on under Cards, then Controls.",
            "",
            "## Old travel notices",
            "",
            "If you set a travel notice before we retired the feature, there's nothing you need to do. Old notices were deleted and don't affect your card.",
            "",
            "## Still worried?",
            "",
            "Adding a virtual card to your phone's wallet before you leave gives you a backup if your physical card is lost or declined.",
        ])

    return body


art("travel-notice-v1", "Do I need a travel notice?", S_TRAVEL, travel_notice(1),
    eff=eff_of("travel.versions.v1.effective_date"), version=1)
art("travel-notice-v2", "Do I need a travel notice?", S_TRAVEL, travel_notice(2),
    eff=eff_of("travel.versions.v2.effective_date"), version=2, supersedes="travel-notice-v1")


def travel_dcc() -> str:
    t("dcc")
    return "\n".join([
        F.mark("travel.dcc_advice") + "Abroad, card terminals and ATMs sometimes offer to charge you in US dollars instead of the local currency. This is called dynamic currency conversion. It sounds convenient, but it's almost always more expensive.",
        "",
        "## What to choose",
        "",
        "Always choose the local currency. Then the conversion happens at the card network's rate, plus your plan's foreign transaction fee.",
        "",
        "## Why dollars cost more",
        "",
        "If you pick US dollars, the merchant or ATM owner converts the amount using its own rate, which usually includes a markup. Your plan's foreign transaction fee may still apply, because the merchant is still abroad.",
        "",
        "## Online shopping",
        "",
        "Foreign websites sometimes show prices in dollars using their own conversion. If you can switch the site to its home currency, that's usually cheaper.",
        "",
        "## Already paid in dollars?",
        "",
        "Dynamic currency conversion isn't a reason to dispute a purchase, because you agreed to it at the terminal. Next time, choose the local currency.",
    ])


art("travel-dcc", "Paying in local currency or dollars", S_TRAVEL, travel_dcc)


def travel_rates() -> str:
    t("fx_rate_source")
    return "\n".join([
        F.mark("travel.fx_rate_source") + "When you pay in a foreign currency, we convert the amount to US dollars using the card network's exchange rate on the day the transaction settles. We don't add a markup to that rate.",
        "",
        "## What you pay on top",
        "",
        "Your plan's foreign transaction fee is added as a percentage of the converted amount:",
        "",
        *[f"- {pn(p)}: {F.fee(p, 'fx_fee_pct')}" for p in PLANS],
        "",
        "## Why the amount changes",
        "",
        "A pending transaction shows an estimate based on the rate when you paid. The final amount uses the rate on the day it settles, usually a day or two later, so it can move slightly up or down.",
        "",
        "## Refunds in a foreign currency",
        "",
        "If a foreign merchant refunds you, we convert the refund at the rate on the day it settles. That can be a little more or less than you originally paid. We refund the foreign transaction fee on the refunded part.",
        "",
        "The fee applies to purchases made in a foreign currency, and to purchases in US dollars processed by a merchant outside the US.",
    ])


art("travel-exchange-rates", "Exchange rates we use", S_TRAVEL, travel_rates)


def travel_blocked() -> str:
    t("blocked_regions")
    return "\n".join([
        F.mark("travel.blocked_regions") + "Tallowbrook cards don't work in comprehensively sanctioned countries and regions. Purchases, ATM withdrawals and online orders from merchants based there are declined.",
        "",
        "## Why",
        "",
        "US banks and card programs must follow US sanctions rules. We can't make exceptions, and support can't unblock a sanctioned location for you.",
        "",
        "## Signing in from abroad",
        "",
        "You can usually use the app from anywhere, but access from a sanctioned location may be blocked and could lead to a review of your account.",
        "",
        "## Declined somewhere else?",
        "",
        "If you're in a country that isn't sanctioned and your card is declined, the cause is more likely a fraud check, your card controls, or your balance. See \"Why was my card declined?\".",
        "",
        "## Sending money abroad",
        "",
        "International wires to sanctioned countries are also blocked. Wires to other countries may still be reviewed before they're sent.",
    ])


art("travel-blocked-regions", "Countries where cards don't work", S_TRAVEL, travel_blocked)


def travel_emergency() -> str:
    t("emergency_card_abroad")
    return "\n".join([
        f"If you're a Premium member and your card is lost, stolen or damaged while you're outside the US, we can send an emergency card to where you're staying. It arrives within {F.v('cards.premium_emergency_card_abroad_business_days')} and costs {F.v('cards.emergency_card_abroad_fee_usd')}.",
        "",
        "## How to request one",
        "",
        "1. Freeze or report your card in the app.",
        "2. Message support and ask for an emergency card.",
        "3. Give the full address where you'll be for the next few days, such as a hotel.",
        "",
        "Delivery can take longer in remote areas. The card is activated in the app like any other card.",
        "",
        "## Basic and Plus members",
        "",
        "Emergency cards abroad aren't available on Basic or Plus. Standard and expedited shipping only go to US addresses. Instead, add a virtual card in the app and use it in your phone's wallet while you're away. Your replacement can be sent to your US address.",
        "",
        "## Cash abroad",
        "",
        "Virtual cards can't be used at ATMs. If you need cash, you can ask someone with a Tallowbrook account to send you money by BrookPay and then use a card-free option at your destination, or rely on card payments in stores.",
    ])


art("travel-emergency-card", "Emergency card while abroad", S_TRAVEL, travel_emergency)


def fx_fees(plan: str, ver: int) -> Callable[[], str]:
    def body() -> str:
        t("fx_fee")
        name = pn(plan)
        fee_s = F.fee(plan, "fx_fee_pct", ver)
        val = F.fee_raw(plan, "fx_fee_pct", ver)
        lines = [
            f"On the {name} plan, the foreign transaction fee is {fee_s} of the purchase amount in US dollars.",
            "",
            "## When it applies",
            "",
            "- Purchases in a foreign currency, in a store or online",
            "- Purchases in US dollars processed by a merchant outside the US",
            "",
            "It doesn't apply to ATM withdrawals abroad. Those have the international ATM fee instead.",
            "",
        ]
        if float(val) == 0:
            lines += ["## Example", "", f"On {name}, a purchase abroad costs exactly the converted amount. No fee is added."]
        else:
            lines += [
                "## Example",
                "",
                f"If a purchase converts to a given dollar amount, the fee is {fee_s} of that amount. It's added as a separate line in your activity feed, called \"Foreign transaction fee\", on the same day the purchase settles.",
            ]
        lines += [
            "",
            "## Exchange rate",
            "",
            "We use the card network's rate on the day the transaction settles, with no markup. Choose the local currency when a terminal offers to charge you in dollars.",
        ]
        lines += [
            "",
            "## Refunds",
            "",
            "If a foreign purchase is refunded, we also refund the foreign transaction fee on the refunded amount. The refund is converted at the rate on the day it settles.",
        ]
        if plan != "premium":
            lines += ["", f"If you travel often, Premium has no foreign transaction fee."]
        return "\n".join(lines)

    return body


art("fx-fees-basic", "Foreign transaction fees on Basic", S_TRAVEL, fx_fees("basic", 2), plans=["basic"],
    eff=eff_of("fees.versions.v1.effective_date"))
art("fx-fees-plus-v1", "Foreign transaction fees on Plus", S_TRAVEL, fx_fees("plus", 1), plans=["plus"],
    eff=eff_of("fees.versions.v1.effective_date"), version=1)
art("fx-fees-plus-v2", "Foreign transaction fees on Plus", S_TRAVEL, fx_fees("plus", 2), plans=["plus"],
    eff=eff_of("fees.versions.v2.effective_date"), version=2, supersedes="fx-fees-plus-v1")
art("fx-fees-premium", "Foreign transaction fees on Premium", S_TRAVEL, fx_fees("premium", 2), plans=["premium"],
    eff=eff_of("fees.versions.v1.effective_date"))


# =====================================================================
# Transfers and deposits
# =====================================================================

def link_external() -> str:
    t("link_external")
    return "\n".join([
        "To move money between Tallowbrook and an account at another bank, link that account first. You can link it instantly by signing in to the other bank, or by confirming two small deposits.",
        "",
        "## Link instantly",
        "",
        "Tap Transfers, then Linked accounts, then Add account. Choose your bank and sign in. Most large banks can be linked this way in under a minute.",
        "",
        "## Link with small deposits",
        "",
        f"If your bank isn't listed, enter its routing and account number. We send two small deposits, which arrive in {F.range_bd('transfers.ach.external_account_microdeposit_min_business_days', 'transfers.ach.external_account_microdeposit_max_business_days')}. Enter the two amounts in the app within {F.v('transfers.ach.microdeposit_confirm_within_days')} to finish linking. After that, the link expires and you need to start again.",
        "",
        "## Rules",
        "",
        "- The external account must be in your name. We can't link accounts that belong to someone else, including a spouse.",
        f"- You can link up to {F.raw('transfers.ach.max_linked_accounts')} external accounts.",
        "- New linked accounts may have lower transfer amounts for the first few days while we confirm them.",
        "",
        "Your Tallowbrook routing and account numbers are shown in the app under Account details, if the other bank asks for them.",
    ])


art("transfers-link-external", "Linking an external bank account", S_TRANSFERS, link_external)


def ach_cutoff(ver: int) -> Callable[[], str]:
    def body() -> str:
        t("ach_cutoff")
        cutoff = F.v(f"transfers.ach.versions.v{ver}.cutoff_time_et")
        return "\n".join([
            "Standard transfers to and from your linked bank accounts use ACH, which only processes on business days.",
            "",
            "## Cutoff time",
            "",
            f"Transfers you submit before {cutoff} on a business day are sent that day. Transfers submitted after {cutoff}, or on a weekend or bank holiday, are sent the next business day.",
            "",
            "## When the money arrives",
            "",
            f"Once sent, ACH transfers usually arrive in {F.range_bd('transfers.ach.arrival_min_business_days', 'transfers.ach.arrival_max_business_days')}. The other bank decides when to make the money available.",
            "",
            "## Example",
            "",
            f"If you send money on a Friday after {cutoff}, it's sent on Monday (or Tuesday if Monday is a holiday).",
            "",
            "## Faster options",
            "",
            "- BrookPay transfers to other Tallowbrook members arrive in seconds, every day.",
            "- Domestic wires arrive the same business day if sent before the wire cutoff.",
            "",
            "## Limits",
            "",
            "Daily and monthly ACH limits depend on your plan. See the transfer limits article for your plan.",
        ])

    return body


art("transfers-ach-cutoff-v1", "ACH transfer cutoff times", S_TRANSFERS, ach_cutoff(1),
    eff=eff_of("transfers.ach.versions.v1.effective_date"), version=1)
art("transfers-ach-cutoff-v2", "ACH transfer cutoff times", S_TRANSFERS, ach_cutoff(2),
    eff=eff_of("transfers.ach.versions.v2.effective_date"), version=2, supersedes="transfers-ach-cutoff-v1")


def brookpay() -> str:
    t("brookpay")
    return "\n".join([
        F.mark("transfers.brookpay.availability", "transfers.brookpay.reversible") + "BrookPay lets you send money to another Tallowbrook member by phone number, email or username. It's free, it works every day, and the money arrives in seconds.",
        "",
        "## Send money",
        "",
        "Tap Transfers, then BrookPay. Choose the person, enter the amount and confirm. The recipient sees the money straight away.",
        "",
        "## Limits",
        "",
        "| Plan | Daily BrookPay limit |",
        "|---|---|",
        *[f"| {pn(p)} | {F.lim(p, 'daily_brookpay_limit_usd')} |" for p in PLANS],
        "",
        f"For {F.v('limits.new_device_hold_hours')} after you sign in on a new device, BrookPay and external transfers are limited to {F.v('limits.new_device_transfer_cap_usd')}.",
        "",
        "## BrookPay can't be reversed",
        "",
        "Once a BrookPay payment is sent, it can't be cancelled or pulled back, even by support. Only send money to people you know. If you sent money to the wrong person, ask them to send it back. If you think you were scammed, report it to us right away.",
        "",
        "## Fees",
        "",
        f"BrookPay costs {F.v('transfers.brookpay.fee_usd')} on every plan. BrookPay only works between Tallowbrook members.",
    ])


art("transfers-brookpay", "Sending money with BrookPay", S_TRANSFERS, brookpay)


def transfer_cancel() -> str:
    t("transfer_cancel")
    return "\n".join([
        "Whether you can cancel a transfer depends on the type of transfer and how far along it is.",
        "",
        "## Scheduled or pending ACH transfers",
        "",
        F.mark("transfers.ach.cancel_rule") + f"You can cancel a scheduled ACH transfer in the app until the ACH cutoff time ({F.v('transfers.ach.versions.v2.cutoff_time_et')}) on its send date. Tap Transfers, choose the transfer and tap Cancel. After the cutoff, the transfer is already on its way and can't be cancelled.",
        "",
        "## BrookPay",
        "",
        "BrookPay payments arrive instantly and can't be cancelled.",
        "",
        "## Wires",
        "",
        F.mark("transfers.wires.recall_rule") + "Wires can't be cancelled once sent. We can ask the receiving bank to return the money, but the receiving bank decides, and it isn't guaranteed.",
        "",
        "## Recurring transfers",
        "",
        "You can stop future transfers in a recurring series at any time. Stopping the series doesn't cancel a transfer that's already past its cutoff.",
        "",
        "## Sent to the wrong account?",
        "",
        "If an ACH transfer went to the wrong external account, message support. We'll try to contact the other bank, but we can't promise the money comes back.",
    ])


art("transfers-cancel", "Cancelling a transfer", S_TRANSFERS, transfer_cancel)


def transfer_pending() -> str:
    t("transfer_pending")
    return "\n".join([
        "A transfer can show as pending for a few reasons. Here's how to tell what's going on.",
        "",
        "## Waiting for the cutoff",
        "",
        f"ACH transfers submitted after {F.v('transfers.ach.versions.v2.cutoff_time_et')}, or on a weekend or holiday, stay pending until they're sent on the next business day.",
        "",
        "## On its way",
        "",
        f"Once sent, ACH transfers take {F.range_bd('transfers.ach.arrival_min_business_days', 'transfers.ach.arrival_max_business_days')} to arrive. The app shows the expected arrival date on the transfer.",
        "",
        "## Under review",
        "",
        "Some transfers are reviewed before they're sent, for example a large first transfer to a newly linked account. The app shows \"In review\". Reviews usually finish within one business day, and we'll message you if we need anything.",
        "",
        "## New device",
        "",
        f"For {F.v('limits.new_device_hold_hours')} after you sign in on a new device, transfers above {F.v('limits.new_device_transfer_cap_usd')} can't be sent.",
        "",
        "## Incoming transfers",
        "",
        "If money sent to you hasn't arrived, check with the sender's bank when it was sent. Direct deposits usually arrive as soon as your employer's payment file reaches us.",
        "",
        "If a transfer has been pending longer than the expected arrival date, message support with the amount and date.",
    ])


art("transfers-pending", "Why is my transfer pending?", S_TRANSFERS, transfer_pending)


def wires_domestic() -> str:
    t("wire_domestic")
    return "\n".join([
        "Domestic wires send money to another US bank account on the same business day.",
        "",
        "## How to send a wire",
        "",
        "1. Tap Transfers, then Wire.",
        "2. Enter the recipient's name, bank routing number, account number and address.",
        "3. Review the fee and confirm with your passcode.",
        "",
        "## Cutoff",
        "",
        f"Wires submitted before {F.v('transfers.wires.domestic_cutoff_time_et')} on a business day are sent the same day. Later wires are sent the next business day.",
        "",
        "## Fees and limits",
        "",
        "| Plan | Fee | Daily limit |",
        "|---|---|---|",
        *[f"| {pn(p)} | {F.fee_or_free(p, 'domestic_wire_out_fee_usd')} | {F.lim(p, 'daily_domestic_wire_limit_usd')} |" for p in PLANS],
        "",
        "## Before you send",
        "",
        "Wires can't be cancelled once sent. Double-check the details, and be careful if someone you've only met online, or someone claiming to be from a bank or the government, asks you to wire money.",
    ])


art("wires-domestic", "Sending a domestic wire", S_TRANSFERS, wires_domestic)


def wires_intl() -> str:
    t("wire_international", "intl_wire_not_on_basic")
    return "\n".join([
        "Plus and Premium members can send international wires from the app. International wires aren't available on Basic.",
        "",
        "## How to send one",
        "",
        "Tap Transfers, then Wire, then International. You'll need the recipient's name and address, their bank's SWIFT code, and their account number or IBAN.",
        "",
        "## Cutoff and timing",
        "",
        f"International wires submitted before {F.v('transfers.wires.international_cutoff_time_et')} on a business day are sent that day. They usually arrive in {F.range_bd('transfers.wires.international_arrival_min_business_days', 'transfers.wires.international_arrival_max_business_days')}, depending on the country and any banks in between.",
        "",
        "## Fees",
        "",
        f"- Plus: {F.fee('plus', 'international_wire_out_fee_usd')} per wire",
        f"- Premium: {F.fee('premium', 'international_wire_out_fee_usd')} per wire",
        "",
        "Banks along the way, and the receiving bank, may take their own fees from the amount.",
        "",
        "## Currency",
        "",
        "International wires are sent in US dollars. The receiving bank converts them at its own rate.",
        "",
        "Wires can't be cancelled once sent. Wires to comprehensively sanctioned countries are blocked.",
    ])


art("wires-international", "Sending an international wire", S_TRANSFERS, wires_intl, plans=["plus", "premium"])


def wires_incoming() -> str:
    t("wire_incoming")
    return "\n".join([
        "You can receive domestic and international wires into your Tallowbrook account on any plan.",
        "",
        "## What to give the sender",
        "",
        "Your routing number, account number, full legal name and address are in the app under Account details, then Wire instructions. International senders also need our SWIFT code, shown on the same screen.",
        "",
        "## Fees",
        "",
        "| Plan | Incoming domestic wire | Incoming international wire |",
        "|---|---|---|",
        *[f"| {pn(p)} | {F.fee_or_free(p, 'incoming_domestic_wire_fee_usd')} | {F.fee_or_free(p, 'incoming_international_wire_fee_usd')} |" for p in PLANS],
        "",
        "## Timing",
        "",
        "Domestic wires usually arrive within hours on a business day. International wires can take longer, depending on the sending bank.",
        "",
        "## If a wire hasn't arrived",
        "",
        "Ask the sender for the wire reference number and the date it was sent, then message support. Wires with a name that doesn't match your account name may be returned to the sender.",
    ])


art("wires-incoming", "Receiving a wire", S_TRANSFERS, wires_incoming)


def direct_deposit() -> str:
    t("direct_deposit_setup")
    return "\n".join([
        "Direct deposit sends your paycheck, or benefits, straight into your Tallowbrook account.",
        "",
        "## Set it up",
        "",
        "- **Automatic:** tap Transfers, then Direct deposit, then Set up automatically. Find your employer or payroll provider and sign in. The switch happens in the payroll system.",
        "- **Manual:** download a pre-filled direct deposit form from the same screen, or copy your routing and account number, and give it to your employer.",
        "",
        "## How long it takes",
        "",
        "Employers usually need one or two pay cycles to make the switch. Your first paycheck may still go to your old account.",
        "",
        "## Why it helps",
        "",
        f"- You may get paid up to {F.v('transfers.direct_deposit.early_direct_deposit_max_days')} early.",
        f"- Plus and Premium members need at least {F.v('overdraft.cushion.min_direct_deposit_usd')} in direct deposits in the last {F.v('overdraft.cushion.direct_deposit_lookback_days')} to use Cushion.",
        "",
        "## Splitting a paycheck",
        "",
        "Your employer may let you split your pay between accounts. You can also set up an automatic transfer from your main balance to a Savings Pocket on payday.",
    ])


art("direct-deposit-setup", "Setting up direct deposit", S_TRANSFERS, direct_deposit)


def early_dd() -> str:
    t("early_direct_deposit")
    return "\n".join([
        f"With early direct deposit, you can get your paycheck up to {F.v('transfers.direct_deposit.early_direct_deposit_max_days')} before your employer's official payday. It's included on every plan at no cost.",
        "",
        "## How it works",
        "",
        "Employers send payroll files to banks a few days before payday. When we receive yours, we make the money available right away instead of waiting for the official pay date.",
        "",
        "## Why timing varies",
        "",
        "How early you're paid depends on when your employer sends the file. Some pay periods you might get paid two days early, others one day, or on payday itself. We can't make an employer send it sooner.",
        "",
        "## Things to know",
        "",
        "- Early direct deposit is automatic. There's nothing to switch on.",
        "- It applies to payroll and government benefits sent by direct deposit. It doesn't apply to transfers you make yourself or to BrookPay.",
        "- If an employer cancels or reverses a payroll file, the deposit may be reversed.",
        "",
        "If your paycheck is late on your official payday, check with your employer first. They can confirm when and where it was sent.",
    ])


art("direct-deposit-early", "Early direct deposit", S_TRANSFERS, early_dd)


def check_deposit_how() -> str:
    t("check_deposit_how")
    return "\n".join([
        "You can deposit paper checks by taking a photo in the app.",
        "",
        "## How to deposit",
        "",
        "1. Sign the back of the check.",
        f"2. Under your signature, write \"{F.raw('transfers.mobile_check_deposit.endorsement')}\".",
        "3. Tap Transfers, then Deposit a check.",
        "4. Enter the amount and photograph the front and back on a dark surface.",
        "",
        "## Limits",
        "",
        "| Plan | Daily mobile check deposit limit |",
        "|---|---|",
        *[f"| {pn(p)} | {F.lim(p, 'daily_mobile_check_deposit_limit_usd')} |" for p in PLANS],
        "",
        "## Keep the check",
        "",
        f"Keep the paper check for {F.v('transfers.mobile_check_deposit.keep_check_days')} after it's deposited, then destroy it. Don't deposit the same check anywhere else.",
        "",
        "## Checks we can't accept",
        "",
        "Checks made out to someone else, checks from outside the US, money orders, and checks without the endorsement above.",
        "",
        "## When the money is available",
        "",
        "See \"When check deposits are available\". The app shows the release date after you submit the deposit.",
    ])


art("check-deposit-how", "Depositing a check in the app", S_TRANSFERS, check_deposit_how)


def check_availability() -> str:
    t("check_deposit_availability")
    mcd = "transfers.mobile_check_deposit"
    return "\n".join([
        "Once your check deposit is accepted, here's when you can use the money.",
        "",
        "## Standard availability",
        "",
        f"- The first {F.v(f'{mcd}.next_business_day_available_usd')} is available the next business day.",
        f"- The rest is available within {F.v(f'{mcd}.remainder_available_business_days')}.",
        "",
        "## New accounts",
        "",
        f"For accounts open less than {F.v(f'{mcd}.new_account_period_days')}, we may hold check deposits for up to {F.v(f'{mcd}.new_account_hold_business_days')}.",
        "",
        "## Longer holds",
        "",
        "We may also hold a check longer if it's unusually large for your account, if the check looks altered, or if it has been returned before. The app shows the release date on the deposit.",
        "",
        "## Returned checks",
        "",
        "If the check bounces after the money was made available, we take the amount back from your balance. There's no returned item fee.",
        "",
        "Deposits made on a weekend or bank holiday count as received on the next business day.",
        "",
        "## Check the status",
        "",
        "Tap the deposit in your activity feed to see whether it's pending, available or returned.",
    ])


art("check-deposit-availability", "When check deposits are available", S_TRANSFERS, check_availability)


def transfer_returned() -> str:
    t("transfer_returned")
    return "\n".join([
        "Sometimes a transfer is sent back. This is called a return. It usually happens a few business days after the transfer was sent.",
        "",
        "## Common reasons",
        "",
        "- The account number or routing number was wrong.",
        "- The receiving account was closed or frozen.",
        "- The name on the receiving account didn't match.",
        "- For money you pulled in from another bank, there wasn't enough money in that account.",
        "",
        "## What happens to the money",
        "",
        "For transfers out, the money comes back to your Tallowbrook balance when the return reaches us. For transfers in, we take back any amount we already made available.",
        "",
        "## Fees",
        "",
        f"We don't charge for returned transfers ({F.v('fees.all_plans.returned_transfer_fee_usd')}). The other bank might.",
        "",
        "## What to do",
        "",
        "Check the details on the linked account, correct anything wrong, and send the transfer again. If a pull from your own account at another bank was returned, ask that bank why before you try again. Repeated returns can lead to lower transfer amounts.",
    ])


art("transfers-returned", "Returned transfers", S_TRANSFERS, transfer_returned)


def transfer_recurring() -> str:
    t("transfer_recurring")
    return "\n".join([
        "You can schedule a transfer for a future date, or set one up to repeat weekly, every two weeks or monthly.",
        "",
        "## Set one up",
        "",
        "Tap Transfers, choose the account, enter the amount, and tap Schedule. Pick the date and how often it repeats.",
        "",
        "## When scheduled transfers are sent",
        "",
        f"On the scheduled date, ACH transfers are sent in the batch before {F.v('transfers.ach.versions.v2.cutoff_time_et')}. If the date falls on a weekend or holiday, they're sent the next business day.",
        "",
        "## Not enough money",
        "",
        "If your balance is too low on the send date, the transfer is skipped and we notify you. We don't retry it, and there's no fee. Future transfers in the series still run.",
        "",
        "## Change or stop",
        "",
        "Edit or cancel a scheduled transfer in the app any time before the cutoff on its send date.",
        "",
        "## Transfers to savings",
        "",
        "Recurring transfers between your main balance and Savings Pockets happen instantly on the scheduled day, including weekends.",
        "",
        "Scheduled transfers count toward your daily and monthly transfer limits on the day they're sent.",
    ])


art("transfers-recurring", "Scheduled and recurring transfers", S_TRANSFERS, transfer_recurring)


def transfer_holidays() -> str:
    t("bank_holidays")
    return "\n".join([
        F.mark("transfers.holidays") + "Some kinds of payment only move on business days. Business days are Monday to Friday, except bank holidays.",
        "",
        "## What pauses on weekends and holidays",
        "",
        "- ACH transfers to and from linked accounts",
        "- Domestic and international wires",
        "- Mobile check deposits (they count as received the next business day)",
        "",
        "## What works every day",
        "",
        "- Card purchases and ATM withdrawals",
        "- BrookPay transfers between members",
        "- Transfers between your main balance and Savings Pockets",
        "- Cash deposits at retail stores",
        "",
        "## Paydays near a holiday",
        "",
        "If your payday falls on a holiday, your employer usually sends payroll earlier, and early direct deposit may pay you sooner still.",
        "",
        "## Timelines in business days",
        "",
        "When we say something takes a number of business days, such as card delivery, dispute credits or transfer arrival, weekends and bank holidays aren't counted.",
        "",
        "Support is open every day, including holidays.",
    ])


art("transfers-holidays", "Weekends and bank holidays", S_TRANSFERS, transfer_holidays)


def transfer_limits(plan: str) -> Callable[[], str]:
    def body() -> str:
        t("transfer_limits", "limits_fixed")
        name = pn(plan)
        return "\n".join([
            f"These are the transfer and deposit limits on the {name} plan.",
            "",
            "| Limit | Amount |",
            "|---|---|",
            f"| ACH transfers out, per day | {F.lim(plan, 'daily_ach_out_limit_usd')} |",
            f"| ACH transfers out, per calendar month | {F.lim(plan, 'monthly_ach_out_limit_usd')} |",
            f"| BrookPay, per day | {F.lim(plan, 'daily_brookpay_limit_usd')} |",
            f"| Domestic wires, per day | {F.lim(plan, 'daily_domestic_wire_limit_usd')} |",
            f"| Mobile check deposits, per day | {F.lim(plan, 'daily_mobile_check_deposit_limit_usd')} |",
            f"| Cash deposits at stores, per day | {F.lim(plan, 'daily_retail_cash_deposit_limit_usd')} |",
            "",
            "Daily limits reset at midnight Eastern Time. The monthly limit resets on the first day of each calendar month.",
            "",
            "## Incoming transfers",
            "",
            "There's no limit on money coming in by ACH, wire, direct deposit or BrookPay.",
            "",
            "## New devices",
            "",
            f"For {F.v('limits.new_device_hold_hours')} after you sign in on a new device, BrookPay and external transfers are capped at {F.v('limits.new_device_transfer_cap_usd')}.",
            "",
            "## Need more?",
            "",
            F.mark("limits.limit_changes") + "Limits are set by plan. Support can't raise them, even for one transfer. Changing plan is the only way to get higher limits.",
        ])

    return body


for _p in PLANS:
    art(f"transfer-limits-{_p}", f"Transfer limits on {pn(_p)}", S_TRANSFERS, transfer_limits(_p), plans=[_p])


def wires_plan(plan: str) -> Callable[[], str]:
    def body() -> str:
        t("wire_fees_plan")
        name = pn(plan)
        intl = F.fee_raw(plan, "international_wire_out_fee_usd")
        lines = [
            f"Here's what wires cost on the {name} plan.",
            "",
            "| Wire | Fee |",
            "|---|---|",
            f"| Outgoing domestic | {F.fee_or_free(plan, 'domestic_wire_out_fee_usd')} |",
            f"| Outgoing international | {'not available' if intl is None else F.fee_or_free(plan, 'international_wire_out_fee_usd')} |",
            f"| Incoming domestic | {F.fee_or_free(plan, 'incoming_domestic_wire_fee_usd')} |",
            f"| Incoming international | {F.fee_or_free(plan, 'incoming_international_wire_fee_usd')} |",
            "",
            f"You can send up to {F.lim(plan, 'daily_domestic_wire_limit_usd')} a day in domestic wires on {name}.",
            "",
            "## Cutoff times",
            "",
            f"Domestic wires sent before {F.v('transfers.wires.domestic_cutoff_time_et')} on a business day go out the same day.",
        ]
        if intl is None:
            t("intl_wire_not_on_basic")
            lines += ["", f"To send money abroad by wire, you'd need to move to Plus or Premium. You can still receive international wires on {name}."]
        else:
            lines += [f"International wires sent before {F.v('transfers.wires.international_cutoff_time_et')} go out the same day and arrive in {F.range_bd('transfers.wires.international_arrival_min_business_days', 'transfers.wires.international_arrival_max_business_days')}."]
        lines += [
            "",
            "## Before you send",
            "",
            "Wires can't be cancelled once they're sent. Check the recipient's name, bank and account number carefully. Be wary of anyone who pressures you to wire money quickly, especially someone you've only met online.",
            "",
            "Other banks involved in the wire may charge their own fees, which can reduce the amount the recipient gets. We show our fee before you confirm.",
        ]
        return "\n".join(lines)

    return body


for _p in PLANS:
    art(f"wires-{_p}", f"Wire fees on {pn(_p)}", S_TRANSFERS, wires_plan(_p), plans=[_p])


# =====================================================================
# Disputes
# =====================================================================

D = "disputes"


def disputes_unauthorized() -> str:
    t("dispute_unauthorized", "dispute_window_unauthorized", "zero_liability")
    return "\n".join([
        "If you see a card transaction you didn't make or authorize, you can dispute it as unauthorized.",
        "",
        "## First, protect your card",
        "",
        F.mark(f"{D}.unauthorized_requires_card_block") + "Before we open an unauthorized dispute, the card used must be frozen or reported lost or stolen. That stops more charges while we investigate. If the card is still in your wallet, freezing is enough.",
        "",
        "## How long you have",
        "",
        F.mark(f"{D}.statement_date_rule") + f"Report an unauthorized transaction within {F.v(f'{D}.unauthorized_window_days')} of the date of the statement it first appeared on. Statements close on the last calendar day of each month. For example, a transaction that posted in March appears on the March statement, so the window runs from the end of March.",
        "",
        "## What happens next",
        "",
        f"- If your dispute passes an initial review, you get a provisional credit within {F.v(f'{D}.provisional_credit_business_days')}.",
        f"- Most investigations finish within {F.v(f'{D}.investigation_days')}. Foreign transactions, and accounts open less than {F.v(f'{D}.new_account_days')}, can take up to {F.v(f'{D}.extended_investigation_days')}.",
        "",
        "## Your liability",
        "",
        "You aren't responsible for unauthorized card transactions reported within the window.",
        "",
        "## Not sure it's unauthorized?",
        "",
        "Unfamiliar names are often a merchant's parent company or a payment processor. Tap the transaction to see the merchant's details before you dispute it.",
    ])


art("disputes-unauthorized", "Disputing a transaction you didn't make", S_DISPUTES, disputes_unauthorized,
    eff=eff_of("disputes.versions.v2.effective_date"))


def disputes_merchant(ver: int) -> Callable[[], str]:
    def body() -> str:
        t("dispute_merchant", "dispute_window_merchant", "dispute_merchant_first")
        window = F.v(f"disputes.versions.v{ver}.merchant_dispute_window_days")
        return "\n".join([
            "If you paid for something with your card and there's a problem with the purchase itself, you can file a merchant dispute. This covers:",
            "",
            "- Items or services you paid for but never received",
            "- Items that were significantly different from what was described",
            "- Subscriptions or recurring charges that continued after you cancelled",
            "",
            "## Contact the merchant first",
            "",
            F.mark(f"{D}.merchant_first_rule") + "Before you dispute, try to sort it out with the merchant, and keep a record of what you sent and what they said. We'll ask for it.",
            "",
            "## How long you have",
            "",
            f"File a merchant dispute within {window} of the transaction date. After that, we can't open a dispute for you.",
            "",
            "## How to file",
            "",
            "Tap the transaction, then Get help, then Problem with a purchase. Tell us what happened and upload your records.",
            "",
            "## What happens next",
            "",
            f"If the dispute passes an initial review, you get a provisional credit within {F.v(f'{D}.provisional_credit_business_days')}. We then contact the merchant through the card network. Most cases finish within {F.v(f'{D}.investigation_days')}.",
            "",
            "Transactions you don't recognize at all are a different kind of dispute. See \"Disputing a transaction you didn't make\".",
        ])

    return body


art("disputes-merchant-v1", "Disputing a purchase with a merchant", S_DISPUTES, disputes_merchant(1),
    eff=eff_of("disputes.versions.v1.effective_date"), version=1)
art("disputes-merchant-v2", "Disputing a purchase with a merchant", S_DISPUTES, disputes_merchant(2),
    eff=eff_of("disputes.versions.v2.effective_date"), version=2, supersedes="disputes-merchant-v1")


def disputes_provisional() -> str:
    t("provisional_credit", "dispute_timeline")
    return "\n".join([
        "A provisional credit is money we put back in your account while we investigate a dispute. It lets you use the money without waiting for the final outcome.",
        "",
        "## When you get it",
        "",
        f"If your dispute passes an initial review, the provisional credit arrives within {F.v(f'{D}.provisional_credit_business_days')} of filing. It shows in your activity as \"Provisional credit\" with the merchant name.",
        "",
        "## How long the investigation takes",
        "",
        f"Most disputes are resolved within {F.v(f'{D}.investigation_days')}. For foreign transactions, and for accounts open less than {F.v(f'{D}.new_account_days')}, it can take up to {F.v(f'{D}.extended_investigation_days')}.",
        "",
        "## If the dispute goes your way",
        "",
        "The provisional credit becomes permanent. You don't need to do anything.",
        "",
        "## If it doesn't",
        "",
        f"We send you a written explanation and take the provisional credit back {F.v(f'{D}.denial_reversal_notice_business_days')} after that notice. If that makes your balance negative, you'll need to add money.",
        "",
        "Not every dispute gets a provisional credit. For example, we may wait for the merchant's response if you already received a refund for part of the purchase.",
    ])


art("disputes-provisional-credit", "Provisional credit during a dispute", S_DISPUTES, disputes_provisional)


def disputes_status() -> str:
    t("dispute_status")
    return "\n".join([
        "You can follow every dispute in the app. Tap Profile, then Disputes, and choose the case.",
        "",
        "## What the statuses mean",
        "",
        "- **Submitted:** we've received it and are doing an initial review.",
        "- **Provisional credit issued:** the money is back in your account while we investigate.",
        "- **Waiting on merchant:** we've contacted the merchant through the card network and are waiting for its response.",
        "- **More information needed:** we've asked you for something. Check your messages.",
        "- **Resolved in your favor:** the credit is permanent.",
        "- **Denied:** we've sent an explanation, and any provisional credit will be taken back.",
        "",
        "## How long it takes",
        "",
        f"Most disputes finish within {F.v(f'{D}.investigation_days')}, and some take up to {F.v(f'{D}.extended_investigation_days')}. We'll message you whenever the status changes.",
        "",
        "## Adding information",
        "",
        "You can upload documents or add notes to an open dispute from the case page at any time.",
        "",
        "## Cancelling a dispute",
        "",
        "If the merchant refunds you directly, withdraw the dispute from the case page so you aren't credited twice. If you were credited twice, we'll take back the extra amount.",
    ])


art("disputes-status", "Checking the status of a dispute", S_DISPUTES, disputes_status)


def disputes_documents() -> str:
    t("dispute_documents")
    return "\n".join([
        "The right documents make a dispute faster and more likely to succeed. Here's what helps for each kind of dispute.",
        "",
        "## Item not received",
        "",
        "- Order confirmation showing the expected delivery date",
        "- Tracking information, if any",
        "- Your messages to the merchant asking about the delivery",
        "",
        "## Item not as described",
        "",
        "- The product listing or description",
        "- Photos of what you received",
        "- Proof you returned it or offered to, if the merchant asked",
        "",
        "## Cancelled subscription still charged",
        "",
        "- The cancellation confirmation, or a screenshot of the date you cancelled",
        "- The merchant's cancellation terms, if you have them",
        "",
        "## Unauthorized transactions",
        "",
        "Usually nothing. Just confirm you didn't make or authorize the transaction and that your card is frozen or reported.",
        "",
        "## How to upload",
        "",
        "Open the dispute under Profile, then Disputes, and tap Add documents. Photos and PDFs work.",
        "",
        "Never send us your full card number, PIN or passwords as part of a dispute. We don't need them.",
    ])


art("disputes-documents", "Documents that help a dispute", S_DISPUTES, disputes_documents)


def disputes_duplicate() -> str:
    t("dispute_duplicate", "dispute_pending")
    return "\n".join([
        "Seeing the same charge twice doesn't always mean you were charged twice.",
        "",
        "## Check if one is pending",
        "",
        F.mark(f"{D}.pending_rule") + f"Hotels, gas stations, car rentals and restaurants often place a temporary authorization, then post the final amount separately. The pending one usually drops off within {F.v(f'{D}.pending_dropoff_days')}. Pending transactions can't be disputed.",
        "",
        "## If both have posted",
        "",
        "If two identical charges have both posted, contact the merchant first. Most will refund a duplicate quickly. If the merchant doesn't help, dispute the second charge in the app as a duplicate. Tap the transaction, then Get help, then Charged twice.",
        "",
        "## Duplicate fees",
        "",
        "If we charged you the same fee twice, message support. A fee charged in error is always refunded.",
        "",
        "## Time limit",
        "",
        f"Duplicate charges follow the merchant dispute window: {F.v('disputes.versions.v2.merchant_dispute_window_days')} from the transaction date.",
        "",
        "## Split payments",
        "",
        "Some merchants split one order into several charges, for example when items ship separately. Check your order confirmation before you dispute.",
    ])


art("disputes-duplicate", "Charged twice for the same thing", S_DISPUTES, disputes_duplicate)


def disputes_pending() -> str:
    t("dispute_pending")
    return "\n".join([
        F.mark(f"{D}.pending_rule") + "You can't dispute a transaction while it's still pending. A pending transaction is an authorization. The merchant hasn't collected the money yet, and the amount can still change or disappear.",
        "",
        "## What to do",
        "",
        f"Wait for the transaction to post. Most pending transactions post or drop off within {F.v(f'{D}.pending_dropoff_days')}. Once it posts, the Dispute option appears when you tap it.",
        "",
        "## If you don't recognize it",
        "",
        "Freeze your card now, even though you can't dispute yet. That stops any more charges. When the transaction posts, dispute it as unauthorized. Your dispute window starts from the statement it posts on, so you won't lose time by waiting.",
        "",
        "## If the amount is wrong",
        "",
        "Pending amounts are often estimates, for example at gas pumps or hotels. The posted amount is usually different. Wait and see what posts.",
        "",
        "## If it's been pending a long time",
        "",
        "Some merchants take longer. If a transaction has been pending much longer than usual, message support and we'll look at it.",
    ])


art("disputes-pending", "Why can't I dispute a pending transaction?", S_DISPUTES, disputes_pending)


def disputes_subscriptions() -> str:
    t("dispute_subscription", "dispute_merchant_first")
    return "\n".join([
        "If a subscription keeps charging you after you cancelled it, here's what to do.",
        "",
        "## 1. Cancel with the merchant",
        "",
        "Cancel through the merchant's website or app, and save the confirmation email or a screenshot. Freezing your card doesn't stop recurring payments to merchants you've already authorized.",
        "",
        "## 2. Ask for a refund",
        "",
        "Contact the merchant with your cancellation confirmation and ask them to refund charges made after that date. Many merchants will.",
        "",
        "## 3. Dispute it",
        "",
        f"If the merchant won't refund you, dispute each charge made after you cancelled. Tap the transaction, then Get help, then Cancelled but still charged. You have {F.v('disputes.versions.v2.merchant_dispute_window_days')} from each transaction date.",
        "",
        "## Stopping future charges",
        "",
        "If a merchant keeps charging after you cancel, you can report the card lost and get a new number. Recurring payments to the old number will then fail. There may be a replacement fee depending on your plan.",
        "",
        "## Free trials",
        "",
        "Charges after a free trial ended aren't unauthorized if you gave your card when signing up. Cancel with the merchant and ask for a refund.",
    ])


art("disputes-subscriptions", "Subscriptions that keep charging", S_DISPUTES, disputes_subscriptions,
    eff=eff_of("disputes.versions.v2.effective_date"))


def disputes_ach() -> str:
    t("dispute_ach")
    return "\n".join([
        "An ACH debit is when a company pulls money from your account using your account and routing number, for example a utility bill or a loan payment.",
        "",
        "## Unauthorized ACH debits",
        "",
        f"If a company took money you didn't authorize, report it within {F.v(f'{D}.ach_unauthorized_window_days')} of the statement it first appeared on. Tap the transaction, then Get help, then I didn't authorize this.",
        "",
        "## Authorized, but wrong",
        "",
        "If you did authorize the company but the amount or date is wrong, contact the company first. If that doesn't work, message support and we'll look at it with you.",
        "",
        "## Stopping future debits",
        "",
        "You can block a company from pulling money in the app. Tap the transaction, then Block future debits. Tell the company too, so they know to bill you another way.",
        "",
        "## What happens next",
        "",
        f"We'll ask the company's bank to return the money. If your claim passes an initial review, you get a provisional credit within {F.v(f'{D}.provisional_credit_business_days')}.",
        "",
        "ACH transfers you sent yourself, and BrookPay payments, can't be disputed this way.",
    ])


art("disputes-ach", "Disputing an ACH debit", S_DISPUTES, disputes_ach)


def disputes_denied() -> str:
    t("dispute_denied")
    return "\n".join([
        "Sometimes a dispute is decided against you. Here's what that means.",
        "",
        "## Why disputes are denied",
        "",
        "- The merchant showed you made or authorized the purchase, for example with delivery proof signed at your address.",
        "- The dispute was filed after its window closed.",
        "- The merchant already refunded you.",
        "- The issue isn't covered, such as a price you agreed to and later changed your mind about.",
        "",
        "## What happens to the provisional credit",
        "",
        f"We send you a written explanation first. {F.v(f'{D}.denial_reversal_notice_business_days').capitalize()} later, we take the provisional credit back. If that leaves your balance negative, add money so your account isn't restricted.",
        "",
        "## Can I appeal?",
        "",
        "Yes, if you have new information. Reply to the decision message with the new documents and we'll review it again. We can't reopen a dispute just because you disagree with the outcome.",
        "",
        "## Negative balance",
        "",
        "See \"Negative balance after a dispute\" for how repayment works.",
    ])


art("disputes-denied", "If your dispute is denied", S_DISPUTES, disputes_denied)


def disputes_how() -> str:
    t("dispute_how_to_file", "dispute_support_limit")
    return "\n".join([
        "Most disputes can be filed yourself in the app in a couple of minutes. Support can also file one for you.",
        "",
        "## File it yourself",
        "",
        "1. Tap the transaction in your activity feed.",
        "2. Tap Get help and choose the reason.",
        "3. Answer the questions and add any documents.",
        "",
        "## Before you file",
        "",
        "- The transaction must have posted. Pending transactions can't be disputed.",
        f"- Unauthorized transactions: within {F.v(f'{D}.unauthorized_window_days')} of the statement date, and the card must be frozen or reported first.",
        f"- Merchant problems: within {F.v('disputes.versions.v2.merchant_dispute_window_days')} of the transaction date, after contacting the merchant.",
        "",
        "## Filing through support",
        "",
        f"Support can open disputes totaling up to {F.v(f'{D}.support_max_dispute_usd')} per case. Larger cases go to our Disputes team, who will contact you within {F.v('support.escalation_response_business_days')}.",
        "",
        "## One dispute per transaction",
        "",
        "If a transaction already has an open dispute, add information to that case instead of opening a new one.",
    ])


art("disputes-how-to-file", "How to file a dispute", S_DISPUTES, disputes_how,
    eff=eff_of("disputes.versions.v2.effective_date"))


# =====================================================================
# Security and fraud
# =====================================================================

def fraud_alerts() -> str:
    t("fraud_alerts")
    return "\n".join([
        "When we see a card transaction that looks unusual, we decline it and send you a fraud alert in the app, and by text if you've turned that on.",
        "",
        "## Answering an alert",
        "",
        "The alert shows the merchant, amount and location. Tap \"Yes, that was me\" or \"No, that wasn't me\".",
        "",
        "- **Yes:** the card is unlocked straight away. Try the purchase again.",
        "- **No:** the card stays locked. We'll guide you through reporting it and disputing any other charges.",
        "",
        "## If you don't answer",
        "",
        F.mark("security.fraud_alert_rule") + f"If you don't answer within {F.v('security.fraud_alert_response_hours')}, the card stays locked until you contact us.",
        "",
        "## Is this alert real?",
        "",
        "Real alerts appear inside the Tallowbrook app. A text alert only ever asks you to reply YES or NO, and never contains a link. We never ask for your card number, PIN, password or a one-time code to confirm an alert. If someone calls you about a fraud alert, hang up and open the app instead.",
        "",
        "## Too many alerts?",
        "",
        "Alerts are more common when you shop somewhere new or abroad. They get less frequent as our systems learn your patterns.",
    ])


art("security-fraud-alerts", "How fraud alerts work", S_SECURITY, fraud_alerts)


def never_ask() -> str:
    t("never_ask")
    F.mark("security.never_ask")
    return "\n".join([
        "Scammers often pretend to be from a bank. Here's what Tallowbrook will never ask you for, by phone, text, email, chat or social media.",
        "",
        "## We will never ask for",
        "",
        "- Your full card number or security code",
        "- Your PIN",
        "- Your password or passcode",
        "- A one-time code we sent you",
        "- Remote access to your phone or computer",
        "- To move your money to a \"safe account\" to protect it",
        "- To buy gift cards or send cryptocurrency",
        "",
        "## What we might ask",
        "",
        "In the app, support may ask you to confirm recent transactions or your address. That's in your signed-in session, where we already know it's you.",
        "",
        "## If someone asks",
        "",
        "Stop, hang up or close the chat. Open the Tallowbrook app yourself and message us. Don't call a number or tap a link the person gave you. Our real phone number is in the app under Help.",
        "",
        "If you already shared a code or password, change your password right away and message us so we can secure your account.",
    ])


art("security-never-ask", "What we'll never ask you", S_SECURITY, never_ask)


def scams() -> str:
    t("scams")
    return "\n".join([
        "These are the scams we see most often. Knowing them is the best protection.",
        "",
        "## Fake bank calls",
        "",
        "Someone calls claiming to be from Tallowbrook's fraud team. They know some of your details and say your account is at risk. They ask for a code or ask you to move money. That's a scam. We never do that.",
        "",
        "## Impersonating someone you know",
        "",
        "A message from a \"family member\" with a new number who urgently needs money. Call them on their old number before you send anything.",
        "",
        "## Marketplace and rental scams",
        "",
        "A seller asks you to pay by BrookPay or wire instead of a protected payment method, often for a deal that seems too good. BrookPay and wires can't be reversed.",
        "",
        "## Job and investment scams",
        "",
        "Someone offers easy money for moving funds through your account, or promises guaranteed returns. Moving money for others can get your account closed.",
        "",
        "## What to do",
        "",
        "If you think you've been scammed, message us straight away. The sooner we know, the better the chance of stopping or recovering the money, although we can't promise that. See \"If you sent money to a scammer\".",
    ])


art("security-scams", "Common scams to watch for", S_SECURITY, scams)


def account_takeover() -> str:
    t("account_takeover")
    return "\n".join([
        "If you think someone else has signed in to your account, act fast.",
        "",
        "## Signs to look for",
        "",
        "- A password change or new-device sign-in you didn't make",
        "- Your email address or phone number changed without you",
        "- Transfers or BrookPay payments you didn't send",
        "- You're suddenly signed out and can't sign back in",
        "",
        "## What to do",
        "",
        "1. If you can still sign in, change your password and freeze your cards.",
        "2. Message support from the app, or call us if you can't sign in.",
        "",
        "## What we'll do",
        "",
        F.mark("security.account_takeover_rule") + f"Support will freeze your cards and pass your case to our Fraud team, who handle account takeovers. The Fraud team contacts you within {F.v('support.escalation_response_business_days')}, and usually much faster. They may sign out every device, reset your credentials and review recent activity with you.",
        "",
        "Support can't reverse transfers or change your email or phone number during a takeover investigation. The Fraud team does that once they've confirmed who you are.",
        "",
        "## Afterward",
        "",
        "Check your email account's security too. Many takeovers start with a compromised email.",
    ])


art("security-account-takeover", "If someone else accessed your account", S_SECURITY, account_takeover)


def two_step() -> str:
    t("two_step")
    return "\n".join([
        "Two-step verification is always on for Tallowbrook accounts. When you sign in on a new device, or do something sensitive, we ask for your password and a second step.",
        "",
        "## The second step",
        "",
        "- A push approval on a device you've already signed in on, or",
        f"- A one-time code sent by text, valid for {F.raw('security.one_time_code_validity_minutes')} minutes",
        "",
        "## When we ask",
        "",
        "- Signing in on a new device",
        "- Changing your password, email or phone number",
        "- Adding a new external account or wire recipient",
        "- Viewing your full virtual card number",
        "",
        "## Keep codes to yourself",
        "",
        "A one-time code is only for you to type into the Tallowbrook app. Nobody from Tallowbrook will ask you to read it out. If someone asks, it's a scam.",
        "",
        "## Changed your phone number?",
        "",
        "Update it in the app while you still have the old number. If you've lost access to your old number, message support. We'll pass it to the team that verifies identity, which can take longer.",
    ])


art("security-two-step", "Two-step verification", S_SECURITY, two_step)


def password_reset() -> str:
    t("password_reset")
    return "\n".join([
        "If you forgot your password, you can reset it from the sign-in screen.",
        "",
        "## Reset your password",
        "",
        "1. Tap Forgot password on the sign-in screen.",
        "2. Enter the email on your account. We send a reset link.",
        "3. Open the link on the same phone, confirm the one-time code we text you, and choose a new password.",
        "",
        "The link expires after an hour. If it has expired, start again.",
        "",
        "## Didn't get the email?",
        "",
        "Check spam and make sure you entered the email you signed up with. If you no longer have access to that email, message support from the sign-in screen. For your security, changing the email on an account you can't sign in to needs extra checks.",
        "",
        "## After a reset",
        "",
        f"Other devices are signed out. The first time you sign in on a new device, BrookPay and external transfers are limited to {F.v('limits.new_device_transfer_cap_usd')} for {F.v('limits.new_device_hold_hours')}.",
        "",
        "If you didn't ask for a reset email, someone may have typed your email by mistake. If you also see a sign-in you don't recognize, see \"If someone else accessed your account\".",
    ])


art("security-password-reset", "Resetting your password", S_SECURITY, password_reset)


def new_device() -> str:
    t("new_device")
    return "\n".join([
        "When you sign in on a new phone or computer, we add a short safety period.",
        "",
        "## What changes",
        "",
        F.mark("limits.new_device_transfer_cap_usd") + f"For {F.v('limits.new_device_hold_hours')} after the first sign-in on a new device, BrookPay payments and transfers to external accounts are limited to {F.v('limits.new_device_transfer_cap_usd')} in total. Card payments, ATM withdrawals and deposits aren't affected.",
        "",
        "## Why",
        "",
        "If someone steals your password, the first thing they usually try is moving money out quickly. The limit gives you time to notice the sign-in alert and tell us.",
        "",
        "## Can support lift it early?",
        "",
        "No. The limit ends on its own. Support can't remove it.",
        "",
        "## Sign-in alerts",
        "",
        "We send an alert to your other devices and your email whenever a new device signs in. If it wasn't you, change your password and message us right away.",
        "",
        "## Switching phones",
        "",
        "If you know you'll need to send a large transfer, do it from your old phone before you switch, or plan around the waiting period.",
    ])


art("security-new-device", "Signing in on a new device", S_SECURITY, new_device)


def zero_liability() -> str:
    t("zero_liability", "dispute_window_unauthorized")
    return "\n".join([
        F.mark("lost_stolen.zero_liability_rule") + "You aren't responsible for unauthorized card transactions if you report them in time.",
        "",
        "## What counts as in time",
        "",
        f"Report unauthorized card transactions within {F.v(f'{D}.unauthorized_window_days')} of the date of the statement they first appear on. Statements close on the last calendar day of each month.",
        "",
        "## What's covered",
        "",
        "- Purchases and ATM withdrawals made with your card or card number without your permission",
        "- Mobile wallet payments made with a device you don't control",
        "",
        "## What isn't covered",
        "",
        "- Purchases made by someone you gave your card or PIN to",
        "- BrookPay payments and transfers you sent yourself, even if you were tricked into sending them",
        "- Transactions reported after the window",
        "",
        "## How to report",
        "",
        "Freeze or report your card first, then dispute each transaction as unauthorized in the app. See \"Disputing a transaction you didn't make\".",
    ])


art("security-zero-liability", "Zero liability for unauthorized card use", S_SECURITY, zero_liability)


def scam_payment() -> str:
    t("scam_payment")
    return "\n".join([
        "If you sent a BrookPay payment, a transfer or a wire and then realized it was a scam, tell us right away.",
        "",
        "## What we can do",
        "",
        F.mark("security.scam_payment_rule") + "Support will pass your report to our Fraud team. They'll contact the receiving account or bank and try to recover the money. BrookPay payments can't be reversed, so recovery depends on the money still being in the other account. We can't promise to get it back.",
        "",
        "## What to include",
        "",
        "- The payment and the date",
        "- How you were contacted and what you were told",
        "- Screenshots of messages, and any names, numbers or websites",
        "",
        "## Protect yourself now",
        "",
        "- If you shared a password or code, change your password.",
        "- Don't send more money to get the first payment back. That's a common second scam.",
        "- Report the scam to the police.",
        "",
        "## Card payments are different",
        "",
        "If you paid a scammer by card, you may be able to dispute the purchase with the merchant. See \"Disputing a purchase with a merchant\".",
        "",
        f"The Fraud team usually replies within {F.v('support.escalation_response_business_days')}.",
    ])


art("security-scam-payment", "If you sent money to a scammer", S_SECURITY, scam_payment)


# =====================================================================
# Overdraft and Cushion
# =====================================================================

C = "overdraft.cushion"


def overdraft_none() -> str:
    t("overdraft_none")
    return "\n".join([
        f"No. Tallowbrook doesn't charge overdraft fees ({F.v('overdraft.overdraft_fee_usd')}) or non-sufficient funds fees ({F.v('fees.all_plans.nsf_fee_usd')}) on any plan.",
        "",
        "## What happens instead",
        "",
        F.mark("overdraft.declined_if_insufficient") + "If you don't have enough money, the transaction is simply declined. There's no charge for a decline.",
        "",
        "## Cushion",
        "",
        f"Plus and Premium members can turn on Cushion, which covers small debit card purchases when your balance runs short, with no fee. Cushion covers up to {F.v(f'{C}.plus_limit_usd')} on Plus and {F.v(f'{C}.premium_limit_usd')} on Premium. It isn't available on Basic.",
        "",
        "## Can my balance still go negative?",
        "",
        "Rarely. It can happen when a merchant charges more than it first authorized, for example a tip added to a restaurant bill, when a deposit is returned, or when a dispute credit is taken back. Add money to bring it back to zero.",
        "",
        "## Returned payments",
        "",
        "If a scheduled transfer or an ACH debit can't be paid, it's returned unpaid, with no fee from us. The company you owe may charge its own late fee.",
    ])


art("overdraft-no-fees", "Do you charge overdraft fees?", S_CUSHION, overdraft_none)


def cushion_how() -> str:
    t("cushion_how", "cushion_basic_not_eligible")
    return "\n".join([
        "Cushion is fee-free overdraft coverage for debit card purchases, available on Plus and Premium.",
        "",
        "## How much it covers",
        "",
        f"- Plus: up to {F.v(f'{C}.plus_limit_usd')}",
        f"- Premium: up to {F.v(f'{C}.premium_limit_usd')}",
        "- Basic: not available",
        "",
        "## Who can use it",
        "",
        f"You need at least {F.v(f'{C}.min_direct_deposit_usd')} in direct deposits in the last {F.v(f'{C}.direct_deposit_lookback_days')}. Turn it on in the app under Profile, then Cushion.",
        "",
        "## What it covers",
        "",
        F.mark(f"{C}.covers") + "Debit card purchases only, including mobile wallet payments. It never covers ATM withdrawals, transfers, BrookPay, wires or plan fees.",
        "",
        "## What it costs",
        "",
        f"Nothing. Cushion has no fees ({F.v(f'{C}.fee_usd')}) and no interest.",
        "",
        "## Paying it back",
        "",
        f"Your next deposit repays it automatically. If your balance stays negative for {F.v(f'{C}.repay_within_days')}, Cushion is suspended. See \"Repaying Cushion\".",
        "",
        "## Turning it off",
        "",
        "You can switch Cushion off at any time in the same place. Purchases larger than your balance are then declined, with no fee.",
    ])


art("cushion-how-it-works", "How Cushion works", S_CUSHION, cushion_how, plans=["plus", "premium"])


def cushion_repay() -> str:
    t("cushion_repay")
    return "\n".join([
        "When Cushion covers a purchase, your balance goes below zero. Here's how it gets back to zero.",
        "",
        "## Automatic repayment",
        "",
        F.mark(f"{C}.repayment_rule") + "Any money that comes in, such as a direct deposit, a transfer or a refund, goes toward the negative balance first. You don't need to do anything.",
        "",
        "## Paying it back yourself",
        "",
        "You can add money at any time by transfer, BrookPay from another member, or a cash deposit at a store.",
        "",
        "## If it stays negative",
        "",
        f"- After {F.v(f'{C}.repay_within_days')}, Cushion is suspended until you're back at zero and meet the direct deposit requirement again.",
        f"- After {F.v(f'{C}.restriction_after_days')}, your account is restricted. You can still add money, but you can't spend or send it until the balance is repaid.",
        "",
        "## Cost",
        "",
        "There are no fees or interest for using Cushion, and none for repaying late. The only consequences are suspension and restriction.",
        "",
        "## Can I close my account while it's negative?",
        "",
        "No. Bring the balance to zero first.",
    ])


art("cushion-repayment", "Repaying Cushion", S_CUSHION, cushion_repay, plans=["plus", "premium"])


def negative_after_dispute() -> str:
    t("negative_balance_dispute")
    return "\n".join([
        "Your balance can go negative when we take back a provisional credit after a dispute is denied, or when a deposit is returned. This isn't Cushion, and it can happen on any plan, including Basic.",
        "",
        "## Why it happens",
        "",
        f"A provisional credit is money we lend you while a dispute is investigated. If the dispute is denied, we give you written notice and take it back {F.v(f'{D}.denial_reversal_notice_business_days')} later. If you've spent it in the meantime, the balance can drop below zero.",
        "",
        "## What to do",
        "",
        "Add money to bring your balance back to zero. Any incoming deposit repays it automatically.",
        "",
        "## Fees",
        "",
        f"There are no fees for a negative balance ({F.v('overdraft.overdraft_fee_usd')}).",
        "",
        "## If it isn't repaid",
        "",
        f"As with Cushion, if a negative balance lasts {F.v(f'{C}.restriction_after_days')}, the account is restricted until it's repaid. On Plus and Premium, Cushion is suspended while the balance is negative for longer than {F.v(f'{C}.repay_within_days')}.",
        "",
        "## Can't pay it right away?",
        "",
        "Message support. If you're going through financial hardship, we can pass your case to our Account Services team.",
    ])


art("cushion-negative-after-dispute", "Negative balance after a dispute", S_CUSHION, negative_after_dispute)


def cushion_plan(plan: str) -> Callable[[], str]:
    def body() -> str:
        t("cushion_plan")
        name = pn(plan)
        if plan == "basic":
            t("cushion_basic_not_eligible")
            return "\n".join([
                "Cushion isn't available on the Basic plan. On Basic, a debit card purchase larger than your balance is declined, with no fee.",
                "",
                "## Why",
                "",
                "Cushion is part of the paid plans. It's how we give members a little room without charging overdraft fees.",
                "",
                "## Getting Cushion",
                "",
                f"Move to Plus for coverage up to {F.v(f'{C}.plus_limit_usd')}, or to Premium for coverage up to {F.v(f'{C}.premium_limit_usd')}. You'll also need at least {F.v(f'{C}.min_direct_deposit_usd')} in direct deposits in the last {F.v(f'{C}.direct_deposit_lookback_days')}.",
                "",
                "## Overdraft fees",
                "",
                f"Basic has no overdraft fees ({F.v('overdraft.overdraft_fee_usd')}). Transactions that would overdraw your account are declined instead.",
                "",
                "## Can a Basic balance go negative?",
                "",
                "Occasionally, for example if a merchant posts more than it authorized or a dispute credit is taken back. Add money to bring it back to zero. Any deposit repays a negative balance automatically.",
                "",
                "There's no fee for a declined transaction.",
            ])
        lim = F.v(f"{C}.{plan}_limit_usd")
        return "\n".join([
            f"On {name}, Cushion covers debit card purchases up to {lim} below zero, with no fee.",
            "",
            "## Turning it on",
            "",
            f"Go to Profile, then Cushion. You need at least {F.v(f'{C}.min_direct_deposit_usd')} in direct deposits in the last {F.v(f'{C}.direct_deposit_lookback_days')}. If you stop meeting that, Cushion pauses until you do again.",
            "",
            "## Example",
            "",
            f"If your balance is low and you buy groceries that cost more than you have, Cushion covers the difference as long as the total below zero stays within {lim}. A purchase that would take you past {lim} below zero is declined.",
            "",
            "## Limits",
            "",
            "Cushion covers debit card purchases only. It never covers ATM withdrawals, transfers or wires.",
            "",
            "## Repaying",
            "",
            f"Your next deposit repays it. If you stay negative for {F.v(f'{C}.repay_within_days')}, Cushion is suspended. If the balance is still negative after {F.v(f'{C}.restriction_after_days')}, the account is restricted until it's repaid.",
            "",
            "## Cost",
            "",
            f"Cushion has no fees and no interest ({F.v(f'{C}.fee_usd')}). It isn't a loan and isn't reported to credit bureaus.",
        ])

    return body


for _p in PLANS:
    art(f"cushion-{_p}", f"Cushion on {pn(_p)}", S_CUSHION, cushion_plan(_p), plans=[_p])


# =====================================================================
# Savings
# =====================================================================

SV = "savings"


def savings_pockets() -> str:
    t("savings_pockets")
    return "\n".join([
        f"Savings Pockets are separate balances inside your account that earn interest. You can have up to {F.raw(f'{SV}.max_pockets')} of them, each with its own name and goal.",
        "",
        "## Create a pocket",
        "",
        "Tap Savings, then New pocket. Name it, set a goal if you like, and move money in.",
        "",
        "## Moving money",
        "",
        "Moving money between your main balance and your pockets is instant and free, every day. There's no limit on how many times you can withdraw from a pocket.",
        "",
        "## Interest",
        "",
        f"All your pockets earn the APY for your plan: {F.v(f'{SV}.versions.v2.basic_apy_pct', decimals=2)} on Basic, {F.v(f'{SV}.versions.v2.plus_apy_pct', decimals=2)} on Plus and {F.v(f'{SV}.versions.v2.premium_apy_pct', decimals=2)} on Premium. Rates are variable.",
        "",
        "## Spending from a pocket",
        "",
        "Your card spends from your main balance, not from pockets. Move money back first.",
        "",
        "## Closing a pocket",
        "",
        "Tap the pocket, then Close. Any money and accrued interest move to your main balance.",
    ])


art("savings-pockets", "How Savings Pockets work", S_SAVINGS, savings_pockets,
    eff=eff_of("savings.versions.v2.effective_date"))


def savings_interest() -> str:
    t("savings_interest_calc")
    return "\n".join([
        F.mark(f"{SV}.compounding", f"{SV}.paid") + "Interest on Savings Pockets is calculated every day and paid once a month.",
        "",
        "## How it's calculated",
        "",
        "Each day we work out interest on the balance in each pocket at your plan's rate, and it compounds daily. The APY already includes the effect of compounding.",
        "",
        "## When it's paid",
        "",
        "Interest for the month is paid into each pocket on the first business day of the next month, and shows as \"Interest\".",
        "",
        "## Balance cap",
        "",
        f"Your plan's APY applies to combined pocket balances up to {F.v(f'{SV}.apy_balance_cap_usd')}. Anything above that earns {F.v(f'{SV}.above_cap_apy_pct', decimals=2)} APY.",
        "",
        "## Changing plans",
        "",
        "If you change plans, the new rate applies from the day the change takes effect. Interest for earlier days in the month uses the old rate.",
        "",
        "## Rate changes",
        "",
        f"Rates are variable. We give at least {F.v(f'{SV}.rate_change_notice_days')} notice by email before we lower a rate.",
        "",
        "Your main balance doesn't earn interest.",
    ])


art("savings-interest-calc", "How savings interest is calculated", S_SAVINGS, savings_interest)


def round_ups() -> str:
    t("round_ups")
    return "\n".join([
        F.mark(f"{SV}.round_ups") + "Round-ups save your spare change. Each time you pay with your card, we round the purchase up to the next dollar and move the difference into a Savings Pocket you choose.",
        "",
        "## Turn on round-ups",
        "",
        "Tap Savings, choose a pocket, and switch on Round-ups.",
        "",
        "## How it works",
        "",
        "- Round-ups are added up during the day and moved once a day.",
        "- Purchases that are already a whole dollar amount aren't rounded.",
        "- Round-ups only apply to card purchases, not to ATM withdrawals, transfers or fees.",
        "- If your main balance is too low, that day's round-up is skipped.",
        "",
        "## Refunds",
        "",
        "If a purchase is refunded, the round-up stays in your pocket. It's your money either way.",
        "",
        "## Turning it off",
        "",
        "Switch off Round-ups on the pocket at any time. Money already saved stays in the pocket.",
        "",
        "Round-ups are available on every plan and earn your plan's savings rate once they're in the pocket.",
    ])


art("savings-round-ups", "Round-ups", S_SAVINGS, round_ups)


def savings_tax() -> str:
    t("tax_forms")
    return "\n".join([
        f"If you earn at least {F.v(f'{SV}.tax_form_threshold_usd')} in interest during a calendar year, we send you a Form {F.raw(f'{SV}.tax_form')}.",
        "",
        "## When it's available",
        "",
        f"By {F.raw(f'{SV}.tax_form_deadline')} of the following year. You'll find it in the app under Profile, then Documents, then Tax forms. We'll notify you when it's ready.",
        "",
        "## Earned less?",
        "",
        "If you earned less than the threshold, we don't send a form, but the interest is still shown on your monthly statements. You may still need to report it.",
        "",
        "## Premium ATM surcharge refunds and bonuses",
        "",
        "ATM surcharge refunds aren't interest and don't appear on the form. Referral bonuses may be reported separately.",
        "",
        "## Tax questions",
        "",
        "We can't give tax advice. For questions about how to report interest, talk to a tax professional.",
        "",
        "## Wrong details on the form",
        "",
        "If your name or taxpayer ID on the form is wrong, message support. Changes to your legal name or taxpayer ID go to our Verification team.",
    ])


art("savings-tax-forms", "Tax forms for interest", S_SAVINGS, savings_tax)


def savings_rate_change() -> str:
    t("savings_rate_change_2026")
    return "\n".join([
        f"On {F.date(f'{SV}.versions.v2.effective_date')}, we changed the rates on Savings Pockets. Here are the old and new rates.",
        "",
        "| Plan | Before | From the change |",
        "|---|---|---|",
        *[f"| {pn(p)} | {F.v(f'{SV}.versions.v1.{p}_apy_pct', decimals=2)} | {F.v(f'{SV}.versions.v2.{p}_apy_pct', decimals=2)} |" for p in PLANS],
        "",
        "## Why",
        "",
        "Our savings rates follow wider interest rates, which came down over the past year. We kept the gap between plans the same.",
        "",
        "## What you need to do",
        "",
        "Nothing. Your pockets started earning the new rate automatically. Interest earned before the change was calculated at the old rate.",
        "",
        "## Notice",
        "",
        f"We emailed every member at least {F.v(f'{SV}.rate_change_notice_days')} before the change.",
        "",
        "## Balance cap",
        "",
        f"The cap didn't change. Plan rates apply up to {F.v(f'{SV}.apy_balance_cap_usd')}.",
        "",
        "## Thinking about switching plans?",
        "",
        "The rate you earn depends on your plan. Upgrading takes effect immediately, so your pockets start earning the higher rate the same day. Compare plans in the app before you decide.",
    ])


art("savings-rate-change-2026", "Savings rate change, June 2026", S_SAVINGS, savings_rate_change,
    eff=eff_of("savings.versions.v2.effective_date"))


def savings_apy(plan: str, ver: int) -> Callable[[], str]:
    def body() -> str:
        t("savings_apy")
        name = pn(plan)
        apy = F.v(f"{SV}.versions.v{ver}.{plan}_apy_pct", decimals=2)
        lines = [
            f"On the {name} plan, Savings Pockets earn {apy} APY, effective {F.date(f'{SV}.versions.v{ver}.effective_date')}.",
            "",
            "## Details",
            "",
            f"- The rate applies to combined pocket balances up to {F.v(f'{SV}.apy_balance_cap_usd')}. Anything above earns {F.v(f'{SV}.above_cap_apy_pct', decimals=2)} APY.",
            "- Interest compounds daily and is paid monthly.",
            "- Your main balance doesn't earn interest. Move money to a pocket to earn it.",
            "- There's no minimum balance and no limit on withdrawals.",
            "",
            "## Rates are variable",
            "",
            f"We may change the rate. We'll email you at least {F.v(f'{SV}.rate_change_notice_days')} before a decrease.",
            "",
            "## How to start earning",
            "",
            "Tap Savings, then New pocket, and move money in from your main balance. Moving money in and out is instant and free. Interest for each month is paid into the pocket on the first business day of the next month.",
            "",
            "## If you change plans",
            "",
            "Your pockets start earning the new plan's rate on the day the change takes effect. Upgrades take effect right away and downgrades on your next billing date.",
        ]
        if plan != "premium":
            nxt = PLANS[PLANS.index(plan) + 1]
            lines += ["", "## Earn more", "", f"{pn(nxt)} members earn {F.v(f'{SV}.versions.v{ver}.{nxt}_apy_pct', decimals=2)} APY on the same balance."]
        return "\n".join(lines)

    return body


for _p in PLANS:
    art(f"savings-apy-{_p}-v1", f"Savings rate on {pn(_p)}", S_SAVINGS, savings_apy(_p, 1), plans=[_p],
        eff=eff_of("savings.versions.v1.effective_date"), version=1)
    art(f"savings-apy-{_p}-v2", f"Savings rate on {pn(_p)}", S_SAVINGS, savings_apy(_p, 2), plans=[_p],
        eff=eff_of("savings.versions.v2.effective_date"), version=2, supersedes=f"savings-apy-{_p}-v1")


# =====================================================================
# Closing your account
# =====================================================================

CL = "closure"


def close_how() -> str:
    t("close_how", "close_requirements")
    return "\n".join([
        "You can close your Tallowbrook account in the app, or ask support to close it for you.",
        "",
        "## Before you close",
        "",
        F.mark(f"{CL}.requirements") + "Your account can only be closed when:",
        "",
        "- No transactions are pending",
        "- No disputes are open",
        "- Your balance isn't negative",
        "",
        "Also move money out of your Savings Pockets and switch your direct deposit and any automatic payments to another account.",
        "",
        "## Close in the app",
        "",
        "Tap Profile, then Close account. The app checks the requirements, shows where any remaining money will go, and asks you to confirm.",
        "",
        "## Closing through support",
        "",
        F.mark(f"{CL}.support_may_close") + f"Support can close your account for you when the balance is {F.v(f'{CL}.inactivity_balance_usd')} and the requirements above are met. If there's money left, move it out first or close in the app, which sends it to you.",
        "",
        "## Plan fees",
        "",
        "Plan fees aren't refunded or prorated when you close, except under the cooling-off rule for a first paid plan charge.",
        "",
        F.mark(f"{CL}.reopen") + "Closing is permanent. A closed account can't be reopened.",
    ])


art("close-how", "How to close your account", S_CLOSE, close_how)


def close_balance() -> str:
    t("close_payout")
    return "\n".join([
        "When you close your account in the app, we send any remaining money to you.",
        "",
        "## How we send it",
        "",
        F.mark(f"{CL}.payout_methods") + "- If you have a linked external account, we send the balance there by ACH.",
        "- If you don't, we mail a check to the address on your profile.",
        "",
        f"Either way, it's sent within {F.v(f'{CL}.payout_business_days')} of closing.",
        "",
        "## Savings Pockets",
        "",
        "Money in your pockets, and any interest earned up to the closing date, is included in the payout.",
        "",
        "## Refunds that arrive later",
        "",
        "If a merchant refund or other credit arrives after your account is closed, we send it to you the same way.",
        "",
        "## Before you close",
        "",
        "Make sure your address is up to date if you'll be paid by check. Checks can't be sent to a PO box.",
        "",
        "## Can support send it somewhere else?",
        "",
        "No. We can only send your balance to an external account in your own name, or by check to your address on file. We can't send it to another person.",
    ])


art("close-balance", "What happens to your money when you close", S_CLOSE, close_balance)


def close_blockers() -> str:
    t("close_requirements")
    return "\n".join([
        "If the app won't let you close your account, one of these is usually the reason.",
        "",
        "## Pending transactions",
        "",
        "Wait until everything has posted. Card authorizations usually post or drop off within a few days.",
        "",
        "## Open disputes",
        "",
        f"We can't close an account while a dispute is open, because the outcome may add or remove money. Most disputes finish within {F.v(f'{D}.investigation_days')}. If the merchant refunded you, you can withdraw the dispute.",
        "",
        "## Negative balance",
        "",
        "Add money to bring the balance back to zero, whether it's from Cushion or from a reversed credit.",
        "",
        "## Account restricted or under review",
        "",
        "If your account is restricted or under review, support can't close it. Message us and we'll pass your request to the team that manages the review.",
        "",
        "## Still stuck?",
        "",
        "Message support with the reason the app gives. We'll tell you exactly what's left to do.",
    ])


art("close-blockers", "Why can't I close my account?", S_CLOSE, close_blockers)


def close_reopen() -> str:
    t("close_reopen")
    return "\n".join([
        F.mark(f"{CL}.reopen") + "Closed accounts can't be reopened. If you'd like to come back, apply for a new account in the app.",
        "",
        "## What a new account means",
        "",
        "- New account and card numbers",
        "- New verification. You'll go through identity checks again.",
        "- Your old transaction history stays with the closed account.",
        "- Any cooling-off refund for a first paid plan charge only applies once per person.",
        "",
        "## Statements from the old account",
        "",
        f"You can download statements for {F.v(f'{CL}.statements_after_closure_days')} after closing. After that, message support to request them.",
        "",
        "## If we closed your account",
        "",
        "If Tallowbrook closed your account, for example after a review, you may not be able to open a new one. Support can't reverse that decision.",
        "",
        "## Inactive accounts",
        "",
        "If your account was closed for inactivity, you can apply again at any time.",
        "",
        "## Your old account details",
        "",
        "Update your employer and any companies that paid into your old account. Payments sent to a closed account are returned to the sender.",
    ])


art("close-reopen", "Reopening a closed account", S_CLOSE, close_reopen)


def close_inactivity() -> str:
    t("close_inactivity")
    return "\n".join([
        F.mark(f"{CL}.inactivity_closure_rule") + f"We may close accounts that have a {F.v(f'{CL}.inactivity_balance_usd')} balance and no activity for 12 months.",
        "",
        "## Notice",
        "",
        f"We email you at least {F.v(f'{CL}.inactivity_notice_days')} before closing. Any activity during that time, such as a deposit or a card purchase, keeps the account open.",
        "",
        "## Fees",
        "",
        f"We don't charge inactivity fees ({F.v('fees.all_plans.inactivity_fee_usd')}) on any plan.",
        "",
        "## Accounts with money in them",
        "",
        "We don't close accounts for inactivity while they hold a balance, including money in Savings Pockets.",
        "",
        "## Paid plans",
        "",
        "A paid plan keeps charging its monthly fee while the account is open, which counts as activity. If you don't use your account, consider moving to Basic.",
        "",
        "## Coming back",
        "",
        "If your account was closed for inactivity, you can apply for a new one whenever you like.",
        "",
        "## Keeping an account open",
        "",
        "Any card purchase, deposit or transfer counts as activity. Signing in to the app on its own doesn't.",
    ])


art("close-inactivity", "Inactive accounts", S_CLOSE, close_inactivity)


def close_statements() -> str:
    t("close_statements")
    return "\n".join([
        f"After you close your account, you can still sign in and download statements for {F.v(f'{CL}.statements_after_closure_days')}.",
        "",
        "## Download them before you go",
        "",
        "We recommend downloading any statements and tax forms you might need before you close. Tap Profile, then Documents.",
        "",
        "## After that",
        "",
        "Once the download period ends, you can't sign in. Message support from the sign-in screen, or email us, and we'll send the statements you need. We keep records for seven years.",
        "",
        "## Tax forms",
        "",
        f"If you earned at least {F.v(f'{SV}.tax_form_threshold_usd')} in interest in the year you closed, we still send your {F.raw(f'{SV}.tax_form')} by {F.raw(f'{SV}.tax_form_deadline')}.",
        "",
        "## Verification letters",
        "",
        "Account verification letters are only available for open accounts.",
        "",
        "## Card and transaction history",
        "",
        "Your statements list every transaction, fee, interest payment and dispute credit for the month, so they're usually all you need for records, taxes or a loan application elsewhere.",
    ])


art("close-statements", "Statements after closing", S_CLOSE, close_statements)


# =====================================================================
# Help and support
# =====================================================================

SP = "support"


def support_contact() -> str:
    t("support_hours")
    return "\n".join([
        "Here's how to reach us.",
        "",
        "## Chat in the app",
        "",
        F.mark(f"{SP}.chat_hours") + "Tap Help, then Message us. Chat is open around the clock, every day. It's the fastest way to reach us, and we already know it's you because you're signed in.",
        "",
        "## Phone",
        "",
        F.mark(f"{SP}.phone_days") + f"Call {F.raw('meta.support_phone')} from {F.v(f'{SP}.phone_open_time_et')} to {F.v(f'{SP}.phone_close_time_et')}, every day. You'll be asked to confirm a code in the app to prove it's you.",
        "",
        "## Email",
        "",
        f"Email {F.raw('meta.support_email')} if you can't sign in. We reply by email, but for your security we can't make changes to your account by email.",
        "",
        "## Specialist teams",
        "",
        f"Some requests go to a specialist team, such as Fraud, Disputes or Verification. They reply within {F.v(f'{SP}.escalation_response_business_days')}.",
        "",
        "## Beware of fakes",
        "",
        "Only use the contact details in the app or on this help center. Scammers post fake support numbers online.",
    ])


art("support-contact", "Contacting support", S_SUPPORT, support_contact)


def support_scope() -> str:
    t("support_scope")
    F.mark(f"{SP}.may_do", f"{SP}.must_escalate", f"{SP}.must_refuse")
    return "\n".join([
        "Our support team can sort out most things during your chat. Some requests go to a specialist team, and a few things we can't do at all.",
        "",
        "## Support can",
        "",
        "- Freeze or unfreeze your card",
        "- Report a card lost or stolen and order a replacement",
        f"- Open a dispute that's inside its window, has posted, and totals up to {F.v('disputes.support_max_dispute_usd')}",
        "- Refund a fee under our refund policy",
        "- Change your plan",
        "- Close your account if the balance is zero and nothing is pending or disputed",
        "- Explain fees, limits and policies",
        "",
        "## Support passes to a specialist team",
        "",
        "- **Fraud:** suspected account takeover, removing a fraud lock, scam payments",
        "- **Disputes:** disputes above the support limit",
        "- **Verification:** identity checks, legal name or date-of-birth changes, accounts under review or restricted",
        "- **Account Services:** power of attorney, a deceased account holder, financial abuse",
        "- **Complaints:** formal complaints",
        "",
        f"Specialist teams reply within {F.v(f'{SP}.escalation_response_business_days')}.",
        "",
        "## Support can't",
        "",
        "- Act for, or share details with, anyone other than the account holder",
        "- Give investment, tax or legal advice",
        "- Raise limits",
        "- Refund fees outside the refund policy, or open disputes after their window",
    ])


art("support-what-we-can-do", "What our support team can help with", S_SUPPORT, support_scope)


def complaints() -> str:
    t("complaints")
    return "\n".join([
        "If you're unhappy with something we've done, tell us. You can make a formal complaint at any time.",
        "",
        "## How to complain",
        "",
        "Message support in the app and say you'd like to make a formal complaint. Tell us what happened and what you'd like us to do. Support will pass it to our Complaints team.",
        "",
        "## What happens next",
        "",
        f"- We acknowledge your complaint within {F.v(f'{SP}.complaint_ack_business_days')}.",
        f"- We send a final response within {F.v(f'{SP}.complaint_final_response_business_days')}.",
        "",
        "If we need more time, for example because we're waiting on another bank, we'll tell you why and when to expect an answer.",
        "",
        "## Our response",
        "",
        "The final response explains what we found, what we'll do, and your options if you're still unhappy, including where you can take your complaint outside Tallowbrook.",
        "",
        "## Legal matters",
        "",
        "If your message involves legal action or a regulator, support passes it straight to the Complaints team.",
        "",
        "Making a complaint never affects how we treat your account.",
    ])


art("support-complaints", "Making a complaint", S_SUPPORT, complaints)


def third_parties() -> str:
    t("third_party", "joint_accounts")
    return "\n".join([
        F.mark(f"{SP}.identity") + "Tallowbrook accounts belong to one person. We don't offer joint accounts or authorized users, and support only acts for the account holder in their own signed-in session.",
        "",
        "## What that means",
        "",
        "- We can't discuss an account with a spouse, partner, parent, child or friend, even if they know the account details.",
        "- We can't make changes because someone else asks, even if they're messaging from the account holder's phone.",
        "- We can't confirm whether a person has a Tallowbrook account.",
        "",
        "## Helping someone with their account",
        "",
        "The account holder can sit with you and message us from their own session. We'll talk to them directly.",
        "",
        "## Legal authority",
        "",
        "If you have power of attorney, or you're handling the account of someone who has died, we can help through our Account Services team. See \"Power of attorney\" and \"If an account holder has died\".",
        "",
        "## If you think someone is misusing your account",
        "",
        "If someone else is using your account without your permission, or pressuring you over your money, tell us in the app. See \"If someone is controlling your money\".",
    ])


art("support-third-parties", "Can someone else manage my account?", S_SUPPORT, third_parties)


def poa() -> str:
    t("power_of_attorney")
    return "\n".join([
        "If you have power of attorney for a Tallowbrook member, our Account Services team can add you to their account as an agent.",
        "",
        "## What we need",
        "",
        "- A copy of the power of attorney document, signed and in effect",
        "- Your government-issued photo ID",
        "- Your contact details",
        "",
        "## How to send it",
        "",
        f"Email {F.raw('meta.support_email')} with the subject Power of attorney, or ask the account holder to message support from their session. Support can't review the documents. They go to Account Services.",
        "",
        "## How long it takes",
        "",
        f"Account Services replies within {F.v(f'{SP}.escalation_response_business_days')} to confirm what they received, and may ask for more information.",
        "",
        "## Until it's approved",
        "",
        "We can't share information or take instructions from you until Account Services has approved the document. That includes freezing cards or changing plans.",
        "",
        "## If the account holder is in danger",
        "",
        "If you think the account holder is being exploited financially, tell us. We treat that as urgent.",
    ])


art("support-power-of-attorney", "Power of attorney", S_SUPPORT, poa)


def deceased() -> str:
    t("deceased")
    return "\n".join([
        "We're sorry for your loss. Here's how to let us know that a Tallowbrook member has died.",
        "",
        "## Tell us",
        "",
        f"Email {F.raw('meta.support_email')} with the subject Bereavement, or message support. Support passes every bereavement case to our Account Services team, who handle it from start to finish.",
        "",
        "## What Account Services will ask for",
        "",
        "- The death certificate",
        "- Proof that you're the executor or administrator of the estate, or the next of kin",
        "- Your photo ID",
        "",
        "## What happens to the account",
        "",
        "Once we're notified, we freeze the account and cards to protect them. Recurring payments stop. Direct deposits received after the date of death may be returned to the sender.",
        "",
        "After Account Services confirms who's authorized, the balance is paid to the estate.",
        "",
        "## Timing",
        "",
        f"Account Services contacts you within {F.v(f'{SP}.escalation_response_business_days')}. Please don't try to sign in to the account or use its cards.",
    ])


art("support-deceased", "If an account holder has died", S_SUPPORT, deceased)


def financial_abuse() -> str:
    t("financial_abuse")
    return "\n".join([
        "If someone is controlling your money, pressuring you to hand it over, or using your account without your permission, you're not alone, and we can help.",
        "",
        "## Tell us safely",
        "",
        "Message support in the app when it's safe. You can use a code phrase like \"I need to talk about my account security\" if someone might see your screen. Support will pass your case to our Account Services team, who are trained to help.",
        "",
        "## What we can do",
        "",
        "- Change your password and sign out other devices",
        "- Freeze or replace your card",
        "- Stop sending paper mail to your address",
        "- Talk about options for moving money you need to keep safe",
        "",
        "## What we won't do",
        "",
        "We won't tell anyone else you contacted us.",
        "",
        "## If you're in danger",
        "",
        "If you're in immediate danger, call emergency services first.",
        "",
        f"Account Services replies within {F.v(f'{SP}.escalation_response_business_days')}, and faster when you tell us it's urgent.",
    ])


art("support-financial-abuse", "If someone is controlling your money", S_SUPPORT, financial_abuse)


def merchant_refund_timing() -> str:
    t("merchant_refund_timing")
    return "\n".join([
        f"When a merchant refunds a card purchase, it usually shows in your account within {F.range_bd('refunds.merchant_refund_min_business_days', 'refunds.merchant_refund_max_business_days')} of the merchant issuing it.",
        "",
        "## Why it takes time",
        "",
        "The merchant sends the refund through the card network, which passes it to us. We credit it as soon as we receive it. The delay is almost always before it reaches us.",
        "",
        "## Pending refunds",
        "",
        "Some refunds show as pending first. The money is available once it posts.",
        "",
        "## If it's been longer",
        "",
        f"If more than {F.v('refunds.merchant_refund_max_business_days')} have passed, ask the merchant for the refund reference number, sometimes called an ARN, and message support. We can trace it.",
        "",
        "## Refunds to a cancelled card",
        "",
        "Refunds to a card you've replaced still reach your account. You don't need to give the merchant your new card.",
        "",
        "## Refunds in a foreign currency",
        "",
        "Foreign refunds are converted on the day they settle, so the amount can differ slightly from what you paid.",
        "",
        "## Merchant won't refund?",
        "",
        "You may be able to file a merchant dispute. See \"Disputing a purchase with a merchant\".",
    ])


art("support-merchant-refunds", "When will my refund arrive?", S_SUPPORT, merchant_refund_timing)


def no_advice() -> str:
    t("no_advice")
    return "\n".join([
        "Our support team can explain how Tallowbrook products work, but we can't give financial, investment, tax or legal advice.",
        "",
        "## What we can't help with",
        "",
        "- Whether to invest, save or pay down debt",
        "- Which stocks, funds or cryptocurrencies to buy or sell",
        "- How much of your money to keep in savings",
        "- How to report interest or other income on your taxes",
        "- Legal questions about wills, divorce or disputes with other people",
        "",
        "## What we can help with",
        "",
        "- How our savings rates and Savings Pockets work",
        "- Fees and limits on each plan, so you can decide which suits you",
        "- Your statements and tax forms",
        "",
        "## Where to get advice",
        "",
        "For advice about your situation, talk to a qualified financial adviser, tax professional or lawyer.",
        "",
        "## Investing",
        "",
        "Tallowbrook doesn't offer investment or brokerage accounts. See \"Investing and cryptocurrency\".",
    ])


art("support-no-advice", "Financial advice", S_SUPPORT, no_advice)


# =====================================================================
# Account opening and verification
# =====================================================================

V = "verification"


def eligibility() -> str:
    t("eligibility", "under_18")
    F.mark(f"{V}.requirements")
    return "\n".join([
        "You can open a Tallowbrook account if you:",
        "",
        f"- Are at least {F.raw(f'{V}.min_age_years')} years old",
        "- Live in the US and have a US residential address (not a PO box)",
        "- Have a Social Security number or ITIN",
        "- Have an unexpired government-issued photo ID",
        "- Have a smartphone that can run the Tallowbrook app",
        "",
        "## Opening an account",
        "",
        "Download the app and tap Get started. You'll enter your details, take a photo of your ID and a selfie. Every account starts on Basic, and you can change plans once you're verified.",
        "",
        "## One account per person",
        "",
        "Each person can have one Tallowbrook account. We don't offer joint accounts, business accounts or accounts for people under 18.",
        "",
        "## Living abroad",
        "",
        "If you move abroad permanently, you may need to close your account. Tell us before you move.",
        "",
        "## Previously closed by us",
        "",
        "If Tallowbrook closed an account of yours in the past, a new application may be declined.",
    ])


art("account-eligibility", "Who can open an account", S_ACC, eligibility)


def documents() -> str:
    t("kyc_documents")
    return "\n".join([
        "To verify your identity, we need a photo ID and a selfie, plus a few details.",
        "",
        "## Photo IDs we accept",
        "",
        "- US driver's license or state ID card",
        "- US passport or passport card",
        "- Permanent resident card",
        "- Foreign passport, together with an ITIN",
        "",
        "The ID must be unexpired, and all four corners must be visible in the photo.",
        "",
        "## Selfie",
        "",
        "The app asks you to take a short selfie video to match your face to the ID. Remove glasses and hats, and use good light.",
        "",
        "## Details",
        "",
        "Your legal name, date of birth, residential address and Social Security number or ITIN. Your name must match your ID.",
        "",
        "## Proof of address",
        "",
        f"If we can't confirm your address automatically, we'll ask for a document dated within the last {F.v(f'{V}.proof_of_address_max_age_days')}, such as a utility bill, lease or bank statement.",
        "",
        "## What we don't accept",
        "",
        "Expired IDs, photocopies, student IDs, or screenshots of an ID.",
    ])


art("account-documents", "Documents for identity verification", S_ACC, documents)


def verification_time() -> str:
    t("verification_time")
    return "\n".join([
        f"Most applications are verified within {F.v(f'{V}.typical_business_days')}, and many within minutes.",
        "",
        "## Manual review",
        "",
        f"If we need to check something by hand, it can take up to {F.v(f'{V}.manual_review_max_business_days')}. We may ask for another document, such as proof of address.",
        "",
        "## While you wait",
        "",
        F.mark(f"{V}.under_review_restrictions") + "You can receive deposits into your new account, but you can't send money, withdraw cash or order a card until you're verified.",
        "",
        "## Checking status",
        "",
        "The app shows your verification status on the home screen. We'll notify you as soon as it's done.",
        "",
        "## Can support speed it up?",
        "",
        F.mark(f"{V}.decisions") + "No. Verification is handled by our Verification team, and support can't approve applications or tell you more than the app shows. If we need anything from you, we'll ask in the app.",
        "",
        "## Common causes of delays",
        "",
        "A blurry ID photo, a name that doesn't match your ID exactly, or a recent move are the most common reasons we need a manual review. Checking these before you apply helps.",
    ])


art("account-verification-time", "How long verification takes", S_ACC, verification_time)


def verification_failed() -> str:
    t("verification_failed")
    return "\n".join([
        "If we couldn't verify your identity, the app tells you what went wrong and whether you can try again.",
        "",
        "## Common reasons",
        "",
        "- The photo of your ID was blurry, cropped or had glare",
        "- The name or date of birth didn't match your ID",
        "- The ID had expired",
        "- The selfie didn't match the ID photo",
        "",
        "## Trying again",
        "",
        f"You can try up to {F.raw(f'{V}.max_attempts')} times. After that, the application is closed and you can apply again after {F.v(f'{V}.reapply_after_days')}.",
        "",
        "## Can support help?",
        "",
        "Support can explain the steps, but can't approve an application, override a decision, or tell you why a decision was made beyond what the app shows. Verification decisions are made by our Verification team.",
        "",
        "## Tips",
        "",
        "Use your ID's exact legal name, photograph it on a dark background in daylight, and take the selfie facing a window.",
    ])


art("account-verification-failed", "If we couldn't verify your identity", S_ACC, verification_failed)


def address_change() -> str:
    t("address_change")
    return "\n".join([
        "You can update your address in the app under Profile, then Personal details.",
        "",
        "## Proof of address",
        "",
        f"Sometimes we can confirm a new address automatically. If we can't, we'll ask you to upload a document dated within the last {F.v(f'{V}.proof_of_address_max_age_days')} that shows your name and new address:",
        "",
        "- Utility bill",
        "- Lease or mortgage statement",
        "- Bank or credit card statement",
        "- Government letter",
        "",
        "Until it's confirmed, cards and checks still go to your old address.",
        "",
        "## Moving abroad",
        "",
        "We can only use US residential addresses. If you're moving abroad, tell us first.",
        "",
        "## Ordering a card after moving",
        "",
        "Update your address before ordering a replacement card. We can't redirect a card that's already been mailed.",
        "",
        "## Name changes",
        "",
        "Changing your name is a different process. See \"Changing your legal name\".",
        "",
        "## Mail we send",
        "",
        "We send very little by post. Cards, and a check if you close your account without a linked external account, go to the address on file, so keep it current.",
    ])


art("account-address-change", "Updating your address", S_ACC, address_change)


def name_change() -> str:
    t("name_change")
    return "\n".join([
        F.mark(f"{V}.name_change") + "If your legal name has changed, for example after marriage or divorce, our Verification team can update it.",
        "",
        "## What you'll need",
        "",
        "- A photo ID in your new name, or your current ID plus a legal document showing the change, such as a marriage certificate or court order",
        "",
        "## How to request it",
        "",
        "Message support in the app and ask to change your legal name. Support can't change it directly, but will pass the request to the Verification team and tell you how to upload the documents.",
        "",
        "## How long it takes",
        "",
        f"Up to {F.v(f'{V}.name_change_business_days')} after we receive your documents.",
        "",
        "## Date of birth",
        "",
        "If your date of birth is wrong on your account, the same process applies.",
        "",
        "## Your card",
        "",
        "Once the change is approved, you can order a card in your new name. Replacement fees follow your plan, but we waive the fee for name changes.",
        "",
        "## Preferred name",
        "",
        "You can set a preferred first name for the app yourself under Profile. It doesn't change your legal name.",
    ])


art("account-name-change", "Changing your legal name", S_ACC, name_change)


def under_review() -> str:
    t("account_under_review")
    return "\n".join([
        "Sometimes we need to review an account, for example to confirm identity details, check unusual activity, or meet legal requirements.",
        "",
        "## What you can do during a review",
        "",
        F.mark(f"{V}.under_review_restrictions") + "You can still receive deposits. Depending on the review, you may not be able to send money, withdraw cash, use your card or order a new card.",
        "",
        "## What you'll see",
        "",
        "A banner in the app tells you the account is under review or restricted, and what we need from you, if anything.",
        "",
        "## How long it takes",
        "",
        "It depends on the reason. If we need documents, sending them quickly is the best way to speed things up.",
        "",
        "## Can support help?",
        "",
        "Support can't lift a review or restriction, and often can't share the reason. Support will pass your message to the Verification team, who reply within " + F.v(f"{SP}.escalation_response_business_days") + ".",
        "",
        "## Negative balance restrictions",
        "",
        "If your account is restricted because of a negative balance, adding money lifts the restriction once the balance is back to zero.",
    ])


art("account-under-review", "Account under review or restricted", S_ACC, under_review)


def contact_change() -> str:
    t("contact_change")
    return "\n".join([
        "You can change your email address and phone number in the app under Profile, then Personal details.",
        "",
        "## Email",
        "",
        "Enter the new email. We send a confirmation link to the new address and a notice to the old one. The change happens when you tap the link.",
        "",
        "## Phone number",
        "",
        "Enter the new number and confirm the code we text to it. You'll also be asked to approve the change from your current device.",
        "",
        "## Lost access to your old phone or email",
        "",
        "Message support. Because this is a common way for fraudsters to take over accounts, we pass these requests to our Verification team, who may ask for a new selfie.",
        "",
        "## After a change",
        "",
        f"For {F.v('limits.new_device_hold_hours')} after a change, if you're also on a new device, BrookPay and external transfers are limited to {F.v('limits.new_device_transfer_cap_usd')}.",
        "",
        "If you get a notice about a change you didn't make, message us right away.",
    ])


art("account-contact-change", "Changing your email or phone number", S_ACC, contact_change)


# =====================================================================
# Statements and documents
# =====================================================================

def statements() -> str:
    t("statements")
    return "\n".join([
        f"Your monthly statement is ready within {F.v('statements.available_business_days_after_month_end')} after the end of each month. We'll notify you when it's available.",
        "",
        "## Download a statement",
        "",
        "Tap Profile, then Documents, then Statements, and choose the month. Statements are PDFs.",
        "",
        "## How far back",
        "",
        F.mark("statements.history") + "Seven years of statements are available in the app.",
        "",
        "## Statement dates",
        "",
        "Statements cover a calendar month and close on the last day of the month. The statement date matters for disputes about unauthorized transactions, which must be reported within " + F.v("disputes.unauthorized_window_days") + " of the statement they first appear on.",
        "",
        "## Paper statements",
        "",
        f"We don't mail paper statements, and there's no fee for any statement ({F.v('fees.all_plans.paper_statement_fee_usd')}).",
        "",
        "## Savings Pockets",
        "",
        "Pocket balances and interest appear on the same statement as your main balance.",
        "",
        "## Something looks wrong",
        "",
        "If a statement shows a transaction you don't recognize, tap it in the app to see the merchant details, and dispute it if you didn't make it.",
    ])


art("docs-statements", "Monthly statements", S_DOCS, statements)


def verification_letter() -> str:
    t("verification_letter")
    return "\n".join([
        F.mark("statements.verification_letter") + "An account verification letter confirms that you have an open account with us. Landlords, employers and government offices sometimes ask for one.",
        "",
        "## Get a letter",
        "",
        "Tap Profile, then Documents, then Account verification letter. The PDF is ready right away and shows your name, account number, account type and the date the account was opened.",
        "",
        "## Showing your balance",
        "",
        "You can choose whether the letter includes your current balance.",
        "",
        "## Custom letters",
        "",
        "We can't write custom letters, sign forms, or fill in third-party verification forms. The standard letter is accepted by most organizations.",
        "",
        "## Closed accounts",
        "",
        "Letters are only available for open accounts. For a closed account, use your final statement.",
        "",
        "## Someone else asking",
        "",
        "We don't confirm account details to landlords, employers or anyone else who contacts us directly. Give them the letter yourself.",
        "",
        "## Is it official?",
        "",
        "Yes. The letter carries a verification code that the recipient can check on our help center.",
    ])


art("docs-verification-letter", "Account verification letter", S_DOCS, verification_letter)


def account_details() -> str:
    t("account_details")
    return "\n".join([
        "Your routing number and account number are in the app under Account details. You'll need them for direct deposit, to link Tallowbrook at another bank, or to receive a wire.",
        "",
        "## Where to find them",
        "",
        "Tap Profile, then Account details. Tap the eye icon to show the full account number. The app asks you to confirm it's you first.",
        "",
        "## Card number versus account number",
        "",
        "Your card number is different from your account number. Don't give your card number to an employer for direct deposit.",
        "",
        "## Keeping them safe",
        "",
        "Only give your account and routing numbers to people and companies you trust, such as your employer. Someone with these numbers can try to pull money by ACH. If that happens, you can dispute it.",
        "",
        "## Support won't read them out",
        "",
        "For security, support won't read your full account number or card number to you in chat or on the phone. You can always see them in the app.",
        "",
        "## Wire instructions",
        "",
        "For incoming wires, use the Wire instructions screen under Account details, which also shows our SWIFT code for international wires.",
    ])


art("docs-account-details", "Finding your account and routing numbers", S_DOCS, account_details)


# =====================================================================
# What we offer
# =====================================================================

def joint_accounts() -> str:
    t("joint_accounts")
    return "\n".join([
        "No. Tallowbrook accounts are individual accounts only. We don't offer joint accounts, and you can't add an authorized user or a second card holder.",
        "",
        "## Sharing money with a partner",
        "",
        "Each of you can open your own Tallowbrook account and send money to each other instantly with BrookPay, for free.",
        "",
        "## Shared goals",
        "",
        "Savings Pockets belong to one account and can't be shared. You can each save toward the same goal in your own pockets.",
        "",
        "## Managing an account for someone else",
        "",
        "If you have power of attorney, see \"Power of attorney\". Otherwise, we can only talk to the account holder.",
        "",
        "## Will you offer joint accounts?",
        "",
        "We don't have plans to share right now.",
        "",
        "## Couples and roommates",
        "",
        "Splitting a bill is easy with BrookPay. One person pays, and the other sends their share straight from the app. BrookPay has no fee and arrives in seconds.",
        "",
        "## Moving money from a joint account elsewhere",
        "",
        "You can link a joint account you hold at another bank as an external account, as long as your name is on it.",
    ])


art("products-joint-accounts", "Do you offer joint accounts?", S_PRODUCTS, joint_accounts)


def business_accounts() -> str:
    t("business_accounts")
    return "\n".join([
        "No. Tallowbrook only offers personal accounts. We don't offer business accounts for companies, partnerships, nonprofits or sole proprietors.",
        "",
        "## Using a personal account for business",
        "",
        "Personal accounts are for personal use. Small amounts of freelance or side income are fine, such as getting paid for occasional gigs. Running a business's regular payments through a personal account isn't allowed, and could lead to a review of your account.",
        "",
        "## Business payments into your account",
        "",
        "You can receive payments from a business, such as your employer or a client, into your personal account.",
        "",
        "## Business debit cards",
        "",
        "Your Tallowbrook card is a personal debit card. It can't carry a business name.",
        "",
        "## Tax forms",
        "",
        "We only issue interest tax forms. We can't issue forms for business income.",
        "",
        "## Looking for a business account?",
        "",
        "You'll need a bank that offers business accounts. We can't recommend a specific one.",
    ])


art("products-business", "Business accounts", S_PRODUCTS, business_accounts)


def credit_products() -> str:
    t("credit_products")
    return "\n".join([
        "Tallowbrook doesn't offer credit cards, personal loans, mortgages, auto loans or lines of credit.",
        "",
        "## Your Tallowbrook card is a debit card",
        "",
        "It spends money that's already in your account. It doesn't build credit history and doesn't appear on your credit report.",
        "",
        "## Is Cushion a loan?",
        "",
        "Cushion covers small debit card purchases on Plus and Premium with no fees or interest. It isn't a line of credit, isn't reported to credit bureaus, and is repaid automatically from your next deposit.",
        "",
        "## Credit checks",
        "",
        "We don't run a hard credit check when you open an account.",
        "",
        "## Paying a loan or card from another lender",
        "",
        "You can pay other lenders from your Tallowbrook account by ACH, using your routing and account number.",
        "",
        "## Looking for credit?",
        "",
        "We can't recommend lenders or give advice about borrowing.",
        "",
        "## Credit reports",
        "",
        "Because we don't lend, we don't report your account to credit bureaus. A Tallowbrook account won't raise or lower your credit score.",
    ])


art("products-credit", "Credit cards and loans", S_PRODUCTS, credit_products)


def crypto_investing() -> str:
    t("crypto_investing")
    return "\n".join([
        "Tallowbrook doesn't offer investing, brokerage or retirement accounts, and you can't buy, sell or hold cryptocurrency with us.",
        "",
        "## Sending money to an investment app",
        "",
        "You can link your Tallowbrook account to an investment or crypto platform and move money by ACH, like any other external account. Some platforms also accept debit card payments.",
        "",
        "## Crypto purchases with your card",
        "",
        "Some crypto platforms treat card purchases as cash advances. We don't charge cash advance fees, but the platform might. Card purchases from crypto platforms can't be disputed if you simply change your mind.",
        "",
        "## Scams",
        "",
        "Be careful of anyone promising guaranteed returns or asking you to move money into crypto to keep it safe. That's a common scam.",
        "",
        "## Advice",
        "",
        "Support can't advise you on investments or cryptocurrency.",
        "",
        "## Earning more on savings",
        "",
        "Savings Pockets earn interest at your plan's rate. See the savings rate article for your plan.",
    ])


art("products-crypto-investing", "Investing and cryptocurrency", S_PRODUCTS, crypto_investing)


def checks() -> str:
    t("paper_checks")
    return "\n".join([
        "Tallowbrook doesn't offer paper checkbooks, cashier's checks or money orders.",
        "",
        "## Paying someone who needs a check",
        "",
        "Options include BrookPay (if they're a Tallowbrook member), an ACH transfer to their bank account, or a wire. Many landlords accept ACH.",
        "",
        "## Depositing checks",
        "",
        "You can deposit paper checks made out to you by taking a photo in the app. See \"Depositing a check in the app\".",
        "",
        "## Money orders",
        "",
        "We can't issue money orders, and money orders can't be deposited through the app.",
        "",
        "## Closing balance checks",
        "",
        "The only time we mail a check is when you close your account without a linked external account. We send the remaining balance by check.",
        "",
        "## Direct deposit forms that ask for a voided check",
        "",
        "Use the pre-filled direct deposit form in the app instead. It has everything a voided check would show.",
    ])


art("products-checks", "Checkbooks, cashier's checks and money orders", S_PRODUCTS, checks)


def under_18() -> str:
    t("under_18")
    return "\n".join([
        f"Tallowbrook accounts are for people aged {F.raw(f'{V}.min_age_years')} and over. We don't offer teen accounts, custodial accounts or accounts for children.",
        "",
        "## Sending money to a young person",
        "",
        "If your child has an account at another bank, you can send money there by ACH.",
        "",
        "## Adding a child to your account",
        "",
        "You can't add a child as an authorized user or give them a card on your account.",
        "",
        "## When they turn 18",
        "",
        "They can apply for their own account in the app with their own ID and Social Security number or ITIN.",
        "",
        "## Using a parent's account",
        "",
        "Please don't let someone else use your card or app. Transactions made by someone you gave your card to aren't covered as unauthorized.",
        "",
        "## Students",
        "",
        "Students aged 18 and over can open an account like anyone else.",
        "",
        "## Gifts for a child",
        "",
        "We can't hold money on a child's behalf. If you want to save for a child, you can create a Savings Pocket in your own account and name it after them. The money stays yours.",
    ])


art("products-under-18", "Accounts for under-18s", S_PRODUCTS, under_18)


def referrals() -> str:
    t("referrals")
    R = "referrals"
    return "\n".join([
        f"When a friend opens a Tallowbrook account with your invite link and qualifies, you each get {F.v(f'{R}.bonus_usd')}.",
        "",
        "## How your friend qualifies",
        "",
        f"They need to receive at least {F.v(f'{R}.friend_min_direct_deposit_usd')} in direct deposits within {F.v(f'{R}.qualify_within_days')} of opening their account.",
        "",
        "## When you're paid",
        "",
        f"Both bonuses are paid within {F.v(f'{R}.payout_business_days')} of your friend qualifying.",
        "",
        "## Your invite link",
        "",
        "Tap Profile, then Invite friends. Share the link any way you like. Your friend has to use the link when they sign up. We can't add a referral afterward.",
        "",
        "## Rules",
        "",
        "- Your friend must be new to Tallowbrook.",
        "- You can't refer yourself or open a second account.",
        "- Referral bonuses may be reported to tax authorities.",
        "",
        "## Didn't get your bonus?",
        "",
        "Check your friend used your link and received the direct deposits in time. If they did, message support.",
    ])


art("referrals", "Refer a friend", S_PRODUCTS, referrals)


# =====================================================================
# Build
# =====================================================================

def build() -> list[dict]:
    out = []
    seen = set()
    for i, meta in enumerate(REGISTRY):
        if meta["article_id"] in seen:
            raise SystemExit(f"duplicate article id {meta['article_id']}")
        seen.add(meta["article_id"])
        with F.recording() as keys:
            for tag in meta["tags"]:
                F.mark(f"tag:{tag}")
            body = meta["body_fn"]().rstrip()
        footer = FOOTERS[sum(map(ord, meta["article_id"])) % len(FOOTERS)]
        body = f"{body}\n\n{footer}"
        out.append(
            dict(
                article_id=meta["article_id"],
                title=meta["title"],
                section=meta["section"],
                plans=meta["plans"],
                effective_date=meta["effective_date"],
                version=meta["version"],
                supersedes=meta["supersedes"],
                body=body,
                _keys=sorted(keys),
            )
        )
    return out


def public(article: dict) -> dict:
    return {k: v for k, v in article.items() if not k.startswith("_")}


def main() -> None:
    articles = build()
    path = ROOT / "corpus" / "articles.jsonl"
    with open(path, "w") as fh:
        for a in articles:
            fh.write(json.dumps(public(a), ensure_ascii=False) + "\n")
    print(f"wrote {len(articles)} articles to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
