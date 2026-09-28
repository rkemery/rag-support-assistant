"""Build rag/questions.jsonl (and the per-split copies) from hand-written questions.

Questions and reference answers are written here by hand. Every number in a
reference answer comes from facts/policies.yaml through `F`, and the facts an
answer uses (plus the topic tags listed with each question) decide its gold
articles: every current article that states one of those facts or covers one
of those topics. Superseded articles are gold only when a question is about
the change itself (sup=True).

Run: uv run python scripts/build_questions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_corpus import F, build as build_corpus  # noqa: E402
from facts import ROOT  # noqa: E402

fee, fof, lim, v, n, vq = F.fee, F.fee_or_free, F.lim, F.v, F.n, F.quiet


def apy(plan: str, ver: int = 2) -> str:
    return v(f"savings.versions.v{ver}.{plan}_apy_pct", decimals=2)


def bd(lo: str, hi: str) -> str:
    return F.range_bd(lo, hi)


ANSWERABLE: list[dict] = []
UNANSWERABLE: list[dict] = []


def A(q: str, ans: Callable[[], str], tags: str = "", plan: str | None = None, d: str = "easy",
      sup: bool = False, gold: str = "") -> None:
    ANSWERABLE.append(dict(q=q, ans=ans, tags=tags.split(), plan=plan, d=d, sup=sup, gold=gold.split()))


def U(split: str, kind: str, q: str, ans: Callable[[], str], tags: str = "", plan: str | None = None, d: str = "medium") -> None:
    UNANSWERABLE.append(dict(split=split, kind=kind, q=q, ans=ans, tags=tags.split(), plan=plan, d=d, sup=False, gold=[]))


# =====================================================================
# Answerable (160)
# =====================================================================

# ---- plans and fees
A("How much does the Plus plan cost per month?",
  lambda: f"Plus costs {fee('plus', 'monthly_fee_usd')} a month.", plan="plus")
A("is basic actually free or is there a catch",
  lambda: f"Basic has no monthly fee ({fee('basic', 'monthly_fee_usd')}) and no minimum balance. Some fees can still apply, such as {vq(F.fee_path('basic', 'out_of_network_atm_fee_usd'))} for out-of-network ATM withdrawals and a {vq(F.fee_path('basic', 'fx_fee_pct'))} foreign transaction fee.",
  plan="basic", d="medium")
A("whats the monthly fee for premium and do i get a metal card with it",
  lambda: f"Premium costs {fee('premium', 'monthly_fee_usd')} a month, and yes, Premium members get a metal card.",
  tags="metal_card", plan="premium", d="medium")
A("If I upgrade from Basic to Premium today, when do I get charged?",
  lambda: f"Upgrades take effect immediately. You're charged Premium's full monthly fee of {fee('premium', 'monthly_fee_usd')} that day, and that day becomes your new billing date.",
  tags="plan_upgrade", plan="premium", d="medium")
A("I downgraded from Plus to Basic yesterday. Do I get the rest of this month's fee back?",
  lambda: f"No. A downgrade takes effect on your next billing date and plan fees aren't prorated, so you keep Plus until then. The only exception is the cooling-off rule: if this was your first paid plan charge and it was less than {v('plan_billing.cooling_off_days')} ago, you can get that charge refunded.",
  tags="plan_downgrade plan_fee_cooling_off", plan="plus", d="medium")
A("Signed up for Premium last week and honestly regret it. Can I get my money back??",
  lambda: f"If this was the first paid plan charge on your account, yes. Leaving the paid plan within {v('plan_billing.cooling_off_days')} of that charge gets it refunded in full, and your plan returns to Basic right away. Ask support to cancel under the cooling-off period.",
  tags="plan_fee_cooling_off", plan="premium", d="medium")
A("What happens if I don't have enough money in my account when the plan fee is due?",
  lambda: f"We try again after {v('plan_billing.failed_payment_retry_days')}. If the second try also fails, your plan moves to Basic. Cushion never covers plan fees.",
  tags="plan_fee_failed_payment", d="medium")
A("Can support refund my monthly Plus fee as a one time courtesy thing?",
  lambda: "No. Plan fees aren't eligible for goodwill refunds. The only plan fee refund is the cooling-off rule for the first paid plan charge on an account.",
  tags="plan_fee_not_goodwill", plan="plus", d="medium")
A("what changed in the march 2026 fee update",
  lambda: f"The Plus monthly fee went from {fee('plus', 'monthly_fee_usd', 1)} to {fee('plus', 'monthly_fee_usd', 2)}, the Plus foreign transaction fee went from {fee('plus', 'fx_fee_pct', 1)} to {fee('plus', 'fx_fee_pct', 2)}, the out-of-network ATM fee on Basic and Plus went from {fee('basic', 'out_of_network_atm_fee_usd', 1)} to {fee('basic', 'out_of_network_atm_fee_usd', 2)}, and Plus free out-of-network withdrawals went from {F.n('fees.versions.v1.plus.free_out_of_network_withdrawals_per_month')} to {F.n('fees.versions.v2.plus.free_out_of_network_withdrawals_per_month')} a month.",
  tags="fee_update_2026", d="hard", sup=True, gold="fee-changes-2026 fees-all-v1 fees-all-v2")
A("Did Plus get more expensive? What did it used to cost?",
  lambda: f"Yes. The Plus monthly fee went from {fee('plus', 'monthly_fee_usd', 1)} to {fee('plus', 'monthly_fee_usd', 2)} in the March 2026 fee update.",
  tags="fee_update_2026", plan="plus", d="hard", sup=True,
  gold="fee-changes-2026 fees-all-v1 fees-all-v2 fees-schedule-plus-v1 fees-schedule-plus-v2")
A("How many free out of network ATM withdrawals do I get on Plus?",
  lambda: f"{F.n('fees.versions.v2.plus.free_out_of_network_withdrawals_per_month')} per calendar month. After that each out-of-network withdrawal costs {fee('plus', 'out_of_network_atm_fee_usd')}.",
  plan="plus")
A("Does Premium charge anything to use another bank's ATM?",
  lambda: f"No. Premium has no Tallowbrook fee at out-of-network ATMs, and it reimburses ATM owner surcharges up to {fee('premium', 'atm_surcharge_reimbursement_monthly_cap_usd')} a month.",
  plan="premium")
A("basic plan atm fee if its not a brooklink one?",
  lambda: f"On Basic, each withdrawal at an ATM outside BrookLink costs {fee('basic', 'out_of_network_atm_fee_usd')}. BrookLink ATMs are free.",
  plan="basic")
A("What's the fee to replace my card on Basic, and how much more if I want it fast?",
  lambda: f"A replacement card costs {fee('basic', 'card_replacement_fee_usd')} on Basic. Expedited shipping adds {fee('basic', 'expedited_shipping_fee_usd')} and takes {bd('cards.expedited_delivery_min_business_days', 'cards.expedited_delivery_max_business_days')}.",
  plan="basic", d="medium")
A("im on plus. do i have to pay for a replacement card?",
  lambda: f"No, replacement cards are free on Plus. You only pay if you choose expedited shipping, which is {fee('plus', 'expedited_shipping_fee_usd')}.",
  plan="plus")
A("What's the foreign transaction fee on Plus right now?",
  lambda: f"{fee('plus', 'fx_fee_pct')} of the purchase amount.", plan="plus")
A("I'm going to Spain and I have a Basic account. What will it cost me to pay by card there, and to take out cash?",
  lambda: f"Card purchases in a foreign currency carry a {fee('basic', 'fx_fee_pct')} foreign transaction fee on Basic. ATM withdrawals abroad cost {fee('basic', 'international_atm_fee_usd')} each instead, and the foreign transaction fee doesn't apply to them. The ATM owner may add its own surcharge.",
  tags="atm_abroad", plan="basic", d="hard")
A("Does the foreign transaction fee apply when I take cash out of an ATM abroad?",
  lambda: "No. The foreign transaction fee only applies to purchases. ATM withdrawals outside the US have your plan's international ATM fee instead.",
  tags="atm_abroad fx_fee", d="medium")
A("How much does sending a domestic wire cost on each plan?",
  lambda: f"{fee('basic', 'domestic_wire_out_fee_usd')} on Basic, {fee('plus', 'domestic_wire_out_fee_usd')} on Plus, and free on Premium.",
  d="medium")
A("can i send an international wire from a basic account",
  lambda: f"No, outgoing international wires aren't available on Basic. They're available on Plus for {fee('plus', 'international_wire_out_fee_usd')} and on Premium for {fee('premium', 'international_wire_out_fee_usd')}. You can still receive international wires on Basic.",
  tags="intl_wire_not_on_basic", plan="basic", d="medium")
A("How much is an international wire on Premium and how long does it take to get there?",
  lambda: f"It costs {fee('premium', 'international_wire_out_fee_usd')}. Wires sent before {v('transfers.wires.international_cutoff_time_et')} on a business day go out that day and usually arrive in {bd('transfers.wires.international_arrival_min_business_days', 'transfers.wires.international_arrival_max_business_days')}.",
  plan="premium", d="medium")
A("Do you guys charge for incoming wires?",
  lambda: f"Incoming domestic wires are free on every plan. Incoming international wires cost {fee('basic', 'incoming_international_wire_fee_usd')} on Basic and Plus and are free on Premium.",
  tags="wire_incoming", d="medium")
A("how much does it cost to deposit cash at a store on plus",
  lambda: f"{fee('plus', 'retail_cash_deposit_fee_usd')} per deposit on Plus.", plan="plus")
A("Are there any fees for returned transfers or for not using my account for a while?",
  lambda: f"No. Returned transfers cost {v('fees.all_plans.returned_transfer_fee_usd')} and there are no inactivity fees ({v('fees.all_plans.inactivity_fee_usd')}) on any plan.",
  d="medium")
A("Got charged an ATM fee last week, can I get that refunded?",
  lambda: f"Possibly. Once in any rolling 12-month period, support can refund one fee of up to {v('refunds.goodwill_max_fee_usd')} as a goodwill gesture, and ATM fees qualify. If the fee was charged in error, for example a fee your plan doesn't have, it's always refunded.",
  tags="fee_refund_goodwill fee_refund_error", d="medium")
A("Support already refunded an ATM fee for me back in May. Can they refund another one now?",
  lambda: "Not as a goodwill refund. Support can only give one goodwill fee refund per rolling 12 months. If the new fee was charged in error, it can still be refunded as an error correction.",
  tags="fee_refund_goodwill fee_refund_error", d="hard")
A("I'm on Premium and got charged an out-of-network ATM fee. That's not supposed to happen, right?",
  lambda: f"Right. Premium has no out-of-network ATM fee ({fee('premium', 'out_of_network_atm_fee_usd')}). A fee your plan doesn't have is a fee charged in error, so it's refunded, and it doesn't count as your goodwill refund.",
  tags="fee_refund_error", plan="premium", d="hard")
A("what are the daily card spending limits on each plan",
  lambda: f"{lim('basic', 'daily_card_spend_limit_usd')} on Basic, {lim('plus', 'daily_card_spend_limit_usd')} on Plus and {lim('premium', 'daily_card_spend_limit_usd')} on Premium.",
  d="medium")
A("Thinking about switching from Basic to Plus. What's the monthly cost, and what would my savings rate go up to?",
  lambda: f"Plus costs {fee('plus', 'monthly_fee_usd')} a month. Your Savings Pockets would go from {apy('basic')} APY on Basic to {apy('plus')} APY on Plus.",
  d="hard")

# ---- cards
A("How do I activate my new card?",
  lambda: "In the app, tap Cards, then Activate card. Scan the card or enter the last 4 digits and the expiry date, then set your PIN. It works right away.",
  tags="card_activate")
A("my card never showed up and it's been like 2 weeks",
  lambda: f"If a standard card hasn't arrived {v('cards.standard_delivery_max_business_days')} after it was mailed, message support. We'll cancel the card in transit and send a new one at no cost.",
  tags="card_delivery", d="medium")
A("How long does standard card delivery take?",
  lambda: f"Standard shipping takes {bd('cards.standard_delivery_min_business_days', 'cards.standard_delivery_max_business_days')}.",
  tags="card_delivery")
A("forgot my pin, what do i do",
  lambda: "Use Change PIN in the app under Cards, then Card settings. You don't need the old PIN because you're already signed in.",
  tags="card_pin")
A("whats the difference between freezing my card and reporting it lost",
  lambda: "Freezing is temporary and reversible. It blocks new purchases until you switch it off. Reporting a card lost or stolen cancels it permanently and lets you order a replacement.",
  tags="card_freeze card_lost_stolen", d="medium")
A("If I freeze my card, will my Netflix subscription still go through?",
  lambda: "Yes. Recurring payments to merchants you've already authorized still go through on a frozen card, as do refunds and deposits.",
  tags="card_freeze", d="medium")
A("I lost my card, what do I do?",
  lambda: "Freeze the card in the app right away. If you can't find it, report it lost or stolen in the app, which cancels it permanently. Add a virtual card to keep spending, order a physical replacement, and check your recent transactions for anything you don't recognize.",
  tags="card_lost_stolen")
A("Do I have to call a phone number to report my stolen card?",
  lambda: "No. You can report a lost or stolen card in the app. Freeze it, then tap Report lost or stolen.",
  tags="card_lost_stolen", d="medium")
A("Can I get a virtual card while I'm waiting on my replacement?",
  lambda: "Yes. Any verified member can create a virtual card in the app. It works right away online and in mobile wallets, but not at ATMs.",
  tags="card_virtual")
A("My card chip is cracked. Will my old card stop working as soon as I order a new one?",
  lambda: "No. A damaged card keeps working until you activate the replacement. Activating the new card switches the old one off.",
  tags="card_damaged", d="medium")
A("When do you send a renewal card before mine expires? Do I get a new number?",
  lambda: f"We mail it automatically about {v('cards.renewal_mailed_before_expiry_days')} before your card expires. It has the same card number, with a new expiry date and security code.",
  tags="card_renewal", d="medium")
A("why was my card declined at the gas station",
  lambda: f"Common reasons are not enough money in your account, hitting your daily card spending limit, the card being frozen or not activated, a card control switched off, or a fraud check. If it was a fraud check, confirm the alert in the app and try again.",
  tags="card_declined", d="medium")
A("how do i turn off international purchases on my card",
  lambda: "In the app, tap Cards, choose the card, then Controls, and switch International purchases off. It takes effect straight away.",
  tags="card_controls")
A("App says my card is Locked by Tallowbrook and the freeze toggle doesn't do anything",
  lambda: f"That's a fraud lock, not a freeze. If there's an open fraud alert, answer it. Otherwise message support, who will pass it to the Fraud team. Support can't remove a fraud lock. The Fraud team replies within {vq('support.escalation_response_business_days')}.",
  tags="fraud_lock", d="medium")
A("If I downgrade from Premium, does my metal card stop working?",
  lambda: "No. It keeps working until it expires. After that, or if you need a replacement, you'll get a plastic card.",
  tags="metal_card", plan="premium", d="medium")
A("How fast can I get a replacement card on Premium and what does it cost?",
  lambda: f"On Premium both the replacement and expedited shipping are free. Expedited shipping takes {bd('cards.expedited_delivery_min_business_days', 'cards.expedited_delivery_max_business_days')}, and the replacement is a metal card.",
  plan="premium", d="medium")
A("Can you ship my replacement card to my office instead of my home address?",
  lambda: "No. Cards only go to the residential address on your profile. Update your address first if you've moved.",
  tags="card_delivery", d="medium")
A("I lost my card and I'm on Basic. How much is a replacement and how long with normal shipping?",
  lambda: f"A replacement costs {fee('basic', 'card_replacement_fee_usd')} on Basic, and standard shipping takes {bd('cards.standard_delivery_min_business_days', 'cards.standard_delivery_max_business_days')}.",
  plan="basic", d="medium")

# ---- ATMs and cash
A("where can I find a brooklink atm",
  lambda: "In the app, tap Cash, then ATM map. It shows BrookLink ATMs near you.",
  tags="atm_network")
A("Whats the most I can take out of an ATM in one day on Plus?",
  lambda: f"{lim('plus', 'daily_atm_withdrawal_limit_usd')} a day, across all ATMs combined.", plan="plus")
A("Can support raise my ATM limit just for today? I'm on Basic and need 800 bucks",
  lambda: f"No. The Basic daily ATM limit is {lim('basic', 'daily_atm_withdrawal_limit_usd')}, and support can't raise limits or make exceptions. The only way to get a higher limit is to change plan: Plus allows {lim('plus', 'daily_atm_withdrawal_limit_usd')} and Premium {lim('premium', 'daily_atm_withdrawal_limit_usd')}.",
  tags="limits_fixed", plan="basic", d="hard")
A("ATM ate my money. It didn't give me the cash but I still got charged",
  lambda: f"Once the withdrawal has posted, open a dispute in the app within {v('atm.cash_not_dispensed_dispute_window_days')}: tap the transaction, then Get help, then Cash not dispensed. Most ATM cases are resolved within {v('atm.cash_not_dispensed_resolution_business_days')}, and if the cash wasn't dispensed we credit the full amount, including any ATM fee.",
  tags="atm_cash_not_dispensed", d="medium")
A("Can I deposit cash at a BrookLink ATM?",
  lambda: "No, BrookLink ATMs don't take cash deposits. You can add cash at a participating retail store by showing a barcode from the app.",
  tags="cash_deposit_retail", d="medium")
A("How much cash can I deposit at a store per day on Premium, and is there a fee?",
  lambda: f"Up to {lim('premium', 'daily_retail_cash_deposit_limit_usd')} a day, and there's no fee on Premium.",
  plan="premium", d="medium")
A("The ATM charged me a surcharge AND Tallowbrook charged me a fee. Why am I paying twice?",
  lambda: "They're two different charges. The surcharge is set by the ATM owner and added to the withdrawal. The out-of-network fee is ours and shows as a separate line. BrookLink ATMs have neither.",
  tags="atm_surcharge", d="medium")
A("How much does Premium pay back for ATM surcharges?",
  lambda: f"Up to {fee('premium', 'atm_surcharge_reimbursement_monthly_cap_usd')} per calendar month, credited automatically on the first business day of the next month.",
  tags="atm_surcharge", plan="premium")

# ---- travel
A("Do I need to set a travel notice before going to Mexico?",
  lambda: "No. Travel notices were retired. Keep app notifications on so you can answer fraud alerts, and check that International purchases is switched on.",
  tags="travel_notice")
A("The card machine in Italy asked if I want to pay in USD or EUR. which one?",
  lambda: "Choose the local currency, euros. If you pick US dollars, the merchant converts it at its own rate, which is usually worse.",
  tags="dcc")
A("What exchange rate do you use for purchases in other currencies?",
  lambda: "The card network's exchange rate on the day the transaction settles, with no markup, plus your plan's foreign transaction fee.",
  tags="fx_rate_source", d="medium")
A("why did the amount of my purchase in london change after it posted",
  lambda: "A pending foreign purchase shows an estimate at the rate when you paid. The final amount uses the card network's rate on the day it settles, so it can move slightly.",
  tags="fx_rate_source", d="medium")
A("I'm on Premium and lost my card in Japan. Can you send me a new one there?",
  lambda: f"Yes. Premium members can get an emergency card delivered abroad within {v('cards.premium_emergency_card_abroad_business_days')}, at no cost. Freeze or report the card, then message support with the address where you're staying.",
  tags="emergency_card_abroad", plan="premium", d="medium")
A("Lost my card while traveling in Portugal, I'm on Plus. Can you ship a replacement to my hotel?",
  lambda: "No. Emergency cards abroad are only for Premium members, and standard and expedited shipping only go to US addresses. Add a virtual card in the app and use it in your phone's wallet until you're home.",
  tags="emergency_card_abroad", plan="plus", d="hard")
A("Will my card work in every country?",
  lambda: "It works anywhere Visa is accepted, except in comprehensively sanctioned countries and regions.",
  tags="blocked_regions")
A("I'm on Plus and going to Canada next week. Do I need to tell you I'm traveling, and what's the fee if I buy stuff there?",
  lambda: f"You don't need a travel notice. Purchases in Canadian dollars carry the Plus foreign transaction fee of {fee('plus', 'fx_fee_pct')}.",
  tags="travel_notice", plan="plus", d="hard")

# ---- transfers and deposits
A("What's the cutoff time for ACH transfers?",
  lambda: f"{v('transfers.ach.versions.v2.cutoff_time_et')} on a business day. Transfers submitted later go out the next business day.",
  tags="ach_cutoff")
A("If I send an ACH transfer at 3pm on a Tuesday, when does it go out and when will it arrive?",
  lambda: f"It goes out the same day, because it's before the {v('transfers.ach.versions.v2.cutoff_time_et')} cutoff, and usually arrives in {bd('transfers.ach.arrival_min_business_days', 'transfers.ach.arrival_max_business_days')}.",
  tags="ach_cutoff", d="hard")
A("How do I link my account at another bank?",
  lambda: f"Tap Transfers, then Linked accounts, then Add account. You can link instantly by signing in to your bank, or enter the routing and account number and confirm two small deposits, which arrive in {bd('transfers.ach.external_account_microdeposit_min_business_days', 'transfers.ach.external_account_microdeposit_max_business_days')}.",
  tags="link_external", d="medium")
A("never confirmed the two little test deposits, is the link still good? it's been 3 weeks",
  lambda: f"No. The small deposits must be confirmed within {v('transfers.ach.microdeposit_confirm_within_days')}. After that the link expires and you need to start again.",
  tags="link_external", d="medium")
A("Can I link my wife's bank account so I can send her money?",
  lambda: "No. A linked external account has to be in your name. If she's a Tallowbrook member, you can send her money with BrookPay.",
  tags="link_external", d="hard")
A("What's the BrookPay daily limit on Basic?",
  lambda: f"{lim('basic', 'daily_brookpay_limit_usd')} a day.", plan="basic")
A("I sent money with BrookPay to the wrong person. can you reverse it?",
  lambda: "No. BrookPay payments can't be cancelled or reversed, even by support. Ask the person to send it back. If you think it was a scam, report it to us right away.",
  tags="brookpay", d="medium")
A("Can I cancel a scheduled ACH transfer?",
  lambda: f"Yes, in the app until the ACH cutoff time ({v('transfers.ach.versions.v2.cutoff_time_et')}) on its send date. After that it can't be cancelled.",
  tags="transfer_cancel", d="medium")
A("Just got a new phone and now I can't send more than a few hundred bucks. Why?",
  lambda: f"For {v('limits.new_device_hold_hours')} after you sign in on a new device, BrookPay and external transfers are limited to {v('limits.new_device_transfer_cap_usd')} in total. The limit ends on its own and support can't remove it.",
  tags="new_device", d="medium")
A("What time does a wire have to be in to go out the same day?",
  lambda: f"Domestic wires need to be submitted before {v('transfers.wires.domestic_cutoff_time_et')} on a business day. International wires need to be in before {v('transfers.wires.international_cutoff_time_et')}.",
  d="medium")
A("Can I cancel a wire I sent this morning?",
  lambda: "No, wires can't be cancelled once sent. We can ask the receiving bank to return the money, but the receiving bank decides and it isn't guaranteed.",
  tags="transfer_cancel", d="medium")
A("How do I set up direct deposit?",
  lambda: "Tap Transfers, then Direct deposit. You can set it up automatically by signing in to your payroll provider, or download a pre-filled form or copy your routing and account number for your employer.",
  tags="direct_deposit_setup")
A("How early can I get my paycheck with early direct deposit?",
  lambda: f"Up to {v('transfers.direct_deposit.early_direct_deposit_max_days')} before your official payday, depending on when your employer sends the payroll file.",
  tags="early_direct_deposit")
A("My paycheck came early last time but only on payday this time. Why?",
  lambda: "How early you're paid depends on when your employer sends its payroll file. We make the money available as soon as we receive it, but we can't make the employer send it sooner.",
  tags="early_direct_deposit", d="medium")
A("how much can i deposit by mobile check per day on plus",
  lambda: f"{lim('plus', 'daily_mobile_check_deposit_limit_usd')} a day.", plan="plus")
A("When can I actually use the money from a check I deposited in the app?",
  lambda: f"The first {v('transfers.mobile_check_deposit.next_business_day_available_usd')} is available the next business day and the rest within {v('transfers.mobile_check_deposit.remainder_available_business_days')}. For accounts open less than {v('transfers.mobile_check_deposit.new_account_period_days')}, checks can be held for up to {v('transfers.mobile_check_deposit.new_account_hold_business_days')}.",
  tags="check_deposit_availability", d="medium")
A("what do I write on the back of a check for mobile deposit",
  lambda: f"Sign it and write \"{F.raw('transfers.mobile_check_deposit.endorsement')}\" under your signature.",
  tags="check_deposit_how")
A("Why did my transfer get returned?",
  lambda: f"Common reasons are a wrong account or routing number, a closed or frozen receiving account, or a name that doesn't match. For money pulled in from another bank, that account may not have had enough money. We don't charge for returns ({v('fees.all_plans.returned_transfer_fee_usd')}).",
  tags="transfer_returned", d="medium")
A("Do ACH transfers go through on Saturdays?",
  lambda: "No. ACH transfers only process on business days, so a transfer made on a Saturday is sent the next business day. BrookPay and card payments work every day.",
  tags="bank_holidays")
A("What's the monthly ACH transfer limit on Premium?",
  lambda: f"{lim('premium', 'monthly_ach_out_limit_usd')} per calendar month.", plan="premium")
A("I'm on Basic and need to send 4k to my landlord by ACH today. Is that possible?",
  lambda: f"Not in one day. The Basic daily ACH limit is {lim('basic', 'daily_ach_out_limit_usd')}, and support can't raise it. Plus allows {lim('plus', 'daily_ach_out_limit_usd')} a day and Premium {lim('premium', 'daily_ach_out_limit_usd')}.",
  plan="basic", d="hard")
A("How much can I wire per day domestically on Plus?",
  lambda: f"{lim('plus', 'daily_domestic_wire_limit_usd')} a day.", plan="plus")
A("My recurring transfer didn't go through because I didn't have enough money. Will it retry?",
  lambda: "No. The transfer is skipped and we notify you, with no fee. Future transfers in the series still run.",
  tags="transfer_recurring", d="medium")
A("what info do I give someone so they can send me an international wire",
  lambda: "Give them your routing number, account number, full legal name and address, plus our SWIFT code. All of it is in the app under Account details, then Wire instructions.",
  tags="wire_incoming", d="medium")

# ---- disputes
A("How do I dispute a charge I didn't make?",
  lambda: f"Freeze or report the card first. Then tap the transaction, then Get help, and dispute it as unauthorized. Do it within {v('disputes.unauthorized_window_days')} of the statement it first appeared on.",
  tags="dispute_unauthorized")
A("How long do I have to dispute a charge from last month I don't recognize?",
  lambda: f"{v('disputes.unauthorized_window_days').capitalize()} from the date of the statement it first appeared on. Statements close on the last calendar day of each month.",
  tags="dispute_window_unauthorized", d="medium")
A("I paid for a jacket that never arrived. What's the deadline to dispute it?",
  lambda: f"{v('disputes.versions.v2.merchant_dispute_window_days').capitalize()} from the transaction date. Contact the merchant first and keep a record, then file a merchant dispute in the app.",
  tags="dispute_merchant", d="medium")
A("Item came broken and the seller won't answer my emails. can i get my money back thru you guys?",
  lambda: f"You can file a merchant dispute for an item not as described, within {v('disputes.versions.v2.merchant_dispute_window_days')} of the transaction date. Keep your messages to the seller and photos of the item, and upload them with the dispute.",
  tags="dispute_merchant dispute_documents", d="medium")
A("when will i get my money back after filing a dispute",
  lambda: f"If the dispute passes an initial review, you get a provisional credit within {v('disputes.provisional_credit_business_days')}. Most investigations finish within {v('disputes.investigation_days')}, and some take up to {v('disputes.extended_investigation_days')}.",
  tags="provisional_credit", d="medium")
A("My dispute has been open longer than 45 days. It was for a purchase in Canada. Is that normal?",
  lambda: f"Yes. Disputes about foreign transactions can take up to {v('disputes.extended_investigation_days')} instead of the usual {vq('disputes.investigation_days')}.",
  tags="provisional_credit", d="hard")
A("My dispute got denied. Are you taking the provisional credit back right away?",
  lambda: f"Not right away. We send a written explanation first and take the provisional credit back {v('disputes.denial_reversal_notice_business_days')} after that notice.",
  tags="dispute_denied", d="medium")
A("Can I dispute a transaction that's still pending?",
  lambda: f"No. Wait until it posts. Most pending transactions post or drop off within {v('disputes.pending_dropoff_days')}. If you don't recognize it, freeze your card now.",
  tags="dispute_pending")
A("hotel charged me twice, one of them says pending",
  lambda: f"The pending one is probably a temporary authorization. It usually drops off within {v('disputes.pending_dropoff_days')}. If both charges end up posted, contact the hotel first, and if that doesn't work, dispute the second one as a duplicate.",
  tags="dispute_duplicate", d="medium")
A("I cancelled my gym membership in June but they keep charging me. What can I do?",
  lambda: f"Ask the gym for a refund with your cancellation confirmation. If they refuse, dispute each charge made after you cancelled, within {v('disputes.versions.v2.merchant_dispute_window_days')} of each transaction date. Freezing your card doesn't stop recurring payments you already authorized.",
  tags="dispute_subscription", d="medium")
A("The electric company pulled money from my account that I never authorized. How do I get it back?",
  lambda: f"Report the ACH debit as unauthorized within {v('disputes.ach_unauthorized_window_days')} of the statement it first appeared on: tap it, then Get help, then I didn't authorize this. You can also block future debits from that company. If the claim passes an initial review, you get a provisional credit within {v('disputes.provisional_credit_business_days')}.",
  tags="dispute_ach", d="medium")
A("Can support open a dispute for 7200 dollars for me?",
  lambda: f"Not directly. Support can open disputes totaling up to {v('disputes.support_max_dispute_usd')} per case. Larger cases go to the Disputes team, who reply within {vq('support.escalation_response_business_days')}.",
  tags="dispute_support_limit", d="hard")
A("What should I upload for an item not received dispute?",
  lambda: "Your order confirmation with the expected delivery date, any tracking information, and your messages to the merchant asking about the delivery.",
  tags="dispute_documents")
A("Do I have to freeze my card before disputing a charge I didn't make?",
  lambda: "Yes. The card must be frozen or reported lost or stolen before we open an unauthorized dispute.",
  tags="dispute_unauthorized", d="medium")
A("The merchant refunded me after I opened a dispute. What should I do?",
  lambda: "Withdraw the dispute from the case page so you aren't credited twice. If you were credited twice, we take back the extra amount.",
  tags="dispute_status", d="medium")

# ---- security and fraud
A("Someone called saying they're from Tallowbrook fraud and asked for the code you just texted me. Legit?",
  lambda: "No. Tallowbrook never asks for a one-time code. Hang up and message us from the app. If you already shared the code, change your password right away.",
  tags="never_ask scams")
A("What happens if I don't respond to a fraud alert?",
  lambda: f"If you don't answer within {v('security.fraud_alert_response_hours')}, the card stays locked until you contact us.",
  tags="fraud_alerts")
A("I think someone got into my account. My password was changed and it wasn't me",
  lambda: f"Change your password if you still can, freeze your cards, and message support. Support will freeze your cards and pass the case to the Fraud team, who contact you within {vq('support.escalation_response_business_days')}.",
  tags="account_takeover", d="medium")
A("How long is the one-time sign in code good for?",
  lambda: f"{F.n('security.one_time_code_validity_minutes')} minutes.", tags="two_step")
A("password reset link isn't working",
  lambda: "Reset links expire after an hour. Start again from Forgot password on the sign-in screen, and open the link on the same phone.",
  tags="password_reset", d="medium")
A("Am I on the hook if someone uses my stolen card?",
  lambda: f"No, as long as you report the unauthorized transactions within {v('disputes.unauthorized_window_days')} of the statement they first appear on.",
  tags="zero_liability", d="medium")
A("sent money on BrookPay to a guy selling concert tickets and now he blocked me. can you get it back?",
  lambda: "Report it to us right away. Support will pass it to the Fraud team, who will try to recover the money. BrookPay payments can't be reversed, so we can't promise to get it back.",
  tags="scam_payment", d="medium")
A("Is a text asking me to click a link to confirm a fraud alert real?",
  lambda: "No. Real fraud alert texts only ask you to reply YES or NO and never contain a link. Open the Tallowbrook app yourself instead.",
  tags="fraud_alerts", d="medium")

# ---- overdraft and Cushion
A("Do you charge overdraft fees?",
  lambda: f"No. There are no overdraft fees ({v('overdraft.overdraft_fee_usd')}) on any plan. A transaction you can't cover is declined, with no fee.",
  tags="overdraft_none")
A("How much does Cushion cover on Premium?",
  lambda: f"Up to {v('overdraft.cushion.premium_limit_usd')} on debit card purchases.", plan="premium")
A("I'm on Basic, can I get Cushion?",
  lambda: f"No, Cushion isn't available on Basic. Plus covers up to {v('overdraft.cushion.plus_limit_usd')} and Premium up to {v('overdraft.cushion.premium_limit_usd')}.",
  tags="cushion_basic_not_eligible", plan="basic")
A("what do I need to qualify for Cushion",
  lambda: f"You need to be on Plus or Premium and have at least {v('overdraft.cushion.min_direct_deposit_usd')} in direct deposits in the last {v('overdraft.cushion.direct_deposit_lookback_days')}.",
  tags="cushion_how", d="medium")
A("Does Cushion cover ATM withdrawals?",
  lambda: "No. Cushion only covers debit card purchases. ATM withdrawals, transfers and wires are never covered.",
  tags="cushion_how")
A("What happens if I don't pay back my Cushion balance?",
  lambda: f"If your balance stays negative for {v('overdraft.cushion.repay_within_days')}, Cushion is suspended. After {v('overdraft.cushion.restriction_after_days')}, the account is restricted until it's repaid. There are no fees or interest.",
  tags="cushion_repay", d="medium")
A("My balance went negative after my dispute was denied. I'm on Basic and don't even have Cushion, how is that possible?",
  lambda: f"When a dispute is denied, we take back the provisional credit {v('disputes.denial_reversal_notice_business_days')} after sending notice. If you'd already spent it, the balance can go negative on any plan. Add money to bring it back to zero. If it stays negative for {v('overdraft.cushion.restriction_after_days')}, the account is restricted.",
  tags="negative_balance_dispute", plan="basic", d="hard")

# ---- savings
A("What's the savings APY on Premium?",
  lambda: f"{apy('premium')} APY.", plan="premium")
A("what interest rate does plus get on savings pockets",
  lambda: f"{apy('plus')} APY.", plan="plus")
A("My Basic savings rate used to be 0.75%. What is it now and when did it change?",
  lambda: f"It's {apy('basic')} APY. It went down from {apy('basic', 1)} in the June 2026 savings rate change.",
  tags="savings_rate_change_2026", plan="basic", d="hard", sup=True,
  gold="savings-rate-change-2026 savings-apy-basic-v1 savings-apy-basic-v2")
A("Is there a cap on how much money earns the savings rate?",
  lambda: f"Yes. Your plan's APY applies to combined pocket balances up to {v('savings.apy_balance_cap_usd')}. Anything above that earns {v('savings.above_cap_apy_pct', decimals=2)} APY.",
  d="medium")
A("How often is savings interest paid?",
  lambda: "Interest is calculated and compounded daily and paid monthly, on the first business day of the next month.",
  tags="savings_interest_calc")
A("How many savings pockets can I have?",
  lambda: f"Up to {F.n('savings.max_pockets')}.", tags="savings_pockets")
A("are there limits on how often I can take money out of savings",
  lambda: "No. Moving money between your pockets and your main balance is instant and free, with no limit on withdrawals.",
  tags="savings_pockets")
A("how do round ups work",
  lambda: "Each card purchase is rounded up to the next dollar and the difference is moved to a Savings Pocket you choose, once a day. Turn it on under Savings for a pocket.",
  tags="round_ups")
A("Will I get a tax form for my savings interest? I only earned a few dollars last year",
  lambda: f"We only send a Form {F.raw('savings.tax_form')} if you earned at least {v('savings.tax_form_threshold_usd')} in interest in the year. The interest still appears on your monthly statements, and you may still need to report it.",
  tags="tax_forms", d="medium")
A("Do you give notice before lowering savings rates?",
  lambda: f"Yes, at least {v('savings.rate_change_notice_days')} by email before a rate decrease.",
  tags="savings_apy")

# ---- closing
A("How do I close my account?",
  lambda: "Tap Profile, then Close account. Nothing can be pending, no disputes can be open and your balance can't be negative. The app shows where any remaining money will go and asks you to confirm.",
  tags="close_how")
A("I want to close my account but I have an open dispute. Can I?",
  lambda: "No. An account can't be closed while a dispute is open. Wait for it to finish, or withdraw it if the merchant already refunded you.",
  tags="close_requirements", d="medium")
A("If I close my account how do I get my remaining balance?",
  lambda: f"We send it to your linked external account by ACH, or by check to your address if no account is linked, within {v('closure.payout_business_days')} of closing.",
  tags="close_payout", d="medium")
A("Can I reopen my closed account?",
  lambda: "No. Closed accounts can't be reopened, but you can apply for a new account.",
  tags="close_reopen")
A("Will you close my account if I don't use it?",
  lambda: f"We may close accounts with a {v('closure.inactivity_balance_usd')} balance and no activity for 12 months, after emailing at least {v('closure.inactivity_notice_days')} before. We don't charge inactivity fees.",
  tags="close_inactivity", d="medium")
A("after i close, can I still get my old statements?",
  lambda: f"You can sign in and download them for {v('closure.statements_after_closure_days')} after closing. After that, message support to request them.",
  tags="close_statements", d="medium")
A("I'm closing my Plus account mid-month. Do I get part of the monthly fee back?",
  lambda: "No. Plan fees aren't refunded or prorated when you close, unless the cooling-off rule for your first paid plan charge applies.",
  tags="closure_no_plan_refund", plan="plus", d="medium")

# ---- support
A("What are your phone support hours?",
  lambda: f"Phone support is open from {v('support.phone_open_time_et')} to {v('support.phone_close_time_et')} every day. Chat in the app is open around the clock.",
  tags="support_hours")
A("How long does it take to get a response to a formal complaint?",
  lambda: f"We acknowledge it within {v('support.complaint_ack_business_days')} and send a final response within {v('support.complaint_final_response_business_days')}.",
  tags="complaints", d="medium")
A("Can my husband call in and ask about my account for me?",
  lambda: "No. Support only talks to the account holder in their own signed-in session. He can sit with you while you message us.",
  tags="third_party", d="medium")
A("I have power of attorney for my mom. How do I get access to her account?",
  lambda: f"Send the power of attorney document, your photo ID and your contact details to our Account Services team by email or through the account holder's session. Support can't review the documents. Account Services replies within {vq('support.escalation_response_business_days')}.",
  tags="power_of_attorney", d="medium")
A("My father passed away and he had a Tallowbrook account. What do I do?",
  lambda: "We're sorry for your loss. Email us with the subject Bereavement or message support. Account Services handles it and will ask for the death certificate, proof you're the executor or next of kin, and your photo ID.",
  tags="deceased", d="medium")
A("A store said they refunded me 3 days ago, when will it show up?",
  lambda: f"Merchant refunds usually show within {bd('refunds.merchant_refund_min_business_days', 'refunds.merchant_refund_max_business_days')} of the merchant issuing them.",
  tags="merchant_refund_timing")
A("Can your support team help me decide how to invest my money?",
  lambda: "No. Support can explain how Tallowbrook products work, but can't give financial, investment, tax or legal advice.",
  tags="no_advice")
A("What can support do for me directly, and what gets sent to another team?",
  lambda: f"Support can freeze or unfreeze cards, report lost or stolen cards and order replacements, open disputes up to {v('disputes.support_max_dispute_usd')}, refund fees under the refund policy, change plans, close eligible accounts and explain policies. Account takeovers, fraud locks and scams go to Fraud, larger disputes to Disputes, identity and restrictions to Verification, power of attorney, bereavement and financial abuse to Account Services, and formal complaints to Complaints.",
  tags="support_scope", d="medium")
A("Someone is pressuring me to hand over access to my money. Who can help?",
  lambda: "Message support in the app when it's safe. We'll pass your case to our Account Services team, who can help secure your account. If you're in immediate danger, call emergency services first.",
  tags="financial_abuse", d="medium")

# ---- account opening
A("what do I need to open an account",
  lambda: f"You need to be at least {F.n('verification.min_age_years')}, live in the US with a residential address, and have a Social Security number or ITIN, an unexpired government photo ID and a smartphone. You'll take a photo of your ID and a selfie in the app.",
  tags="eligibility kyc_documents")
A("Can I open an account with a PO box address?",
  lambda: "No. You need a US residential address, not a PO box.",
  tags="eligibility")
A("How long does identity verification take?",
  lambda: f"Usually within {v('verification.typical_business_days')}. If it needs a manual review, up to {v('verification.manual_review_max_business_days')}.",
  tags="verification_time")
A("I failed verification twice. How many more tries do I get?",
  lambda: f"You can try up to {F.n('verification.max_attempts')} times in total, so one more. After that the application is closed and you can apply again after {v('verification.reapply_after_days')}.",
  tags="verification_failed", d="hard")
A("My account is under review. Can I still get my paycheck deposited?",
  lambda: "Yes, you can receive deposits while under review. You may not be able to send money, withdraw cash or use your card until the review ends, and support can't lift it.",
  tags="account_under_review", d="medium")
A("I just got married and changed my last name. How do I update it on my account?",
  lambda: f"Message support and ask to change your legal name. Support passes it to the Verification team, who need an ID in your new name or a legal document such as a marriage certificate. It takes up to {v('verification.name_change_business_days')} after they get the documents.",
  tags="name_change", d="medium")
A("What counts as proof of address and how recent does it have to be?",
  lambda: f"A utility bill, lease or mortgage statement, bank or credit card statement, or government letter showing your name and address, dated within the last {v('verification.proof_of_address_max_age_days')}.",
  tags="address_change")
A("lost access to my old phone number, how do i change it on my account",
  lambda: "Message support. Because this is a common fraud route, the request goes to the Verification team, who may ask for a new selfie.",
  tags="contact_change", d="medium")

# ---- statements, documents, products
A("When is my monthly statement ready?",
  lambda: f"Within {v('statements.available_business_days_after_month_end')} after the end of each month.",
  tags="statements")
A("Can I get a letter proving I have an account, for my landlord?",
  lambda: "Yes. Download an account verification letter in the app under Profile, then Documents. It's ready right away.",
  tags="verification_letter")
A("Can I open a joint account with my partner?",
  lambda: "No. Tallowbrook only offers individual accounts, with no joint accounts or authorized users. You can each open an account and send money to each other with BrookPay.",
  tags="joint_accounts")
A("Do you have business checking?",
  lambda: "No. Tallowbrook only offers personal accounts.",
  tags="business_accounts")
A("Can I buy bitcoin in the Tallowbrook app?",
  lambda: "No. You can't buy, sell or hold cryptocurrency with Tallowbrook, and there are no investing accounts.",
  tags="crypto_investing")
A("can my 16 year old get a card on my account",
  lambda: "No. Accounts are for people 18 and over, and you can't add a child as an authorized user or give them a card on your account.",
  tags="under_18")
A("how does the refer a friend bonus work",
  lambda: f"You and your friend each get {v('referrals.bonus_usd')} when they sign up with your link and receive at least {v('referrals.friend_min_direct_deposit_usd')} in direct deposits within {v('referrals.qualify_within_days')} of opening. Bonuses are paid within {v('referrals.payout_business_days')} of qualifying.",
  tags="referrals", d="medium")
A("hey so if i report my card stolen and then find it in my couch can i just keep using it",
  lambda: "No. Reporting a card lost or stolen cancels it permanently, even if you find it later. If you're not sure it's gone, freeze it instead, since a freeze can be undone.",
  tags="card_lost_stolen card_freeze", d="medium")
A("Do I get a new card number when you replace my damaged card, and do I have to update my subscriptions?",
  lambda: "A damaged-card replacement keeps the same card number, with a new expiry date and security code. Merchants that store your card may need the new expiry date.",
  tags="card_damaged", d="medium")


# =====================================================================
# Unanswerable (40)
# =====================================================================

NO_INFO = "I can't answer that from the Tallowbrook help center."

# ---- out of scope (14)
U("dev", "out_of_scope", "What's the weather going to be like in Denver tomorrow?",
  lambda: f"{NO_INFO} I can help with questions about your Tallowbrook account.", d="easy")
U("dev", "out_of_scope", "Can you help me dispute a charge on my Chase credit card?",
  lambda: "I can only help with Tallowbrook accounts. Contact the bank that issued that card about its dispute process.", d="medium")
U("dev", "out_of_scope", "Should I pay off my car loan early or put the money into savings?",
  lambda: "Tallowbrook support can't give financial advice. A qualified financial adviser can help you weigh that decision.", d="medium")
U("dev", "out_of_scope", "write me a short poem about my bank balance lol",
  lambda: f"{NO_INFO} I can help with questions about your Tallowbrook account.", d="easy")
U("test", "out_of_scope", "What stock should I buy with my tax refund?",
  lambda: "Tallowbrook support can't give investment advice. Talk to a qualified financial adviser.", d="easy")
U("test", "out_of_scope", "How do I file taxes on my freelance income?",
  lambda: "Tallowbrook support can't give tax advice. A tax professional can help with that.", d="medium")
U("test", "out_of_scope", "What's the current federal funds rate?",
  lambda: f"{NO_INFO} It doesn't cover general interest rate news.", d="medium")
U("test", "out_of_scope", "is bitcoin a good investment right now",
  lambda: "Tallowbrook support can't give investment advice about cryptocurrency or anything else.", d="easy")
U("test", "out_of_scope", "What's the best travel rewards credit card?",
  lambda: "Tallowbrook doesn't compare or recommend other companies' products and can't give financial advice.", d="medium")
U("test", "out_of_scope", "Can you translate a letter from my landlord into Spanish for me?",
  lambda: f"{NO_INFO} I can help with questions about your Tallowbrook account.", d="easy")
U("test", "out_of_scope", "how do I raise my credit score fast",
  lambda: "Tallowbrook support can't give credit or financial advice. A nonprofit credit counselor or financial adviser can help.", d="medium")
U("test", "out_of_scope", "What time does the DMV open on Saturdays?",
  lambda: f"{NO_INFO} Check with your local DMV.", d="easy")
U("test", "out_of_scope", "Can you recommend a good tax accountant near me?",
  lambda: "Tallowbrook can't recommend tax professionals or give tax advice.", d="easy")
U("test", "out_of_scope", "Is it smarter to rent or buy a house in this market?",
  lambda: "Tallowbrook support can't give financial advice. A qualified financial adviser can help with that decision.", d="medium")

# ---- false premise (13)
U("dev", "false_premise", "What's the overdraft fee on the Basic plan?",
  lambda: f"Tallowbrook doesn't charge overdraft fees on any plan ({v('overdraft.overdraft_fee_usd')}). A transaction you can't cover is declined, with no fee.",
  tags="overdraft_none", plan="basic", d="easy")
U("dev", "false_premise", "How many days before my trip do I need to set a travel notice?",
  lambda: "You don't. Travel notices and the Travel mode setting were retired. Keep app notifications on so you can confirm fraud alerts quickly.",
  tags="travel_notice", d="medium")
U("dev", "false_premise", "How do I add my wife as an authorized user on my card?",
  lambda: "Tallowbrook doesn't offer authorized users or joint accounts. She can open her own account, and you can send each other money with BrookPay.",
  tags="joint_accounts", d="medium")
U("test", "false_premise", "What's the interest rate on Tallowbrook personal loans?",
  lambda: "Tallowbrook doesn't offer personal loans, credit cards or lines of credit.",
  tags="credit_products", d="easy")
U("test", "false_premise", "How much does it cost to send an international wire on the Basic plan?",
  lambda: f"Outgoing international wires aren't available on Basic. They're available on Plus for {fee('plus', 'international_wire_out_fee_usd')} and on Premium for {fee('premium', 'international_wire_out_fee_usd')}.",
  tags="intl_wire_not_on_basic", plan="basic", d="hard")
U("test", "false_premise", "whats the cushion limit on basic",
  lambda: f"Cushion isn't available on Basic. It covers up to {v('overdraft.cushion.plus_limit_usd')} on Plus and {v('overdraft.cushion.premium_limit_usd')} on Premium.",
  tags="cushion_basic_not_eligible", plan="basic", d="medium")
U("test", "false_premise", "Why did you charge me an inactivity fee?",
  lambda: f"Tallowbrook doesn't charge inactivity fees ({v('fees.all_plans.inactivity_fee_usd')}). Tap the fee in your activity feed to see what it actually is, and message support if it looks wrong.",
  tags="close_inactivity", d="medium")
U("test", "false_premise", "How do I order the metal card on the Plus plan?",
  lambda: "Metal cards are only for Premium members. Plus cards are plastic. You'd need to upgrade to Premium to get one.",
  tags="metal_card", plan="plus", d="medium")
U("test", "false_premise", "What's the BrookPay fee when I send more than $1,000?",
  lambda: f"BrookPay has no fee ({v('transfers.brookpay.fee_usd')}) on any plan and any amount. Your daily BrookPay limit depends on your plan.",
  tags="brookpay", d="medium")
U("test", "false_premise", "My order that never arrived is 100 days old, so I'm past the 90-day dispute window, right?",
  lambda: f"No. The merchant dispute window is {v('disputes.versions.v2.merchant_dispute_window_days')} from the transaction date, so you can still file. Contact the merchant first and keep a record.",
  tags="dispute_window_merchant", d="hard")
U("test", "false_premise", "Plus charges 1.5% on foreign purchases, right? Just checking before my trip.",
  lambda: f"Not anymore. The Plus foreign transaction fee is {fee('plus', 'fx_fee_pct')}.",
  plan="plus", d="hard")
U("test", "false_premise", "Can I get my paycheck 5 days early with early direct deposit?",
  lambda: f"No. Early direct deposit can pay you up to {v('transfers.direct_deposit.early_direct_deposit_max_days')} early, depending on when your employer sends the payroll file.",
  tags="early_direct_deposit", d="medium")
U("test", "false_premise", "How do I move money from my Tallowbrook credit card to my checking?",
  lambda: "Tallowbrook doesn't offer credit cards. Your Tallowbrook card is a debit card that spends from your main balance.",
  tags="credit_products", d="easy")

# ---- near miss (13)
U("dev", "near_miss", "What's the maximum for a contactless payment without entering a PIN?",
  lambda: "The help center doesn't give a contactless amount limit. It only says you can switch contactless on or off under card controls, and your plan's daily card spending limit applies.",
  d="medium")
U("dev", "near_miss", "What is Tallowbrook's SWIFT code?",
  lambda: "The help center doesn't list the SWIFT code. It says the code is shown in the app under Account details, then Wire instructions.",
  d="medium")
U("dev", "near_miss", "What's today's exchange rate for euros?",
  lambda: "The help center doesn't give live exchange rates. Purchases are converted at the card network's rate on the day they settle.",
  d="medium")
U("test", "near_miss", "Which stores can I deposit cash at?",
  lambda: "The help center doesn't list the stores. It says participating retail stores accept cash deposits using a barcode from the app.",
  d="medium")
U("test", "near_miss", "How many BrookLink ATMs are there in Ohio?",
  lambda: "The help center doesn't give ATM counts. The ATM map in the app shows BrookLink ATMs near you.",
  d="medium")
U("test", "near_miss", "How long does Account Services take to approve a power of attorney?",
  lambda: f"The help center doesn't give an approval time. It only says Account Services replies within {vq('support.escalation_response_business_days')} to confirm what they received and may ask for more information.",
  d="hard")
U("test", "near_miss", "Can I pick a custom design for my Premium metal card?",
  lambda: "The help center doesn't mention card designs. It only covers getting and replacing the metal card.",
  plan="premium", d="medium")
U("test", "near_miss", "Is there a cap on how many friends I can refer for the bonus?",
  lambda: "The help center doesn't mention a cap on referrals. It covers the bonus amount and how a friend qualifies.",
  d="medium")
U("test", "near_miss", "What was the Premium savings APY back in 2024?",
  lambda: "The help center doesn't cover rates before July 2025. It lists the rate in effect from July 2025 and the current rate.",
  plan="premium", d="hard")
U("test", "near_miss", "can i set a spending limit just for restaurants",
  lambda: "The help center doesn't describe category spending limits. Card controls only switch online, international, ATM and contactless use on or off.",
  d="medium")
U("test", "near_miss", "How long does it take for a deceased member's balance to be paid to the estate?",
  lambda: "The help center doesn't give a payout time. It says Account Services pays the balance to the estate after confirming who's authorized.",
  d="hard")
U("test", "near_miss", "What's the direct phone number for the Fraud team?",
  lambda: "The help center doesn't give a separate Fraud team number. Support passes fraud cases to the Fraud team, and you reach support in the app or on the main support line.",
  d="medium")
U("test", "near_miss", "Can I get my statements in Spanish?",
  lambda: "The help center doesn't say whether statements are available in other languages.",
  d="medium")


# =====================================================================
# Build
# =====================================================================

IGNORE_KEY_SUFFIXES = ("effective_date",)


def main() -> None:
    articles = build_corpus()
    superseded = {a["supersedes"] for a in articles if a["supersedes"]}
    key_index: dict[str, set[str]] = {}
    for a in articles:
        for k in a["_keys"]:
            key_index.setdefault(k, set()).add(a["article_id"])

    plans_of = {a["article_id"]: set(a["plans"]) for a in articles}

    def gold_for(keys: set[str], sup: bool, plan: str | None) -> list[str]:
        """Articles covering a listed topic tag, plus articles stating a fact the
        answer uses. When the question is about one plan, fact matches from
        another plan's variant are dropped (tag matches are kept)."""
        ids: set[str] = set()
        for k in keys:
            if k.endswith(IGNORE_KEY_SUFFIXES):
                continue
            hits = key_index.get(k, set())
            if plan and not k.startswith("tag:"):
                hits = {a for a in hits if plan in plans_of[a]}
            ids |= hits
        if not sup:
            ids -= superseded
        return sorted(ids)

    rows = []
    # dev gets every 4th answerable question, so topics spread over both splits
    for i, item in enumerate(ANSWERABLE):
        split = "dev" if i % 4 == 0 else "test"
        with F.recording() as keys:
            answer = item["ans"]()
        keys |= {f"tag:{t}" for t in item["tags"]}
        gold = sorted(item["gold"]) or gold_for(keys, item["sup"], item["plan"])
        if not gold:
            raise SystemExit(f"no gold articles for answerable question: {item['q']}")
        rows.append((split, item, answer, gold, True, None))
    for item in UNANSWERABLE:
        with F.recording() as keys:
            answer = item["ans"]()
        keys |= {f"tag:{t}" for t in item["tags"]}
        gold = gold_for(keys, False, item["plan"]) if item["kind"] == "false_premise" else []
        if item["kind"] == "false_premise" and not gold:
            raise SystemExit(f"false premise question needs refuting articles: {item['q']}")
        rows.append((item["split"], item, answer, gold, False, item["kind"]))

    out = []
    counters = {"dev": 0, "test": 0}
    for split in ("dev", "test"):
        for s, item, answer, gold, answerable, kind in rows:
            if s != split:
                continue
            counters[split] += 1
            out.append(
                dict(
                    question_id=f"q-{split}-{counters[split]:03d}",
                    split=split,
                    question=item["q"],
                    gold_article_ids=gold,
                    reference_answer=answer,
                    answerable=answerable,
                    unanswerable_type=kind,
                    plan=item["plan"],
                    difficulty=item["d"],
                )
            )

    rag = ROOT / "rag"
    with open(rag / "questions.jsonl", "w") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    for split in ("dev", "test"):
        with open(rag / f"questions_{split}.jsonl", "w") as fh:
            for row in out:
                if row["split"] == split:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_ans = sum(r["answerable"] for r in out)
    print(f"wrote {len(out)} questions ({counters['dev']} dev, {counters['test']} test, "
          f"{len(out) - n_ans} unanswerable) to rag/")


if __name__ == "__main__":
    main()
