---
license: cc-by-4.0
language:
- en
pretty_name: Tallowbrook Neobank Support (synthetic)
size_categories:
- n<1K
task_categories:
- question-answering
- text-retrieval
- text-generation
tags:
- synthetic
- banking
- customer-support
- rag
- agents
configs:
- config_name: corpus
  data_files:
  - split: train
    path: corpus/articles.jsonl
- config_name: rag_questions
  data_files:
  - split: dev
    path: rag/questions_dev.jsonl
  - split: test
    path: rag/questions_test.jsonl
- config_name: agent_tasks
  data_files:
  - split: test
    path: agents/tasks.jsonl
---

# Tallowbrook Neobank Support (synthetic)

A small, fully synthetic customer-support dataset for a fictional US neobank called Tallowbrook. It has a help-center corpus, retrieval questions with gold articles, and support tickets for testing agents against a fake bank backend. It was built as shared data for a set of portfolio projects on RAG, multi-agent triage and LLM guardrails.

## Read this first

- **Everything here is synthetic.** Tallowbrook is not a real bank. A web search in September 2026 found no bank, credit union or fintech with that name. Its policies, fees and products are invented.
- **Written by an AI.** Claude (Anthropic) wrote the facts file, the article text, the questions, the reference answers and the tickets. Numbers in the text come from one structured facts file through code, so they agree with each other. The judgments (which article answers a question, what the right support action is) are Claude's.
- **No human labels, by design.** Nobody labels or audits this data by hand. Gold labels are checked by code instead: `scripts/validate.py` checks every question and task against the facts file, the agent tasks' final states are computed by `scripts/bank_sim.py`, an independent reimplementation of the bank in support-triage-agents reproduces all 50 gold states, and that repo cross-checks the tasks' gold actions with a second model. Treat the labels as machine-checked, not human-validated.
- **Identifiers are fake.** Emails use `example.com`, phone numbers use the 555-01xx range reserved for fiction, towns are made up, and ZIP codes are placeholders. Card numbers are fake Luhn-valid numbers in the `411111` Visa test range. There are no routing numbers, full account numbers or Social Security numbers. Customer names are invented, and any match with a real person is a coincidence.

## Files

| Path | What it is | Rows |
|---|---|---|
| `facts/policies.yaml` | The single source of truth: plans, fees, limits, windows, rules | 1 file |
| `corpus/articles.jsonl` | Help-center articles | 151 |
| `rag/questions.jsonl` | All RAG questions (split in a column) | 200 |
| `rag/questions_dev.jsonl`, `rag/questions_test.jsonl` | The same questions, one file per split | 50 and 150 |
| `agents/bank_seed.json` | The fake bank: customers, accounts, cards, transactions, disputes | 30 customers |
| `agents/tasks.jsonl` | Support tickets with gold actions and gold final state | 50 |
| `scripts/` | Builders, the bank model and the validator | |
| `tests/` | pytest tests | |

### `facts/policies.yaml`

Three plans (Basic, Plus, Premium) with their fees and limits, plus card replacement, ATM and foreign transaction fees, transfer limits and cutoff times, dispute windows, lost and stolen card steps, identity verification, Cushion (fee-free overdraft coverage on paid plans), savings rates, account closure, travel, fraud handling, refunds, and the support escalation rules: what support may do, what it must escalate and what it must refuse.

Key suffixes carry units (`_usd`, `_pct`, `_days`, `_business_days`, `_hours`, `_time_et`). Six policies are versioned, and each keeps the old version with the date it was superseded:

| Policy | Old version | New version | What changed |
|---|---|---|---|
| Fee schedule | 2025-01-01 | 2026-03-01 | Plus fee, Plus foreign fee, out-of-network ATM fee, Plus free withdrawals |
| Savings rates | 2025-07-01 | 2026-06-01 | All three plan rates went down |
| Merchant dispute window | 2024-06-01 | 2026-01-15 | Longer window |
| ACH cutoff | 2024-01-01 | 2026-05-01 | Later cutoff |
| Travel notices | 2024-03-01 | 2025-09-01 | No longer needed |
| Lost or stolen card | 2024-01-01 | 2025-11-01 | Phone line replaced by in-app steps |

### `corpus/articles.jsonl`

| Field | Meaning |
|---|---|
| `article_id` | Stable ID. Versioned articles end in `-v1` or `-v2`. |
| `title`, `section` | Help-center title and one of 14 sections |
| `plans` | Plans the article applies to |
| `effective_date`, `version` | When this version took effect |
| `supersedes` | The `article_id` this version replaces, or null |
| `body` | Markdown, 150 to 230 words |

An article is superseded if another article's `supersedes` points to it. Superseded articles keep their original wording and don't say they are out of date, the way an unmaintained help center would look.

### `rag/questions.jsonl`

| Field | Meaning |
|---|---|
| `question_id` | `q-dev-NNN` or `q-test-NNN` |
| `split` | `dev` (50) or `test` (150) |
| `question` | Customer wording, with typos, casual tone and some multi-part questions |
| `gold_article_ids` | Current articles that contain what the answer needs (see below) |
| `reference_answer` | A short correct answer, or the right way to decline |
| `answerable` | False for 40 questions |
| `unanswerable_type` | `out_of_scope`, `false_premise`, `near_miss`, or null |
| `plan` | The plan the question is about, when the answer depends on it. The plan is also named in the question text, so a system that ignores this field is not penalized. |
| `difficulty` | `easy` (70), `medium` (107) or `hard` (23) |

Unanswerable questions by split:

| Type | dev | test | Expected behavior |
|---|---|---|---|
| `out_of_scope` | 4 | 10 | Decline. Not about Tallowbrook, or asks for financial advice. Gold is empty. |
| `false_premise` | 3 | 10 | Correct the premise (for example, there is no overdraft fee). Gold lists the articles that refute it. |
| `near_miss` | 3 | 10 | Say the help center doesn't cover it. A related article exists but lacks the specific fact. Gold is empty. |

**How gold IDs were chosen.** Each reference answer pulls its numbers from the facts file, and each question lists the topics it needs. Gold is every current article that states one of those facts or covers one of those topics. So gold is inclusive: a question about the Plus foreign transaction fee has seven gold articles, because seven current articles state that fee. For plan-specific questions, another plan's variant is not gold. Superseded articles are gold only for the 3 questions that ask about a change. Answerable questions have 1 to 10 gold articles (74 have exactly one, mean 2.7). If you report recall@k, report it next to a hit-rate or MRR number, since inclusive gold lowers recall@k.

### `agents/bank_seed.json` and `agents/tasks.jsonl`

The seed is a snapshot as of 2026-09-15: 30 customers, 33 accounts, 33 cards, 196 recent transactions (a mix of hand-written scenario transactions and seeded random filler), 2 disputes and an empty escalations table. Balances are snapshots, so recent transactions don't sum to them.

| Task field | Meaning |
|---|---|
| `task_id`, `customer_id` | The ticket and the signed-in customer it comes from |
| `ticket_text` | What the customer wrote |
| `hidden_facts` | What a simulated customer knows but didn't say, to reveal only when asked |
| `needs_clarification` | True for 10 of 50 tasks, where acting without asking would be a guess |
| `should_escalate` | True for 10 tasks, per the escalation rules |
| `gold_actions` | The expected write actions with arguments. Empty for the 13 tasks where the right move is to explain, decline or do nothing. |
| `gold_final_state` | The exact change to the seed as `updated` (with `from` and `to` values) and `inserted` (fields to match) |
| `policy_refs` | Current article IDs that justify the resolution |
| `rationale` | One line on why, for reviewers |

Actions are `freeze_card`, `unfreeze_card`, `report_card_lost_stolen`, `order_replacement_card`, `open_dispute`, `refund_fee`, `change_plan`, `close_account` and `escalate_to_human`. `scripts/bank_sim.py` defines what each one does to the state and which support rules it enforces (dispute windows, the support dispute limit, one goodwill fee refund per rolling 12 months, the cooling-off period, closure requirements and so on). Each task's `gold_final_state` was computed by running its `gold_actions` through that model, not typed by hand. Inserted records match on content only. IDs, dates and free-text descriptions are left to the implementation.

The tasks include cases that test reading the right policy version (a purchase 102 days old is inside the current merchant window but was outside the old one), plan-specific fees, fees charged in error versus goodwill refunds, requests from people who aren't the account holder, and requests support must refuse.

## Distractor design

- **Per-plan variants.** 37 articles apply to a single plan and differ from their siblings mostly in the numbers, for example "ATM fees on Basic" and "ATM fees on Plus".
- **Superseded versions.** 14 articles are older versions with the same title as their replacement and different numbers or rules.
- **Overlapping summaries.** Plan overviews, the all-plans fee table, per-plan fee schedules and topic articles restate the same facts in different shapes.
- **Near-miss topics.** Articles on products Tallowbrook doesn't offer (joint accounts, credit, crypto, checkbooks) and on neighboring topics give near-miss and false-premise questions something plausible to retrieve.

## Checks

`make validate` runs `scripts/validate.py`, which checks among other things that every money amount, percentage, day count, hour count and clock time in an article or reference answer appears in the facts file with the same unit, that superseded articles point to real, older articles, that split sizes and unanswerable counts are exact, that every gold article exists, that no question is repeated within or across splits, and that every task's `gold_final_state` applies cleanly to the seed and is reproduced by its `gold_actions` without breaking policy.

`make test` runs the validator plus tests showing that it catches wrong numbers and broken diffs, that the bank model rejects 17 plausible wrong moves, and that rebuilding from the facts gives byte-identical files.

```
uv sync
make build      # regenerate data from facts/policies.yaml
make validate
make test
```

## Known limitations

- **One author voice.** Articles, questions and tickets were all written by the same model. Customer phrasing varies (typos, slang, multi-part questions) but is still synthetic and cleaner than real support traffic.
- **US-centric.** US dollars, ACH, US-style disputes and US addresses only.
- **Short, templated articles.** Articles run 150 to 230 words and per-plan variants share templates, so they are more regular than a real help center.
- **Simplified policy.** The rules are plausible but invented, and they are not legal or regulatory guidance.
- **Inclusive gold.** See above. Some gold articles mention a needed fact in passing.
- **No human audit.** Gold article IDs, unanswerable labels and agent resolutions reflect one model's reading of the facts file, checked by code and a second model but never by a person.
- **No attack suite in this release.** A set of bank-specific prompt-injection attacks and poisoned articles for guardrail testing was planned but is not included.

## License

CC-BY-4.0.
