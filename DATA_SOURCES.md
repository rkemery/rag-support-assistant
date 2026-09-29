# Data and model sources

Everything this repo downloads or vendors, pinned, with its license. Licenses were checked on 2026-09-28 from the source named in each row.

## Datasets

| Name | Where it lives here | Revision | License | Source | How the license was checked |
|---|---|---|---|---|---|
| Tallowbrook Neobank Support (synthetic) | `data/tallowbrook/` (corpus, facts, RAG questions) | tag [`tallowbrook-v0.1`](https://github.com/rkemery/rag-support-assistant/tree/tallowbrook-v0.1) (commit `3c72058e8cc7d5dd6e224ecdec7b814341a3d48f`) | CC-BY-4.0 | The portfolio's shared dataset, built on the [`claude/tallowbrook-dataset`](https://github.com/rkemery/rag-support-assistant/tree/claude/tallowbrook-dataset) branch of this repo. That branch is kept after merges so the pin stays reachable. Its canonical home will be a Hugging Face dataset. | Dataset card in the source repo. `scripts/sync_data.py` checks every file's sha256. |
| RAGTruth, QA task, test split, 200-response seeded subset | `data/ragtruth/` | `c103204b9ce28d6bbad859304bf30de72b8ed8fe` | MIT | https://github.com/ParticleMedia/RAGTruth | The `LICENSE` file at that commit (copied to `data/ragtruth/LICENSE`). Source file hashes are in `data/ragtruth/MANIFEST.json`. |
| Judge-validation perturbations | `data/judge_validation/` | built by `eval build-validation` from the Tallowbrook files above | CC-BY-4.0 (derived from Tallowbrook) | This repo | Derived data, same license as its source. |

## Models (Hugging Face, all run on CPU)

| Model | Role | Revision | License (Hub tag) |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | dense retrieval baseline, snapshot embeddings | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | MIT |
| `ibm-granite/granite-embedding-small-english-r2` | dense retrieval arm | `2ab6fa8ea2d674564defd37171ae19079b864b33` | Apache-2.0 |
| `ibm-granite/granite-embedding-reranker-english-r2` | reranker on the top 10 chunks | `d09d3d6971b689bf9c23839e45a470874d46e13a` | Apache-2.0 |
| `Qwen/Qwen3-Embedding-0.6B` | strong-arm embedding, offline benchmark only | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | Apache-2.0 |
| `Qwen/Qwen3-Reranker-0.6B` | strong-arm reranker, offline benchmark only | `e61197ed45024b0ed8a2d74b80b4d909f1255473` | Apache-2.0 |
| `vectara/hallucination_evaluation_model` (HHEM-2.1-Open) | non-LLM faithfulness check | `8e4a2e6e96c708cc76c2344f7e4757df2515292c` | Apache-2.0 |
| `google/flan-t5-base` | HHEM's base config and tokenizer | `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2` | Apache-2.0 |

HHEM's checkpoint expects `trust_remote_code=True`, because its model class ships with the checkpoint. We read that remote code at the pinned revision (72 lines of modeling code and 21 of config: transformers' `T5ForTokenClassification` over the flan-t5-base config, a fixed prompt, and a softmax over the first token's two logits). It does not load under transformers 5 (the class never calls `post_init`), so `hhem.py` rebuilds the same computation from transformers' own class, with flan-t5-base's config and tokenizer and HHEM's weights, each at its pinned revision. No remote code runs. On the seven example pairs on the model card, it reproduces the card's printed scores to four decimals (`tests/test_models_download.py`).

## API models (Azure AI Foundry, live runs only)

`gpt-6-luna` (answers, contextual retrieval contexts, Ragas), `Llama-3.3-70B-Instruct` (primary judge), `gpt-5-mini` (second judge). Called by deployment name through the harness `FoundryClient`. Nothing in the repo names a resource, endpoint or key.
