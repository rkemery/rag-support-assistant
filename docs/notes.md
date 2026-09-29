# Engineering notes

Snags hit while building rag-support-assistant, and how each was handled. The README's [What didn't work](../README.md#what-didnt-work) covers the experiments.

- **HHEM's own loading code.** Its `trust_remote_code` class fails under transformers 5 (it never calls `post_init`). Rebuilding the same computation from transformers' `T5ForTokenClassification` reproduces the scores printed on the model card to four decimals, and runs no remote code.
- **Ragas 0.4.3 with current LangChain.** It imports `langchain_community.chat_models.vertexai`, which langchain-community 0.4.2 removed. The `ragas` extra pins langchain-community 0.4.1, the release that was current when Ragas 0.4.3 shipped.
- **Four torch threads on a shared 4-vCPU box.** With other jobs running, a single bge-small query took 0.8 to 2 seconds instead of about 30 ms, because OpenMP threads spin while they wait for cores. Torch is pinned to 2 threads, and the grid was rerun from scratch.
- **Smaller snags.** LlamaIndex's bundled NLTK stopword file is refused by NLTK's hardlink check when uv installs it, so the list is inlined. A literal "$5.00" in the judge prompt broke Python's `string.Template`, which reads `$5` as a placeholder. A unit test caught it before any live call. Few reference answers state a versioned number, so superseded-value perturbations came to 15 items, and word-level swaps (travel notices, the old lost-card phone line) only brought them to 17.
