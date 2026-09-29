"""Open models used offline, each pinned to a Hugging Face commit.

Revisions were read from the Hugging Face API on 2026-09-28. Licenses come
from each model's `license:` tag on the Hub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EmbeddingSpec:
    key: str
    hf_id: str
    revision: str
    license: str
    params: str
    query_prompt: str | None = None  # literal prefix added to queries
    query_prompt_name: str | None = (
        None  # a named prompt in the model's sentence-transformers config
    )
    role: Literal["runtime", "strong_arm"] = "runtime"


@dataclass(frozen=True)
class RerankerSpec:
    key: str
    hf_id: str
    revision: str
    license: str
    params: str
    max_length: int = 512
    role: Literal["runtime", "strong_arm"] = "runtime"


EMBEDDINGS: dict[str, EmbeddingSpec] = {
    "bge-small": EmbeddingSpec(
        key="bge-small",
        hf_id="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        license="MIT",
        params="33M",
        # From the model card: short queries retrieving passages get this instruction.
        query_prompt="Represent this sentence for searching relevant passages: ",
    ),
    "granite-small": EmbeddingSpec(
        key="granite-small",
        hf_id="ibm-granite/granite-embedding-small-english-r2",
        revision="2ab6fa8ea2d674564defd37171ae19079b864b33",
        license="Apache-2.0",
        params="47M",
    ),
    "qwen3-emb": EmbeddingSpec(
        key="qwen3-emb",
        hf_id="Qwen/Qwen3-Embedding-0.6B",
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        license="Apache-2.0",
        params="0.6B",
        query_prompt_name="query",
        role="strong_arm",
    ),
}

RERANKERS: dict[str, RerankerSpec] = {
    "granite-rerank": RerankerSpec(
        key="granite-rerank",
        hf_id="ibm-granite/granite-embedding-reranker-english-r2",
        revision="d09d3d6971b689bf9c23839e45a470874d46e13a",
        license="Apache-2.0",
        params="149M",
    ),
    "qwen3-rerank": RerankerSpec(
        key="qwen3-rerank",
        hf_id="Qwen/Qwen3-Reranker-0.6B",
        revision="e61197ed45024b0ed8a2d74b80b4d909f1255473",
        license="Apache-2.0",
        params="0.6B",
        role="strong_arm",
    ),
}

# HHEM-2.1-Open. Its checkpoint expects trust_remote_code. We read that code at
# this revision, and hhem.py rebuilds the same computation from transformers' own
# T5ForTokenClassification instead of running it (it fails to load under
# transformers 5). flan-t5-base supplies the config and tokenizer, pinned too.
HHEM_ID = "vectara/hallucination_evaluation_model"
HHEM_REVISION = "8e4a2e6e96c708cc76c2344f7e4757df2515292c"
HHEM_FOUNDATION_ID = "google/flan-t5-base"
HHEM_FOUNDATION_REVISION = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
