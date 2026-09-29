"""HHEM-2.1-Open as a non-LLM second opinion on faithfulness. Offline, CPU.

Vectara's model scores (premise, hypothesis) pairs from 0 (not supported) to 1
(supported). The score is the softmax probability of its "consistent" class
against its "hallucinated" class, so 0.5 is where the model's own prediction
flips. An answer counts as grounded when the score is at least 0.5. The
threshold is not tuned here.

The checkpoint ships its own model class and expects `trust_remote_code`. I
read that code at the pinned revision (72 lines): a transformers
`T5ForTokenClassification` over the google/flan-t5-base config, a fixed
prompt, and a softmax over the first token's two logits. Under transformers 5
it no longer loads (its class never calls `post_init`, so
`all_tied_weights_keys` is missing). Instead of running it, `HHEM` rebuilds the
same computation from transformers' own class: flan-t5-base config and
tokenizer at a pinned revision, HHEM's weights at a pinned revision, the same
prompt. No remote code runs. `tests/test_models_download.py` checks the scores
against the ones printed on the model card.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from llm_eval_harness import EvalRecord

from rag_support_assistant.models import (
    HHEM_FOUNDATION_ID,
    HHEM_FOUNDATION_REVISION,
    HHEM_ID,
    HHEM_REVISION,
)

THRESHOLD = 0.5
FINGERPRINT = f"hhem-2.1-open@{HHEM_REVISION[:12]}-threshold{THRESHOLD}"
# From configuration_hhem_v2.py at HHEM_REVISION.
PROMPT = (
    "<pad> Determine if the hypothesis is true given the premise?"
    "\n\nPremise: {text1}\n\nHypothesis: {text2}"
)


@dataclass(frozen=True)
class HHEMInput:
    item_id: str
    premise: str
    hypothesis: str
    cluster: str | None


class HHEM:
    def __init__(self, batch_size: int = 4) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoTokenizer, T5ForTokenClassification

        config = AutoConfig.from_pretrained(
            HHEM_FOUNDATION_ID, revision=HHEM_FOUNDATION_REVISION, num_labels=2
        )
        model = T5ForTokenClassification(config)
        weights = hf_hub_download(HHEM_ID, "model.safetensors", revision=HHEM_REVISION)
        state = {k.removeprefix("t5."): v for k, v in load_file(weights).items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        # The encoder's token embedding is tied to `shared`, which the checkpoint holds.
        if unexpected or set(missing) - {"transformer.encoder.embed_tokens.weight"}:
            raise RuntimeError(
                f"HHEM weights do not fit: missing {missing}, unexpected {unexpected}"
            )
        model.eval()
        self._torch = torch
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            HHEM_FOUNDATION_ID, revision=HHEM_FOUNDATION_REVISION
        )
        self.batch_size = batch_size

    def _predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        prompts = [PROMPT.format(text1=premise, text2=hypothesis) for premise, hypothesis in pairs]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        with self._torch.no_grad():
            logits = self.model(**inputs).logits[:, 0, :]
        return self._torch.softmax(logits, dim=-1)[:, 1].tolist()

    def scores(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Scores in input order. Batches are length-sorted to limit padding."""
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        out = [0.0] * len(pairs)
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            for i, value in zip(idx, self._predict([pairs[i] for i in idx]), strict=True):
                out[i] = float(value)
        return out


def score_items(model: HHEM, items: Sequence[HHEMInput], run_id: str) -> list[EvalRecord]:
    start = time.perf_counter()
    values = model.scores([(i.premise, i.hypothesis) for i in items])
    per_item_ms = (time.perf_counter() - start) * 1000.0 / max(len(items), 1)
    return [
        EvalRecord(
            run_id=run_id,
            item_id=item.item_id,
            config="hhem",
            model=f"{HHEM_ID}@{HHEM_REVISION[:12]}",
            scores={"grounded": value >= THRESHOLD, "hhem_score": value},
            cluster=item.cluster,
            latency_ms=per_item_ms,
            meta={"judge_fingerprint": FINGERPRINT},
        )
        for item, value in zip(items, values, strict=True)
    ]
