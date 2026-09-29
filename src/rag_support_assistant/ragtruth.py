"""A fixed, seeded RAGTruth subset: an outside check of the judge on natural hallucinations.

RAGTruth (Niu et al., ACL 2024, arXiv 2401.00396) has human span annotations of
hallucinations in answers that real LLMs wrote from retrieved passages. This repo uses
its QA task (MS MARCO questions with passages), test split, responses of
quality "good" only, and draws 100 responses with at least one annotated span
and 100 with none. A response with any span counts as "not grounded", which
matches this repo's `grounded` check: RAGTruth's "implicit true" spans (true, but not
in the passages) are unsupported by the context too.

The builder downloads the two source files at a pinned commit, checks their
sha256, and writes the subset with the license. The subset is committed, so
nothing needs the network afterwards.
"""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = "ParticleMedia/RAGTruth"
COMMIT = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
LICENSE = "MIT"
SOURCES = {
    "response.jsonl": "dataset/response.jsonl",
    "source_info.jsonl": "dataset/source_info.jsonl",
}
SEED = 20260928
N_PER_CLASS = 100


@dataclass(frozen=True)
class RagTruthItem:
    item_id: str
    response_id: str
    source_id: str
    model: str
    question: str
    passages: str
    response: str
    grounded: bool
    spans: list[dict[str, Any]]


def raw_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}"


def download(dest_dir: Path) -> dict[str, str]:
    """Fetch the source files at the pinned commit. Returns {name: sha256}."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, path in SOURCES.items():
        with urllib.request.urlopen(raw_url(path), timeout=120) as resp:
            data = resp.read()
        (dest_dir / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    return hashes


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def select(raw_dir: Path, seed: int = SEED, n_per_class: int = N_PER_CLASS) -> list[RagTruthItem]:
    sources = {s["source_id"]: s for s in _read(raw_dir / "source_info.jsonl")}
    pools: dict[bool, list[RagTruthItem]] = {True: [], False: []}
    for r in _read(raw_dir / "response.jsonl"):
        source = sources[r["source_id"]]
        if source["task_type"] != "QA" or r["split"] != "test" or r["quality"] != "good":
            continue
        info = source["source_info"]
        grounded = not r["labels"]
        pools[grounded].append(
            RagTruthItem(
                item_id=f"ragtruth-{r['id']}",
                response_id=str(r["id"]),
                source_id=str(r["source_id"]),
                model=r["model"],
                question=info["question"],
                passages=info["passages"].strip(),
                response=r["response"],
                grounded=grounded,
                spans=[
                    {k: span.get(k) for k in ("text", "label_type", "implicit_true")}
                    for span in r["labels"]
                ],
            )
        )
    rng = random.Random(seed)
    chosen = []
    for grounded in (False, True):
        pool = sorted(pools[grounded], key=lambda item: int(item.response_id))
        if len(pool) < n_per_class:
            raise ValueError(f"only {len(pool)} responses with grounded={grounded}")
        chosen.extend(rng.sample(pool, n_per_class))
    return sorted(chosen, key=lambda item: int(item.response_id))


def write_subset(items: list[RagTruthItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def read_subset(path: Path) -> list[RagTruthItem]:
    return [RagTruthItem(**row) for row in _read(path)]
