"""Build and load the two judge-validation sets, both committed under data/.

- data/judge_validation/: the perturbation set (items, labels in the harness
  label format, the dev/test split).
- data/ragtruth/: the seeded RAGTruth QA subset, its license and a manifest.

Labels are written as harness `LabelRecord`s with labeler
"construction:perturb-v1", so calibration runs through the harness
(`pair_labels`, `judge_agreement`, `corrected_pass_rate`) with its judge
fingerprint checks. Every constructed item is labeled (a census, not a sample),
so the harness's "uniform" sampling tag holds.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_eval_harness.calibration import Split, load_split, save_split
from llm_eval_harness.labeling import LabelRecord, items_fingerprint, read_labels

from rag_support_assistant import perturb, ragtruth
from rag_support_assistant.data import REPO_ROOT, load_articles, load_facts, load_questions

PERTURB_DIR = REPO_ROOT / "data" / "judge_validation"
RAGTRUTH_DIR = REPO_ROOT / "data" / "ragtruth"
LABELER = "construction:perturb-v1"
CREATED_AT = "2026-09-28T00:00:00+00:00"


@dataclass(frozen=True)
class PerturbationSet:
    items: list[perturb.Perturbation]
    labels: list[LabelRecord]
    split: Split

    def by_split(self, name: str) -> list[perturb.Perturbation]:
        wanted = set(self.split.dev if name == "dev" else self.split.test)
        return [item for item in self.items if item.item_id in wanted]


def build_perturbation_set(out_dir: Path = PERTURB_DIR) -> PerturbationSet:
    items = perturb.build(load_questions(), load_facts(), load_articles())
    ids = [item.item_id for item in items]
    fingerprint = items_fingerprint(ids)
    labels = [
        LabelRecord(
            item_id=item.item_id,
            labels=dict(item.labels),
            labeler=LABELER,
            sampling="uniform",
            seed=perturb.SEED,
            items_sha256=fingerprint,
            n_target=len(ids),
            created_at=CREATED_AT,
        )
        for item in items
    ]
    split = Split(
        seed=perturb.SEED,
        dev=tuple(sorted(i.item_id for i in items if i.split == "dev")),
        test=tuple(sorted(i.item_id for i in items if i.split == "test")),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    perturb.write_jsonl(out_dir / "perturbations.jsonl", items)
    with (out_dir / "labels.jsonl").open("w", encoding="utf-8") as fh:
        for label in labels:
            fh.write(json.dumps(asdict(label), ensure_ascii=False) + "\n")
    save_split(split, out_dir / "split.json")
    return PerturbationSet(items, labels, split)


def load_perturbation_set(out_dir: Path = PERTURB_DIR) -> PerturbationSet:
    return PerturbationSet(
        items=perturb.read_jsonl(out_dir / "perturbations.jsonl"),
        labels=read_labels(out_dir / "labels.jsonl"),
        split=load_split(out_dir / "split.json"),
    )


def build_ragtruth_subset(
    raw_dir: Path, out_dir: Path = RAGTRUTH_DIR, download: bool = True
) -> dict:
    """Download (or reuse) the pinned source files, select the subset, write it with a manifest."""
    if download:
        source_hashes = ragtruth.download(raw_dir)
        with urllib.request.urlopen(ragtruth.raw_url("LICENSE"), timeout=60) as resp:
            license_text = resp.read().decode("utf-8")
    else:
        source_hashes = {
            name: hashlib.sha256((raw_dir / name).read_bytes()).hexdigest()
            for name in ragtruth.SOURCES
        }
        license_text = (raw_dir / "LICENSE").read_text(encoding="utf-8")
    items = ragtruth.select(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = out_dir / "qa_subset.jsonl"
    ragtruth.write_subset(items, subset)
    (out_dir / "LICENSE").write_text(license_text, encoding="utf-8")
    manifest = {
        "dataset": "RAGTruth",
        "repo": f"https://github.com/{ragtruth.REPO}",
        "commit": ragtruth.COMMIT,
        "license": ragtruth.LICENSE,
        "source_sha256": source_hashes,
        "selection": {
            "task_type": "QA",
            "split": "test",
            "quality": "good",
            "per_class": ragtruth.N_PER_CLASS,
            "seed": ragtruth.SEED,
        },
        "subset_sha256": hashlib.sha256(subset.read_bytes()).hexdigest(),
        "n_items": len(items),
        "n_not_grounded": sum(not item.grounded for item in items),
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_ragtruth_subset(out_dir: Path = RAGTRUTH_DIR) -> list[ragtruth.RagTruthItem]:
    manifest = json.loads((out_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    subset = out_dir / "qa_subset.jsonl"
    digest = hashlib.sha256(subset.read_bytes()).hexdigest()
    if digest != manifest["subset_sha256"]:
        raise ValueError(f"{subset} does not match its manifest hash")
    return ragtruth.read_subset(subset)
