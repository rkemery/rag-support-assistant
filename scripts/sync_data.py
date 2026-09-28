"""Vendor and verify the pinned Tallowbrook dataset snapshot in data/tallowbrook/.

    uv run python scripts/sync_data.py            # verify vendored files against MANIFEST.json
    uv run python scripts/sync_data.py --from ../neobank-support-data   # copy, rewrite manifest

The copy mode refuses a source checkout that is not at the pinned commit or has
uncommitted changes, so the manifest always names a commit the files came from.
The dataset's canonical home will be a Hugging Face dataset. Until then the
source is a git checkout of neobank-support-data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "tallowbrook"
MANIFEST = DEST / "MANIFEST.json"

PINNED_COMMIT = "3c72058e8cc7d5dd6e224ecdec7b814341a3d48f"
# source path in neobank-support-data -> vendored path under data/tallowbrook/
FILES = {
    "corpus/articles.jsonl": "articles.jsonl",
    "facts/policies.yaml": "policies.yaml",
    "rag/questions_dev.jsonl": "questions_dev.jsonl",
    "rag/questions_test.jsonl": "questions_test.jsonl",
}


class SyncError(RuntimeError):
    """The vendored data does not match its manifest, or the source is not the pinned commit."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(dest: Path = DEST) -> list[str]:
    """Check every vendored file against MANIFEST.json. Returns the verified file names."""
    manifest = json.loads((dest / "MANIFEST.json").read_text(encoding="utf-8"))
    problems = []
    for entry in manifest["files"]:
        path = dest / entry["path"]
        if not path.exists():
            problems.append(f"missing {entry['path']}")
        elif sha256(path) != entry["sha256"]:
            problems.append(f"hash mismatch for {entry['path']}")
    listed = {entry["path"] for entry in manifest["files"]}
    extra = sorted(
        p.name for p in dest.iterdir() if p.is_file() and p.name not in listed | {"MANIFEST.json"}
    )
    if extra and set(extra) - {"README.md"}:
        problems.append(f"files not in the manifest: {extra}")
    if problems:
        raise SyncError("; ".join(problems))
    return sorted(listed)


def _git(source: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def copy_from(source: Path, dest: Path = DEST) -> None:
    head = _git(source, "rev-parse", "HEAD")
    if head != PINNED_COMMIT:
        raise SyncError(f"{source} is at {head}, expected the pinned commit {PINNED_COMMIT}")
    dirty = _git(source, "status", "--porcelain", "--", *FILES)
    if dirty:
        raise SyncError(f"{source} has uncommitted changes to the vendored files:\n{dirty}")
    dest.mkdir(parents=True, exist_ok=True)
    entries = []
    for src_rel, dest_name in FILES.items():
        shutil.copyfile(source / src_rel, dest / dest_name)
        entries.append(
            {"path": dest_name, "source_path": src_rel, "sha256": sha256(dest / dest_name)}
        )
    manifest = {
        "dataset": "Tallowbrook Neobank Support (synthetic)",
        "license": "CC-BY-4.0",
        "source_repo": "neobank-support-data",
        "source_commit": PINNED_COMMIT,
        "files": entries,
    }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="source", type=Path, help="checkout to copy from")
    args = parser.parse_args(argv)
    try:
        if args.source is not None:
            copy_from(args.source)
        names = verify()
    except SyncError as exc:
        print(f"sync_data: {exc}", file=sys.stderr)
        return 1
    print(f"sync_data: {len(names)} files match MANIFEST.json ({', '.join(names)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
