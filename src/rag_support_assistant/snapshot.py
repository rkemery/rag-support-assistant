"""Export a frozen retrieval snapshot that the agents repo can vendor.

snapshot/
  chunks.jsonl      one chunk per line, in row order of the embeddings
  embeddings.npy    float32 [n_chunks, 384], bge-small-en-v1.5, L2-normalized
  config.json       chunking, model and revision, query prompt, BM25 settings, dataset commit
  MANIFEST.json     sha256 of the three files above

The chunking is the one chosen on dev. The embedding model is always
bge-small-en-v1.5 (the runtime baseline in the plan), whichever model won the
grid, so a consumer only needs that 33M model to embed queries.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rag_support_assistant.bm25 import BM25Encoder
from rag_support_assistant.chunking import (
    FIXED_CHUNK_TOKENS,
    FIXED_OVERLAP_TOKENS,
    chunk_articles,
    embed_text,
    to_chunk,
)
from rag_support_assistant.data import DATA_DIR, REPO_ROOT, load_articles
from rag_support_assistant.embeddings import SentenceTransformerEmbedding, embed_documents
from rag_support_assistant.models import EMBEDDINGS

SNAPSHOT_EMBEDDING = "bge-small"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(out_dir: Path, results: Path, cache_dir: Path | None) -> dict[str, str]:
    selection = json.loads((results / "retrieval" / "selection.json").read_text(encoding="utf-8"))
    final = selection["steps"][-1]["chosen"]
    chosen = selection["configs"][final]["config"]
    chunking = chosen["chunking"]
    spec = EMBEDDINGS[SNAPSHOT_EMBEDDING]
    nodes = chunk_articles(load_articles(), chunking)
    texts = [embed_text(n) for n in nodes]
    vectors = embed_documents(SentenceTransformerEmbedding(spec), texts, cache_dir)
    bm25 = BM25Encoder()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for node in nodes:
            fh.write(json.dumps(asdict(to_chunk(node)), ensure_ascii=False) + "\n")
    np.save(out_dir / "embeddings.npy", vectors.astype(np.float32))
    data_manifest = json.loads((DATA_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
    config = {
        "chunking": chunking,
        "fixed_chunk_tokens": FIXED_CHUNK_TOKENS if chunking != "header" else None,
        "fixed_overlap_tokens": FIXED_OVERLAP_TOKENS if chunking != "header" else None,
        "title_in_embedding_text": chunking in ("fixed-title", "header"),
        "n_chunks": len(nodes),
        "embedding_model": spec.hf_id,
        "embedding_revision": spec.revision,
        "embedding_dim": int(vectors.shape[1]),
        "normalized": True,
        "query_prompt": spec.query_prompt,
        "similarity": "cosine (dot product of normalized vectors)",
        "bm25": {"k1": bm25.k1, "b": bm25.b, "tokenizer": "rag_support_assistant.bm25.tokenize"},
        "dev_chosen_config": chosen,
        "dataset_commit": data_manifest["source_commit"],
        "dataset_license": data_manifest["license"],
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    files = {
        name: _sha256(out_dir / name) for name in ("chunks.jsonl", "embeddings.npy", "config.json")
    }
    (out_dir / "MANIFEST.json").write_text(
        json.dumps({"files": files}, indent=2) + "\n", encoding="utf-8"
    )
    return files


def verify(out_dir: Path = REPO_ROOT / "snapshot") -> None:
    manifest = json.loads((out_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    bad = [name for name, digest in manifest["files"].items() if _sha256(out_dir / name) != digest]
    if bad:
        raise ValueError(f"snapshot files do not match MANIFEST.json: {bad}")
