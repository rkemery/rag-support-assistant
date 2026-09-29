"""LlamaIndex embedding models: sentence-transformers wrappers and a hashing model for tests.

Document embeddings are cached on disk as .npy files keyed by the model, its
revision and a hash of the texts, so the retrieval grid embeds each chunk set
once. Queries are embedded fresh each time, which is what the latency column
measures.
"""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import PrivateAttr

from rag_support_assistant.models import EmbeddingSpec


class SentenceTransformerEmbedding(BaseEmbedding):
    """A pinned sentence-transformers bi-encoder behind LlamaIndex's embedding interface.

    Embeddings are L2-normalized, so Qdrant's cosine distance is a dot product.
    Queries get the model's query instruction, documents get none.
    """

    spec_key: str
    hf_id: str
    revision: str
    query_prompt: str | None = None
    query_prompt_name: str | None = None
    _model: Any = PrivateAttr()

    def __init__(self, spec: EmbeddingSpec, batch_size: int = 32, **kwargs: Any) -> None:
        super().__init__(
            model_name=spec.hf_id,
            embed_batch_size=batch_size,
            spec_key=spec.key,
            hf_id=spec.hf_id,
            revision=spec.revision,
            query_prompt=spec.query_prompt,
            query_prompt_name=spec.query_prompt_name,
            **kwargs,
        )
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(spec.hf_id, revision=spec.revision, device="cpu")

    @classmethod
    def class_name(cls) -> str:
        return "SentenceTransformerEmbedding"

    def _encode(self, texts: list[str], *, query: bool) -> list[list[float]]:
        kwargs: dict[str, Any] = {
            "batch_size": self.embed_batch_size,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        if query and self.query_prompt_name:
            kwargs["prompt_name"] = self.query_prompt_name
        elif query and self.query_prompt:
            kwargs["prompt"] = self.query_prompt
        vectors = self._model.encode(texts, **kwargs)
        return vectors.astype(np.float32).tolist()

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._encode([query], query=True)[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._encode([text], query=False)[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, query=False)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)


class HashingEmbedding(BaseEmbedding):
    """Bag-of-words feature hashing into `dim` buckets, L2-normalized. For tests only.

    Deterministic, needs no download, and ranks texts that share words above
    texts that don't, which is enough to exercise the pipeline end to end.
    """

    model_name: str = "hashing-bow"
    dim: int = 64
    revision: str = "hashing-v1"

    @classmethod
    def class_name(cls) -> str:
        return "HashingEmbedding"

    def _vector(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float64)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(word.encode()) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return (v / norm if norm > 0 else v).tolist()

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vector(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._vector(text)


def cache_key(model_name: str, revision: str, texts: Sequence[str]) -> str:
    blob = json.dumps([model_name, revision, list(texts)], ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:20]


def embed_documents(
    model: BaseEmbedding, texts: Sequence[str], cache_dir: Path | None = None
) -> np.ndarray:
    """Document embeddings as a float32 array, read from or written to `cache_dir`."""
    revision = getattr(model, "revision", "unknown")
    path = None
    if cache_dir is not None:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model.model_name)
        path = cache_dir / f"{safe}-{cache_key(model.model_name, revision, texts)}.npy"
        if path.exists():
            cached = np.load(path)
            if cached.shape[0] == len(texts):
                return cached
    vectors = np.asarray(model.get_text_embedding_batch(list(texts)), dtype=np.float32)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
    return vectors
