"""Retrieval pipeline: LlamaIndex over Qdrant local mode, dense, BM25 or hybrid, optional rerank.

One in-memory Qdrant collection per (chunking, embedding model) holds a dense
vector and a BM25 sparse vector for every chunk. Dense, BM25-only and hybrid
queries all run against that one collection through LlamaIndex's
`QdrantVectorStore`, with the fusion function passed explicitly.

Results are chunk lists. `rank_articles` collapses them to articles (first
appearance wins), which is the level the gold labels are at.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from rag_support_assistant.bm25 import BM25Encoder
from rag_support_assistant.chunking import Chunking, embed_text
from rag_support_assistant.fusion import convex_fusion, rrf_fusion
from rag_support_assistant.rerank import CrossEncoderRerank

Mode = Literal["dense", "bm25", "hybrid"]
Fusion = Literal["convex", "rrf"]

# First-stage depth in chunks for each of dense and BM25, and after fusion.
FIRST_STAGE_DEPTH = 50


@dataclass(frozen=True)
class RetrievalConfig:
    """One cell of the retrieval grid."""

    chunking: Chunking
    embedding: str  # key in models.EMBEDDINGS (BM25-only configs still build the dense index)
    mode: Mode
    fusion: Fusion | None = None
    alpha: float | None = None  # convex fusion weight on dense, tuned on dev
    reranker: str | None = None  # key in models.RERANKERS
    contextual: bool = False  # chunks carry an LLM-written context (live-only cell)

    def __post_init__(self) -> None:
        if (self.mode == "hybrid") != (self.fusion is not None):
            raise ValueError("fusion is set exactly when mode is 'hybrid'")
        if (self.fusion == "convex") != (self.alpha is not None):
            raise ValueError("alpha is set exactly when fusion is 'convex'")
        if self.alpha is not None and not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")

    @property
    def name(self) -> str:
        parts = [self.chunking + ("+ctx" if self.contextual else "")]
        if self.mode == "dense":
            parts.append(self.embedding)
        elif self.mode == "bm25":
            parts.append("bm25")
        else:
            fusion = f"convex(a={self.alpha:.1f})" if self.fusion == "convex" else "rrf"
            parts.append(f"{self.embedding}+bm25 {fusion}")
        if self.reranker:
            parts.append(f"rerank {self.reranker}")
        return " / ".join(parts)

    @property
    def slug(self) -> str:
        parts = [self.chunking + ("-ctx" if self.contextual else ""), self.mode]
        if self.mode != "bm25":
            parts.append(self.embedding)
        if self.fusion == "convex":
            parts.append(f"convex{self.alpha:.1f}")
        elif self.fusion == "rrf":
            parts.append("rrf")
        if self.reranker:
            parts.append(self.reranker)
        return "-".join(parts)

    def with_(self, **changes: object) -> RetrievalConfig:
        return replace(self, **changes)  # type: ignore[arg-type]


@dataclass
class ChunkIndex:
    """Qdrant collection plus the LlamaIndex objects that query it."""

    chunking: Chunking
    embedding: str
    nodes: list[TextNode]
    embed_model: BaseEmbedding
    bm25: BM25Encoder
    client: QdrantClient
    collection: str
    dim: int
    _indexes: dict[str, VectorStoreIndex] = field(default_factory=dict)

    def _store(self, fusion: Fusion) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection,
            enable_hybrid=True,
            sparse_doc_fn=self.bm25.encode_documents,
            sparse_query_fn=self.bm25.encode_queries,
            hybrid_fusion_fn=convex_fusion if fusion == "convex" else rrf_fusion,
        )

    def index_for(self, fusion: Fusion) -> VectorStoreIndex:
        if fusion not in self._indexes:
            self._indexes[fusion] = VectorStoreIndex.from_vector_store(
                self._store(fusion), embed_model=self.embed_model
            )
        return self._indexes[fusion]


def build_index(
    nodes: Sequence[TextNode],
    doc_vectors: np.ndarray,
    embed_model: BaseEmbedding,
    *,
    chunking: Chunking,
    embedding: str,
) -> ChunkIndex:
    """Fit BM25 on the chunk texts and load dense and sparse vectors into Qdrant (in memory)."""
    if len(nodes) != len(doc_vectors):
        raise ValueError(f"{len(nodes)} nodes but {len(doc_vectors)} vectors")
    texts = [embed_text(n) for n in nodes]
    bm25 = BM25Encoder().fit(texts)
    loaded = []
    for node, vector in zip(nodes, doc_vectors, strict=True):
        copy = node.model_copy(deep=True)
        copy.embedding = [float(x) for x in vector]
        loaded.append(copy)
    client = QdrantClient(location=":memory:")
    collection = f"chunks-{chunking}-{embedding}"
    store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        enable_hybrid=True,
        sparse_doc_fn=bm25.encode_documents,
        sparse_query_fn=bm25.encode_queries,
        hybrid_fusion_fn=convex_fusion,
        batch_size=256,
    )
    index = VectorStoreIndex(
        loaded,
        storage_context=StorageContext.from_defaults(vector_store=store),
        embed_model=embed_model,
    )
    chunk_index = ChunkIndex(
        chunking=chunking,
        embedding=embedding,
        nodes=list(nodes),
        embed_model=embed_model,
        bm25=bm25,
        client=client,
        collection=collection,
        dim=int(doc_vectors.shape[1]),
    )
    chunk_index._indexes["convex"] = index
    return chunk_index


@dataclass(frozen=True)
class RetrievalResult:
    question_id: str
    chunks: list[NodeWithScore]
    latency_ms: float
    rerank_ms: float = 0.0

    @property
    def articles(self) -> list[str]:
        return rank_articles(self.chunks)


def rank_articles(chunks: Sequence[NodeWithScore]) -> list[str]:
    """Article IDs in order of their best-ranked chunk."""
    seen: dict[str, None] = {}
    for item in chunks:
        seen.setdefault(item.node.metadata["article_id"], None)
    return list(seen)


class Retriever:
    """Runs one config against its chunk index, timing each query."""

    def __init__(
        self,
        config: RetrievalConfig,
        index: ChunkIndex,
        reranker: CrossEncoderRerank | None = None,
        depth: int = FIRST_STAGE_DEPTH,
    ) -> None:
        if (config.reranker is None) != (reranker is None):
            raise ValueError("pass a reranker exactly when the config names one")
        if (index.chunking, index.embedding) != (config.chunking, config.embedding):
            raise ValueError(f"index {index.collection} does not match config {config.name}")
        self.config = config
        self.index = index
        self.reranker = reranker
        mode = {
            "dense": VectorStoreQueryMode.DEFAULT,
            "bm25": VectorStoreQueryMode.SPARSE,
            "hybrid": VectorStoreQueryMode.HYBRID,
        }[config.mode]
        vector_index = index.index_for(config.fusion or "convex")
        self._retriever = VectorIndexRetriever(
            vector_index,
            similarity_top_k=depth,
            sparse_top_k=depth,
            hybrid_top_k=depth,
            vector_store_query_mode=mode,
            alpha=config.alpha,
        )

    def retrieve(self, question_id: str, query: str) -> RetrievalResult:
        start = time.perf_counter()
        bundle = QueryBundle(query_str=query)
        if self.config.mode == "bm25":
            # BM25 needs no query embedding. A placeholder keeps LlamaIndex from computing one.
            bundle.embedding = [0.0] * self.index.dim
        chunks = self._retriever.retrieve(bundle)
        rerank_ms = 0.0
        if self.reranker is not None:
            rerank_start = time.perf_counter()
            chunks = self.reranker.postprocess_nodes(chunks, query_bundle=bundle)
            rerank_ms = (time.perf_counter() - rerank_start) * 1000.0
        latency_ms = (time.perf_counter() - start) * 1000.0
        return RetrievalResult(question_id, chunks, latency_ms, rerank_ms)
