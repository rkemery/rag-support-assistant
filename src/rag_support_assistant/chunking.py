"""Chunking with LlamaIndex node parsers.

Three strategies, so the grid can change one thing at a time:

- `fixed`: `TokenTextSplitter`, 128 tokens with 16 overlap (tiktoken cl100k,
  which llama-index-core ships), body text only.
- `fixed-title`: the same chunks with the article title prepended.
- `header`: `MarkdownNodeParser`, one chunk per `##` section (the intro is its
  own chunk), with the article title prepended.

The title goes in through LlamaIndex metadata that is allowed into the
embedding text (`MetadataMode.EMBED`), so the dense and BM25 encoders see
exactly the same string. Chunk IDs are UUIDv5 values derived from the article
ID and chunk position, so indexes and snapshots are reproducible (Qdrant
needs UUID or integer point IDs).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, TokenTextSplitter
from llama_index.core.schema import MetadataMode, TextNode

from rag_support_assistant.data import Article

Chunking = Literal["fixed", "fixed-title", "header"]
CHUNKINGS: tuple[Chunking, ...] = ("fixed", "fixed-title", "header")

FIXED_CHUNK_TOKENS = 128
FIXED_OVERLAP_TOKENS = 16

_NAMESPACE = uuid.UUID("0d7f5a52-3c1e-4b8f-9a61-6f2b1c9e7d40")
_META_KEYS = ("article_id", "title", "section", "effective_date", "chunk_index", "chunking")


@dataclass(frozen=True)
class Chunk:
    """A chunk as plain data, for snapshots and prompts."""

    chunk_id: str
    article_id: str
    title: str
    effective_date: str
    chunk_index: int
    text: str  # the chunk body, without the prepended title
    embed_text: str  # exactly what the encoders see


def chunk_id(article_id: str, chunking: str, index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{chunking}/{article_id}/{index}"))


def _document(article: Article, chunking: Chunking) -> Document:
    metadata = {
        "article_id": article.article_id,
        "title": article.title,
        "section": article.section,
        "effective_date": article.effective_date,
    }
    embed_title = chunking in ("fixed-title", "header")
    excluded = [k for k in metadata if not (embed_title and k == "title")]
    return Document(
        text=article.body,
        id_=f"doc/{article.article_id}",
        metadata=metadata,
        excluded_embed_metadata_keys=excluded,
        excluded_llm_metadata_keys=list(metadata),
        metadata_template="{key}: {value}",
    )


def _parser(chunking: Chunking) -> TokenTextSplitter | MarkdownNodeParser:
    if chunking in ("fixed", "fixed-title"):
        return TokenTextSplitter(
            chunk_size=FIXED_CHUNK_TOKENS,
            chunk_overlap=FIXED_OVERLAP_TOKENS,
            include_prev_next_rel=False,
        )
    if chunking == "header":
        return MarkdownNodeParser(include_metadata=True, include_prev_next_rel=False)
    raise ValueError(f"unknown chunking {chunking!r}, expected one of {CHUNKINGS}")


def chunk_articles(articles: Sequence[Article], chunking: Chunking) -> list[TextNode]:
    """LlamaIndex nodes for every article, in corpus order, with stable IDs."""
    parser = _parser(chunking)
    nodes: list[TextNode] = []
    for article in articles:
        parsed = parser.get_nodes_from_documents([_document(article, chunking)])
        for index, node in enumerate(parsed):
            if not isinstance(node, TextNode):
                raise TypeError(f"expected TextNode, got {type(node).__name__}")
            node.id_ = chunk_id(article.article_id, chunking, index)
            # MarkdownNodeParser adds header_path. Keep only our keys, in a fixed order.
            node.metadata = {
                **{k: node.metadata[k] for k in _META_KEYS if k in node.metadata},
                "chunk_index": index,
                "chunking": chunking,
            }
            keep_title = chunking in ("fixed-title", "header")
            node.excluded_embed_metadata_keys = [
                k for k in node.metadata if not (keep_title and k == "title")
            ]
            node.excluded_llm_metadata_keys = list(node.metadata)
            node.metadata_template = "{key}: {value}"
            node.relationships = {}
            nodes.append(node)
    return nodes


def embed_text(node: TextNode) -> str:
    return node.get_content(metadata_mode=MetadataMode.EMBED)


def to_chunk(node: TextNode) -> Chunk:
    return Chunk(
        chunk_id=node.node_id,
        article_id=node.metadata["article_id"],
        title=node.metadata["title"],
        effective_date=node.metadata["effective_date"],
        chunk_index=int(node.metadata["chunk_index"]),
        text=node.get_content(metadata_mode=MetadataMode.NONE),
        embed_text=embed_text(node),
    )


def add_contexts(nodes: Sequence[TextNode], contexts: dict[str, str]) -> list[TextNode]:
    """Copies of `nodes` with an LLM-written `context` prepended to the embedding text.

    This is the contextual retrieval recipe (Anthropic, 2024): a short
    situating context goes in front of each chunk before both the dense and
    the BM25 encoders see it. The context sits in metadata, so the text shown
    to the answer model is unchanged.
    """
    out = []
    for node in nodes:
        if node.node_id not in contexts:
            raise KeyError(f"no context for chunk {node.node_id} ({node.metadata['article_id']})")
        copy = node.model_copy(deep=True)
        copy.metadata = {"context": contexts[node.node_id].strip(), **node.metadata}
        keep = (
            {"context", "title"}
            if "title" not in node.excluded_embed_metadata_keys
            else {"context"}
        )
        copy.excluded_embed_metadata_keys = [k for k in copy.metadata if k not in keep]
        copy.excluded_llm_metadata_keys = list(copy.metadata)
        out.append(copy)
    return out
