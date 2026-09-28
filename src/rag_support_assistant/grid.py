"""The retrieval grid: pick a path on dev one factor at a time, then score test once.

Steps (each keeps the dev winner of the step before, by mean dev nDCG@10):

1. Chunking, dense bge-small: fixed tokens, fixed tokens with the title, header-aware with
   the title.
2. Embedding: bge-small vs granite-small on the winning chunking.
3. First stage: dense vs BM25 alone vs hybrid with convex fusion (alpha tuned on dev) vs
   hybrid with RRF.
4. Reranker: the granite cross-encoder on the top 10 chunks of the winning first stage.

Strong-arm rows (Qwen3 embedding and reranker, offline only) are extra rows on
the chosen path. They never change the path and are not eligible for generation.

Every config then runs on test in one pass. Nothing is tuned on test.
"""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from llama_index.core.base.embeddings.base import BaseEmbedding
from llm_eval_harness import EvalRecord, read_records, write_records

from rag_support_assistant.chunking import Chunking, add_contexts, chunk_articles, embed_text
from rag_support_assistant.data import (
    Article,
    Question,
    Split,
    cluster_key,
    load_articles,
    load_questions,
    retrieval_questions,
)
from rag_support_assistant.embeddings import SentenceTransformerEmbedding, embed_documents
from rag_support_assistant.models import EMBEDDINGS, RERANKERS
from rag_support_assistant.rerank import CrossEncoderRerank, ScoreFn, load_cross_encoder
from rag_support_assistant.retrieval import (
    ChunkIndex,
    RetrievalConfig,
    RetrievalResult,
    Retriever,
    build_index,
)
from rag_support_assistant.scoring import PRIMARY_METRIC, per_query_scores

ALPHAS = tuple(round(a * 0.1, 1) for a in range(11))

EmbedFactory = Callable[[str], BaseEmbedding]
RerankFactory = Callable[[str], ScoreFn]


def default_embed_factory(key: str) -> BaseEmbedding:
    return SentenceTransformerEmbedding(EMBEDDINGS[key])


def default_rerank_factory(key: str) -> ScoreFn:
    return load_cross_encoder(RERANKERS[key])


@dataclass
class GridContext:
    """Builds and caches models, indexes and rerankers for the grid."""

    articles: Sequence[Article]
    cache_dir: Path | None = None
    embed_factory: EmbedFactory = default_embed_factory
    rerank_factory: RerankFactory = default_rerank_factory
    contexts: dict[str, str] | None = None  # chunk_id -> LLM-written context (contextual cell)
    timings: dict[str, float] = field(default_factory=dict)
    _embed_models: dict[str, BaseEmbedding] = field(default_factory=dict)
    _indexes: dict[tuple[str, str, bool], ChunkIndex] = field(default_factory=dict)
    _rerankers: dict[str, CrossEncoderRerank] = field(default_factory=dict)

    def embed_model(self, key: str) -> BaseEmbedding:
        if key not in self._embed_models:
            self._embed_models[key] = self.embed_factory(key)
        return self._embed_models[key]

    def index(self, chunking: Chunking, embedding: str, contextual: bool = False) -> ChunkIndex:
        key = (chunking, embedding, contextual)
        if key not in self._indexes:
            start = time.perf_counter()
            nodes = chunk_articles(self.articles, chunking)
            if contextual:
                if self.contexts is None:
                    raise ValueError("a contextual config needs chunk contexts (run the live step)")
                nodes = add_contexts(nodes, self.contexts)
            model = self.embed_model(embedding)
            vectors = embed_documents(model, [embed_text(n) for n in nodes], self.cache_dir)
            with warnings.catch_warnings():
                # Qdrant local mode warns that payload indexes do nothing there.
                warnings.filterwarnings("ignore", message="Payload indexes have no effect")
                self._indexes[key] = build_index(
                    nodes, vectors, model, chunking=chunking, embedding=embedding
                )
            label = f"index {chunking}/{embedding}" + (" +context" if contextual else "")
            self.timings[label] = time.perf_counter() - start
        return self._indexes[key]

    def reranker(self, key: str) -> CrossEncoderRerank:
        if key not in self._rerankers:
            self._rerankers[key] = CrossEncoderRerank(self.rerank_factory(key), model_key=key)
        return self._rerankers[key]

    def retriever(self, config: RetrievalConfig) -> Retriever:
        index = self.index(config.chunking, config.embedding, config.contextual)
        reranker = self.reranker(config.reranker) if config.reranker else None
        return Retriever(config, index, reranker)


def run_config(
    ctx: GridContext, config: RetrievalConfig, questions: Sequence[Question]
) -> list[RetrievalResult]:
    retriever = ctx.retriever(config)
    start = time.perf_counter()
    results = [retriever.retrieve(q.question_id, q.question) for q in questions]
    ctx.timings[f"queries {config.slug} x{len(questions)}"] = time.perf_counter() - start
    return results


def model_label(config: RetrievalConfig) -> str:
    base = "bm25" if config.mode == "bm25" else EMBEDDINGS[config.embedding].hf_id
    if config.mode == "hybrid":
        base += "+bm25"
    if config.reranker:
        base += f"+{RERANKERS[config.reranker].hf_id}"
    return base


def to_records(
    config: RetrievalConfig,
    split: Split,
    questions: Sequence[Question],
    results: Sequence[RetrievalResult],
) -> list[EvalRecord]:
    by_id = {r.question_id: r for r in results}
    scores = per_query_scores(questions, {r.question_id: r.articles for r in results})
    return [
        EvalRecord(
            run_id=f"retrieval/{split}/{config.slug}",
            item_id=q.question_id,
            config=config.name,
            model=model_label(config),
            scores=scores[q.question_id],
            cluster=cluster_key(q),
            latency_ms=by_id[q.question_id].latency_ms,
            meta={
                "split": split,
                "slug": config.slug,
                "retrieval_config": asdict(config),
                "top_articles": by_id[q.question_id].articles[:10],
                "rerank_ms": round(by_id[q.question_id].rerank_ms, 3),
            },
        )
        for q in questions
        if q.has_gold
    ]


def mean_metric(records: Sequence[EvalRecord], metric: str = PRIMARY_METRIC) -> float:
    return float(np.mean([float(r.scores[metric]) for r in records]))


@dataclass
class GridRun:
    """Everything the grid decided and measured, saved as selection.json."""

    steps: list[dict[str, Any]] = field(default_factory=list)
    alpha_sweep: dict[str, float] = field(default_factory=dict)
    configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    generation_configs: list[str] = field(default_factory=list)
    compute: dict[str, Any] = field(default_factory=dict)


def _best(candidates: dict[str, float]) -> str:
    """Highest dev score. Ties go to the earlier (simpler) candidate."""
    best_slug, best_score = None, -1.0
    for slug, score in candidates.items():
        if score > best_score + 1e-12:
            best_slug, best_score = slug, score
    assert best_slug is not None
    return best_slug


class Grid:
    """Runs the dev selection, then every chosen config on test, writing records as it goes."""

    def __init__(self, ctx: GridContext, out_dir: Path, questions: dict[Split, list[Question]]):
        self.ctx = ctx
        self.out_dir = out_dir
        self.questions = {s: retrieval_questions(qs) for s, qs in questions.items()}
        self.run = GridRun()
        self._dev_cache: dict[str, list[EvalRecord]] = {}
        self._configs: dict[str, RetrievalConfig] = {}

    def dev_score(self, config: RetrievalConfig, *, keep: bool = True) -> float:
        if config.slug not in self._dev_cache:
            results = run_config(self.ctx, config, self.questions["dev"])
            self._dev_cache[config.slug] = to_records(config, "dev", self.questions["dev"], results)
        if keep:
            self._configs[config.slug] = config
        return mean_metric(self._dev_cache[config.slug])

    def _step(self, name: str, configs: Sequence[RetrievalConfig]) -> RetrievalConfig:
        candidates = {c.slug: self.dev_score(c) for c in configs}
        chosen = _best(candidates)
        self.run.steps.append(
            {
                "step": name,
                "metric": PRIMARY_METRIC,
                "dev": {s: round(v, 4) for s, v in candidates.items()},
                "chosen": chosen,
            }
        )
        return next(c for c in configs if c.slug == chosen)

    def tune_alpha(self, base: RetrievalConfig) -> float:
        scores = {}
        for alpha in ALPHAS:
            config = base.with_(mode="hybrid", fusion="convex", alpha=alpha)
            scores[f"{alpha:.1f}"] = self.dev_score(config, keep=False)
        self.run.alpha_sweep = {k: round(v, 4) for k, v in scores.items()}
        return float(_best(scores))

    def select(self) -> RetrievalConfig:
        """Steps 1 to 4 on dev. Returns the final (reranked) config."""
        base = RetrievalConfig(chunking="fixed", embedding="bge-small", mode="dense")
        chunked = self._step(
            "chunking",
            [base, base.with_(chunking="fixed-title"), base.with_(chunking="header")],
        )
        dense = self._step("embedding", [chunked, chunked.with_(embedding="granite-small")])
        alpha = self.tune_alpha(dense)
        first_stage = self._step(
            "first stage",
            [
                dense,
                dense.with_(mode="bm25"),
                dense.with_(mode="hybrid", fusion="convex", alpha=alpha),
                dense.with_(mode="hybrid", fusion="rrf"),
            ],
        )
        return self._step("reranker", [first_stage, first_stage.with_(reranker="granite-rerank")])

    def strong_arm(self, final: RetrievalConfig) -> list[RetrievalConfig]:
        """Offline-only Qwen3 rows on the chosen path. They never change the selection."""
        first_stage = final.with_(reranker=None)
        dense = first_stage.with_(mode="dense", fusion=None, alpha=None)
        rows = [
            dense.with_(embedding="qwen3-emb"),
            first_stage.with_(reranker="qwen3-rerank"),
        ]
        for config in rows:
            self.dev_score(config)
        return rows

    def choose_generation_configs(self, strong: Sequence[RetrievalConfig]) -> list[str]:
        """Top 3 runtime configs by dev nDCG@10 (strong-arm rows are offline only)."""
        excluded = {c.slug for c in strong}
        ranked = sorted(
            (s for s in self._configs if s not in excluded),
            key=lambda s: -mean_metric(self._dev_cache[s]),
        )
        return ranked[:3]

    def write(self, split: Split, config: RetrievalConfig, records: list[EvalRecord]) -> None:
        write_records(self.out_dir / split / f"{config.slug}.jsonl", records)

    def run_all(self, *, strong: bool) -> GridRun:
        wall, cpu = time.perf_counter(), time.process_time()
        final = self.select()
        strong_rows = self.strong_arm(final) if strong else []
        self.run.generation_configs = self.choose_generation_configs(strong_rows)
        dev_done = time.perf_counter()
        for slug, config in self._configs.items():
            self.write("dev", config, self._dev_cache[slug])
        for slug, config in self._configs.items():
            results = run_config(self.ctx, config, self.questions["test"])
            self.write("test", config, to_records(config, "test", self.questions["test"], results))
            self.run.configs[slug] = {
                "name": config.name,
                "config": asdict(config),
                "role": "strong_arm" if config in strong_rows else "grid",
                "model": model_label(config),
            }
        self.run.compute = {
            "wall_s": round(time.perf_counter() - wall, 1),
            "cpu_s": round(time.process_time() - cpu, 1),
            "dev_selection_wall_s": round(dev_done - wall, 1),
            "timings_s": {k: round(v, 2) for k, v in self.ctx.timings.items()},
        }
        (self.out_dir / "selection.json").write_text(
            json.dumps(asdict(self.run), indent=2) + "\n", encoding="utf-8"
        )
        return self.run


def set_torch_threads(threads: int | None = None) -> int:
    """Pin torch's CPU threads. Defaults to RAG_TORCH_THREADS, else 2.

    Oversubscribed OpenMP threads spin and wait, so on a shared machine a
    single query can take 20 times longer with 4 threads than with 2.
    """
    import os

    import torch

    n = threads or int(os.environ.get("RAG_TORCH_THREADS", "2"))
    torch.set_num_threads(n)
    return n


def run_grid(out_dir: Path, cache_dir: Path | None, *, strong: bool) -> GridRun:
    set_torch_threads()
    ctx = GridContext(load_articles(), cache_dir=cache_dir)
    questions = {"dev": load_questions("dev"), "test": load_questions("test")}
    out_dir.mkdir(parents=True, exist_ok=True)
    return Grid(ctx, out_dir, questions).run_all(strong=strong)


def load_selection(out_dir: Path) -> dict[str, Any]:
    return json.loads((out_dir / "selection.json").read_text(encoding="utf-8"))


def load_split_records(out_dir: Path, split: Split) -> dict[str, list[EvalRecord]]:
    """{slug: records} for every config file in results/retrieval/<split>/."""
    return {p.stem: read_records(p) for p in sorted((out_dir / split).glob("*.jsonl"))}
