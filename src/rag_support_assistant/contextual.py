"""One contextual-retrieval cell: an LLM-written context per chunk, then the same retrieval.

Live step: gpt-6-luna writes one short context per chunk of the chosen
chunking, from the whole article and the chunk, with the prompt Anthropic
published with contextual retrieval (September 2024). Contexts are saved to
results/contextual/contexts.jsonl.

Offline step: the chosen retrieval config runs again with each chunk's context
prepended for both the dense and BM25 encoders, on dev and test, and is
compared with the same config without contexts, paired on test.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llm_eval_harness import ModelClient, ModelRequest, write_records
from llm_eval_harness.client import DEFAULT_PRICES, cost_usd

from rag_support_assistant import analysis
from rag_support_assistant.chunking import chunk_articles
from rag_support_assistant.data import REPO_ROOT, Article, load_articles, load_questions
from rag_support_assistant.retrieval import RetrievalConfig

CONTEXT_MODEL = "gpt-6-luna"
CONTEXT_MAX_TOKENS = 150
PROMPT = (
    "<document>\n{document}\n</document>\n"
    "Here is the chunk we want to situate within the whole document\n"
    "<chunk>\n{chunk}\n</chunk>\n"
    "Please give a short succinct context to situate this chunk within the overall document "
    "for the purposes of improving search retrieval of the chunk. Answer only with the "
    "succinct context and nothing else."
)


def document_text(article: Article) -> str:
    return f"# {article.title}\n\n{article.body}"


def context_request(document: str, chunk: str) -> ModelRequest:
    return ModelRequest(
        model=CONTEXT_MODEL,
        input=PROMPT.format(document=document, chunk=chunk),
        max_output_tokens=CONTEXT_MAX_TOKENS,
        reasoning_effort="none",
    )


def chosen_config(results: Path) -> RetrievalConfig:
    selection = json.loads((results / "retrieval" / "selection.json").read_text(encoding="utf-8"))
    final = selection["steps"][-1]["chosen"]
    return RetrievalConfig(**selection["configs"][final]["config"])


def contextualize(client: ModelClient, results: Path) -> Path:
    """Write one context per chunk of the chosen chunking. Live or replay."""
    config = chosen_config(results)
    articles = {a.article_id: a for a in load_articles()}
    nodes = chunk_articles(list(articles.values()), config.chunking)
    price = DEFAULT_PRICES[CONTEXT_MODEL]
    rows = []
    for node in nodes:
        article = articles[node.metadata["article_id"]]
        request = context_request(document_text(article), node.get_content())
        response = client.complete(request)
        rows.append(
            {
                "chunk_id": node.node_id,
                "article_id": article.article_id,
                "context": response.text.strip(),
                "tokens_in": response.input_tokens,
                "tokens_out": response.output_tokens,
                "cost_usd": cost_usd(price, response),
            }
        )
    path = results / "contextual" / "contexts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def run_cell(results: Path) -> dict[str, Any]:
    """Offline: score the chosen config with and without contexts. Needs contexts.jsonl."""
    from rag_support_assistant.grid import GridContext, run_config, to_records

    path = results / "contextual" / "contexts.jsonl"
    with path.open(encoding="utf-8") as fh:
        contexts = {row["chunk_id"]: row["context"] for row in map(json.loads, fh)}
    base = chosen_config(results)
    cell = base.with_(contextual=True)
    ctx = GridContext(
        load_articles(), cache_dir=REPO_ROOT / ".cache" / "embeddings", contexts=contexts
    )
    records = {}
    for split in ("dev", "test"):
        questions = [q for q in load_questions(split) if q.has_gold]
        for config in (base, cell):
            recs = to_records(config, split, questions, run_config(ctx, config, questions))
            write_records(results / "contextual" / split / f"{config.slug}.jsonl", recs)
            records[(split, config.slug)] = recs
    test_base, test_cell = records[("test", base.slug)], records[("test", cell.slug)]
    ci = analysis.cluster_bootstrap(test_cell, "ndcg@10").interval
    diff = analysis.paired(test_base, test_cell, "ndcg@10").comparison
    summary = (
        f"{ci.estimate:.3f} ({ci.low:.3f} to {ci.high:.3f}), "
        f"{diff.diff:+.3f} vs no context ({diff.low:+.3f} to {diff.high:+.3f}, p={diff.pvalue:.3f})"
    )
    info = {"name": cell.name, "config": asdict(cell), "summary": summary}
    (results / "contextual" / "selection.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    return info
