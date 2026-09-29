"""Answer generation: structured output with citations and an explicit abstain flag.

The answer model is `gpt-6-luna` with reasoning effort "none". The Responses
API enforces the JSON schema (`strict: true`), so a reply is an object with
`answer`, `citations` (article IDs) and `abstain`.

Two ways to build the prompt:

- RAG: the top chunks a retrieval config returned, frozen in
  results/contexts/ so a replay sends byte-identical requests.
- Full context: every article in the corpus as one static prefix, question
  last, so the provider's prompt cache serves the prefix after the first call.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Any

from llm_eval_harness import ModelRequest

from rag_support_assistant.data import Article

ANSWER_MODEL = "gpt-6-luna"
ANSWER_EFFORT = "none"
ANSWER_MAX_TOKENS = 600
CONTEXT_CHUNKS = 8

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "abstain": {"type": "boolean"},
    },
    "required": ["answer", "citations", "abstain"],
    "additionalProperties": False,
}
TEXT_FORMAT: dict[str, Any] = {
    "format": {
        "type": "json_schema",
        "name": "support_answer",
        "schema": ANSWER_SCHEMA,
        "strict": True,
    }
}


def instructions() -> str:
    return (
        resources.files("rag_support_assistant") / "prompts" / "answer_instructions.txt"
    ).read_text(encoding="utf-8")


@dataclass(frozen=True)
class ContextChunk:
    chunk_id: str
    article_id: str
    title: str
    effective_date: str
    text: str


def format_chunks(chunks: Sequence[ContextChunk]) -> str:
    return "\n\n".join(
        f"[{c.article_id}] {c.title} (effective {c.effective_date})\n{c.text.strip()}"
        for c in chunks
    )


def format_corpus(articles: Sequence[Article]) -> str:
    return "\n\n".join(
        f"[{a.article_id}] {a.title} (effective {a.effective_date})\n{a.body.strip()}"
        for a in articles
    )


def rag_request(question: str, chunks: Sequence[ContextChunk]) -> ModelRequest:
    prompt = f"Help-center excerpts:\n\n{format_chunks(chunks)}\n\nCustomer question:\n{question}"
    return ModelRequest(
        model=ANSWER_MODEL,
        instructions=instructions(),
        input=[{"role": "user", "content": prompt}],
        max_output_tokens=ANSWER_MAX_TOKENS,
        reasoning_effort=ANSWER_EFFORT,
        extra={"text": TEXT_FORMAT},
    )


def full_context_request(question: str, corpus_text: str) -> ModelRequest:
    """Corpus first as a developer message that is identical on every call, question last."""
    prefix = f"{instructions()}\nHelp-center articles (the whole help center):\n\n{corpus_text}"
    return ModelRequest(
        model=ANSWER_MODEL,
        input=[
            {"role": "developer", "content": prefix},
            {"role": "user", "content": f"Customer question:\n{question}"},
        ],
        max_output_tokens=ANSWER_MAX_TOKENS,
        reasoning_effort=ANSWER_EFFORT,
        extra={"text": TEXT_FORMAT},
    )


class AnswerParseError(ValueError):
    """The model's reply is not the structured answer the schema asks for."""


@dataclass(frozen=True)
class Answer:
    answer: str
    citations: tuple[str, ...]
    abstain: bool


def parse_answer(text: str) -> Answer:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AnswerParseError(f"reply is not JSON ({exc.msg})") from exc
    if not isinstance(data, dict) or set(data) != {"answer", "citations", "abstain"}:
        raise AnswerParseError("reply must have exactly answer, citations and abstain")
    answer, citations, abstain = data["answer"], data["citations"], data["abstain"]
    if not isinstance(answer, str) or not isinstance(abstain, bool):
        raise AnswerParseError("answer must be a string and abstain a boolean")
    if not isinstance(citations, list) or not all(isinstance(c, str) for c in citations):
        raise AnswerParseError("citations must be a list of strings")
    return Answer(answer=answer.strip(), citations=tuple(citations), abstain=abstain)
