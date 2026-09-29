"""Real models at their pinned revisions. Needs downloads, so it only runs with -m download."""

from __future__ import annotations

import numpy as np
import pytest

from rag_support_assistant.embeddings import SentenceTransformerEmbedding
from rag_support_assistant.models import EMBEDDINGS, RERANKERS
from rag_support_assistant.rerank import load_cross_encoder

pytestmark = pytest.mark.download


def test_bge_small_embeds_normalized_vectors_and_ranks_sensibly():
    model = SentenceTransformerEmbedding(EMBEDDINGS["bge-small"])
    q = np.array(model.get_query_embedding("How much does the Plus plan cost?"))
    good = np.array(model.get_text_embedding("Plus costs $5.00 a month."))
    bad = np.array(model.get_text_embedding("Wires can't be cancelled once sent."))
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-4)
    assert q @ good > q @ bad


def test_granite_reranker_prefers_the_relevant_passage():
    score = load_cross_encoder(RERANKERS["granite-rerank"])
    s = score(
        "How much does the Plus plan cost?",
        ["Plus costs $5.00 a month.", "Wires can't be cancelled."],
    )
    assert s[0] > s[1]


def test_hhem_rebuild_matches_the_model_card():
    from rag_support_assistant.hhem import HHEM

    pairs = [
        ("The capital of France is Berlin.", "The capital of France is Paris."),
        ("I am in California", "I am in United States."),
        ("I am in United States", "I am in California."),
        (
            "A person on a horse jumps over a broken down airplane.",
            "A person is outdoors, on a horse.",
        ),
        (
            "A boy is jumping on skateboard in the middle of a red bridge.",
            "The boy skates down the sidewalk on a red bridge",
        ),
        (
            "A man with blond-hair, and a brown shirt drinking out of a public water fountain.",
            "A blond man wearing a brown shirt is reading a book.",
        ),
        ("Mark Wahlberg was a fan of Manny.", "Manny was a fan of Mark Wahlberg."),
    ]
    card = [0.0111, 0.6474, 0.1290, 0.8969, 0.1846, 0.0050, 0.0543]  # printed on the model card
    assert HHEM().scores(pairs) == pytest.approx(card, abs=1e-4)
