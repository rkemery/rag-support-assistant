from __future__ import annotations

import math

import pytest

from rag_support_assistant.bm25 import BM25Encoder, tokenize


def test_tokenize_keeps_amounts_and_times_and_stems():
    assert tokenize("What's the ATM fee? $2.50 at 4:00 PM, $3,000 withdrawals") == [
        "atm", "fee", "2.50", "4:00", "pm", "3,000", "withdraw",
    ]  # fmt: skip


def test_score_matches_the_bm25_formula():
    docs = ["plus plan fee is five dollars", "basic plan is free", "wire fee"]
    enc = BM25Encoder(k1=1.2, b=0.75).fit(docs)
    n, avgdl = 3, sum(len(tokenize(d)) for d in docs) / 3
    df_fee, df_plan = 2, 2
    tokens = tokenize(docs[0])
    norm = 1.2 * (1 - 0.75 + 0.75 * len(tokens) / avgdl)

    def idf(df: int) -> float:
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    expected = sum(idf(df) * 1 * 2.2 / (1 + norm) for df in (df_fee, df_plan))
    assert enc.score("plan fee", docs[0]) == pytest.approx(expected)


def test_unknown_query_terms_are_dropped_and_idf_is_positive():
    enc = BM25Encoder().fit(["alpha beta", "alpha gamma", "alpha delta"])
    (idx,), (vals,) = enc.encode_queries(["zeta alpha"])
    assert len(idx) == 1
    assert vals == [1.0]
    assert all(v > 0 for v in enc.idf)


def test_encoding_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        BM25Encoder().encode_queries(["x"])
