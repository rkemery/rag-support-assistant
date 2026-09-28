"""BM25 as Qdrant sparse vectors, so BM25 and hybrid search run inside Qdrant.

Each chunk gets a sparse vector whose value for term t is its full BM25 term
weight, IDF included:

    w(t, d) = idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
    idf(t)  = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

A query's sparse vector holds its term counts, so the dot product Qdrant
computes is exactly the BM25 score (Robertson and Zaragoza 2009, with the
non-negative Lucene IDF). k1 = 1.2 and b = 0.75 are the usual defaults.
Tokens are lowercased words and numbers (so "$2.50" matches "2.50"), NLTK
English stopwords are dropped, and words are Porter-stemmed.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache

from nltk.stem import PorterStemmer

TOKEN_PATTERN = r"[a-z0-9]+(?:[.,:][0-9]+)*"
_TOKEN = re.compile(TOKEN_PATTERN)
# NLTK's English stopword list (the 179-word version), inlined so no corpus download or
# file lookup is needed. Newer NLTK lists add contractions such as "we're", which never
# reach this list because the tokenizer splits on apostrophes.
STOPWORDS = frozenset(
    """
    i me my myself we our ours ourselves you you're you've you'll you'd your yours yourself
    yourselves he him his himself she she's her hers herself it it's its itself they them
    their theirs themselves what which who whom this that that'll these those am is are was
    were be been being have has had having do does did doing a an the and but if or because as
    until while of at by for with about against between into through during before after above
    below to from up down in out on off over under again further then once here there when
    where why how all any both each few more most other some such no nor not only own same so
    than too very s t can will just don don't should should've now d ll m o re ve y ain aren
    aren't couldn couldn't didn didn't doesn doesn't hadn hadn't hasn hasn't haven haven't isn
    isn't ma mightn mightn't mustn mustn't needn needn't shan shan't shouldn shouldn't wasn
    wasn't weren weren't won won't wouldn wouldn't
    """.split()  # noqa: SIM905 (a word list reads better as text)
)

_stemmer = PorterStemmer()


@lru_cache(maxsize=65536)
def _stem(word: str) -> str:
    return _stemmer.stem(word)


def tokenize(text: str) -> list[str]:
    words = _TOKEN.findall(text.lower().replace("\u2019", "'"))
    return [_stem(w) for w in words if w not in STOPWORDS]


class BM25Encoder:
    """Fit on the chunk texts once, then encode documents and queries as sparse vectors.

    The call signatures match what `QdrantVectorStore` expects for
    `sparse_doc_fn` and `sparse_query_fn`: a list of texts in, a pair of lists
    (indices per text, values per text) out.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError(f"need k1 >= 0 and 0 <= b <= 1, got k1={k1}, b={b}")
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.idf: list[float] = []
        self.avgdl = 0.0
        self.n_docs = 0

    def fit(self, texts: Sequence[str]) -> BM25Encoder:
        if not texts:
            raise ValueError("cannot fit BM25 on an empty corpus")
        docs = [tokenize(t) for t in texts]
        df: Counter[str] = Counter()
        for tokens in docs:
            df.update(set(tokens))
        self.n_docs = len(docs)
        self.avgdl = sum(len(d) for d in docs) / self.n_docs
        self.vocab = {term: i for i, term in enumerate(sorted(df))}
        n = self.n_docs
        self.idf = [math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in sorted(df)]
        return self

    def _check_fitted(self) -> None:
        if not self.vocab:
            raise RuntimeError("BM25Encoder.fit must be called before encoding")

    def encode_documents(self, texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
        self._check_fitted()
        indices: list[list[int]] = []
        values: list[list[float]] = []
        for text in texts:
            tokens = tokenize(text)
            norm = self.k1 * (1 - self.b + self.b * len(tokens) / self.avgdl)
            idx, val = [], []
            for term, tf in sorted(Counter(tokens).items()):
                i = self.vocab.get(term)
                if i is None:  # a term the fitted corpus never saw carries no IDF
                    continue
                idx.append(i)
                val.append(self.idf[i] * tf * (self.k1 + 1) / (tf + norm))
            indices.append(idx)
            values.append(val)
        return indices, values

    def encode_queries(self, texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
        self._check_fitted()
        indices: list[list[int]] = []
        values: list[list[float]] = []
        for text in texts:
            counts = Counter(t for t in tokenize(text) if t in self.vocab)
            pairs = sorted((self.vocab[t], float(c)) for t, c in counts.items())
            indices.append([i for i, _ in pairs])
            values.append([v for _, v in pairs])
        return indices, values

    def score(self, query: str, document: str) -> float:
        """BM25 score of one pair, the same number Qdrant's dot product gives. For tests."""
        (qi,), (qv,) = self.encode_queries([query])
        (di,), (dv,) = self.encode_documents([document])
        doc = dict(zip(di, dv, strict=True))
        return sum(v * doc.get(i, 0.0) for i, v in zip(qi, qv, strict=True))
