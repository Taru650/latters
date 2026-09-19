"""Retrieval: find the 3-5 past letters a new request should be drafted from.

Built cheapest-first, and deliberately so. The plan assumed a neural encoder
was needed; Phase 4 measures whether it is, because on this hardware every
dependency costs RAM the LLM needs and every model costs a download the
target office cannot do.

Three scorers, all optional and all composable:

* **BM25** via SQLite FTS5. Already present from Phase 2, costs nothing.
* **TF-IDF over character n-grams**, pure numpy. No model, no download,
  ~2 MB of vectors for 547 letters. Character n-grams matter here because
  this corpus spells स्थानान्तरण and स्थानांतरण interchangeably, so word
  features miss matches that character features catch.
* **A dense encoder**, behind the `Encoder` protocol, for when a neural model
  earns its place. Nothing in this module imports onnxruntime.

Fused with Reciprocal Rank Fusion, which needs no score calibration -- BM25
scores and cosine similarities are not comparable quantities, and any attempt
to weight them directly is a tuning parameter with no principled value.

Metadata filters run BEFORE scoring, not after. In a departmental corpus a
hard department filter does more for precision than any scoring change, and
filtering first also means the scorers only see candidates that could
plausibly be right.
"""

from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence

import numpy as np

_WS = re.compile(r"\s+")
#: Devanagari punctuation lives INSIDE the Devanagari block: danda
#: U+0964, double danda U+0965, the abbreviation sign U+0970, the high
#: spacing dot U+0971. A naive [\u0900-\u097F] word class therefore makes
#: "।" a searchable term that matches every document in the corpus.
_DEVA_WORD = re.compile(r"[ऀ-ॣ०-९ॲ-ॿA-Za-z0-9]+")


class Encoder(Protocol):
    """Anything that turns text into unit-norm vectors.

    Deliberately minimal so a Granite/E5 ONNX session, a stub, or a test
    double all satisfy it without this module knowing they exist.
    """

    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class Letter:
    id: int
    text: str
    subject: str | None = None
    department: str | None = None
    letter_type: str | None = None
    trust: float = 0.0
    source_file: str = ""


@dataclass
class Filters:
    department: str | None = None
    letter_type: str | None = None
    min_trust: float = 0.0
    exclude_ids: frozenset[int] = frozenset()

    def keep(self, l: Letter) -> bool:
        if l.id in self.exclude_ids:
            return False
        if l.trust < self.min_trust:
            return False
        if self.department and l.department != self.department:
            return False
        if self.letter_type and l.letter_type != self.letter_type:
            return False
        return True


@dataclass
class Hit:
    letter: Letter
    score: float
    #: Rank in each contributing scorer, for explaining the result to a user.
    ranks: dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------
def char_ngrams(text: str, lo: int = 3, hi: int = 5, limit: int = 1500) -> list[str]:
    t = _WS.sub(" ", text)[:limit]
    return [t[i:i + n] for n in range(lo, hi + 1) for i in range(len(t) - n + 1)]


class TfidfIndex:
    """Character n-gram TF-IDF with cosine similarity, on an inverted index.

    Standard lnc.ltc weighting: sublinear term frequency, inverse document
    frequency, L2 normalisation. Grams are hashed into a fixed bucket space
    so the vocabulary cannot grow without bound.

    THE DATA STRUCTURE IS THE POINT. A dense documents x buckets matrix is
    71 MB for 547 letters and 655 MB for 5,000 -- unusable on a machine with
    3 GB free. The same information as an inverted index is ~6 MB for 547 and
    ~56 MB for 5,000, because each letter touches only a couple of thousand
    of the 32,768 buckets. Query cost drops too: scoring touches only the
    buckets the query actually contains, not all of them.
    """

    def __init__(self, *, lo: int = 3, hi: int = 5, buckets: int = 2 ** 15):
        self.lo, self.hi, self.buckets = lo, hi, buckets
        self.ids: list[int] = []
        self.idf: np.ndarray | None = None
        # Postings sorted by bucket: parallel arrays plus a start offset per
        # bucket. This is a CSC matrix without the scipy dependency.
        self._doc_of_posting: np.ndarray | None = None
        self._weight_of_posting: np.ndarray | None = None
        self._bucket_start: np.ndarray | None = None

    @property
    def matrix(self):
        """Present so callers can test 'is the index built'."""
        return self._bucket_start

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self._doc_of_posting, self._weight_of_posting,
                                      self._bucket_start, self.idf) if a is not None)

    def _bucket(self, gram: str) -> int:
        # FNV-1a: deterministic across processes, unlike hash().
        h = 2166136261
        for ch in gram:
            h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
        return h % self.buckets

    def _term_counts(self, text: str) -> dict[int, int]:
        out: dict[int, int] = {}
        for g in char_ngrams(text, self.lo, self.hi):
            b = self._bucket(g)
            out[b] = out.get(b, 0) + 1
        return out

    @staticmethod
    def _doc_text(l: Letter) -> str:
        # The subject is the strongest single signal, so give it weight
        # without discarding the body's context.
        return ((l.subject or "") + " ") * 2 + l.text

    def fit(self, letters: Sequence[Letter]) -> "TfidfIndex":
        self.ids = [l.id for l in letters]
        n = max(len(letters), 1)

        per_doc: list[dict[int, int]] = [self._term_counts(self._doc_text(l))
                                         for l in letters]
        df = np.zeros(self.buckets, dtype=np.int32)
        for counts in per_doc:
            df[list(counts)] += 1
        self.idf = (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)

        buckets_l: list[int] = []
        docs_l: list[int] = []
        weights_l: list[float] = []
        for doc_index, counts in enumerate(per_doc):
            if not counts:
                continue
            b = np.fromiter(counts.keys(), dtype=np.int32, count=len(counts))
            c = np.fromiter(counts.values(), dtype=np.float32, count=len(counts))
            w = (1.0 + np.log(c)) * self.idf[b]
            norm = float(np.linalg.norm(w))
            if norm <= 0:
                continue
            w /= norm
            buckets_l.append(b)
            docs_l.append(np.full(len(b), doc_index, dtype=np.int32))
            weights_l.append(w.astype(np.float32))

        if not buckets_l:
            self._bucket_start = np.zeros(self.buckets + 1, dtype=np.int64)
            self._doc_of_posting = np.zeros(0, dtype=np.int32)
            self._weight_of_posting = np.zeros(0, dtype=np.float32)
            return self

        all_buckets = np.concatenate(buckets_l)
        all_docs = np.concatenate(docs_l)
        all_weights = np.concatenate(weights_l)
        order = np.argsort(all_buckets, kind="stable")
        all_buckets = all_buckets[order]
        self._doc_of_posting = all_docs[order]
        self._weight_of_posting = all_weights[order]
        counts_per_bucket = np.bincount(all_buckets, minlength=self.buckets)
        self._bucket_start = np.zeros(self.buckets + 1, dtype=np.int64)
        np.cumsum(counts_per_bucket, out=self._bucket_start[1:])
        return self

    def query(self, text: str) -> np.ndarray:
        """Cosine similarity of `text` against every indexed letter."""
        scores = np.zeros(len(self.ids), dtype=np.float32)
        if self._bucket_start is None or self.idf is None:
            return scores
        counts = self._term_counts(text)
        if not counts:
            return scores
        b = np.fromiter(counts.keys(), dtype=np.int32, count=len(counts))
        c = np.fromiter(counts.values(), dtype=np.float32, count=len(counts))
        q = (1.0 + np.log(c)) * self.idf[b]
        norm = float(np.linalg.norm(q))
        if norm <= 0:
            return scores
        q /= norm
        for bucket, qw in zip(b.tolist(), q.tolist()):
            lo, hi = self._bucket_start[bucket], self._bucket_start[bucket + 1]
            if hi > lo:
                np.add.at(scores, self._doc_of_posting[lo:hi],
                          qw * self._weight_of_posting[lo:hi])
        return scores


class DenseIndex:
    """Brute-force cosine over encoder vectors.

    At under ~50k letters this is a matmul of a few milliseconds, so FAISS
    would add a dependency, a build step and an index-corruption failure mode
    to save nothing. Revisit past ~200k.
    """

    def __init__(self, encoder: Encoder):
        self.encoder = encoder
        self.matrix: np.ndarray | None = None
        self.ids: list[int] = []

    def fit(self, letters: Sequence[Letter], *, batch: int = 32) -> "DenseIndex":
        self.ids = [l.id for l in letters]
        texts = [TfidfIndex._doc_text(l) for l in letters]
        chunks = [self.encoder.encode(texts[i:i + batch])
                  for i in range(0, len(texts), batch)]
        m = np.vstack(chunks).astype(np.float32) if chunks else np.zeros((0, self.encoder.dim), np.float32)
        self.matrix = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
        return self

    def query(self, text: str) -> np.ndarray:
        assert self.matrix is not None
        q = self.encoder.encode([text])[0].astype(np.float32)
        q /= max(float(np.linalg.norm(q)), 1e-9)
        return self.matrix @ q


# --------------------------------------------------------------------------
def rrf(rankings: dict[str, Sequence[int]], *, k: int = 60,
        weights: dict[str, float] | None = None) -> dict[int, float]:
    """Reciprocal Rank Fusion.

    Uses ranks, not scores, which is the point: BM25 scores and cosine
    similarities live on incomparable scales, and normalising them introduces
    a tuning parameter with no principled value. k=60 is the constant from
    the original paper and is not sensitive.
    """
    out: dict[int, float] = {}
    for name, ids in rankings.items():
        w = (weights or {}).get(name, 1.0)
        for rank, doc_id in enumerate(ids, start=1):
            out[doc_id] = out.get(doc_id, 0.0) + w / (k + rank)
    return out


_FTS_SPECIAL = re.compile(r'["\'()*:^-]')


def fts_query(text: str, *, max_terms: int = 24) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    Unquoted user text is a syntax error waiting to happen: a stray colon,
    quote or hyphen makes FTS5 raise, and in Devanagari punctuation those
    appear constantly. Every term is quoted and OR-joined.
    """
    terms = [t for t in _DEVA_WORD.findall(text) if len(t) > 1][:max_terms]
    if not terms:
        return ""
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


class Retriever:
    def __init__(self, db: sqlite3.Connection, letters: Sequence[Letter],
                 *, tfidf: TfidfIndex | None = None, dense: DenseIndex | None = None):
        self.db = db
        self.letters = list(letters)
        self.by_id = {l.id: l for l in self.letters}
        self.tfidf = tfidf
        self.dense = dense

    # --- individual scorers ----------------------------------------------
    def bm25(self, query: str, filters: Filters, limit: int) -> list[int]:
        expr = fts_query(query)
        if not expr:
            return []
        try:
            rows = self.db.execute(
                """SELECT letters_fts.rowid AS id FROM letters_fts
                   JOIN letters l ON l.id = letters_fts.rowid
                   WHERE letters_fts MATCH ? ORDER BY bm25(letters_fts) LIMIT ?""",
                (expr, limit * 8)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [r["id"] for r in rows
                if r["id"] in self.by_id and filters.keep(self.by_id[r["id"]])][:limit]

    def _vector(self, index, query: str, filters: Filters, limit: int) -> list[int]:
        if index is None or index.matrix is None or not len(index.ids):
            return []
        sims = index.query(query)
        order = np.argsort(-sims)
        out = []
        for i in order:
            lid = index.ids[int(i)]
            if filters.keep(self.by_id[lid]):
                out.append(lid)
                if len(out) >= limit:
                    break
        return out

    # --- fusion ------------------------------------------------------------
    def search(self, query: str, *, filters: Filters | None = None, limit: int = 5,
               use: Iterable[str] = ("bm25", "tfidf", "dense"),
               weights: dict[str, float] | None = None) -> list[Hit]:
        filters = filters or Filters()
        use = set(use)
        pool = limit * 6

        rankings: dict[str, list[int]] = {}
        if "bm25" in use:
            r = self.bm25(query, filters, pool)
            if r:
                rankings["bm25"] = r
        if "tfidf" in use and self.tfidf is not None:
            r = self._vector(self.tfidf, query, filters, pool)
            if r:
                rankings["tfidf"] = r
        if "dense" in use and self.dense is not None:
            r = self._vector(self.dense, query, filters, pool)
            if r:
                rankings["dense"] = r
        if not rankings:
            return []

        fused = rrf(rankings, weights=weights)
        ordered = sorted(fused, key=lambda i: -fused[i])[:limit]
        hits = []
        for lid in ordered:
            ranks = {name: ids.index(lid) + 1
                     for name, ids in rankings.items() if lid in ids}
            hits.append(Hit(self.by_id[lid], fused[lid], ranks))
        return hits


# --------------------------------------------------------------------------
def load_letters(db: sqlite3.Connection, *, min_trust: float = 0.0) -> list[Letter]:
    rows = db.execute(
        "SELECT id, text, subject, department, letter_type, trust, source_file "
        "FROM letters WHERE trust >= ?", (min_trust,)).fetchall()
    return [Letter(id=r["id"], text=r["text"], subject=r["subject"],
                   department=r["department"], letter_type=r["letter_type"],
                   trust=r["trust"] or 0.0, source_file=r["source_file"])
            for r in rows]
