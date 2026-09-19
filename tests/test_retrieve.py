import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latters.encoders import RandomProjectionEncoder
from latters.evaluate import (Result, build_queries, degrade, evaluate,
                              random_baseline, render)
from latters.retrieve import (DenseIndex, Filters, Letter, Retriever,
                              TfidfIndex, char_ngrams, fts_query, rrf)
from latters.store import LetterRow, Store

SUBJECTS = [
    "भूमि दाखिल खारिज की जाँच के संबंध में",
    "अनुशासनिक कार्यवाही हेतु कारण बताओ सूचना",
    "मासिक समीक्षा बैठक की सूचना",
    "वेतन भुगतान की स्वीकृति के संबंध में",
    "सेवानिवृत्ति उपादान के भुगतान हेतु",
    "अतिक्रमण हटाने के संबंध में प्रतिवेदन",
]


def _letters(n_per=3):
    out, lid = [], 1
    for i, s in enumerate(SUBJECTS):
        for j in range(n_per):
            out.append(Letter(
                id=lid, subject=s,
                text=f"कार्यालय जिला शाखा\nविषय:- {s}।\nमहाशय,\n"
                     f"{s} के प्रसंग में आवश्यक कार्यवाही हेतु प्रति संख्या {j} "
                     f"प्रेषित की जा रही है।\nविश्वासभाजन",
                department=f"dept{i % 2}", letter_type=f"type{i}", trust=0.9))
            lid += 1
    return out


def _store(letters):
    s = Store()
    s.add([LetterRow(source_file="f.docx", seq=l.id, text=l.text, trust=l.trust,
                     subject=l.subject) for l in letters])
    # Align the in-memory ids with the rowids the store assigned.
    rows = s.db.execute("SELECT id, subject FROM letters ORDER BY id").fetchall()
    for l, r in zip(letters, rows):
        l.id = r["id"]
    return s


# --- TF-IDF ---------------------------------------------------------------
def test_char_ngrams_span_the_range():
    g = char_ngrams("कार्यालय", 3, 5)
    assert {len(x) for x in g} == {3, 4, 5}


def test_identical_text_scores_near_one():
    letters = _letters()
    idx = TfidfIndex().fit(letters)
    sims = idx.query(TfidfIndex._doc_text(letters[0]))
    assert sims[0] == pytest.approx(1.0, abs=1e-4)


def test_similar_beats_dissimilar():
    letters = _letters()
    idx = TfidfIndex().fit(letters)
    sims = idx.query("भूमि दाखिल खारिज की जाँच")
    best = int(np.argmax(sims))
    assert "भूमि" in letters[best].subject


def test_index_is_sparse_not_dense():
    """Regression: a dense documents x buckets matrix was 71 MB for 547
    letters and would be 655 MB for 5,000 -- unusable at 3 GB free."""
    letters = _letters(n_per=6)
    idx = TfidfIndex().fit(letters)
    dense_equivalent = len(letters) * idx.buckets * 4
    assert idx.nbytes < dense_equivalent / 5


def test_query_produces_no_nan():
    """Regression: np.where(x>0, log(x, where=x>0), 0) reads uninitialised
    memory and produced NaN scores."""
    idx = TfidfIndex().fit(_letters())
    for q in ["", "x", "भूमि", "क" * 500, "!!!", "123"]:
        sims = idx.query(q)
        assert not np.isnan(sims).any() and not np.isinf(sims).any()


def test_hashing_is_stable_across_instances():
    """Python's hash() is salted per process; the index must not be."""
    a, b = TfidfIndex(), TfidfIndex()
    assert a._bucket("कार्यालय") == b._bucket("कार्यालय")


def test_empty_corpus_does_not_crash():
    idx = TfidfIndex().fit([])
    assert idx.query("कुछ भी").shape == (0,)


# --- FTS query escaping ---------------------------------------------------
@pytest.mark.parametrize("raw", [
    'विषय: भूमि "विवाद" के संबंध में',
    "पत्रांक-802/स्था०, दिनांक 22.05.2025",
    "NOT AND OR ( ) * ^ :",
    "अनु०ः-यथोपरि।",
    "",
])
def test_fts_query_never_produces_a_syntax_error(raw):
    """Unquoted user text is a syntax error waiting to happen: a stray colon,
    quote or hyphen makes FTS5 raise, and Devanagari punctuation is full of
    them."""
    letters = _letters()
    with _store(letters) as s:
        expr = fts_query(raw)
        if expr:
            s.db.execute("SELECT rowid FROM letters_fts WHERE letters_fts MATCH ?",
                         (expr,)).fetchall()


def test_fts_query_is_empty_for_empty_input():
    assert fts_query("") == "" and fts_query("।।।") == ""


# --- RRF ------------------------------------------------------------------
def test_rrf_rewards_agreement():
    fused = rrf({"a": [1, 2, 3], "b": [1, 3, 2]})
    assert max(fused, key=fused.get) == 1


def test_rrf_uses_ranks_not_scores():
    """Two runs with identical ranks must fuse identically however the
    underlying scores were scaled."""
    assert rrf({"a": [7, 8]}) == rrf({"b": [7, 8]})


def test_rrf_weights_shift_the_order():
    a = rrf({"x": [1, 2], "y": [2, 1]})
    b = rrf({"x": [1, 2], "y": [2, 1]}, weights={"y": 5.0})
    assert max(a, key=a.get) == 1 and max(b, key=b.get) == 2


# --- filters --------------------------------------------------------------
def test_filters_compose():
    l = Letter(id=1, text="t", department="A", letter_type="x", trust=0.8)
    assert Filters().keep(l)
    assert not Filters(department="B").keep(l)
    assert not Filters(letter_type="y").keep(l)
    assert not Filters(min_trust=0.9).keep(l)
    assert not Filters(exclude_ids=frozenset({1})).keep(l)


def test_filters_run_before_scoring():
    letters = _letters()
    with _store(letters) as s:
        r = Retriever(s.db, letters, tfidf=TfidfIndex().fit(letters))
        hits = r.search("भूमि दाखिल खारिज", filters=Filters(department="dept1"), limit=5)
        assert hits and all(h.letter.department == "dept1" for h in hits)


def test_search_reports_which_scorer_ranked_what():
    letters = _letters()
    with _store(letters) as s:
        r = Retriever(s.db, letters, tfidf=TfidfIndex().fit(letters))
        hits = r.search("अनुशासनिक कार्यवाही", limit=3)
        assert hits and any(h.ranks for h in hits)


def test_search_with_no_indexes_returns_nothing_rather_than_raising():
    letters = _letters()
    with _store(letters) as s:
        assert Retriever(s.db, letters).search("कुछ", use=("tfidf", "dense")) == []


# --- dense path -----------------------------------------------------------
def test_random_projection_encoder_is_a_valid_encoder():
    enc = RandomProjectionEncoder(dim=64)
    v = enc.encode(["कार्यालय ज्ञापन", "भूमि विवाद"])
    assert v.shape == (2, 64)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_encoder_is_deterministic():
    a = RandomProjectionEncoder(dim=32).encode(["भूमि"])
    b = RandomProjectionEncoder(dim=32).encode(["भूमि"])
    assert np.allclose(a, b)


def test_dense_index_round_trips():
    letters = _letters()
    idx = DenseIndex(RandomProjectionEncoder(dim=64)).fit(letters)
    sims = idx.query(TfidfIndex._doc_text(letters[0]))
    assert int(np.argmax(sims)) == 0


def test_dense_fuses_with_lexical():
    letters = _letters()
    with _store(letters) as s:
        r = Retriever(s.db, letters, tfidf=TfidfIndex().fit(letters),
                      dense=DenseIndex(RandomProjectionEncoder(dim=64)).fit(letters))
        hits = r.search("भूमि दाखिल खारिज", limit=3, use=("bm25", "tfidf", "dense"))
        assert hits and len(hits[0].ranks) >= 2


# --- evaluation -----------------------------------------------------------
def test_degrade_drops_words_deterministically():
    s = "भूमि दाखिल खारिज की जाँच के संबंध में प्रतिवेदन"
    a, b = degrade(s, keep=0.5, seed=1), degrade(s, keep=0.5, seed=1)
    assert a == b
    assert 0 < len(a.split()) < len(s.split())


def test_degrade_leaves_very_short_text_alone():
    assert degrade("भूमि विवाद", keep=0.5) == "भूमि विवाद"


def test_build_queries_needs_cell_mates():
    letters = _letters(n_per=2)
    assert build_queries(letters, min_cell=5) == []
    assert build_queries(_letters(n_per=6), min_cell=5)


def test_query_excludes_itself_from_its_cell_mates():
    q = build_queries(_letters(n_per=6), min_cell=5)[0]
    assert q.letter_id not in q.cell_mates


def test_retrieval_beats_random_by_a_lot():
    letters = _letters(n_per=6)
    with _store(letters) as s:
        r = Retriever(s.db, letters, tfidf=TfidfIndex().fit(letters))
        Q = build_queries(letters, min_cell=5)
        got = evaluate(r, Q, name="tfidf", use=("tfidf",))
        base = random_baseline(letters, Q)
        assert got.same_cell_p5 > base.same_cell_p5 + 4 * got.stderr(got.same_cell_p5)


def test_type_filter_is_flagged_as_a_degenerate_metric():
    """Regression: filtering by letter_type makes same-cell precision 1.000
    by construction. Reporting that as a result would be a fake win."""
    letters = _letters(n_per=6)
    with _store(letters) as s:
        r = Retriever(s.db, letters, tfidf=TfidfIndex().fit(letters))
        Q = build_queries(letters, min_cell=5)
        res = evaluate(r, Q, name="t", use=("tfidf",), filter_type=True)
        assert res.cell_metric_degenerate
        assert "n/a" in render([res])
        assert "1.000" not in render([res]).split("n/a")[0]


def test_render_prints_the_noise_floor():
    """Without a standard error printed, small differences get reported as
    findings and tuning chases noise."""
    r = Result(name="x", n=300, same_cell_p5=0.5)
    out = render([r])
    assert "standard error" in out and "±" in out


def test_stderr_shrinks_with_sample_size():
    assert Result("a", 100).stderr(0.5) > Result("a", 1000).stderr(0.5)
