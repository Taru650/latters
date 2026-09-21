"""Phase 7: editing effort, the scorecards, and backup.

The theme of these tests is that a measurement which quietly reports success
when it has no data is worse than no measurement at all. Most of what is
asserted here is about the *absence* of a number being visible.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latters.cli import main  # noqa: E402
from latters.effort import (FAILED_ITS_PURPOSE, Effort, levenshtein, measure,  # noqa: E402
                            summarise)
from latters.scorecard import build, render  # noqa: E402
from latters.store import Store  # noqa: E402

LETTER = ("विषय: मासिक समीक्षा बैठक की सूचना।\n\n"
          "महाशय,\n\n"
          "उपर्युक्त विषय के प्रसंग में कहना है कि दिनांक 25.03.2024 को "
          "बैठक आयोजित की जा रही है।\n\nविश्वासभाजन")


# --- edit distance --------------------------------------------------------
@pytest.mark.parametrize("a,b,d", [
    ("", "", 0), ("क", "", 1), ("", "क", 1),
    ("कार्यालय", "कार्यालय", 0),
    ("बैठक", "बैठकें", 2),
])
def test_levenshtein(a, b, d):
    assert levenshtein(a, b) == d


def test_levenshtein_is_symmetric():
    assert levenshtein("समीक्षा बैठक", "बैठक समीक्षा") == \
           levenshtein("बैठक समीक्षा", "समीक्षा बैठक")


# --- what counts as an edit ----------------------------------------------
def test_an_untouched_draft_costs_nothing():
    assert measure(LETTER, LETTER).score == 0.0
    assert measure(LETTER, LETTER).verdict == "as-drafted"


@pytest.mark.parametrize("dispatched", [
    LETTER.replace("\n\n", "\n\n\n\n"),   # extra blank lines
    LETTER.replace(" ", "  "),            # doubled spaces
    "\n".join(l + "   " for l in LETTER.split("\n")),   # trailing spaces
])
def test_whitespace_is_not_editing(dispatched):
    """A writer pressing Enter twice did not rewrite the letter. Before this
    was folded, Word's trailing spaces alone scored as a substantial edit."""
    assert measure(LETTER, dispatched).score == 0.0


def test_unicode_form_is_not_editing():
    """The browser submits NFC; the converter emits decomposed sequences for
    some conjuncts. Identical Hindi in two normal forms would otherwise score
    as a full rewrite of every affected word."""
    import unicodedata
    assert measure(LETTER, unicodedata.normalize("NFD", LETTER)).score == 0.0


def test_a_wholesale_replacement_fails_its_purpose():
    other = "विषय: भूमि अधिग्रहण।\n\nमहाशय,\n\nअन्य विषय पर पत्र।\n\nविश्वासभाजन"
    e = measure(LETTER, other)
    assert e.failed_its_purpose and e.verdict == "failed"


def test_a_small_correction_stays_in_the_good_band():
    dispatched = LETTER.replace("25.03.2024", "27.03.2024")
    assert measure(LETTER, dispatched).score < 0.10


def test_an_empty_dispatch_is_total_failure():
    """Division by the dispatched length; the guard has to return 1.0, not
    raise, because an export of an emptied box is a real user action."""
    assert measure(LETTER, "").score == 1.0


def test_normalising_by_the_dispatched_length_is_not_flattering():
    """A draft mostly deleted must score high. Normalising by the LONGER of
    the two would divide by the draft and hide it."""
    e = measure(LETTER, LETTER[:40])
    assert e.score > FAILED_ITS_PURPOSE


# --- summarising ----------------------------------------------------------
def test_summary_uses_the_median_not_the_mean():
    """One pasted-over draft would drag a mean above the failure line while
    every other letter went out untouched."""
    scores = [0.02, 0.03, 0.05, 0.04, 4.0]
    s = summarise(scores)
    assert s.median == 0.04 and s.failed == 1
    # The same data read as a mean would condemn a tool that worked on four
    # letters out of five.
    assert sum(scores) / len(scores) > FAILED_ITS_PURPOSE


def test_summary_of_nothing_is_none_not_zero():
    """Zero would render as a perfect score on an empty table."""
    assert summarise([]) is None


# --- the drafts table -----------------------------------------------------
def test_a_draft_is_recorded_before_anyone_exports_it():
    """A table that only held exported drafts would report the tool as
    flawless while people quietly stopped using it."""
    with Store() as st:
        st.record_draft(request="बैठक की सूचना भेजनी है", draft_text=LETTER)
        assert st.draft_stats() == {"generated": 1, "dispatched": 0,
                                    "abandoned": 1}
        assert st.effort_scores() == []


def test_dispatch_scores_the_edit():
    with Store() as st:
        i = st.record_draft(request="बैठक की सूचना भेजनी है", draft_text=LETTER)
        score = st.record_dispatch(i, LETTER.replace("25.03", "27.03"), "docx")
        assert 0 < score < 0.10
        assert st.draft_stats()["dispatched"] == 1


def test_an_unknown_draft_id_does_not_fail_the_export():
    """The letter is already written. Refusing to hand it over to protect a
    statistic would be the wrong trade."""
    with Store() as st:
        assert st.record_dispatch(9999, LETTER, "docx") is None


# --- the scorecards -------------------------------------------------------
@pytest.fixture
def corpus(tmp_path):
    db = tmp_path / "corpus.db"
    with Store(db) as st:
        st.record_draft(request="बैठक की सूचना भेजनी है", draft_text=LETTER)
    return str(db)


def test_missing_scorecards_are_printed_as_loudly_as_failing_ones(corpus):
    text = render(build(corpus, None))
    assert "NO DATA" in text
    assert "has not been evaluated" in text
    # and the exact command that would fill each one
    assert "latters gold extract" in text


def test_the_seed_set_cannot_clear_the_conversion_card(corpus):
    """The packaged seed pairs were written by the person who wrote the
    mapping table. Counting them printed 'usable, 100%' -- the precise
    self-deception this module exists to prevent."""
    cards = {c.name: c for c in build(corpus, None)}
    conversion = cards["conversion"]
    assert conversion.blocked is not None
    assert conversion.verdict != "usable"
    # the seed row is still shown, just not allowed to be the verdict
    assert any("seed regression" in label for label, _ in conversion.rows)


def test_scorecard_exits_nonzero_while_anything_is_unmeasured(corpus, capsys):
    assert main(["scorecard", "--db", corpus]) == 1
    assert "NO DATA" in capsys.readouterr().out


def test_scorecard_refuses_a_mistyped_db(tmp_path, capsys):
    assert main(["scorecard", "--db", str(tmp_path / "nope.db")]) == 2
    assert "no corpus database" in capsys.readouterr().err


def test_the_drafting_card_reports_abandonment_alongside_the_median(tmp_path):
    """A low median over three exports out of ninety drafts is not success,
    and the median alone would read as though it were."""
    db = tmp_path / "corpus.db"
    with Store(db) as st:
        for _ in range(12):
            i = st.record_draft(request="बैठक की सूचना भेजनी है",
                                draft_text=LETTER)
            st.record_dispatch(i, LETTER, "docx")
        for _ in range(30):
            st.record_draft(request="छोड़ा गया", draft_text=LETTER)

    card = {c.name: c for c in build(str(db), None)}["drafting"]
    rows = dict(card.rows)
    assert rows["drafts abandoned"] == "30"
    assert card.verdict == "saving time"


# --- backup ---------------------------------------------------------------
def test_backup_snapshots_a_live_database(tmp_path, capsys):
    """Not a file copy: the database runs in WAL mode, and copying the file
    while the server writes silently loses the newest transactions."""
    db = tmp_path / "corpus.db"
    store = Store(db)                     # deliberately left OPEN
    store.record_draft(request="बैठक की सूचना भेजनी है", draft_text=LETTER)

    assert main(["backup", "--db", str(db), "-o", str(tmp_path / "b")]) == 0
    out = capsys.readouterr().out
    assert "integrity ok" in out

    copies = sorted((tmp_path / "b").glob("corpus-*.db"))
    assert len(copies) == 1
    with sqlite3.connect(copies[0]) as c:
        assert c.execute("SELECT COUNT(*) FROM drafts").fetchone()[0] == 1
    store.close()


def test_backup_keep_prunes_older_copies(tmp_path):
    db = tmp_path / "corpus.db"
    Store(db).close()
    out = tmp_path / "b"
    out.mkdir()
    for stamp in ("20200101-0000", "20200102-0000", "20200103-0000"):
        (out / f"corpus-{stamp}.db").write_bytes(b"")
    assert main(["backup", "--db", str(db), "-o", str(out), "--keep", "2"]) == 0
    assert len(list(out.glob("corpus-*.db"))) == 2


def test_backup_refuses_a_mistyped_db(tmp_path, capsys):
    assert main(["backup", "--db", str(tmp_path / "nope.db")]) == 2
    assert "no corpus database" in capsys.readouterr().err
