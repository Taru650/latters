"""Opening a corpus built by an older version.

Found in the field: a corpus built before the form-aware segmentation of
Phase 2 was opened by the current code, and the web app died on its first
page load with `no such column: form`.

The cause was that SCHEMA_VERSION had existed from the beginning and was
never *used*. Every statement in the schema script is
`CREATE TABLE IF NOT EXISTS`, which silently does nothing when the table is
already there -- so a new TABLE appeared on upgrade, which is why the v2→v3
`drafts` migration worked and got a passing test, but a new COLUMN never
did.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latters.store import _LETTER_COLUMNS, SCHEMA_VERSION, Store  # noqa: E402

#: The shape before Phase 2 added form-aware segmentation and Phase 3 added
#: the label columns. This is what was actually on the office machine.
_V1 = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE letters (
  id INTEGER PRIMARY KEY, source_file TEXT NOT NULL, seq INTEGER NOT NULL,
  text TEXT NOT NULL, text_hash TEXT NOT NULL UNIQUE,
  source_tier TEXT NOT NULL, trust REAL, verdict TEXT,
  created_at TEXT NOT NULL);
INSERT INTO meta VALUES('schema_version','1');
"""


def _old_db(path: Path, rows: int = 3) -> Path:
    c = sqlite3.connect(path)
    c.executescript(_V1)
    for i in range(rows):
        c.execute("INSERT INTO letters VALUES(?,?,?,?,?,?,?,?,?)",
                  (i + 1, "a.docx", i + 1,
                   f"कार्यालय समीक्षा बैठक संख्या {i}", f"h{i}",
                   "docx", 0.9, "index", "2026-01-01"))
    c.commit()
    c.close()
    return path


def test_an_old_corpus_opens_at_all(tmp_path):
    """It used to fail inside the constructor: the schema script builds an
    index on `department`, which an old database has no column for."""
    with Store(_old_db(tmp_path / "old.db")) as store:
        assert store.count() == 3


def test_the_letters_survive_the_migration(tmp_path):
    """ALTER TABLE ADD COLUMN is non-destructive, and this asserts it. The
    alternative anyone reaches for -- delete and rebuild -- also loses the
    admin page's hand corrections, which are not in the archive."""
    with Store(_old_db(tmp_path / "old.db")) as store:
        assert "समीक्षा" in store.get(1)["text"]
        assert store.get(2)["trust"] == 0.9
        assert store.get(3)["verdict"] == "index"


def test_the_page_that_crashed_now_works(tmp_path):
    """`stats()` reads `form`, which is what the office's web app died on."""
    with Store(_old_db(tmp_path / "old.db")) as store:
        stats = store.stats()
        assert stats["letters"] == 3
        # A letter that predates the column reads NULL, rendered as '?'.
        assert stats["by_form"] == {"?": 3}


def test_search_still_works_after_a_column_is_added(tmp_path):
    """The FTS table is external-content over `letters`. Adding a column it
    indexes leaves the index stale unless it is rebuilt."""
    with Store(_old_db(tmp_path / "old.db")) as store:
        assert len(store.search("समीक्षा")) == 3


def test_the_version_is_written_after_migrating(tmp_path):
    with Store(_old_db(tmp_path / "old.db")) as store:
        got = store.db.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        assert int(got) == SCHEMA_VERSION


def test_migrating_twice_is_a_no_op(tmp_path):
    db = _old_db(tmp_path / "old.db")
    Store(db).close()
    with Store(db) as store:
        assert store.count() == 3
        assert store.stats()["letters"] == 3


def test_a_current_database_needs_no_migration(tmp_path):
    db = tmp_path / "new.db"
    Store(db).close()
    with Store(db) as store:
        assert store._migrate() == []


@pytest.mark.parametrize("column", [c for c, _ in _LETTER_COLUMNS])
def test_every_expected_column_exists_after_migrating(tmp_path, column):
    with Store(_old_db(tmp_path / "old.db")) as store:
        have = {r[1] for r in store.db.execute("PRAGMA table_info(letters)")}
        assert column in have


def test_the_migration_list_matches_the_schema(tmp_path):
    """The two could drift, and the drift would be invisible until someone
    with an old corpus upgraded -- which is exactly how this bug reached
    the office."""
    db = tmp_path / "fresh.db"
    with Store(db) as store:
        actual = {r[1] for r in store.db.execute("PRAGMA table_info(letters)")}
    listed = {c for c, _ in _LETTER_COLUMNS}
    # Columns deliberately excluded: NOT NULL since the first version.
    always = {"id", "source_file", "seq", "text", "text_hash",
              "source_tier", "created_at"}
    assert listed | always == actual, (
        f"_LETTER_COLUMNS is out of step with the schema: "
        f"missing {actual - listed - always}, extra {listed - actual}")


# --- migration is not enough on its own -----------------------------------
def test_resegmenting_does_not_fix_a_migrated_row(tmp_path):
    """Dedupe is content-addressed on text_hash, so re-running `segment`
    over the same archive correctly skips everything -- right when nothing
    changed, wrong after an upgrade. A migrated row keeps NULL `form` and
    `subject` for ever, and a letter with no subject is invisible to the
    retrieval query set and to the FTS subject column."""
    from latters.store import LetterRow

    db = tmp_path / "c.db"
    row = LetterRow(source_file="a.docx", seq=1, text="कार्यालय समीक्षा बैठक")
    with Store(db) as store:
        store.add([row])
        store.db.execute("UPDATE letters SET form=NULL, subject=NULL")
        store.db.commit()

        scored = LetterRow(source_file="a.docx", seq=1,
                           text="कार्यालय समीक्षा बैठक",
                           form="letter", subject="समीक्षा बैठक")
        inserted, skipped = store.add([scored])
        assert (inserted, skipped) == (0, 1)
        assert store.get(1)["form"] is None, "plain add must not update"


def test_refresh_rescores_a_migrated_row(tmp_path):
    from latters.store import LetterRow

    db = tmp_path / "c.db"
    with Store(db) as store:
        store.add([LetterRow(source_file="a.docx", seq=1,
                             text="कार्यालय समीक्षा बैठक")])
        store.db.execute("UPDATE letters SET form=NULL, subject=NULL, "
                         "trust=0.1, verdict='review'")
        store.db.commit()

        store.add([LetterRow(source_file="a.docx", seq=1,
                             text="कार्यालय समीक्षा बैठक",
                             form="letter", subject="समीक्षा बैठक",
                             trust=0.93, verdict="index")], refresh=True)
        got = store.get(1)
        assert got["form"] == "letter"
        assert got["subject"] == "समीक्षा बैठक"
        assert got["verdict"] == "index"


def test_refresh_never_rewrites_the_letter_itself(tmp_path):
    """The row is matched BY its text, so there is nothing to change -- and
    a letter corrected on the admin page must keep the correction. Only the
    derived scores are rewritten."""
    from latters.store import LetterRow

    db = tmp_path / "c.db"
    text = "कार्यालय समीक्षा बैठक"
    with Store(db) as store:
        store.add([LetterRow(source_file="a.docx", seq=1, text=text)])
        store.add([LetterRow(source_file="b.docx", seq=9, text=text,
                             form="order")], refresh=True)
        got = store.get(1)
        assert got["text"] == text
        assert got["form"] == "order"


def test_refresh_still_inserts_letters_that_are_genuinely_new(tmp_path):
    from latters.store import LetterRow

    with Store(tmp_path / "c.db") as store:
        store.add([LetterRow(source_file="a.docx", seq=1, text="पहला पत्र")])
        ins, dup = store.add(
            [LetterRow(source_file="a.docx", seq=1, text="पहला पत्र"),
             LetterRow(source_file="a.docx", seq=2, text="दूसरा पत्र")],
            refresh=True)
        assert (ins, dup) == (1, 1)
        assert store.count() == 2


def test_refresh_does_not_clobber_the_classifier_labels(tmp_path):
    """`department` and `letter_type` are written by `classify --write`,
    not by segmentation, so a re-score must leave them alone -- otherwise
    `segment --refresh` would silently undo the labelling step and the
    department retrieval filter would go dark."""
    from latters.store import LetterRow

    db = tmp_path / "c.db"
    text = "कार्यालय समीक्षा बैठक"
    with Store(db) as store:
        store.add([LetterRow(source_file="a.docx", seq=1, text=text)])
        store.db.execute("UPDATE letters SET department=?, letter_type=? "
                         "WHERE id=1", ("राजस्व", "जाँच"))
        store.db.commit()

        store.add([LetterRow(source_file="a.docx", seq=1, text=text,
                             form="letter", subject="समीक्षा")], refresh=True)
        got = store.get(1)
        assert got["form"] == "letter"          # re-scored
        assert got["department"] == "राजस्व"     # untouched
        assert got["letter_type"] == "जाँच"      # untouched
