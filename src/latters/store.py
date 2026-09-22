"""SQLite corpus store.

One file holds everything: letters, provenance, quality scores and the
full-text index. That is a deliberate deployment choice, not laziness — the
whole corpus can be copied to a USB stick and handed to the next office, and
there is no server to install on a locked-down government desktop.

FTS5 gives the lexical half of Phase 4's hybrid retrieval with no extra
process and no extra dependency.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = 3

#: THE TOKENIZER IS LOAD-BEARING AND THE DEFAULT IS WRONG FOR DEVANAGARI.
#:
#: FTS5's unicode61 counts only categories `L* N* Co` as token characters.
#: Devanagari matras, virama and nukta are combining marks (Mn and Mc), so by
#: default they are treated as separators and silently deleted:
#:
#:     समीक्षा   -> ["सम", "ष"]
#:     कार्यवाही -> ["क", "यव", "ह"]
#:     की        -> ["क"]
#:
#: Every word collapses to its bare consonants, कि/की/कु/कू/के/कै/को/कौ all
#: become क, and a search for की matches कार्यवाही. `remove_diacritics 0` does
#: not help -- it governs Latin diacritic folding, not which categories count
#: as letters. The `trigram` tokenizer preserves the text but cannot match
#: queries shorter than three characters, which rules out most Hindi
#: function words.
#:
#: Adding Mn and Mc to `categories` is the fix, verified by inspecting the
#: fts5vocab token table; see test_store_tokenizer_keeps_matras.
_SCHEMA = f"""
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS letters (
    id                    INTEGER PRIMARY KEY,
    source_file           TEXT    NOT NULL,
    seq                   INTEGER NOT NULL,
    start_line            INTEGER,
    end_line              INTEGER,
    text                  TEXT    NOT NULL,
    text_hash             TEXT    NOT NULL UNIQUE,
    source_tier           TEXT    NOT NULL,
    conversion_confidence REAL,
    completeness          REAL,
    trust                 REAL,
    verdict               TEXT,
    opened_by             TEXT,
    form                  TEXT,
    anchors               TEXT,
    violations            TEXT,
    missing               TEXT,
    -- Phase 3 fills these; kept here so the schema does not churn later.
    letter_number         TEXT,
    letter_date           TEXT,
    subject               TEXT,
    department            TEXT,
    letter_type           TEXT,
    created_at            TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_letters_trust      ON letters(trust);
CREATE INDEX IF NOT EXISTS ix_letters_verdict    ON letters(verdict);
CREATE INDEX IF NOT EXISTS ix_letters_source     ON letters(source_file);
CREATE INDEX IF NOT EXISTS ix_letters_department ON letters(department, letter_type);

CREATE VIRTUAL TABLE IF NOT EXISTS letters_fts USING fts5(
    text, subject,
    content='letters', content_rowid='id',
    tokenize="unicode61 remove_diacritics 0 categories 'L* N* Mn Mc Co'"
);

-- Phase 7. Every draft that leaves through the export button is half of a
-- (request, dispatched letter) pair, and the exported text is the other half.
-- Recording both is what turns editing effort -- the metric the plan says
-- decides success -- into a number that accrues by itself, instead of one
-- that waits on a 30-pair study nobody ever runs. See effort.py.
--
-- dispatched_text and effort stay NULL until an export happens: a draft that
-- was generated and abandoned is a real outcome and deleting the row would
-- hide it.
CREATE TABLE IF NOT EXISTS drafts (
    id               INTEGER PRIMARY KEY,
    created_at       TEXT    NOT NULL,
    request          TEXT    NOT NULL,
    department       TEXT,
    letter_type      TEXT,
    draft_text       TEXT    NOT NULL,
    dispatched_text  TEXT,
    dispatched_at    TEXT,
    export_format    TEXT,
    effort           REAL,
    seconds          REAL,
    model            TEXT,
    needs_review     INTEGER
);

CREATE INDEX IF NOT EXISTS ix_drafts_effort ON drafts(effort);

CREATE TRIGGER IF NOT EXISTS letters_ai AFTER INSERT ON letters BEGIN
    INSERT INTO letters_fts(rowid, text, subject)
    VALUES (new.id, new.text, COALESCE(new.subject, ''));
END;
CREATE TRIGGER IF NOT EXISTS letters_ad AFTER DELETE ON letters BEGIN
    INSERT INTO letters_fts(letters_fts, rowid, text, subject)
    VALUES ('delete', old.id, old.text, COALESCE(old.subject, ''));
END;
CREATE TRIGGER IF NOT EXISTS letters_au AFTER UPDATE ON letters BEGIN
    INSERT INTO letters_fts(letters_fts, rowid, text, subject)
    VALUES ('delete', old.id, old.text, COALESCE(old.subject, ''));
    INSERT INTO letters_fts(rowid, text, subject)
    VALUES (new.id, new.text, COALESCE(new.subject, ''));
END;
"""


#: Every column `letters` must have, for `Store._migrate`. Kept beside the
#: schema rather than parsed out of it: a regex over CREATE TABLE would go
#: quietly wrong the first time someone wrote a constraint across two lines,
#: and a test asserts the two agree.
#:
#: Order matters only for readability. `id`, `source_file`, `seq`, `text`,
#: `text_hash`, `source_tier` and `created_at` are NOT listed: they are NOT
#: NULL and have existed since the first version, so a database without them
#: is not an old corpus, it is a different table.
_LETTER_COLUMNS: tuple[tuple[str, str], ...] = (
    ("start_line", "INTEGER"),
    ("end_line", "INTEGER"),
    ("conversion_confidence", "REAL"),
    ("completeness", "REAL"),
    ("trust", "REAL"),
    ("verdict", "TEXT"),
    ("opened_by", "TEXT"),
    ("form", "TEXT"),
    ("anchors", "TEXT"),
    ("violations", "TEXT"),
    ("missing", "TEXT"),
    ("letter_number", "TEXT"),
    ("letter_date", "TEXT"),
    ("subject", "TEXT"),
    ("department", "TEXT"),
    ("letter_type", "TEXT"),
)


@dataclass
class LetterRow:
    source_file: str
    seq: int
    text: str
    source_tier: str = "docx"
    start_line: int | None = None
    end_line: int | None = None
    conversion_confidence: float | None = None
    completeness: float | None = None
    trust: float | None = None
    verdict: str | None = None
    opened_by: str | None = None
    form: str | None = None
    anchors: dict | None = None
    violations: dict | None = None
    missing: list | None = None
    subject: str | None = None

    def hash(self) -> str:
        # Content-addressed, so re-ingesting the same archive is idempotent
        # and the same letter filed in two folders is stored once.
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


class Store:
    """
    Thread safety
    -------------
    The web application runs conversion, segmentation and generation in
    worker threads so a long upload cannot freeze the page someone else is
    drafting on. sqlite3 refuses cross-thread use of a connection by
    default, which surfaced as ProgrammingError the first time a draft was
    requested through the browser.

    `check_same_thread=False` is safe here because CPython links SQLite in
    serialized mode (``sqlite3.threadsafety == 3``), so the library
    serialises access internally. Writes additionally take a lock: SQLite
    permits one writer at a time, and without the lock a concurrent upload
    and correction produce "database is locked" rather than waiting.
    """

    def __init__(self, path: Path | str = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False,
                                  timeout=30.0)
        self.db.row_factory = sqlite3.Row
        self._write_lock = threading.RLock()
        # Columns BEFORE the schema script, or the script itself fails: it
        # builds indexes and FTS triggers that reference columns an older
        # database does not have. See _migrate.
        added = self._migrate()
        self.db.executescript(_SCHEMA)
        if added:
            # The FTS table is external-content over `letters`. If a column
            # it indexes was only just added, its index is stale.
            self.db.execute(
                "INSERT INTO letters_fts(letters_fts) VALUES('rebuild')")
        self.db.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),))
        self.db.commit()

    def _migrate(self) -> list[str]:
        """Add columns an older database is missing. Returns what was added.

        SCHEMA_VERSION existed from the beginning and was never *used*: the
        schema script is all `CREATE TABLE IF NOT EXISTS`, which silently
        does nothing when the table is already there. A new TABLE therefore
        appeared on upgrade -- which is why the v2 to v3 `drafts` migration
        seemed to work and got a passing test -- but a new COLUMN never did.

        Found in the field, not here: a corpus built before the form-aware
        segmentation of Phase 2 was opened by the current code and the web
        app died with `no such column: form` on the first page load. An
        older one fails harder, inside this constructor, because the schema
        script's own index on `department` cannot be built.

        ALTER TABLE ADD COLUMN is non-destructive and O(1) in SQLite; every
        column added here is nullable with no default, so existing rows read
        back NULL, which is exactly what "this letter predates the field"
        should mean.
        """
        have = {r[1] for r in self.db.execute("PRAGMA table_info(letters)")}
        if not have:
            return []                      # fresh database, nothing to do
        added = []
        for name, decl in _LETTER_COLUMNS:
            if name not in have:
                self.db.execute(f"ALTER TABLE letters ADD COLUMN {name} {decl}")
                added.append(name)
        if added:
            self.db.commit()
        return added

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- writing ----------------------------------------------------------
    def add(self, rows: Iterable[LetterRow]) -> tuple[int, int]:
        """Insert letters. Returns (inserted, skipped_as_duplicate)."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        inserted = skipped = 0
        with self._write_lock:
          for r in rows:
              try:
                  self.db.execute(
                      """INSERT INTO letters
                         (source_file, seq, start_line, end_line, text, text_hash,
                          source_tier, conversion_confidence, completeness, trust,
                          verdict, opened_by, form, anchors, violations,
                          missing, subject, created_at)
                         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                      (r.source_file, r.seq, r.start_line, r.end_line, r.text,
                       r.hash(), r.source_tier, r.conversion_confidence,
                       r.completeness, r.trust, r.verdict, r.opened_by, r.form,
                       json.dumps(r.anchors or {}, ensure_ascii=False),
                       json.dumps(r.violations or {}, ensure_ascii=False),
                       json.dumps(r.missing or [], ensure_ascii=False),
                       r.subject, now))
                  inserted += 1
              except sqlite3.IntegrityError:
                  skipped += 1
          self.db.commit()
        return inserted, skipped

    def delete(self, letter_id: int) -> bool:
        with self._write_lock:
            cur = self.db.execute("DELETE FROM letters WHERE id = ?", (letter_id,))
            self.db.commit()
        return cur.rowcount > 0

    def delete_source(self, source_file: str) -> int:
        with self._write_lock:
            cur = self.db.execute("DELETE FROM letters WHERE source_file = ?",
                                  (source_file,))
            self.db.commit()
        return cur.rowcount

    def write(self, sql: str, params: tuple = ()) -> int:
        """Run a statement that changes rows, under the write lock."""
        with self._write_lock:
            cur = self.db.execute(sql, params)
            self.db.commit()
        return cur.rowcount

    # --- reading ----------------------------------------------------------
    def get(self, letter_id: int) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM letters WHERE id = ?", (letter_id,)).fetchone()

    def count(self, *, verdict: str | None = None) -> int:
        if verdict:
            return self.db.execute(
                "SELECT COUNT(*) FROM letters WHERE verdict = ?", (verdict,)).fetchone()[0]
        return self.db.execute("SELECT COUNT(*) FROM letters").fetchone()[0]

    def search(self, query: str, *, limit: int = 10,
               min_trust: float = 0.0) -> list[sqlite3.Row]:
        """Lexical search. The dense half arrives in Phase 4."""
        return self.db.execute(
            """SELECT l.*, bm25(letters_fts) AS rank
               FROM letters_fts
               JOIN letters l ON l.id = letters_fts.rowid
               WHERE letters_fts MATCH ? AND l.trust >= ?
               ORDER BY rank LIMIT ?""",
            (query, min_trust, limit)).fetchall()

    def quarantined(self, limit: int = 50) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM letters WHERE verdict != 'index' "
            "ORDER BY trust ASC LIMIT ?", (limit,)).fetchall()

    # --- drafts (Phase 7) -------------------------------------------------
    def record_draft(self, *, request: str, draft_text: str,
                     department: str | None = None,
                     letter_type: str | None = None,
                     seconds: float | None = None, model: str | None = None,
                     needs_review: bool | None = None) -> int:
        """Log a generated draft and return its id.

        Called for every generation, including ones nobody exports. A draft
        that was produced and abandoned is a real outcome -- probably the
        worst one -- and a table that only held exported drafts would report
        the tool as flawless while people quietly stopped using it.
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._write_lock:
            cur = self.db.execute(
                """INSERT INTO drafts
                   (created_at, request, department, letter_type, draft_text,
                    seconds, model, needs_review)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (now, request, department, letter_type, draft_text, seconds,
                 model, None if needs_review is None else int(needs_review)))
            self.db.commit()
        return int(cur.lastrowid)

    def record_dispatch(self, draft_id: int, dispatched_text: str,
                        export_format: str) -> float | None:
        """Attach the exported text to its draft and score the edit.

        Returns the effort score, or None if the id is unknown -- which is
        not an error worth failing an export over: the letter is already
        written and refusing to hand it over to protect a statistic would be
        the wrong trade.
        """
        from .effort import measure

        row = self.db.execute(
            "SELECT draft_text FROM drafts WHERE id = ?", (draft_id,)).fetchone()
        if row is None:
            return None
        score = measure(row["draft_text"], dispatched_text).score
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._write_lock:
            self.db.execute(
                """UPDATE drafts SET dispatched_text = ?, dispatched_at = ?,
                   export_format = ?, effort = ? WHERE id = ?""",
                (dispatched_text, now, export_format, score, draft_id))
            self.db.commit()
        return score

    def effort_scores(self) -> list[float]:
        return [r[0] for r in self.db.execute(
            "SELECT effort FROM drafts WHERE effort IS NOT NULL")]

    def draft_stats(self) -> dict:
        """Counts that say how the tool is actually being used.

        `generated` minus `dispatched` is the abandonment count, and it is
        the honest companion to the effort median: a low median over three
        exports out of ninety drafts is not a success.
        """
        n = self.db.execute("SELECT COUNT(*) FROM drafts").fetchone()[0]
        d = self.db.execute(
            "SELECT COUNT(*) FROM drafts WHERE dispatched_text IS NOT NULL"
        ).fetchone()[0]
        return {"generated": n, "dispatched": d, "abandoned": n - d}

    def stats(self) -> dict:
        row = self.db.execute(
            """SELECT COUNT(*) n, AVG(trust) t, AVG(completeness) c,
                      AVG(conversion_confidence) v FROM letters""").fetchone()
        by_verdict = {r["verdict"]: r["n"] for r in self.db.execute(
            "SELECT verdict, COUNT(*) n FROM letters GROUP BY verdict")}
        by_tier = {r["source_tier"]: r["n"] for r in self.db.execute(
            "SELECT source_tier, COUNT(*) n FROM letters GROUP BY source_tier")}
        by_form = {r["form"] or "?": r["n"] for r in self.db.execute(
            "SELECT form, COUNT(*) n FROM letters GROUP BY form")}
        return {
            "letters": row["n"],
            "mean_trust": round(row["t"], 4) if row["t"] is not None else None,
            "mean_completeness": round(row["c"], 4) if row["c"] is not None else None,
            "mean_conversion": round(row["v"], 4) if row["v"] is not None else None,
            "by_verdict": by_verdict,
            "by_source_tier": by_tier,
            "by_form": by_form,
        }
