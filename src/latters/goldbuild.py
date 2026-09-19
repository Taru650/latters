"""Build a real gold set from the office's own archive.

The seed gold set proves the converter *engine* works, because both of its
columns were derived from the mapping table's own logic. Only pairs
transcribed from real rendered letters by a Hindi reader can prove the
*table* is right. This module makes producing those pairs as cheap as it can
honestly be made.

The workflow
------------
1. ``latters gold extract <archive> -o review/`` mines candidate lines from
   the DOCX archive, picks a set that exercises as many distinct mapping slots
   as possible, and writes two files:

   * ``review.docx`` -- a two-column table. Left column shows each line in its
     **original legacy font**, so a reader with the font installed sees what
     the page actually said. Right column shows this converter's Unicode
     output. The reviewer compares the two visually.
   * ``review.tsv`` -- the form. For each line the reviewer writes ``ok`` or
     the corrected Unicode.

2. A Hindi reader fills ``review.tsv`` in Excel.

3. ``latters gold collect review.tsv -o tests/gold/office_x.tsv`` validates
   and converts it into a gold file the regression harness picks up.

Two deliberate design choices
-----------------------------
**Coverage-greedy selection, not random sampling.** 200 random lines are
mostly the same boilerplate and would leave rare conjuncts, reph forms and
ligature slots completely untested. Selection instead repeatedly takes the
line covering the most as-yet-uncovered mapping keys. The resulting set is
smaller *and* tests more.

**A blind subset.** Showing the reviewer our conversion makes the job fast,
but invites rubber-stamping: plausible-looking wrong output gets approved. A
random fraction of rows is therefore marked blind -- no suggestion shown, in
either file -- and the reviewer transcribes those from the legacy rendering
alone. If the blind rows disagree with the converter far more often than the
sighted rows do, the sighted rows were being approved rather than checked,
and the whole set needs redoing.
"""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import docx_writer as D
from .extract import UnsupportedFormat, classify_font, read_document
from .fonts.convert import Converter
from .fonts.tables import load_table

#: Lines shorter than this carry too little signal to be worth a reviewer's
#: time; longer than this and transcription errors creep in.
MIN_LEN = 12
MAX_LEN = 140

_WS = re.compile(r"\s+")


@dataclass
class Candidate:
    legacy: str
    table: str
    font: str
    source: str
    suggested: str = ""
    keys: frozenset[str] = frozenset()
    blind: bool = False
    #: New mapping keys this line contributed when it was selected. Shown in
    #: the review sheet so the reviewer knows which lines are load-bearing.
    novel: tuple[str, ...] = ()


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip()


def mine_candidates(root: Path) -> tuple[list[Candidate], list[str]]:
    """Pull every distinct legacy-font line out of a DOCX archive."""
    seen: set[str] = set()
    out: list[Candidate] = []
    problems: list[str] = []
    converters: dict[str, Converter] = {}

    files = [root] if root.is_file() else sorted(root.rglob("*.docx"))
    for path in files:
        if path.name.startswith("~$"):
            continue
        try:
            doc = read_document(path)
        except (UnsupportedFormat, Exception) as exc:
            problems.append(f"{path.name}: {str(exc).splitlines()[0]}")
            continue
        for block in doc.blocks:
            for r in block.runs:
                table, _ = classify_font(r.font)
                if table is None:
                    continue
                text = _norm(r.text)
                if not (MIN_LEN <= len(text) <= MAX_LEN) or text in seen:
                    continue
                seen.add(text)
                if table not in converters:
                    converters[table] = Converter(table)
                conv = converters[table].convert(text)
                out.append(Candidate(
                    legacy=text, table=table, font=r.font or "(unspecified)",
                    source=path.name, suggested=conv.text, keys=conv.used_keys))
    return out, problems


def select_by_coverage(candidates: list[Candidate], n: int,
                       *, seed: int = 0) -> list[Candidate]:
    """Greedy maximum-coverage over mapping-table keys.

    Each pick is the line contributing the most keys not yet covered, with
    shorter lines winning ties because they are faster to verify. Once every
    key seen in the archive is covered, the remainder are filled by longest
    line -- long lines exercise more *combinations* of slots, which is where
    the reordering passes break.
    """
    rng = random.Random(seed)
    pool = list(candidates)
    rng.shuffle(pool)  # stable tie-breaking that is not filesystem order
    covered: set[str] = set()
    chosen: list[Candidate] = []

    while pool and len(chosen) < n:
        best, best_gain = None, -1
        for c in pool:
            gain = len(c.keys - covered)
            if gain > best_gain or (gain == best_gain and best is not None
                                    and len(c.legacy) < len(best.legacy)):
                best, best_gain = c, gain
        if best is None:
            break
        if best_gain == 0:
            break
        best.novel = tuple(sorted(best.keys - covered))
        covered |= best.keys
        chosen.append(best)
        pool.remove(best)

    # Everything distinct is covered; top up with the longest remaining lines.
    pool.sort(key=lambda c: -len(c.legacy))
    for c in pool[: n - len(chosen)]:
        chosen.append(c)
    return chosen


def mark_blind(candidates: list[Candidate], fraction: float, *, seed: int = 0) -> int:
    rng = random.Random(seed)
    k = int(round(len(candidates) * fraction))
    for c in rng.sample(candidates, k) if k else []:
        c.blind = True
    return k


_INSTRUCTIONS_HI = (
    "निर्देश: बाईं ओर मूल पत्र की पंक्ति उसी पुराने फ़ॉन्ट में दिखाई गई है। "
    "दाईं ओर हमारे द्वारा किया गया यूनिकोड रूपांतरण है। दोनों की तुलना कीजिए। "
    "यदि रूपांतरण सही है तो review.tsv में उस पंक्ति के सामने ok लिखिए; "
    "यदि गलत है तो सही हिन्दी टाइप कीजिए।"
)
_INSTRUCTIONS_HI_BLIND = (
    "जिन पंक्तियों के सामने (जाँच पंक्ति) लिखा है, उनका रूपांतरण जानबूझकर "
    "नहीं दिखाया गया है। कृपया बाईं ओर दिख रही पंक्ति को पढ़कर स्वयं हिन्दी में टाइप कीजिए।"
)


def write_review_docx(candidates: list[Candidate], path: Path,
                      *, unicode_font: str = "Mangal") -> Path:
    parts = [
        D.heading("Gold set review / गोल्ड सेट जाँच"),
        D.para(D.run(_INSTRUCTIONS_HI, unicode_font, size_pt=11)),
        D.para(D.run(_INSTRUCTIONS_HI_BLIND, unicode_font, size_pt=11, color="B00000")),
        D.para(D.run(
            "If the left column shows Latin gibberish rather than Hindi, the "
            "legacy font named in each row is not installed on this machine. "
            "Install it before reviewing -- the left column IS the ground truth.",
            "Calibri", size_pt=9, color="808080")),
    ]
    rows: list[list[list[str]]] = [[
        [D.para(D.run("#", "Calibri", bold=True, size_pt=10))],
        [D.para(D.run("मूल (legacy font)", unicode_font, bold=True, size_pt=10))],
        [D.para(D.run("रूपांतरण (our Unicode)", unicode_font, bold=True, size_pt=10))],
    ]]
    for i, c in enumerate(candidates, 1):
        left = [D.para(D.run(c.legacy, c.font, size_pt=14)),
                D.para(D.run(f"{c.font} · {c.source}", "Calibri", size_pt=7, color="909090"))]
        if c.blind:
            right = [D.para(D.run("(जाँच पंक्ति — स्वयं टाइप कीजिए)", unicode_font,
                                  size_pt=10, color="B00000"))]
        else:
            right = [D.para(D.run(c.suggested, unicode_font, size_pt=13))]
        rows.append([[D.para(D.run(f"{i:03d}", "Calibri", size_pt=10))], left, right])
    parts.append(D.table(rows, [700, 5100, 5100]))
    return D.write(path, parts)


REVIEW_HEADER = ["id", "verdict_or_correction", "legacy", "suggested", "blind",
                 "table", "font", "source", "novel_slots"]


def write_review_tsv(candidates: list[Candidate], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel opens Devanagari correctly on a Windows machine
    # without the import wizard mangling it.
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(REVIEW_HEADER)
        for i, c in enumerate(candidates, 1):
            w.writerow([f"{i:03d}", "", c.legacy,
                        "" if c.blind else c.suggested,
                        "1" if c.blind else "", c.table, c.font, c.source,
                        " ".join(c.novel)])
    return path


_ENCODINGS = ("utf-8-sig", "utf-8", "utf-16", "cp1252")


def _read_text_any(path: Path) -> str:
    """Excel saves TSV as UTF-16 or ANSI depending on how it was saved."""
    raw = path.read_bytes()
    for enc in _ENCODINGS:
        try:
            text = raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        if "�" not in text:
            return text
    return raw.decode("utf-8", errors="replace")


_OK_WORDS = {"ok", "OK", "Ok", "ठीक", "सही", "y", "Y", "yes", "हाँ", "हां"}


@dataclass
class Collected:
    pairs: list[tuple[str, str, str, str]] = field(default_factory=list)
    n_ok: int = 0
    n_typed: int = 0
    n_blank: int = 0
    #: Rows where the reviewer's answer differs from what the converter
    #: actually produces, split by whether they could see our suggestion.
    #: These, not "did the reviewer type something", are the control:
    #: a blind reviewer must type an answer for every row, so counting typed
    #: rows would make the blind disagreement rate 100% by construction.
    n_blind: int = 0
    n_blind_disagree: int = 0
    n_sighted: int = 0
    n_sighted_disagree: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def sighted_disagreement(self) -> float | None:
        return self.n_sighted_disagree / self.n_sighted if self.n_sighted else None

    @property
    def blind_disagreement(self) -> float | None:
        return self.n_blind_disagree / self.n_blind if self.n_blind else None


def collect_review(path: Path) -> Collected:
    """Turn a filled review sheet into gold pairs, and audit the filling.

    Disagreement is measured against a *freshly recomputed* conversion, not
    against the ``suggested`` column, so that a mapping table edited since the
    sheet was generated -- or a reviewer who overwrote the column -- cannot
    quietly skew the control.
    """
    from .fonts.convert import normalize_devanagari
    out = Collected()
    converters: dict[str, Converter] = {}
    rows = list(csv.DictReader(_read_text_any(path).splitlines(), delimiter="\t"))
    if not rows:
        out.problems.append("no data rows found")
        return out
    missing = set(REVIEW_HEADER) - set(rows[0])
    if missing:
        out.problems.append(f"missing columns: {sorted(missing)}")
        return out

    for row in rows:
        rid = (row.get("id") or "?").strip()
        legacy = row.get("legacy") or ""
        verdict = (row.get("verdict_or_correction") or "").strip()
        blind = bool((row.get("blind") or "").strip())
        # n_blind is incremented only once a row yields a pair, below --
        # counting it here too double-counted every blind row and inflated
        # the control's denominator.

        if not verdict:
            out.n_blank += 1
            out.problems.append(f"row {rid}: not reviewed (left blank) -- excluded")
            continue

        table_name = row.get("table") or "krutidev010"

        if verdict in _OK_WORDS:
            if blind:
                out.problems.append(
                    f"row {rid}: marked ok but it is a blind row with no "
                    "suggestion shown -- nothing to approve; excluded")
                continue
            expected = row.get("suggested") or ""
            out.n_ok += 1
        else:
            expected = verdict
            out.n_typed += 1

        if not expected.strip():
            out.problems.append(f"row {rid}: empty expected value -- excluded")
            continue

        if table_name not in converters:
            try:
                converters[table_name] = Converter(table_name)
            except Exception as exc:
                out.problems.append(f"row {rid}: unknown table {table_name!r} ({exc})")
                continue
        actual = converters[table_name].convert(legacy).text
        disagrees = normalize_devanagari(expected) != normalize_devanagari(actual)

        if blind:
            out.n_blind += 1
            out.n_blind_disagree += disagrees
        else:
            out.n_sighted += 1
            out.n_sighted_disagree += disagrees

        out.pairs.append((legacy, expected, table_name,
                          f"{row.get('source','')} row {rid}"))
    return out


def write_gold(collected: Collected, path: Path, *, title: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Gold set: {title}",
        "#",
        "# Transcribed from real archive letters by a Hindi reader via",
        "# `latters gold extract` / `latters gold collect`. Unlike the seed set,",
        "# these pairs test the mapping TABLE, not just the converter engine.",
        "#",
        f"# rows: {len(collected.pairs)}  "
        f"(approved {collected.n_ok}, typed {collected.n_typed}; "
        f"disagreed with the converter: "
        f"{collected.n_sighted_disagree + collected.n_blind_disagree})",
        "#",
        "# Format: legacy<TAB>expected<TAB>table<TAB>note",
        "",
    ]
    for legacy, expected, table, note in collected.pairs:
        lines.append(f"{legacy}\t{expected}\t{table}\t{note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
@dataclass
class Coverage:
    table: str
    total: int
    covered: set[str]

    @property
    def uncovered(self) -> list[str]:
        return sorted(set(load_table(self.table).mapping) - self.covered)

    @property
    def fraction(self) -> float:
        return len(self.covered) / self.total if self.total else 0.0


def coverage(pairs: list[tuple[str, str, str, str]], table_name: str) -> Coverage:
    """Which mapping slots does this gold set actually exercise?

    The number that matters is not how many pairs there are, it is how many
    slots go untested. A 200-pair gold set that never exercises ``द्य`` proves
    nothing about ``द्य``.
    """
    conv = Converter(table_name)
    used: set[str] = set()
    for legacy, _, t, _ in pairs:
        if t == table_name:
            used |= conv.convert(legacy).used_keys
    return Coverage(table_name, len(conv.table.mapping), used)
