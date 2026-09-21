"""Text extraction that preserves which font each run was typed in.

This is the single most important design decision in Phase 1.

A legacy departmental letter is almost never uniformly legacy-encoded. The
Hindi body is Kruti Dev; the letter number, the English designation in the
signature block, the date, and anything typed after 2015 are plain Latin or
already Unicode -- all inside the same paragraph. Converting the whole
document with one table turns ``Dy. Secretary`` into Devanagari noise, because
almost every ASCII letter is a valid Kruti Dev slot.

So extraction keeps runs separate, tags each with the font actually applied to
it, and conversion happens per run with the table that font maps to. Runs in a
Unicode font pass through untouched.

DOCX is parsed with the standard library only (it is a zip of XML), which
keeps ``latters ingest`` runnable on an office machine with no wheelhouse.
"""

from __future__ import annotations

import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

#: Font name (lower-cased, whitespace-collapsed) -> mapping table name.
#: ``None`` means "already Unicode, do not convert".
FONT_TABLES: dict[str, str | None] = {}


def _register(names: str, table: str | None) -> None:
    for n in names.split(","):
        FONT_TABLES[n.strip().lower()] = table


# Legacy Devanagari families. The numeric suffixes are separate fonts with
# different ligature slots, but they share the Remington base layout, so they
# all start on the base table and get overrides added as the gold set demands.
_register("kruti dev 010,kruti dev 011,kruti dev 016,kruti dev 050,krutidev,kruti dev,kruti dev 010 condensed", "krutidev010")
_register("devlys 010,devlys 020,devlys,dev lys 010", "devlys010")
# Variants confirmed present in a real district archive. They share the
# Remington base layout, so they inherit the base table; give each its own
# file the moment a gold pair proves a slot differs.
_register("devlys 040,devlys 050,dev lys 040", "devlys040")
_register("kruti dev 041,kruti dev 045,kruti dev 040", "krutidev041")
_register("chanakya,shree dev,shree-dev-0714,shivaji,agra,walkman-chanakya", "krutidev010")
# Unicode Devanagari -- never convert these.
_register("mangal,nirmala ui,noto sans devanagari,noto serif devanagari,aparajita,kokila,utsaah,sanskrit text,arial unicode ms,kalimati,samyak devanagari", None)
# Latin faces. Text in these is English (or a stray Unicode paste); leave it.
_register("times new roman,arial,calibri,cambria,verdana,tahoma,courier new,georgia,book antiqua,bookman old style,segoe ui", None)


def classify_font(font: str | None) -> tuple[str | None, str]:
    """Return ``(table_name_or_None, confidence_label)`` for a font name."""
    if not font:
        return None, "unknown-font"
    key = re.sub(r"\s+", " ", font).strip().lower()
    if key in FONT_TABLES:
        return FONT_TABLES[key], "known"
    # Unseen family, but the name gives it away.
    for stem, table in (("kruti", "krutidev010"), ("devlys", "devlys010"),
                        ("chanakya", "krutidev010"), ("shree", "krutidev010"),
                        ("shusha", "krutidev010"), ("agra", "krutidev010")):
        if stem in key:
            return table, "guessed-by-name"
    return None, "unknown-font"


@dataclass
class Run:
    text: str
    font: str | None


@dataclass
class Block:
    """A paragraph or table cell -- a unit that ends with a line break."""
    runs: list[Run] = field(default_factory=list)

    @property
    def raw_text(self) -> str:
        return "".join(r.text for r in self.runs)


@dataclass
class Document:
    path: Path
    blocks: list[Block]
    #: Font name -> number of characters typed in it. This is the Phase 1.1
    #: triage histogram: one pass over the archive tells you exactly how much
    #: of it is legacy and which families you actually have to support.
    font_histogram: Counter[str] = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)


#: Scans and photographs. Read through ocr.py; see read_document.
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class UnsupportedFormat(Exception):
    """Raised for formats that need an external converter."""


def _style_fonts(styles_xml: bytes) -> tuple[dict[str, str], str | None]:
    """Map style id -> font, plus the document default font."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(styles_xml)
    out: dict[str, str] = {}
    default = None

    dflt = root.find(f"{W}docDefaults/{W}rPrDefault/{W}rPr/{W}rFonts")
    if dflt is not None:
        default = dflt.get(f"{W}ascii") or dflt.get(f"{W}hAnsi")

    for style in root.findall(f"{W}style"):
        sid = style.get(f"{W}styleId")
        fonts = style.find(f"{W}rPr/{W}rFonts")
        if sid and fonts is not None:
            name = fonts.get(f"{W}ascii") or fonts.get(f"{W}hAnsi")
            if name:
                out[sid] = name
    return out, default


def read_docx(path: Path) -> Document:
    import xml.etree.ElementTree as ET

    doc = Document(path=path, blocks=[])
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        if "word/document.xml" not in names:
            raise UnsupportedFormat(f"{path.name}: not a Word document package")
        style_fonts, default_font = ({}, None)
        if "word/styles.xml" in names:
            try:
                style_fonts, default_font = _style_fonts(z.read("word/styles.xml"))
            except ET.ParseError:
                doc.warnings.append("styles.xml unparseable; falling back to run-level fonts only")
        root = ET.fromstring(z.read("word/document.xml"))

    for para in root.iter(f"{W}p"):
        block = Block()
        p_style = para.find(f"{W}pPr/{W}pStyle")
        para_font = style_fonts.get(p_style.get(f"{W}val")) if p_style is not None else None

        for run in para.findall(f"{W}r"):
            rpr = run.find(f"{W}rPr")
            font = None
            if rpr is not None:
                fonts_el = rpr.find(f"{W}rFonts")
                if fonts_el is not None:
                    font = (fonts_el.get(f"{W}ascii") or fonts_el.get(f"{W}hAnsi")
                            or fonts_el.get(f"{W}cs"))
                if font is None:
                    rstyle = rpr.find(f"{W}rStyle")
                    if rstyle is not None:
                        font = style_fonts.get(rstyle.get(f"{W}val"))
            font = font or para_font or default_font

            pieces = []
            for child in run:
                tag = child.tag
                if tag == f"{W}t":
                    pieces.append(child.text or "")
                elif tag in (f"{W}br", f"{W}cr"):
                    pieces.append("\n")
                elif tag == f"{W}tab":
                    pieces.append("\t")
            text = "".join(pieces)
            if text:
                block.runs.append(Run(text, font))
                doc.font_histogram[font or "(unspecified)"] += len(text)

        if block.runs:
            doc.blocks.append(block)

    if not doc.blocks:
        doc.warnings.append("no text runs found -- document may be entirely images (scanned)")
    return doc


_MAGIC_DOC = b"\xd0\xcf\x11\xe0"


def read_document(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return read_docx(path)
    if suffix in (".doc", ".dot"):
        head = path.open("rb").read(4)
        if head == _MAGIC_DOC:
            raise UnsupportedFormat(
                f"{path.name}: binary Word 97 format. Convert first with:\n"
                f"  soffice --headless --convert-to docx --outdir <dir> '{path}'\n"
                "LibreOffice preserves run-level font information, which is what "
                "this pipeline needs; antiword and `catdoc` do not."
            )
        raise UnsupportedFormat(f"{path.name}: .doc extension but unrecognised header")
    if suffix == ".rtf":
        raise UnsupportedFormat(
            f"{path.name}: RTF. Convert with `soffice --headless --convert-to docx` "
            "so the \\fonttbl font assignments survive as run properties."
        )
    if suffix == ".pdf" or suffix in _IMAGE_SUFFIXES:
        # Handled in ocr.py, which also decides the source tier: a PDF with a
        # text layer is 'pdf', a scan or photograph is 'ocr'. Imported lazily
        # so the core pipeline does not depend on poppler or Tesseract being
        # installed -- an archive of .docx needs neither.
        #
        # The original warning stands and is repeated by ocr.read_pdf on every
        # text-layer read: a PDF made from a legacy Hindi font can carry
        # reordered matras that look right in a viewer (और typed `vkSj` can
        # extract as `vkjS` and convert to आरै).
        from .ocr import read_image, read_pdf
        doc, _report = (read_pdf(path) if suffix == ".pdf"
                        else read_image(path))
        return doc
    raise UnsupportedFormat(f"{path.name}: unhandled extension {suffix!r}")


def convert_document(doc: Document, *, latin_digits: bool = False,
                     force_table: str | None = None,
                     rescue_latin: bool = False) -> tuple[str, dict[str, int]]:
    """Convert a Document run by run. Returns text and a per-table char count.

    ``rescue_latin`` passes suspected mis-fonted Latin spans through untouched
    instead of converting them; see :func:`detect_misfonted_latin`. Without it,
    such spans are still recorded in ``doc.warnings``.
    """
    from .fonts.convert import Converter, normalize_devanagari

    cache: dict[str, Converter] = {}
    used: Counter[str] = Counter()
    out_blocks: list[str] = []

    for block in doc.blocks:
        pieces = []
        for run in _coalesce(block.runs, force_table):
            table = force_table if force_table else classify_font(run.font)[0]
            if table is None:
                pieces.append(run.text)
                used["(passthrough)"] += len(run.text)
                continue
            if table not in cache:
                cache[table] = Converter(table, latin_digits=latin_digits)
            conv = cache[table]

            spans = detect_misfonted_latin(run.text)
            if spans:
                doc.warnings.append(
                    "suspected Latin left in a legacy font run: "
                    + ", ".join(repr(s) for _, _, s in spans)
                    + ("" if rescue_latin else " (not rescued; pass rescue_latin=True)")
                )
            if spans and rescue_latin:
                cursor, parts = 0, []
                for start, end, span in spans:
                    parts.append(conv.convert(run.text[cursor:start], normalize=False).text)
                    parts.append(span)
                    cursor = end
                parts.append(conv.convert(run.text[cursor:], normalize=False).text)
                pieces.append("".join(parts))
                used["(rescued-latin)"] += sum(e - s for s, e, _ in spans)
                used[table] += len(run.text) - sum(e - s for s, e, _ in spans)
            else:
                pieces.append(conv.convert(run.text, normalize=False).text)
                used[table] += len(run.text)
        out_blocks.append("".join(pieces))

    return normalize_devanagari("\n".join(out_blocks)), dict(used)


def _coalesce(runs: list[Run], force_table: str | None) -> list[Run]:
    """Merge adjacent runs that convert with the same table.

    Word splits a paragraph into runs at revision-id and proofing boundaries
    with no regard for word boundaries, so a single Hindi word routinely
    arrives as several runs. Observed in a real archive file::

        run 1: 'f'        <- the pre-base chhoti-i matra, alone
        run 2: 'tyk'      <- the rest of जिला

    Converting those separately puts ``ि`` at the end of one fragment and
    ``जला`` at the start of the next, so the reordering pass never sees them
    adjacent and the output is ``िजला`` instead of ``जिला``. The same applies
    to reph, to multi-character slots such as ``'k``, and to any conjunct a
    run boundary happens to fall inside.

    Merging first is the only correct order: reordering is a property of the
    text, not of the formatting runs it happens to be stored in.
    """
    out: list[Run] = []
    for run in runs:
        table = force_table if force_table else classify_font(run.font)[0]
        if out:
            prev_table = (force_table if force_table
                          else classify_font(out[-1].font)[0])
            if prev_table == table:
                out[-1] = Run(out[-1].text + run.text, out[-1].font)
                continue
        out.append(Run(run.text, run.font))
    return out


#: A span of characters that are *only* uppercase Latin, digits and the
#: punctuation used in file numbers. Kruti Dev Hindi is overwhelmingly
#: lowercase (``Hkkjr``, ``dk;kZy;``), so a run of three or more of these with
#: two or more capitals is almost certainly English or a file number that the
#: typist left in the Hindi font by accident.
_MISFONTED_RE = re.compile(r"[A-Z0-9./\-]{3,}")

#: A rescued span must contain at least one of these. Legacy Devanagari
#: produces plenty of all-caps runs -- `mi;qZDRk` (उपर्युक्त) contains the
#: three-capital run `ZDR`, which the first version of this heuristic
#: "rescued" straight into the output as Latin. Real file numbers and
#: abbreviations essentially always carry a digit or a separator, so
#: requiring one removes that whole class of false positive while keeping
#: DEO/RPR/2024/1187 and F.No.
_RESCUE_REQUIRES = re.compile(r"[0-9./\-]")


def detect_misfonted_latin(text: str, *, min_uppercase: int = 2) -> list[tuple[int, int, str]]:
    """Find spans in a *legacy* run that look like Latin left in the wrong font.

    This is a detector, not an auto-fix. A file number such as
    ``DEO/RPR/2024/1187`` typed in a Kruti Dev run converts to half-consonant
    noise: faithful to the font, useless as data. But the heuristic cannot be
    made both safe and complete -- ``F.No.`` breaks at the lowercase ``o`` --
    so the pipeline surfaces these spans for review rather than silently
    rewriting them. Pass ``rescue_latin=True`` to ``convert_document`` to act
    on them anyway, which is the right call for an archive where letter
    numbers matter more than a handful of Hindi false positives.
    """
    out = []
    for m in _MISFONTED_RE.finditer(text):
        span = m.group(0)
        if (sum(1 for c in span if c.isupper()) >= min_uppercase
                and _RESCUE_REQUIRES.search(span)):
            out.append((m.start(), m.end(), span))
    return out
