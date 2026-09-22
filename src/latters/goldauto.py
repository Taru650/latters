"""Validate the mapping tables against OCR, with no Hindi reader.

The conversion is the one thing in this project that has never been checked,
and the reason given everywhere is that only a person who reads Hindi can
check it. That is nearly true, and the "nearly" is worth a lot.

Three routes to an automatic check were tried and closed:

* **Download the real Kruti Dev fonts and render them here.** Not reachable
  from this environment.
* **Synthesise a font from the mapping table.** Circular: OCR of a font
  built from the table can only confirm the table against itself.
* **Use Unicode Devanagari already in the archive.** Measured: the five real
  files contain 488,928 legacy characters and **zero** Unicode Devanagari.
  There is no free parallel text.

The fourth route works, and it costs the office thirty seconds.

**Open one legacy .docx in Word on a machine where the fonts ARE installed,
and export it to PDF.** Word renders the Kruti Dev glyphs correctly, because
it has the font. OCR then reads those rendered glyphs and produces Unicode
Hindi **without consulting the mapping table at all**. Two independent
readings of the same letter:

    letter.docx  --(mapping tables)-->  text A
    letter.pdf   --(Tesseract)------->  text B

Where they agree, the mapping is almost certainly right -- two unrelated
methods do not make the same mistake. Where they disagree, one of them is
wrong and a human has to say which.

What this is and is not
-----------------------
It is **not** a replacement for the gold set. OCR has its own 2-4% error
rate, and when the two disagree this cannot say which is correct.

It is a **localiser**. It turns "review 200 blind lines of Hindi" into
"adjudicate these 15 words where two independent readings differ", which is
minutes of a Hindi reader's time instead of hours -- and it finds systematic
mapping errors, which are the ones that matter, because a wrong slot
disagrees on every single occurrence.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

#: A disagreement seen this many times is a mapping bug, not an OCR slip.
#: OCR errors are scattered; a wrong table slot is wrong every time.
SYSTEMATIC_AT = 3

_WORD = re.compile(r"[ऀ-ॿ]+")


def _words(text: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFC", text))


@dataclass
class Disagreement:
    converted: str
    ocr: str
    count: int = 1

    @property
    def systematic(self) -> bool:
        return self.count >= SYSTEMATIC_AT


@dataclass
class AutoGold:
    """Two independent readings of one letter, compared."""

    words_converted: int = 0
    words_ocr: int = 0
    agreed: int = 0
    disagreements: list[Disagreement] = field(default_factory=list)
    ocr_confidence: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def agreement(self) -> float:
        n = max(self.words_converted, 1)
        return self.agreed / n

    @property
    def systematic(self) -> list[Disagreement]:
        return [d for d in self.disagreements if d.systematic]

    def render(self, *, show: int = 25) -> str:
        out = [
            f"words, font conversion : {self.words_converted}",
            f"words, OCR             : {self.words_ocr}",
            f"agreed                 : {self.agreed} "
            f"({self.agreement:.1%})",
        ]
        if self.ocr_confidence is not None:
            out.append(f"OCR confidence (median): {self.ocr_confidence:.0f}/100")
        out.append("")

        if self.agreement >= 0.95 and not self.systematic:
            out.append("Two independent readings agree on almost every word.")
            out.append("That is not proof the Hindi is right, but a wrong "
                       "mapping slot would")
            out.append("disagree on every occurrence, and none does.")
        elif self.systematic:
            out.append(f"!! {len(self.systematic)} REPEATED disagreement(s). "
                       f"OCR errors are scattered;")
            out.append("!! a wrong mapping slot is wrong every time. These are "
                       "the suspects:")
            out.append("")
            out.append(f"   {'seen':>4}  {'our conversion':<24} OCR reads")
            for d in sorted(self.systematic, key=lambda d: -d.count)[:show]:
                out.append(f"   {d.count:>4}  {d.converted:<24} {d.ocr}")
        else:
            out.append("No repeated disagreement. The differences below "
                       "occur once each,")
            out.append("which is the shape of OCR noise rather than a "
                       "mapping error.")

        others = [d for d in self.disagreements if not d.systematic]
        if others:
            out.append("")
            out.append(f"   {len(others)} one-off difference(s), first few:")
            for d in others[:8]:
                out.append(f"         {d.converted:<24} {d.ocr}")

        out.extend(["", "This cannot say WHICH reading is right when they "
                        "differ -- OCR has its own",
                    "2-4% error rate. It says where to look. A Hindi reader "
                    "adjudicating the",
                    "repeated rows above is minutes of work; a blind review "
                    "of 200 lines is hours."])
        return "\n".join(out)


def compare(converted: str, ocr: str, *,
            ocr_confidence: float | None = None) -> AutoGold:
    """Align two readings of the same letter and report where they differ.

    Word-level, and aligned with `SequenceMatcher` rather than zipped:
    OCR drops and inserts words (a stamp read as a word, a line missed), so
    position `i` in one is not position `i` in the other. Zipping them would
    report every word after the first drop as a disagreement.
    """
    a, b = _words(converted), _words(ocr)
    result = AutoGold(words_converted=len(a), words_ocr=len(b),
                      ocr_confidence=ocr_confidence)
    if not a or not b:
        result.notes.append(
            "one side has no Devanagari at all -- check that the PDF was "
            "exported from the SAME letter, and on a machine with the "
            "legacy fonts installed.")
        return result

    seen: dict[tuple[str, str], Disagreement] = {}
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=a, b=b).get_opcodes():
        if tag == "equal":
            result.agreed += i2 - i1
        elif tag == "replace":
            # Pair them off positionally inside the replaced block. It is
            # imperfect where the block is long, but a systematic mapping
            # error produces short blocks of one or two words and those
            # pair correctly.
            for k in range(max(i2 - i1, j2 - j1)):
                lhs = a[i1 + k] if i1 + k < i2 else ""
                rhs = b[j1 + k] if j1 + k < j2 else ""
                key = (lhs, rhs)
                if key in seen:
                    seen[key].count += 1
                else:
                    seen[key] = Disagreement(lhs, rhs)
        elif tag == "delete":
            for w in a[i1:i2]:
                key = (w, "")
                if key in seen:
                    seen[key].count += 1
                else:
                    seen[key] = Disagreement(w, "(missing in OCR)")
        # `insert` is OCR seeing something the document does not contain --
        # a stamp, a page number, a logo read as text. Not a mapping signal.

    result.disagreements = list(seen.values())
    return result


def from_files(docx: Path, pdf: Path, *, latin_digits: bool = False) -> AutoGold:
    """Convert the .docx by the mapping tables, OCR the PDF, compare."""
    from .extract import convert_document, read_document
    from .ocr import read_pdf

    doc = read_document(docx)
    converted, _ = convert_document(doc, latin_digits=latin_digits,
                                    rescue_latin=True)

    # force_ocr: the PDF was exported from Word, so it HAS a text layer --
    # and that text layer is the legacy bytes again, which would compare the
    # mapping table against itself. The rendered glyphs are the whole point.
    ocr_doc, report = read_pdf(pdf, force_ocr=True)
    ocr_text, _ = convert_document(ocr_doc)

    out = compare(converted, ocr_text,
                  ocr_confidence=report.mean_word_confidence)
    out.notes.extend(report.warnings)
    return out
