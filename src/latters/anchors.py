"""Structural anchors in departmental Hindi letters.

Government letters are extraordinarily formulaic, which is the single reason
segmentation does not need a language model. Each letter carries a small,
predictable set of landmarks in a near-fixed order:

    header / office name
    letter number                    पत्र संख्या / क्रमांक / फा.सं. / F.No.
    date                             दिनांक
    addressee                        सेवा में / प्रति
    subject                          विषय:
    reference                        संदर्भ:
    salutation                       महोदय,
    body
    closing                          भवदीय
    signatory
    distribution                     प्रतिलिपि:

Detecting those is a regex problem, not an inference problem.

TUNING THIS FILE IS EXPECTED WORK, NOT A BUG
--------------------------------------------
These patterns cover forms common across central and state offices, but every
office has house style: its own letter-number prefixes, its own closing
formula, its own transliteration habits. `latters segment audit` reports which
anchors never fire across your archive and which lines look like boundaries
but match nothing -- work that list before trusting any segmentation.

Offices can extend without editing code: pass a JSON file to
``load_anchors(extra=...)`` mapping anchor names to extra regex strings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Role(Enum):
    """What an anchor means for segmentation."""
    START = "start"        #: plausibly begins a letter
    END = "end"            #: plausibly ends one
    CONTENT = "content"    #: a landmark inside a letter
    NOISE = "noise"        #: page furniture, never content or a boundary


@dataclass(frozen=True)
class Anchor:
    name: str
    role: Role
    pattern: re.Pattern[str]
    #: Contribution to the completeness score, 0 for anchors that do not count.
    weight: float = 0.0


def _p(*alts: str) -> str:
    return "(?:" + "|".join(alts) + ")"


# Devanagari or Latin digits, the separators offices actually use, and the
# optional ordinal/abbreviation dots that appear in scanned text.
_D = r"[0-9०-९]"
_SEP = r"[\-/.–—]"
_DATE = rf"{_D}{{1,2}}\s*{_SEP}\s*{_D}{{1,2}}\s*{_SEP}\s*{_D}{{2,4}}"
_MONTHS = _p("जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई",
             "अगस्त", "सितम्बर", "सितंबर", "अक्टूबर", "नवम्बर", "नवंबर", "दिसम्बर", "दिसंबर")

_ANCHOR_SPECS: list[tuple[str, Role, str, float]] = [
    # --- letter identity ---------------------------------------------------
    ("letter_number", Role.START, _p(
        r"पत्र\s*(?:क्रमांक|संख्या|सं0?\.?|सङ्ख्या)",
        r"पत्रांक",
        r"क्रमांक\s*[:\-ः]",
        r"(?:आदेश|ज्ञापन|अधिसूचना|परिपत्र)\s*(?:क्रमांक|संख्या|सं0?\.?)",
        r"ज्ञापांक",          # 228 occurrences in the Saran archive
        r"पृष्ठांकित\s*पत्रांक",
        r"\bफा\s*\.?\s*सं\s*\.?",
        r"\bF\s*\.?\s*No\s*\.?",
        r"\bNo\s*\.\s*(?=[A-Za-z0-9])",
        r"\bसं\s*\.\s*(?=[0-9०-९])",
    ), 0.20),

    # --- header / letterhead ----------------------------------------------
    ("header", Role.START, _p(
        r"भारत\s*सरकार", r"राज्य\s*सरकार", r"शासन\s*[,\s]",
        r"कार्यालय\s*[,\-–]?\s*\S", r"मुख्य\s*कार्यालय",
        r"(?:महा)?निदेशालय", r"मंत्रालय", r"सचिवालय",
        r"(?:जिला|संभाग|मंडल|नगर)\s*(?:कार्यालय|पंचायत|निगम|परिषद)",
        r"आयुक्त\s*कार्यालय", r"कलेक्टर\s*कार्यालय",
        r"\bOffice\s+of\s+the\b", r"\bGovernment\s+of\b",
    ), 0.0),

    # --- content landmarks -------------------------------------------------
    ("date", Role.CONTENT, _p(
        rf"दिनांक\s*[:\-ः]?\s*{_D}", rf"दि\s*\.\s*{_D}",
        # A blank template date is still a date line. 95% of the sample
        # archive reads `दिनांक------------------`, and requiring a digit
        # meant none of them counted -- which cost the completeness score and
        # left the template skeleton unable to order its own date line.
        r"दिनांक\s*[:\-ः]?\s*[-\u2013\u2014_.]{3,}",
        r"(?:दिनंाक|दिनाक)\s*[:\-ः]?",
        rf"{_D}{{1,2}}\s+{_MONTHS}\s+{_D}{{2,4}}",
        rf"(?<![0-9\u0966-\u096F{_SEP}]){_DATE}(?![0-9\u0966-\u096F])",
        r"\bDated?\s*[:\-]?\s*",
    ), 0.15),

    ("addressee", Role.CONTENT, _p(
        r"^\s*सेवा\s*में\s*[,ः:]?\s*$", r"^\s*सेवा\s*में\s*[,ः:]",
        r"^\s*प्रति\s*[,ः:]?\s*$", r"^\s*प्रति\s*[,ः:]",
        r"^\s*प्रेषिती\s*[,ः:]?", r"^\s*To\s*[,:]?\s*$",
    ), 0.15),

    ("subject", Role.CONTENT, _p(
        r"विषय\s*[:\-ः–]", r"^\s*विषय\s+", r"\bSub(?:ject)?\s*[:\-]",
    ), 0.20),

    ("reference", Role.CONTENT, _p(
        # प्रसंग is this archive's word for it: 270 occurrences against zero
        # for संदर्भ. Measured, not assumed -- see docs/PHASE2_FINDINGS.md.
        r"प्रसंग\s*[:\-ः–]", r"प्रसङ्ग\s*[:\-ः–]", r"विषयक\s*[:ः]",
        r"संदर्भ\s*[:\-ः–]", r"सन्दर्भ\s*[:\-ः–]",
        r"कृपया\s+(?:उपरोक्त\s+)?संदर्भ", r"\bRef(?:erence)?\s*[:\-]",
        r"आपके\s+पत्र\s+(?:क्रमांक|संख्या)",
    ), 0.0),

    ("salutation", Role.CONTENT, _p(
        # महाशय is the Bihar/eastern-UP form and was 356 of 356 salutations in
        # the sample archive; महोदय did not appear once. Both are kept: this
        # file has to serve more than one office.
        r"^\s*महाशय(?:ा)?\s*[,ः:]?\s*$", r"^\s*महाशय(?:ा)?\s*[,ः:]",
        r"^\s*महोदय(?:ा)?\s*[,ः:]?\s*$", r"^\s*महोदय(?:ा)?\s*[,ः:]",
        r"^\s*महाभाग\s*[,ः:]?", r"^\s*प्रिय\s+मह(?:ोदय|ाशय)(?:ा)?",
        r"^\s*माननीय\s+मह(?:ोदय|ाशय)", r"^\s*Sir\s*[,:]", r"^\s*Madam\s*[,:]",
    ), 0.0),

    # --- terminators -------------------------------------------------------
    ("closing", Role.END, _p(
        # MEASURED, not assumed. In the sample archive विश्वासभाजन appears on
        # 382 lines entirely BARE; requiring the आपका prefix, as the first
        # version of this file did, matched none of them and pushed 45% of all
        # boundary decisions onto the repeated-subject fallback.
        # Line counts from that archive are given per form.
        r"विश्वासभाजन",                    # 382
        r"विश्वासपात्र",
        r"हस्ताक्षर(?:ित)?",               # 98
        # अनु० (अनुलग्नक) sits in the signature block here, as अनु०यथोक्त।,
        # अनु०ः-यथोपरि। and bare अनु०:-
        r"अनु\s*[०0]",                     # 93
        r"भवदीय(?:ा)?",                    # 52
        r"सादर", r"शुभकामनाओं\s+सहित",
        r"\(\s*हस्ता(?:।|\.|़)?\s*\)", r"ह\s*/\s*-",
        r"\bYours\s+(?:faithfully|sincerely|truly)\b",
    ), 0.15),

    ("distribution", Role.END, _p(
        r"प्रतिलिपि\s*[:\-ः–]", r"प्रतिलिपि\s+(?:सूचनार्थ|प्रेषित)",
        r"पृष्ठांकन\s*(?:क्रमांक|संख्या)?", r"पृष्ठांकित",
        r"\bCopy\s+(?:to|forwarded)\b", r"संलग्न\s*[:\-ः–]",
    ), 0.0),

    # --- page furniture ----------------------------------------------------
    ("noise", Role.NOISE, _p(
        rf"^\s*[\-–(]?\s*{_D}{{1,3}}\s*[\-–)]?\s*$",          # bare page number
        rf"^\s*पृष्ठ\s*[:\-]?\s*{_D}", rf"^\s*Page\s+{_D}",
        rf"^\s*{_D}{{1,3}}\s*/\s*{_D}{{1,3}}\s*$",            # 2/5
        r"^\s*क्रमशः\s*[.।]*\s*$", r"^\s*जारी\s*[.।]*\s*$",
        r"^\s*\.{3,}\s*$", r"^\s*[-–_=*]{3,}\s*$",
    ), 0.0),
]

#: Anchors whose presence contributes to the completeness score, and how much.
#: They sum to 1.0; `header`, `reference`, `salutation` and `distribution`
#: deliberately contribute nothing because they are too often absent in
#: genuine, complete letters to penalise.
COMPLETENESS_WEIGHTS = {name: w for name, _, _, w in _ANCHOR_SPECS if w}


def load_anchors(extra: Path | dict[str, list[str]] | None = None) -> list[Anchor]:
    """Compile the anchor set, optionally extended with office-specific forms."""
    extras: dict[str, list[str]] = {}
    if isinstance(extra, Path):
        extras = json.loads(extra.read_text(encoding="utf-8"))
    elif isinstance(extra, dict):
        extras = extra

    out = []
    for name, role, pattern, weight in _ANCHOR_SPECS:
        if name in extras:
            pattern = _p(pattern, *extras[name])
        out.append(Anchor(name, role, re.compile(pattern, re.MULTILINE), weight))

    unknown = set(extras) - {a.name for a in out}
    if unknown:
        raise ValueError(f"unknown anchor name(s) in extras: {sorted(unknown)}")
    return out


ANCHORS = load_anchors()
BY_NAME = {a.name: a for a in ANCHORS}
