"""Post-conversion repairs for artefacts of how legacy typists actually worked.

These are not font-table bugs. They are cases where the legacy encoding had no
slot for the character the typist wanted, so they used a visually identical
one. The glyph looked right on paper and nobody ever noticed. Converted
faithfully, the result is correct Devanagari that means the wrong thing.

Every rule here is narrow and reversible, and every one is off by default in
the converter -- call ``repair()`` explicitly so the raw conversion stays
auditable against the gold set.
"""

from __future__ import annotations

import re

VISARGA = "ः"

#: Words where a trailing visarga is genuine Sanskrit-derived orthography and
#: must never be turned into a colon. Extend this from your own corpus.
_GENUINE_VISARGA_WORDS = frozenset(
    """अतः प्रायः स्वतः मुख्यतः विशेषतः सामान्यतः पूर्णतः अंशतः क्रमशः
    यथासंभवतः संभवतः वस्तुतः मूलतः प्रथमतः अंततः निःसंदेह दुःख
    पुनः शनैः प्रातः""".split()
)

#: Labels that are followed by a colon in every departmental letter ever
#: written. A visarga here is a colon the typist could not type.
_COLON_LABELS = (
    "विषय", "संदर्भ", "सन्दर्भ", "प्रतिलिपि", "संख्या", "क्रमांक",
    "दिनांक", "प्रेषक", "सेवा में", "प्रति", "उत्तर", "अनुलग्नक",
    "टिप्पणी", "कृते", "पृष्ठांकन",
)

_LABEL_VISARGA_RE = re.compile(
    r"(" + "|".join(re.escape(w) for w in _COLON_LABELS) + r")" + VISARGA
)
_ISOLATED_VISARGA_RE = re.compile(r"(?<=[\s\n])" + VISARGA)
_WORD_FINAL_VISARGA_RE = re.compile(r"([ऀ-ॿ]+)" + VISARGA + r"(?=\s|$)")


def visarga_to_colon(text: str) -> tuple[str, int]:
    """Turn typed-as-visarga colons back into colons.

    Kruti Dev has no ASCII colon slot -- ``:`` is ``रू`` -- so typists used
    ``%`` (visarga), which renders as two stacked dots and reads as a colon on
    paper. ``विषयः`` in a converted letter is ``विषय:`` nine times out of ten,
    and the exception list keeps ``अतः`` and friends intact.

    Returns the repaired text and the number of substitutions made.
    """
    count = 0

    def _label(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return m.group(1) + ":"

    text, n = _LABEL_VISARGA_RE.subn(_label, text)

    def _word_final(m: re.Match[str]) -> str:
        nonlocal count
        word = m.group(1)
        # The exception list stores words *with* their visarga, but the capture
        # group stops before it -- compare the full form.
        if word + VISARGA in _GENUINE_VISARGA_WORDS:
            return m.group(0)
        count += 1
        return word + ":"

    text = _WORD_FINAL_VISARGA_RE.sub(_word_final, text)
    text, n2 = _ISOLATED_VISARGA_RE.subn(":", text)
    count += n2
    return text, count


_SPACED_PUNCT_RE = re.compile(r"\s+([,;:।॥)\]])")
_MISSING_SPACE_RE = re.compile(r"([,;:।])(?=[ऀ-ॿA-Za-z])")


def tidy_punctuation(text: str) -> str:
    """Fix spacing that legacy line-breaking left behind."""
    text = _SPACED_PUNCT_RE.sub(r"\1", text)
    text = _MISSING_SPACE_RE.sub(r"\1 ", text)
    return text


def repair(text: str) -> tuple[str, dict[str, int]]:
    """Apply all repairs. Returns text and a per-rule count for the audit log."""
    text, n_visarga = visarga_to_colon(text)
    before = text
    text = tidy_punctuation(text)
    return text, {
        "visarga_to_colon": n_visarga,
        "punctuation_tidied": int(before != text),
    }
