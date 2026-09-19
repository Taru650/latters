"""Legacy Devanagari font -> Unicode conversion.

Three passes, in this order:

1. **Substitution** -- maximal-munch replacement using the mapping table.
   Longest legacy key wins, so ``vks`` becomes ``ओ`` rather than ``आ`` + ``े``.

2. **chhoti-i reordering** -- Kruti Dev stores ``ि`` *before* the consonant
   cluster it attaches to (that is where the glyph is drawn); Unicode stores it
   *after*. ``fdZ`` -> ``ि क REPH`` -> ``क ि REPH``.

3. **Reph reordering** -- the ``र्`` that renders as a hook above the syllable
   is typed *after* the cluster. Unicode puts ``र`` + virama *before* it.
   ``deZ`` -> ``क म REPH`` -> ``कर्म``.

Worked example, ``कीर्ति``::

    dhfrZ  ->  क ी ि त REPH        (substitution)
           ->  क ी त ि REPH        (pass 2: ि moves after त)
           ->  क ी र् त ि          (pass 3: REPH becomes र् before त)

Passes 2 and 3 are why a naive character-for-character table replacement --
which is what most online converters and every hand-rolled ``str.replace``
loop does -- silently produces wrong words.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .tables import REPH_MARK, FontTable, load_table

# --- Devanagari character classes ----------------------------------------
CONSONANTS = r"क-हक़-य़ॸ-ॿ"
NUKTA = "़"
VIRAMA = "्"
MATRAS = r"ऺऻा-ौॎॏॕ-ॗॢॣ"
SIGNS = r"ऀ-ः"
I_MATRA = "ि"
REPH = "र" + VIRAMA  # र्

#: A consonant cluster: base consonant, optional nukta, then any number of
#: virama-joined consonants. This is the unit both reordering passes pivot on.
_CLUSTER = rf"[{CONSONANTS}]{NUKTA}?(?:{VIRAMA}[{CONSONANTS}]{NUKTA}?)*"

_I_MATRA_RE = re.compile(rf"{I_MATRA}({_CLUSTER})")
_REPH_RE = re.compile(rf"({_CLUSTER})([{MATRAS}{SIGNS}]*){REPH_MARK}")


@dataclass(frozen=True)
class Conversion:
    text: str
    #: Characters the table had no entry for, with counts. A high count on a
    #: character that is not whitespace or ASCII punctuation means the table is
    #: wrong for this font -- investigate before trusting the output.
    unmapped: dict[str, int]
    table: str


class Converter:
    def __init__(self, table: FontTable | str = "krutidev010", *, latin_digits: bool = False):
        if isinstance(table, str):
            table = load_table(table)
        if latin_digits:
            table = table.without_digits()
        self.table = table
        keys = sorted(table.mapping, key=len, reverse=True)
        # Alternation is first-match-wins in Python's re, and the keys are
        # sorted longest-first, which gives maximal munch.
        self._pattern = re.compile("|".join(re.escape(k) for k in keys))

    def convert(self, text: str, *, normalize: bool = True) -> Conversion:
        if not text:
            return Conversion("", {}, self.table.name)

        unmapped: dict[str, int] = {}
        matched_spans: list[tuple[int, int]] = []

        def _sub(m: re.Match[str]) -> str:
            matched_spans.append(m.span())
            return self.table.mapping[m.group(0)]

        out = self._pattern.sub(_sub, text)

        # Record what the table did not cover, so callers can tell "this font
        # is not Kruti Dev" apart from "this letter had some English in it".
        covered = bytearray(len(text))
        for start, end in matched_spans:
            for i in range(start, end):
                covered[i] = 1
        for i, ch in enumerate(text):
            if not covered[i] and not ch.isspace():
                unmapped[ch] = unmapped.get(ch, 0) + 1

        out = _I_MATRA_RE.sub(r"\1" + I_MATRA, out)
        out = _REPH_RE.sub(REPH + r"\1\2", out)
        # A reph with no cluster in front of it (line start, or a table gap)
        # still has to become something readable rather than a PUA character.
        out = out.replace(REPH_MARK, REPH)

        if normalize:
            out = normalize_devanagari(out)
        return Conversion(out, unmapped, self.table.name)


#: U+0958..U+095F (क़ ख़ ग़ ज़ ड़ ढ़ फ़ य़) are Unicode composition exclusions: NFC
#: will not create them, and text in the wild uses both forms. Retrieval breaks
#: if the corpus mixes them, so everything is normalised to the decomposed
#: (consonant + nukta) form, which is what NFD/NFC round-trips stably.
_NUKTA_DECOMPOSE = {chr(cp): unicodedata.normalize("NFD", chr(cp)) for cp in range(0x0958, 0x0960)}
_NUKTA_DECOMPOSE.update({chr(cp): unicodedata.normalize("NFD", chr(cp)) for cp in (0x0929, 0x0931, 0x0934)})


def normalize_devanagari(text: str) -> str:
    """Canonical form for indexing and comparison.

    Idempotent. Safe to apply to already-Unicode archive text as well, which is
    the point -- native-Unicode and converted letters must normalise alike or
    retrieval will silently miss matches.
    """
    text = unicodedata.normalize("NFC", text)
    for composed, decomposed in _NUKTA_DECOMPOSE.items():
        text = text.replace(composed, decomposed)
    # Collapse the two danda variants' surrounding whitespace, strip ZWJ/ZWNJ
    # that legacy converters scatter around, and normalise line endings.
    text = text.replace("‌", "").replace("‍", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def convert(text: str, table: str = "krutidev010", *, latin_digits: bool = False) -> str:
    """Convenience wrapper for one-off conversions."""
    return Converter(table, latin_digits=latin_digits).convert(text).text
