"""Deterministic quality signals for converted Devanagari text.

No model, no dictionary required. Devanagari has a small, strict set of
sequence rules; a mis-mapped font slot or a glyph-order-scrambled PDF extract
violates them constantly, while correctly converted text almost never does.
That asymmetry is the cheapest corruption detector available, and it is the
gate that keeps a garbled letter out of the retrieval index.

WHAT THIS DOES NOT CATCH
------------------------
Sequence rules catch *illegal* text, not *wrong but legal* text. A font table
that maps one valid consonant to a different valid consonant produces perfectly
well-formed Devanagari that says the wrong thing, and this module will score it
1.0. Measured on a deliberate corruption that simply deletes every ``ि``, the
score barely moves.

So this is a tripwire, not a proof of correctness. The things that actually
establish correctness are, in order of strength:

1. the hand-verified gold set (``latters fonts gold``) -- the only real check;
2. the corpus-derived vocabulary hit rate, which does catch consonant swaps
   because the resulting words appear nowhere else in the archive;
3. these sequence rules, which catch reordering and slot-collision bugs
   instantly and for free.

Do not ship a converter on the strength of this module alone.

The score produced here is ``conversion_confidence`` in the trust formula:

    trust = 0.45 * conversion_confidence
          + 0.35 * completeness      (Phase 2, segmentation)
          + 0.20 * source_tier       (native Unicode > DOCX > PDF > OCR)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .fonts.convert import CONSONANTS, MATRAS, NUKTA, SIGNS, VIRAMA, normalize_devanagari

_CONS = rf"[{CONSONANTS}]"
_MATRA = rf"[{MATRAS}]"
_SIGN = rf"[{SIGNS}]"
_VOWEL = r"[ऄ-औॠॡॲ-ॷ]"
_DEVANAGARI = r"[ऀ-ॿ꣠-ꣿ]"

#: Words whose citation form genuinely ends in a virama -- common in official
#: Hindi. Stored WITHOUT the trailing virama: the lookbehind is evaluated at
#: the virama's own position, so it can only see the characters before it.
#: Extend from your own corpus; `latters ingest --json` reports the contexts
#: that fired this rule.
_GENUINE_VIRAMA_FINAL = (
    "एतद", "तहत", "विधिवत", "सम्यक", "महत", "जगत", "विद्युत",
    "भवत", "श्रीमत", "किञ्चित", "पश्चात", "कदाचित", "बृहत",
    "परिषद", "आयुष्मत", "विद्वत", "सक्षमत", "श्रीमान",
)
_NOT_GENUINE_FINAL = "".join(f"(?<!{w})" for w in _GENUINE_VIRAMA_FINAL)

#: NOTE: the two word-initial rules compile with re.MULTILINE. Without it `^`
#: anchors to the start of the whole document, so a word-initial matra was
#: only ever detected if it was the first character of the entire text --
#: which meant the single most important corruption signal fired essentially
#: never. Found when a real archive file containing `िजला` scored "clean".
#:
#: Each rule is (name, compiled regex, weight, human-readable explanation).
#: Weight is how many "error points" one occurrence contributes. Rules that
#: are merely unusual carry a low weight; rules that are impossible in
#: well-formed Devanagari carry a high one.
_RULES: list[tuple[str, re.Pattern[str], float, str]] = [
    (
        "matra_word_initial",
        re.compile(rf"(?:^|(?<=[\s\n(\[]))(?:{_MATRA}|{VIRAMA}|{NUKTA})", re.M),
        3.0,
        "dependent vowel sign or virama with nothing to attach to -- the classic "
        "symptom of pre-base ि not being reordered, or of PDF glyph reordering",
    ),
    (
        "double_matra",
        re.compile(rf"{_MATRA}{_MATRA}"),
        3.0,
        "two dependent vowel signs in a row -- impossible in Devanagari",
    ),
    (
        "virama_then_matra",
        re.compile(rf"{VIRAMA}{_MATRA}"),
        3.0,
        "virama followed by a vowel sign -- a dead consonant cannot take a matra",
    ),
    (
        "matra_after_vowel",
        re.compile(rf"{_VOWEL}{_MATRA}"),
        2.5,
        "dependent vowel sign after an independent vowel",
    ),
    (
        "vowel_after_virama",
        re.compile(rf"{VIRAMA}{_VOWEL}"),
        2.5,
        "independent vowel after a virama",
    ),
    (
        "nukta_misplaced",
        re.compile(rf"(?<!{_CONS}){NUKTA}"),
        2.0,
        "nukta not attached to a consonant",
    ),
    (
        "sign_word_initial",
        re.compile(rf"(?:^|(?<=[\s\n]))(?:{_SIGN})", re.M),
        2.0,
        "anusvara / visarga / candrabindu at the start of a word",
    ),
    (
        "virama_word_final",
        # Excluding the genuine virama-final words keeps this rule pointed at
        # real corruption instead of at correct official prose.
        re.compile(_NOT_GENUINE_FINAL + rf"{VIRAMA}(?=[\s\n.,।॥)\]]|$)"),
        0.5,
        "virama at word end -- legitimate in some abbreviations, so low weight, "
        "but a high rate means half-forms are being emitted where full "
        "consonants belong",
    ),
    (
        "latin_inside_word",
        # The unambiguous signature of a mapping slot we do not have: the
        # converter leaves an unknown key untouched, so a Latin letter ends up
        # welded to Devanagari with no space. Found by auditing a real archive
        # for legacy characters with no table entry -- `NqÎh` came out as
        # छुÎी ("छुट्टी") and scored *clean* at 0.965, because every other rule
        # here only looks at Devanagari. One rule catches every future missing
        # slot, which is worth more than the two slots it found.
        #
        # Two carve-outs, both measured against the archive rather than
        # guessed: U+00D7/U+00F7 are the multiplication and division signs,
        # not letters ("२० फीट×१० फीट" is a measurement), and Devanagari
        # DIGITS abut Latin legitimately in case numbers (११२१२२५५८८६८८/१A).
        # With both, the rule fired 8 times in 72,921 words of a real
        # archive and every hit was genuine corruption.
        re.compile(r"(?:[A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f]"
                   r"(?=[\u0900-\u0965\u0970-\u097f\ua8e0-\ua8ff])"
                   r"|[\u0900-\u0965\u0970-\u097f\ua8e0-\ua8ff]"
                   r"(?=[A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f]))"),
        3.0,
        "a Latin letter welded to Devanagari with no space -- nearly always a "
        "legacy key with no entry in the mapping table, passed through raw",
    ),
    (
        "pua_leak",
        re.compile(r"[-]"),
        5.0,
        "private-use character reached the output -- a converter internal "
        "sentinel escaped, which is a bug, not a data problem",
    ),
    (
        "replacement_char",
        re.compile(r"[�]"),
        5.0,
        "U+FFFD replacement character -- the source was decoded with the wrong "
        "codec before conversion even started",
    ),
]

_WORD_RE = re.compile(rf"{_DEVANAGARI}+")

#: A corpus-derived vocabulary below this size is not evidence of anything --
#: it just makes every document look full of unknown words. A real archive of
#: a few thousand letters yields tens of thousands of entries, so this only
#: ever trips on tiny test corpora and first-day pilot runs.
MIN_USEFUL_VOCABULARY = 500


@dataclass
class Quality:
    #: 0.0 (certainly garbage) .. 1.0 (clean). This is conversion_confidence.
    score: float
    devanagari_ratio: float
    violations: dict[str, int] = field(default_factory=dict)
    #: Fraction of Devanagari words found in the supplied vocabulary. ``None``
    #: when no vocabulary was given.
    vocab_hit_rate: float | None = None
    n_devanagari_words: int = 0
    #: Rendered explanations for the violations that actually fired, so the
    #: admin UI can say *why* a letter was quarantined.
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.score >= 0.85:
            return "clean"
        if self.score >= 0.60:
            return "review"
        return "quarantine"


def devanagari_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    deva = sum(1 for c in letters if "ऀ" <= c <= "ॿ")
    return deva / len(letters)


def assess(text: str, *, vocabulary: set[str] | None = None, expect_devanagari: bool = True) -> Quality:
    """Score converted text. Cheap enough to run on every segment at ingest."""
    text = normalize_devanagari(text)
    words = _WORD_RE.findall(text)
    n_words = len(words)
    ratio = devanagari_ratio(text)

    violations: dict[str, int] = {}
    notes: list[str] = []
    error_points = 0.0
    for name, pattern, weight, explanation in _RULES:
        hits = len(pattern.findall(text))
        if hits:
            violations[name] = hits
            error_points += hits * weight
            notes.append(f"{name} x{hits}: {explanation}")

    # Normalise error points per word so long letters are not punished for
    # being long. A clean letter scores ~0; one bad slot in a common word
    # scores well above 1.
    denom = max(n_words, 1)
    error_rate = error_points / denom
    sequence_score = max(0.0, 1.0 - error_rate)

    vocab_rate: float | None = None
    if vocabulary and n_words and len(vocabulary) >= MIN_USEFUL_VOCABULARY:
        vocab_rate = sum(1 for w in words if w in vocabulary) / n_words
    elif vocabulary and n_words:
        # A vocabulary derived from a handful of documents is mostly empty, so
        # every real word looks unknown and clean letters get marked "review".
        # Below the threshold the signal is worse than no signal at all.
        notes.append(
            f"vocabulary has only {len(vocabulary)} words "
            f"(< {MIN_USEFUL_VOCABULARY}); ignoring the vocabulary signal"
        )

    # Weighted blend. Sequence legality is the primary signal because it needs
    # no resources and has almost no false positives; vocabulary is a strong
    # secondary signal when a wordlist exists; script ratio catches the case
    # where conversion did not happen at all.
    if expect_devanagari:
        if n_words == 0:
            # No Devanagari at all. Sequence legality is vacuously perfect here,
            # so blending it in would score untouched Latin gibberish as
            # "review" instead of "quarantine". Fall through to the ratio.
            score = ratio
            notes.append(
                "no Devanagari words found -- conversion did not run, ran with "
                "the wrong table, or this run was never in a legacy font"
            )
        else:
            score = 0.65 * sequence_score + 0.35 * ratio
            if vocab_rate is not None:
                score = 0.50 * sequence_score + 0.30 * vocab_rate + 0.20 * ratio
    else:
        score = sequence_score

    if 0 < n_words < 3:
        notes.append("very short sample -- score is not meaningful")

    return Quality(
        score=round(min(1.0, max(0.0, score)), 4),
        devanagari_ratio=round(ratio, 4),
        violations=violations,
        vocab_hit_rate=None if vocab_rate is None else round(vocab_rate, 4),
        n_devanagari_words=n_words,
        notes=notes,
    )


def build_vocabulary(texts: list[str], *, min_count: int = 3) -> set[str]:
    """Derive a vocabulary from the corpus itself.

    An office archive is its own best wordlist: departmental Hindi is full of
    terms no general Hindi dictionary contains (scheme names, post titles,
    local place names). Words seen ``min_count`` times across many documents
    are almost certainly real, because a conversion bug produces *different*
    garbage each time rather than the same word repeatedly.
    """
    counts: dict[str, int] = {}
    for t in texts:
        for w in set(_WORD_RE.findall(normalize_devanagari(t))):
            counts[w] = counts.get(w, 0) + 1
    return {w for w, c in counts.items() if c >= min_count}
