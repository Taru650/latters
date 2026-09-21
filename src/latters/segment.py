"""Split a file containing many letters into individual letters.

Office archives routinely hold one .doc per year per clerk, with dozens of
letters concatenated and no reliable separator -- no page break, no rule, no
blank-line convention. Segmentation therefore works off the structural
anchors in `anchors.py` rather than off layout.

The rule, in one sentence: a letter begins at a header or letter-number line
that follows a closing or distribution line.

That single rule handles the common case, because the multi-line letterhead
block at the top of each letter contains several candidate starts and only
the first one follows a closing. Two further passes catch what it misses:

* **Second-subject split.** A truncated letter with no closing merges into the
  next one. A segment carrying two ``विषय:`` lines is therefore split before
  the second, backing up to the nearest preceding header, letter number or
  addressee so the second letter keeps its own letterhead.
* **Noise suppression.** Page numbers, ``क्रमशः``, rules and ``2/5`` markers
  are dropped before boundary detection, so a page break inside one letter
  cannot look like the end of it.

Nothing here uses a language model. Rules are auditable, instant, and can be
corrected by an office clerk reading `anchors.py`; a 1B model's segmentation
decisions can be neither explained nor fixed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .anchors import ANCHORS, COMPLETENESS_WEIGHTS, Anchor, Role

# --------------------------------------------------------------------------
# Document form
# --------------------------------------------------------------------------
# Not every document in an office archive is a letter, and scoring them all
# against a letter's anatomy is wrong in a way that matters. Measured on a
# real district archive: 211 of 589 segments have no addressee at all, and
# 169 of those scored below 0.7 -- 29% of the corpus marked defective for
# being the wrong genre.
#
# An आदेश (order) is addressed to nobody, carries no विषय line and ends with
# a signature and a distribution list rather than भवदीय. Those are not
# missing parts. They are the form.


class Form(Enum):
    #: Addressed to someone: सेवा में / प्रति, विषय, महाशय, भवदीय.
    LETTER = "letter"
    #: आदेश, ज्ञापन, अधिसूचना, परिपत्र -- issued, not sent.
    ORDER = "order"
    #: No addressee and none of an order's own signals either. Most likely a
    #: truncated segment, and scored as one.
    FRAGMENT = "fragment"


#: An explicit form marker standing alone on its line.
_ORDER_MARKER = re.compile(
    r"^\s*(?:कार्यालय\s*)?(?:आदेश|ज्ञापन|अधिसूचना|परिपत्र|संकल्प|"
    r"पृष्ठांकन|अनुदेश)\s*[।:\-–]?\s*$", re.M)
_ADDRESSEE_LINE = re.compile(r"^\s*(?:सेवा\s*में|प्रति|प्रेषिती|To)\s*[,ः:]?\s*$", re.M)
#: How far into a document a form marker still counts as its heading.
_MARKER_WINDOW_LINES = 14


def detect_form(text: str, anchors: dict[str, int] | None = None) -> Form:
    """Classify the genre before scoring completeness against it.

    Conservative on purpose. A document with no addressee is only treated as
    an order if it either says so or behaves like one -- a number, a date and
    a real body. Otherwise it stays a FRAGMENT and keeps its low score,
    because the alternative is laundering every truncated segment into
    "complete" by calling it an order.
    """
    if _ADDRESSEE_LINE.search(text):
        return Form.LETTER
    head = "\n".join(text.splitlines()[:_MARKER_WINDOW_LINES])
    if _ORDER_MARKER.search(head):
        return Form.ORDER
    a = anchors or {}
    if a.get("letter_number") and a.get("date") and len(text) >= BODY_TARGET_CHARS:
        return Form.ORDER
    return Form.FRAGMENT


#: Per-form completeness weights. Each set sums to 1.0 with BODY_WEIGHT.
FORM_WEIGHTS: dict[Form, dict[str, float]] = {
    Form.LETTER: dict(COMPLETENESS_WEIGHTS),
    # An order is identified by its number and date, carries the issuing
    # office's header, and ends in a distribution list. It has no addressee,
    # no subject line and no valediction, and must not be docked for them.
    Form.ORDER: {"letter_number": 0.30, "date": 0.25,
                 "header": 0.15, "distribution": 0.15},
    Form.FRAGMENT: dict(COMPLETENESS_WEIGHTS),
}

#: Completeness is the anchor weights plus a body-length term. They are kept
#: separate so the anchor weights can be retuned without silently changing
#: what a completeness score of 1.0 means.
BODY_WEIGHT = 0.15
#: A body shorter than this scores proportionally less. Departmental letters
#: run 400-2000 characters; a 60-character "segment" is a stray heading.
BODY_TARGET_CHARS = 300

assert abs(sum(COMPLETENESS_WEIGHTS.values()) + BODY_WEIGHT - 1.0) < 1e-9
for _form, _w in FORM_WEIGHTS.items():
    assert abs(sum(_w.values()) + BODY_WEIGHT - 1.0) < 1e-9, _form

#: Below this many characters a candidate is folded into its neighbour rather
#: than emitted -- it is a letterhead fragment, not a letter.
MIN_SEGMENT_CHARS = 120

#: How far back from a second subject line to look for that letter's real
#: start (its letterhead or number).
BACKUP_WINDOW_LINES = 8

#: Walking back from a repeated subject, these mean we have crossed into the
#: previous letter and must stop.
_PREVIOUS_LETTER_MARKERS = frozenset({"closing", "distribution", "salutation"})

#: A line this long is prose, not letterhead.
BODY_LINE_CHARS = 200

SOURCE_TIERS = {"unicode": 1.0, "docx": 0.85, "pdf": 0.6, "ocr": 0.4}

TRUST_WEIGHTS = {"conversion": 0.45, "completeness": 0.35, "source": 0.20}
assert abs(sum(TRUST_WEIGHTS.values()) - 1.0) < 1e-9


@dataclass
class TaggedLine:
    index: int
    text: str
    anchors: frozenset[str] = frozenset()

    @property
    def is_noise(self) -> bool:
        return "noise" in self.anchors

    @property
    def is_blank(self) -> bool:
        return not self.text.strip()


@dataclass
class Segment:
    start_line: int
    end_line: int          #: exclusive
    text: str
    anchors: dict[str, int] = field(default_factory=dict)
    #: Which rule produced the boundary that opened this segment.
    opened_by: str = "file-start"
    _form: "Form | None" = None

    @property
    def form(self) -> "Form":
        if self._form is None:
            self._form = detect_form(self.text, self.anchors)
        return self._form

    @property
    def body_chars(self) -> int:
        return len(self.text)

    def completeness(self) -> float:
        weights = FORM_WEIGHTS[self.form]
        score = sum(w for name, w in weights.items() if self.anchors.get(name))
        score += BODY_WEIGHT * min(1.0, self.body_chars / BODY_TARGET_CHARS)
        return round(min(1.0, score), 4)

    def missing(self) -> list[str]:
        """Parts this document's own form requires and does not have."""
        return sorted(n for n in FORM_WEIGHTS[self.form] if not self.anchors.get(n))


def tag_lines(text: str, anchors: list[Anchor] = ANCHORS) -> list[TaggedLine]:
    out: list[TaggedLine] = []
    for i, line in enumerate(text.split("\n")):
        names = {a.name for a in anchors if a.pattern.search(line)}
        # A reference to some *other* letter's number is not this letter's
        # number. Without this, `संदर्भ: आपके पत्र क्रमांक 44` reads as a
        # letter-number line and opens a spurious segment mid-letter.
        if "reference" in names:
            names.discard("letter_number")
        # Page furniture is never also a landmark.
        if "noise" in names:
            names = {"noise"}
        out.append(TaggedLine(i, line, frozenset(names)))
    return out


def _role_names(role: Role, anchors: list[Anchor]) -> set[str]:
    return {a.name for a in anchors if a.role is role}


def find_boundaries(lines: list[TaggedLine],
                    anchors: list[Anchor] = ANCHORS) -> list[tuple[int, str]]:
    """Line indices where a new letter begins, with the reason for each."""
    starts = _role_names(Role.START, anchors)
    ends = _role_names(Role.END, anchors)

    boundaries: list[tuple[int, str]] = []
    seen_end = False
    chars_since = 0
    first_content = next((l.index for l in lines
                          if not l.is_blank and not l.is_noise), None)
    if first_content is None:
        return []
    boundaries.append((first_content, "file-start"))

    for line in lines[first_content + 1:]:
        if line.is_noise or line.is_blank:
            continue
        chars_since += len(line.text)
        if line.anchors & ends:
            seen_end = True
            continue
        if (line.anchors & starts) and seen_end and chars_since >= MIN_SEGMENT_CHARS:
            reason = "letter_number" if "letter_number" in line.anchors else "header"
            boundaries.append((line.index, f"{reason}-after-close"))
            seen_end = False
            chars_since = 0
    return boundaries


def _split_on_repeated_subject(lines: list[TaggedLine],
                               boundaries: list[tuple[int, str]],
                               anchors: list[Anchor]) -> list[tuple[int, str]]:
    """Recover letters that merged because the first one had no closing."""
    starts = _role_names(Role.START, anchors) | {"addressee"}
    bounds = list(boundaries)
    edges = [b for b, _ in bounds] + [len(lines)]
    extra: list[tuple[int, str]] = []

    for lo, hi in zip(edges, edges[1:]):
        subjects = [l.index for l in lines[lo:hi] if "subject" in l.anchors]
        for second in subjects[1:]:
            # Back up to this letter's own letterhead, if it has one. Scan
            # the whole window and keep the EARLIEST start anchor: a letterhead
            # block interleaves header, number and date lines, so breaking at
            # the first non-start line stops at the addressee and leaves the
            # new segment without its own number and header. Stop only at
            # something that definitely belongs to the previous letter.
            split_at = second
            for j in range(second - 1, max(lo, second - BACKUP_WINDOW_LINES) - 1, -1):
                line = lines[j]
                if line.anchors & _PREVIOUS_LETTER_MARKERS or len(line.text) > BODY_LINE_CHARS:
                    break
                if line.anchors & starts:
                    split_at = j
            if split_at > lo:
                extra.append((split_at, "repeated-subject"))
    if not extra:
        return bounds
    return sorted(set(bounds + extra), key=lambda t: t[0])


def segment(text: str, anchors: list[Anchor] = ANCHORS) -> list[Segment]:
    """Split converted letter text into individual letters."""
    lines = tag_lines(text, anchors)
    bounds = find_boundaries(lines, anchors)
    if not bounds:
        return []
    bounds = _split_on_repeated_subject(lines, bounds, anchors)

    edges = [b for b, _ in bounds] + [len(lines)]
    reasons = [r for _, r in bounds]
    segments: list[Segment] = []

    for (lo, hi), reason in zip(zip(edges, edges[1:]), reasons):
        kept = [l for l in lines[lo:hi] if not l.is_noise]
        body = "\n".join(l.text for l in kept).strip()
        if not body:
            continue
        counts: dict[str, int] = {}
        for l in kept:
            for name in l.anchors:
                counts[name] = counts.get(name, 0) + 1
        segments.append(Segment(lo, hi, body, counts, reason))

    return _merge_runts(segments)


def _merge_runts(segments: list[Segment]) -> list[Segment]:
    """Fold sub-minimum fragments into the following segment.

    A letterhead block that got separated from its letter is a fragment, not a
    letter. Merging forward rather than dropping keeps the letterhead attached
    to the letter it belongs to.
    """
    out: list[Segment] = []
    pending: Segment | None = None
    for seg in segments:
        if pending is not None:
            seg = Segment(pending.start_line, seg.end_line,
                          pending.text + "\n" + seg.text,
                          {k: pending.anchors.get(k, 0) + seg.anchors.get(k, 0)
                           for k in set(pending.anchors) | set(seg.anchors)},
                          pending.opened_by)
            pending = None
        if seg.body_chars < MIN_SEGMENT_CHARS:
            pending = seg
            continue
        out.append(seg)
    if pending is not None:
        if out:
            last = out[-1]
            out[-1] = Segment(last.start_line, pending.end_line,
                              last.text + "\n" + pending.text,
                              {k: last.anchors.get(k, 0) + pending.anchors.get(k, 0)
                               for k in set(last.anchors) | set(pending.anchors)},
                              last.opened_by)
        else:
            out.append(pending)
    return out


def trust(conversion_confidence: float, completeness: float,
          source_tier: str = "docx") -> float:
    """Combine the three independent quality signals into one gate.

    They are independent on purpose: conversion confidence says the characters
    are right, completeness says the letter is whole, and the source tier says
    how much the extraction path can be believed. A letter can fail any one of
    them while looking fine on the other two.
    """
    tier = SOURCE_TIERS.get(source_tier, 0.4)
    return round(
        TRUST_WEIGHTS["conversion"] * conversion_confidence
        + TRUST_WEIGHTS["completeness"] * completeness
        + TRUST_WEIGHTS["source"] * tier, 4)


def verdict(score: float) -> str:
    if score >= 0.80:
        return "index"
    if score >= 0.60:
        return "review"
    return "quarantine"
