"""Editing effort: the number that decides whether this project was worth building.

The plan names it directly -- "the metric that decides success is editing
effort, measured as edit distance between draft and the letter actually
dispatched". Everything else (conversion accuracy, retrieval P@5, a human's
1-5 rating of format fidelity) is diagnostic. This one is the verdict, because
it is the only number that answers the question the office will actually ask:
*is it faster than typing it?*

**It is measured in production, not in a study.** The plan's original shape --
collect 30 (request, dispatched letter) pairs, run a script, get a number --
has a failure mode that this project cannot afford: nobody ever collects the
30 pairs, so the number never exists, and the project ships on the strength of
a demo. Every draft that leaves through the export button is already one half
of a pair, and the exported text is the other half. Recording both costs one
INSERT and turns the success metric into something that accrues by itself.

What "effort" means here
------------------------
``effort = levenshtein(draft, dispatched) / len(dispatched)``

Character-level, on the body only -- the letterhead and the signature block
come from the skeleton and are identical by construction, so including them
would flatter the score by diluting it with text nobody typed.

Reading the number:

===========  ===============================================================
 <= 0.10     the draft went out nearly as written. This is the target.
 0.10-0.30   ordinary editing; still much faster than composing from nothing.
 0.30-0.70   heavy rewriting, but the structure survived.
 > 0.70      **the project has failed its purpose for that letter.** More was
             changed than kept; the person would have been faster typing it.
===========  ===============================================================

The 0.70 line is not arbitrary and it is not a pass mark to be negotiated
down later. Above it, the draft cost the writer time instead of saving it.

Why Levenshtein and not something cleverer
------------------------------------------
A semantic similarity score would say a rewritten letter is "close" to the
draft. That is exactly the wrong reading: the writer's keystrokes are the
cost, and keystrokes are edit distance. We want the pessimistic measure.

The implementation is the two-row dynamic program -- O(n*m) time, O(min(n,m))
memory. A dispatched letter is ~1,500 characters, so this is about 2 million
cell updates, a few milliseconds, and it runs once per export. Bringing in a
C extension for that would be a dependency the target office has no internet
to install.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

#: Above this, more of the letter was rewritten than kept. See the module
#: docstring: this is a failure line, not a target to be relaxed.
FAILED_ITS_PURPOSE = 0.70

#: Below this, the draft went out substantially as written.
GOOD = 0.30

#: An edit distance over two texts this long is not worth the milliseconds,
#: and a "letter" this size is a pasted document, not a draft we produced.
MAX_CHARS = 20_000


def _normalise(text: str) -> str:
    """Fold away differences that cost the writer no keystrokes.

    Three of them, and each one showed up as noise before it was folded:

    * **Unicode form.** The browser submits NFC; our converter emits
      decomposed sequences for some conjuncts. Identical Hindi with different
      code points would score as a full rewrite of every affected word.
    * **Whitespace runs.** A writer pressing Enter twice instead of once is
      not editing the letter.
    * **Trailing space per line.** Invisible, and Word adds it.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def levenshtein(a: str, b: str) -> int:
    """Edit distance, two-row dynamic program."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Iterate over the shorter string in the inner loop: the row we keep in
    # memory is len(inner)+1 long.
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,            # deletion
                current[j - 1] + 1,         # insertion
                previous[j - 1] + (ca != cb),  # substitution
            ))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class Effort:
    """One (draft, dispatched) comparison."""

    distance: int
    draft_chars: int
    dispatched_chars: int

    @property
    def score(self) -> float:
        """Edit distance per character of the dispatched letter.

        Normalising by the *dispatched* length, not by the longer of the two,
        keeps the reading honest in the case that matters most: a draft that
        is deleted and replaced wholesale scores at least 1.0, and a draft
        that was mostly deleted scores high rather than being flattered by a
        long denominator.
        """
        if not self.dispatched_chars:
            return 1.0
        return self.distance / self.dispatched_chars

    @property
    def verdict(self) -> str:
        if self.score > FAILED_ITS_PURPOSE:
            return "failed"
        if self.score > GOOD:
            return "heavy"
        if self.score > 0.10:
            return "edited"
        return "as-drafted"

    @property
    def failed_its_purpose(self) -> bool:
        return self.score > FAILED_ITS_PURPOSE


def measure(draft: str, dispatched: str) -> Effort:
    """Compare what we produced with what actually went out."""
    d = _normalise(draft)[:MAX_CHARS]
    s = _normalise(dispatched)[:MAX_CHARS]
    return Effort(distance=levenshtein(d, s),
                  draft_chars=len(d), dispatched_chars=len(s))


@dataclass(frozen=True)
class Summary:
    """Effort across many letters.

    The median, not the mean. One pasted-over draft scoring 3.0 would drag a
    mean above the failure line while most letters went out untouched, and
    the office would be told the tool does not work when it does.
    """

    n: int
    median: float
    p90: float
    failed: int
    as_drafted: int

    @property
    def failure_rate(self) -> float:
        return self.failed / self.n if self.n else 0.0


def summarise(scores: list[float]) -> Summary | None:
    if not scores:
        return None
    s = sorted(scores)
    mid = len(s) // 2
    median = s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
    # Nearest-rank p90: with fewer than ten samples this is simply the
    # maximum, which is the right answer rather than an interpolated fiction.
    p90 = s[min(len(s) - 1, int(round(0.9 * len(s))) - 1 if len(s) >= 10
               else len(s) - 1)]
    return Summary(
        n=len(s), median=median, p90=p90,
        failed=sum(1 for x in s if x > FAILED_ITS_PURPOSE),
        as_drafted=sum(1 for x in s if x <= 0.10),
    )
