"""Mine the invariant skeleton of each (department, letter-type) cell.

This is the highest-leverage artefact in the project, and the reason a 1B
model is viable at all.

Free composition is the hardest thing to ask of a small model and the thing
it does worst. Slot-filling is the easiest. So instead of asking the model to
write a letter, the system hands it a skeleton mined from the office's own
past letters and asks it to write only the parts that actually vary. The
letterhead, the addressee block, the closing formula and the distribution
list are then *copied*, not generated -- which also means they are exactly
right rather than approximately right, and cost zero tokens of generation.

Measured on a real archive: 545 of 547 letters had a unique body, so there
are no whole-letter templates to reuse. But line-level boilerplate is
enormous -- `प्रभारी पदाधिकारी,` appears in 204 letters, `जिला राजस्व शाखा,`
in 125. Mining at line level is what works; mining at document level finds
nothing.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .fields import BLANK_RUN

_WS = re.compile(r"\s+")
_NUM = re.compile(r"[0-9०-९]+")
_BLANK = re.compile(BLANK_RUN)

#: A line shorter than this is punctuation or a stray fragment.
MIN_LINE_CHARS = 4
#: Appearing in at least this fraction of a cell's letters makes a line
#: boilerplate rather than content.
BOILERPLATE_THRESHOLD = 0.30
#: Below this many letters, "appears in 30% of them" is not evidence.
MIN_CELL_SIZE = 8


def canonical(line: str) -> str:
    """Normalise away the things that vary between two copies of one line.

    Numbers, dates and fill-in dash runs differ between every letter; the
    line is still 'the same line'. Without this the letterhead of every
    letter looks unique because the date differs.
    """
    s = _BLANK.sub("░", line)
    s = _NUM.sub("#", s)
    return _WS.sub(" ", s).strip()


@dataclass
class Slot:
    """A place in the skeleton where content varies."""
    after: str          #: the canonical line this slot follows
    label: str          #: what the office calls it, when it has a label
    examples: list[str] = field(default_factory=list)


@dataclass
class Skeleton:
    department: str
    letter_type: str
    n_letters: int
    #: Canonical lines that appear in at least BOILERPLATE_THRESHOLD of the
    #: cell, with the count and the most common literal rendering.
    boilerplate: list[tuple[str, int, str]] = field(default_factory=list)
    #: Labels that are followed by a fill-in blank, e.g. पत्रांक, दिनांक.
    blanks: list[tuple[str, int]] = field(default_factory=list)
    #: Median body length, so generation can be told how long to write.
    median_body_chars: int = 0

    @property
    def coverage(self) -> float:
        """Share of a typical letter's lines that the skeleton supplies."""
        return self._coverage

    _coverage: float = 0.0

    def render(self) -> str:
        head = [f"# {self.department} / {self.letter_type}   "
                f"({self.n_letters} letters, skeleton covers "
                f"{self.coverage:.0%} of a typical letter)"]
        head.append("")
        head.append("## fixed lines")
        for canon, n, literal in self.boilerplate:
            head.append(f"  [{n:4d}/{self.n_letters}]  {literal}")
        if self.blanks:
            head.append("")
            head.append("## slots to fill")
            for label, n in self.blanks:
                head.append(f"  [{n:4d}]  {label}")
        head.append("")
        head.append(f"## body: write roughly {self.median_body_chars} characters")
        return "\n".join(head)


_LABEL_BEFORE_BLANK = re.compile(
    rf"([ऀ-ॿ]{{2,16}})\s*[:ः]?\s*{BLANK_RUN}")


def mine(letters: list[str], department: str, letter_type: str) -> Skeleton:
    """Find what every letter in this cell has in common."""
    n = len(letters)
    line_counts: Counter[str] = Counter()
    literals: dict[str, Counter[str]] = defaultdict(Counter)
    blank_labels: Counter[str] = Counter()
    body_lens: list[int] = []

    for text in letters:
        seen: set[str] = set()
        for raw in text.splitlines():
            line = raw.strip()
            if len(line) < MIN_LINE_CHARS:
                continue
            c = canonical(line)
            if c and c not in seen:
                seen.add(c)
                line_counts[c] += 1
            literals[c][line] += 1
        for m in _LABEL_BEFORE_BLANK.finditer(text):
            blank_labels[m.group(1)] += 1
        body_lens.append(len(text))

    cutoff = max(2, int(n * BOILERPLATE_THRESHOLD))
    boiler = [(c, k, literals[c].most_common(1)[0][0])
              for c, k in line_counts.most_common() if k >= cutoff]

    # How much of a typical letter does the skeleton account for?
    boiler_set = {c for c, _, _ in boiler}
    shares = []
    for text in letters:
        lines = [canonical(l.strip()) for l in text.splitlines()
                 if len(l.strip()) >= MIN_LINE_CHARS]
        if lines:
            shares.append(sum(1 for l in lines if l in boiler_set) / len(lines))
    body_lens.sort()

    sk = Skeleton(
        department=department, letter_type=letter_type, n_letters=n,
        boilerplate=boiler,
        blanks=[(l, k) for l, k in blank_labels.most_common() if k >= cutoff],
        median_body_chars=body_lens[len(body_lens) // 2] if body_lens else 0)
    sk._coverage = sum(shares) / len(shares) if shares else 0.0
    return sk


def mine_all(rows: list[tuple[str, str, str]], *,
             min_cell: int = MIN_CELL_SIZE) -> list[Skeleton]:
    """`rows` is (text, department, letter_type). Cells too small are skipped.

    Skipping is the point: a "skeleton" mined from three letters is three
    letters' idiosyncrasies, and using it would make every future draft in
    that category look like those three.
    """
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for text, dept, ltype in rows:
        cells[(dept, ltype)].append(text)
    out = [mine(texts, d, t) for (d, t), texts in cells.items()
           if len(texts) >= min_cell]
    return sorted(out, key=lambda s: -s.n_letters)
