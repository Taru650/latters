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

from .anchors import ANCHORS
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
    s = _WS.sub(" ", s)
    # Collapse whitespace AROUND the blank marker too. Without this,
    # "छपरा, दिनांक ░" and "छपरा, दिनांक░" are different
    # canonical lines, both survive deduplication, and the assembled
    # letter carries its date twice.
    s = re.sub(r"\s*░\s*", "░", s)
    return s.strip()


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
    #: cell, with the count and the most common literal rendering, IN
    #: DOCUMENT ORDER. Ordering by frequency instead -- which the first
    #: version did, because Counter.most_common() is the obvious call --
    #: produced letters with the addressee above the letterhead and the
    #: salutation above the letter number.
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


#: A letter's parts come in a fixed order, and the Phase 2 anchors already
#: know which part each line is. Median position alone is too weak a signal:
#: a cell of 53 letters contains several layouts, so the medians interleave
#: and the skeleton comes out with the salutation above the letterhead.
#: Sorting by role first and position only within a role fixes that.
ROLE_ORDER: dict[str, int] = {
    "header": 0,
    "letter_number": 1,
    "date": 2,
    "addressee": 3,
    "subject": 5,
    "reference": 6,
    "salutation": 7,
    "closing": 9,
    "distribution": 10,
}
#: Unanchored lines -- an addressee's designation, a district name, the
#: signatory's post -- sit between the landmarks. They inherit the role of
#: whichever anchored line they sit nearest to by median position.
_UNANCHORED = 4


def line_role(line: str) -> int | None:
    for anchor in ANCHORS:
        if anchor.name in ROLE_ORDER and anchor.pattern.search(line):
            return ROLE_ORDER[anchor.name]
    return None


_LABEL_BEFORE_BLANK = re.compile(
    rf"([ऀ-ॿ]{{2,16}})\s*[:ः]?\s*{BLANK_RUN}")


def mine(letters: list[str], department: str, letter_type: str) -> Skeleton:
    """Find what every letter in this cell has in common."""
    n = len(letters)
    line_counts: Counter[str] = Counter()
    literals: dict[str, Counter[str]] = defaultdict(Counter)
    blank_labels: Counter[str] = Counter()
    body_lens: list[int] = []
    # Where each line sits in a letter, as a fraction of the letter's length.
    positions: dict[str, list[float]] = defaultdict(list)

    for text in letters:
        seen: set[str] = set()
        raw_lines = text.splitlines()
        n_lines = max(len(raw_lines), 1)
        for index, raw in enumerate(raw_lines):
            line = raw.strip()
            if len(line) < MIN_LINE_CHARS:
                continue
            c = canonical(line)
            if c and c not in seen:
                seen.add(c)
                line_counts[c] += 1
                positions[c].append(index / n_lines)
            literals[c][line] += 1
        for m in _LABEL_BEFORE_BLANK.finditer(text):
            blank_labels[m.group(1)] += 1
        body_lens.append(len(text))

    cutoff = max(2, int(n * BOILERPLATE_THRESHOLD))

    def _median(xs: list[float]) -> float:
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else 1.0

    candidates = [(c, k, literals[c].most_common(1)[0][0])
                  for c, k in line_counts.most_common() if k >= cutoff]

    # Deduplicate on the CANONICAL form, not the literal: `दिनांक-----` and
    # `दिनांक-----------` are the same line with different dash runs, and
    # keying on the literal let both through, so the assembled letter carried
    # the date twice.
    by_canon: dict[str, tuple[str, int, str]] = {}
    for c, k, literal in candidates:
        if c not in by_canon or k > by_canon[c][1]:
            by_canon[c] = (c, k, literal)

    anchored = [(row, line_role(row[2])) for row in by_canon.values()]
    landmarks = [(_median(positions[row[0]]), role)
                 for row, role in anchored if role is not None]

    def _rank(row, role) -> tuple[int, float]:
        pos = _median(positions[row[0]])
        if role is not None:
            return role, pos
        if not landmarks:
            return _UNANCHORED, pos
        # Sit where the nearest landmark sits.
        nearest = min(landmarks, key=lambda lm: abs(lm[0] - pos))
        return nearest[1], pos

    boiler = [row for row, role in sorted(anchored, key=lambda x: _rank(*x))]

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


# --------------------------------------------------------------------------
# Human-corrected skeletons
# --------------------------------------------------------------------------
#: Mining cannot fully recover line order. A cell of 53 letters contains
#: several layouts, so unanchored lines -- an addressee's designation, a
#: district name, the signatory's post -- can only be placed relative to the
#: landmarks around them, and sometimes land on the wrong side of the
#: salutation. That is not a solvable inference problem on this data.
#:
#: It is a five-minute editing task, once per category, done by someone who
#: knows what the office's letters look like. So `latters templates -o dir/`
#: writes the mined skeletons out, staff fix the order, and the edited file
#: wins from then on. Machine-mined is the starting point; human-corrected is
#: the artefact.
_FIXED_HEADER = "## fixed lines"
_SLOT_HEADER = "## slots to fill"
_BODY_HEADER = "## body:"
_COUNT_PREFIX = re.compile(r"^\s*\[\s*\d+\s*/?\s*\d*\s*\]\s?")


def parse_skeleton(text: str, department: str, letter_type: str) -> Skeleton:
    """Read a skeleton back from its rendered form, after human editing.

    Tolerant on purpose: a clerk editing this file will delete the `[ 43/53]`
    counts, reorder lines and add ones the archive never contained. All of
    that is intended. Only the `## fixed lines` heading has to survive.
    """
    section = None
    boiler: list[tuple[str, int, str]] = []
    blanks: list[tuple[str, int]] = []
    body_chars = 0
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith(_FIXED_HEADER):
            section = "fixed"
            continue
        if stripped.startswith(_SLOT_HEADER):
            section = "slots"
            continue
        if stripped.startswith(_BODY_HEADER):
            section = None
            m = re.search(r"(\d+)", stripped)
            if m:
                body_chars = int(m.group(1))
            continue
        if not stripped or stripped.startswith("#"):
            continue
        content = _COUNT_PREFIX.sub("", line).strip()
        if not content:
            continue
        if section == "fixed":
            boiler.append((canonical(content), 0, content))
        elif section == "slots":
            blanks.append((content, 0))

    sk = Skeleton(department=department, letter_type=letter_type,
                  n_letters=0, boilerplate=boiler, blanks=blanks,
                  median_body_chars=body_chars or 600)
    sk._coverage = 0.0
    return sk


def load_overrides(directory) -> dict[tuple[str, str], Skeleton]:
    """Load human-corrected skeletons from a directory of rendered files."""
    from pathlib import Path

    out: dict[tuple[str, str], Skeleton] = {}
    d = Path(directory)
    if not d.is_dir():
        return out
    for path in sorted(d.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        m = re.search(r"^#\s*(.+?)\s*/\s*(.+?)\s*(?:\(|$)", text, re.M)
        if not m:
            continue
        dept, ltype = m.group(1).strip(), m.group(2).strip()
        out[(dept, ltype)] = parse_skeleton(text, dept, ltype)
    return out
