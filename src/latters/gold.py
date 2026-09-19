"""Gold-set regression harness for font conversion.

This is the only mechanism in Phase 1 that establishes the converter is
*correct* rather than merely self-consistent. The sequence validator catches
illegal output; the gold set catches output that is legal and wrong.

A gold pair is one line of TSV: ``legacy<TAB>expected_unicode<TAB>table<TAB>note``

How to build the real one
-------------------------
1. Open a real letter from the archive in Word, with the legacy font installed.
2. Copy a line of the *raw* text (what you see with the font switched to
   Courier -- the ASCII gibberish). That is the ``legacy`` column.
3. Read the line as rendered, and type it in Unicode Hindi. That is
   ``expected_unicode``. A Hindi-reading colleague must do this, not a tool.
4. Aim for 200 lines spanning every department and every decade in the
   archive. Prioritise lines containing conjuncts, reph, chhoti-i, numerals
   and file numbers -- the places conversion breaks.

Ship nothing below 98% character accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .fonts.convert import Converter, normalize_devanagari

DEFAULT_GOLD = Path(__file__).resolve().parent.parent.parent / "tests" / "gold"


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass
class PairResult:
    legacy: str
    expected: str
    actual: str
    table: str
    note: str
    distance: int

    @property
    def ok(self) -> bool:
        return self.distance == 0

    @property
    def accuracy(self) -> float:
        n = max(len(self.expected), 1)
        return max(0.0, 1.0 - self.distance / n)


@dataclass
class GoldReport:
    results: list[PairResult]

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def exact(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def char_accuracy(self) -> float:
        total = sum(len(r.expected) for r in self.results)
        errors = sum(r.distance for r in self.results)
        return max(0.0, 1.0 - errors / total) if total else 0.0

    @property
    def failures(self) -> list[PairResult]:
        return [r for r in self.results if not r.ok]

    def render(self, *, show: int = 20) -> str:
        lines = [
            f"gold pairs:      {self.n}",
            f"exact matches:   {self.exact}/{self.n} ({self.exact / max(self.n,1):.1%})",
            f"char accuracy:   {self.char_accuracy:.4%}",
        ]
        if self.failures:
            lines.append("")
            lines.append(f"failures (showing up to {show}):")
            for r in self.failures[:show]:
                lines.append(f"  [{r.table}] {r.note}")
                lines.append(f"    legacy   {r.legacy!r}")
                lines.append(f"    expected {r.expected}")
                lines.append(f"    actual   {r.actual}")
                lines.append(f"    distance {r.distance}")
        return "\n".join(lines)


def load_pairs(path: Path) -> list[tuple[str, str, str, str]]:
    pairs = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\n")
        if not line.strip() or (line.startswith("#") and (len(line) == 1 or line[1] != "\t")):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise ValueError(f"{path.name}: expected 2+ tab-separated fields in {raw!r}")
        legacy, expected = parts[0], parts[1]
        table = parts[2] if len(parts) > 2 and parts[2] else "krutidev010"
        note = parts[3] if len(parts) > 3 else ""
        pairs.append((legacy, expected, table, note))
    return pairs


def run(paths: list[Path], *, latin_digits: bool = False) -> GoldReport:
    cache: dict[str, Converter] = {}
    results: list[PairResult] = []
    for path in paths:
        for legacy, expected, table, note in load_pairs(path):
            if table not in cache:
                cache[table] = Converter(table, latin_digits=latin_digits)
            actual = cache[table].convert(legacy).text
            expected_n = normalize_devanagari(expected)
            results.append(
                PairResult(legacy, expected_n, actual, table, note,
                           levenshtein(expected_n, actual))
            )
    return GoldReport(results)


def discover(root: Path = DEFAULT_GOLD) -> list[Path]:
    return sorted(root.glob("*.tsv")) if root.exists() else []
