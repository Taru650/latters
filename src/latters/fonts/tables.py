"""Loading and validation of legacy-font -> Unicode mapping tables.

A table is a TSV file: ``legacy<TAB>unicode<TAB>comment``.

Comment lines start with ``#`` *not* followed by a tab -- the tab test matters
because ``#`` is itself a legacy character (Kruti Dev slot for ``रु``).

A table may start with ``#inherit <name>`` to load another table first and
apply its own rows as overrides. This keeps DevLys from forking the whole
Kruti Dev table for a handful of differing ligature slots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fonts"

#: Legacy characters whose *position* is wrong, not just their value. The
#: converter replaces them with private-use sentinels and fixes placement in a
#: later pass; see ``convert.py``.
REPH_LEGACY = "Z"
REPH_MARK = ""

#: Kruti Dev renders ASCII digits as Devanagari digits. Some offices typed
#: numerals in a Latin font run instead, so this is switchable.
_DIGIT_KEYS = frozenset("0123456789")


class TableError(ValueError):
    """Raised when a mapping table is malformed."""


@dataclass(frozen=True)
class FontTable:
    name: str
    mapping: dict[str, str]
    source_files: tuple[Path, ...] = field(default=())

    def without_digits(self) -> "FontTable":
        """Return a copy that leaves ASCII digits untouched."""
        trimmed = {k: v for k, v in self.mapping.items() if k not in _DIGIT_KEYS}
        return FontTable(self.name + "+latin-digits", trimmed, self.source_files)


def _is_comment(line: str) -> bool:
    return line.startswith("#") and (len(line) == 1 or line[1] != "\t")


def _parse(path: Path, seen: set[str]) -> tuple[dict[str, str], list[Path]]:
    if not path.exists():
        raise TableError(f"mapping table not found: {path}")

    mapping: dict[str, str] = {}
    files: list[Path] = []

    raw = path.read_text(encoding="utf-8").splitlines()

    # Resolve #inherit first so local rows override the parent's.
    for line in raw:
        stripped = line.strip()
        if stripped.startswith("#inherit "):
            parent = stripped[len("#inherit ") :].strip()
            if parent in seen:
                raise TableError(f"circular #inherit involving {parent!r}")
            seen.add(parent)
            parent_map, parent_files = _parse(DATA_DIR / f"{parent}.tsv", seen)
            mapping.update(parent_map)
            files.extend(parent_files)

    for lineno, line in enumerate(raw, 1):
        line = line.rstrip("\n")
        if not line.strip() or _is_comment(line):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise TableError(f"{path.name}:{lineno}: expected at least 2 tab-separated fields")
        legacy, unicode_val = parts[0], parts[1]
        if not legacy:
            raise TableError(f"{path.name}:{lineno}: empty legacy key")
        if not unicode_val:
            raise TableError(f"{path.name}:{lineno}: empty unicode value for {legacy!r}")
        mapping[legacy] = unicode_val

    files.append(path)
    return mapping, files


def load_table(name: str = "krutidev010") -> FontTable:
    """Load a mapping table by name (a stem under ``data/fonts``)."""
    mapping, files = _parse(DATA_DIR / f"{name}.tsv", {name})
    if REPH_LEGACY in mapping:
        mapping[REPH_LEGACY] = REPH_MARK
    return FontTable(name, mapping, tuple(files))


def available_tables() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob("*.tsv"))
