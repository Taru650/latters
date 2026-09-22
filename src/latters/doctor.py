"""One command that tells the office why it does not work yet.

This exists because of a sentence in `MAINTENANCE.md` that should embarrass
anyone shipping software: *"treat the first install as a debugging session,
not a deployment."* The installer has never run on Windows, the target
machine has no internet to look anything up, and the person in front of it
is a clerk, not an engineer.

`latters doctor` is the answer to that. It checks every external thing the
application needs, in the order that a failure would bite, and for each one
prints what is wrong and the exact command that fixes it. It never guesses:
each check runs the real thing rather than looking for a file and hoping.

Two rules the checks follow.

**A missing optional part is not a failure.** Tesseract is needed only to
read scans; LibreOffice only to export PDF. An office with a .docx archive
that prints from Word needs neither, and telling it the install is broken
would be wrong. Those report WARN and the exit code stays 0.

**A check that cannot run is not a pass.** If Ollama is unreachable, the
model check does not silently succeed -- it reports that it could not be
tested. The scorecard learned this lesson the hard way; see
`scorecard.py`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

OK, WARN, FAIL = "ok", "warn", "FAIL"

#: The model must fit in RAM alongside Windows, the browser and Python. The
#: measured budget in the plan leaves ~2.4 GB spare with the 1B model on an
#: 8 GB machine; below this the machine swaps to a spinning disk and the
#: application looks broken rather than slow.
MIN_FREE_MB = 1200

#: A corpus below this is not representative; draft.py refuses to pretend.
MIN_USEFUL_CORPUS = 30


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    fix: str = ""
    optional: bool = False

    @property
    def blocking(self) -> bool:
        return self.status == FAIL and not self.optional


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, *a, **kw) -> None:
        self.checks.append(Check(*a, **kw))

    @property
    def blocking(self) -> list[Check]:
        return [c for c in self.checks if c.blocking]

    def render(self) -> str:
        width = max(len(c.name) for c in self.checks) + 2
        out = []
        for c in self.checks:
            mark = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[c.status]
            out.append(f"[{mark}] {c.name:<{width}} {c.detail}")
            if c.fix and c.status != OK:
                for line in c.fix.splitlines():
                    out.append(f"          {line}")
        out.append("")
        bad = self.blocking
        warned = [c for c in self.checks if c.status != OK and not c.blocking]
        if bad:
            out.append(f"{len(bad)} blocking problem(s). The application will "
                       f"not work until these are fixed.")
        elif warned:
            out.append(f"Usable. {len(warned)} optional part(s) missing -- see "
                       f"the warn lines for what you lose.")
        else:
            out.append("Everything checks out.")
        return "\n".join(out)


# --------------------------------------------------------------------------
def _python(r: Report) -> None:
    v = sys.version_info
    if v >= (3, 10):
        r.add("python", OK, f"{v.major}.{v.minor}.{v.micro}")
    else:
        r.add("python", FAIL, f"{v.major}.{v.minor}",
              fix="This application needs Python 3.10 or newer.\n"
                  "Install it from python.org and tick "
                  '"Add python.exe to PATH".')


def _console_encoding(r: Report) -> None:
    """A cp1252 console crashes on the first Devanagari character.

    `cli._prepare_streams` reconfigures stdout, so this is a warning about
    what the user will SEE rather than a failure -- but a clerk who sees
    boxes instead of Hindi will conclude the software is broken.
    """
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    if "utf" in enc:
        r.add("console encoding", OK, enc)
    else:
        r.add("console encoding", WARN, enc or "unknown", optional=True,
              fix="Hindi may show as boxes or question marks in this window.\n"
                  "Run  chcp 65001  first, or use the web pages instead.")


def _free_memory_mb() -> float | None:
    """Free RAM, without psutil: not a dependency this project will add."""
    if hasattr(os, "sysconf") and "SC_AVPHYS_PAGES" in os.sysconf_names:
        return (os.sysconf("SC_AVPHYS_PAGES")
                * os.sysconf("SC_PAGE_SIZE") / 1e6)
    if sys.platform == "win32":
        try:
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return ms.ullAvailPhys / 1e6
        except Exception:
            return None
    return None


def _disk_and_memory(r: Report, db: Path) -> None:
    try:
        usage = shutil.disk_usage(db.parent if db.parent.exists() else Path.cwd())
        free_mb = usage.free / 1e6
        if free_mb < 500:
            r.add("free disk", FAIL, f"{free_mb:.0f} MB",
                  fix="Under 500 MB free. The corpus, the model and the "
                      "exports all need room.")
        else:
            r.add("free disk", OK, f"{free_mb / 1000:.1f} GB")
    except OSError as exc:
        r.add("free disk", WARN, str(exc), optional=True)

    free_mb = _free_memory_mb()
    if free_mb is None:
        r.add("free memory", WARN, "could not measure", optional=True)
    elif free_mb < MIN_FREE_MB:
        r.add("free memory", WARN, f"{free_mb:.0f} MB free", optional=True,
              fix=f"Under {MIN_FREE_MB} MB free. Drafting will swap to disk "
                  f"and feel broken rather than slow.\n"
                  f"Close other programs, especially browser tabs.")
    else:
        r.add("free memory", OK, f"{free_mb / 1000:.1f} GB free")


def _corpus(r: Report, db: Path) -> None:
    if not db.exists():
        r.add("corpus", FAIL, f"no database at {db}",
              fix=f"Build one from this folder:\n"
                  f"  latters segment archive --db {db.name}\n"
                  f"  latters classify --db {db.name} --write\n"
                  f"  latters templates --db {db.name} -o skeletons")
        return
    try:
        from .store import Store
        with Store(db) as store:
            # stats() is the call the drafting page makes first, and it
            # reads the newest columns. Opening the database is not enough
            # of a check: an old corpus opened fine and died on page load.
            stats = store.stats()
    except Exception as exc:
        r.add("corpus", FAIL, f"unreadable: {exc}",
              fix="If this mentions a missing column, the corpus predates "
                  "the code and\n"
                  "the migration did not run -- `pip install -e .` in the "
                  "repo, then retry.\n"
                  "Otherwise restore the newest file from backups\\ over "
                  "this one, or rebuild:\n"
                  "  latters segment archive --db corpus.db")
        return

    n = stats["letters"]
    indexed = stats["by_verdict"].get("index", 0)
    if n == 0:
        r.add("corpus", FAIL, "0 letters",
              fix="The database exists but is empty. Re-run `latters segment`.")
    elif n < MIN_USEFUL_CORPUS:
        r.add("corpus", WARN, f"{n} letters ({indexed} indexed)", optional=True,
              fix=f"Below {MIN_USEFUL_CORPUS} letters there is nothing "
                  f"representative to draft from.\n"
                  f"Add more of the office's old letters on the admin page.")
    else:
        r.add("corpus", OK, f"{n} letters, {indexed} indexed, "
                            f"mean trust {stats['mean_trust']}")


def _skeletons(r: Report, path: Path) -> None:
    files = sorted(path.glob("*.md")) if path.is_dir() else []
    if not files:
        r.add("skeletons", WARN, f"none in {path}", optional=True,
              fix="Drafts will have no letterhead, addressee block or "
                  "signature.\n"
                  "  latters templates --db corpus.db -o skeletons")
    else:
        r.add("skeletons", OK, f"{len(files)} in {path}")


def _ollama(r: Report, model: str, host: str, *, stub: bool) -> None:
    if stub:
        r.add("model", WARN, "running with --stub", optional=True,
              fix="Everything works except writing the letter body.")
        return
    from .llm import Ollama

    client = Ollama(model=model, host=host)
    import json
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=10) as resp:
            tags = json.loads(resp.read())
    except Exception as exc:
        r.add("ollama", FAIL, f"cannot reach {host}: {exc}",
              fix="Start it:  ollama serve\n"
                  "start.bat does this for you -- use that instead of "
                  "running `latters serve` by hand.")
        # A check that cannot run is not a pass. See the module docstring.
        r.add("model", FAIL, "not tested -- Ollama is unreachable",
              fix="Fix Ollama first, then run this again.")
        return
    r.add("ollama", OK, host)

    names = {m.get("name", "") for m in tags.get("models", [])}
    if client.available():
        # Report the SIZE, and whether it leaves room. On an 8 GB
        # single-channel machine the difference between the 1B and the 4B
        # is the difference between working and paging a 3 GB model to a
        # spinning disk -- and that is not a judgement anyone should have
        # to make from a blog post about someone else's laptop.
        size_gb = next((m.get("size", 0) / 1e9 for m in tags.get("models", [])
                        if m.get("name", "") in (model, f"{model}:latest")), 0)
        detail = f"{model}" + (f"  {size_gb:.1f} GB" if size_gb else "")
        free = _free_memory_mb()
        if size_gb and free is not None:
            # The model is already resident if it was loaded, so this is a
            # lower bound on the squeeze, not an upper one.
            spare = free / 1000 - size_gb
            detail += f", {free / 1000:.1f} GB free now"
            if spare < 0.5:
                r.add("model", WARN, detail, optional=True,
                      fix=f"This model needs about {size_gb:.1f} GB and there "
                          f"is {free / 1000:.1f} GB free.\n"
                          f"Windows will page it to disk, and on a spinning "
                          f"disk that is minutes, not seconds.\n"
                          f"Close the browser, or drop --num-ctx to 2048, or "
                          f"fit the second memory module.")
                return
        r.add("model", OK, detail)
    else:
        have = ", ".join(sorted(names)[:4]) or "none"
        r.add("model", FAIL, f"{model} not installed (have: {have})",
              fix=f"Offline, from the USB bundle:\n"
                  f"  cd bundle\\model && ollama create {model} -f Modelfile\n"
                  f"With internet:  ollama pull {model}")


def _ocr(r: Report) -> None:
    from .ocr import available, ocr_languages

    tools = available()
    missing = [n for n, ok in tools.items() if not ok]
    if not missing:
        r.add("pdf + scan reading", OK, "pdftotext, pdftoppm, tesseract")
    else:
        r.add("pdf + scan reading", WARN, f"missing: {', '.join(missing)}",
              optional=True,
              fix="PDFs and scans cannot be added on the admin page.\n"
                  ".docx still works. See packaging/install.bat.")
        if "tesseract" in missing:
            return

    langs = ocr_languages()
    if not langs:
        return
    if "hin" not in langs:
        r.add("tesseract hindi", FAIL, f"installed: {', '.join(langs)}",
              optional=True,
              fix="Devanagari will come out as Latin gibberish.\n"
                  "The Tesseract installer does NOT select Hindi by default.\n"
                  "Copy hin.traineddata into the tessdata folder.")
    elif "eng" not in langs:
        r.add("tesseract hindi", WARN, "hin present, eng missing",
              optional=True,
              fix="Letter numbers will be mangled: the language order "
                  "eng+hin is what keeps DEO/SRN/2024/1187 intact.")
    else:
        r.add("tesseract hindi", OK, "hin + eng")


def _libreoffice(r: Report) -> None:
    """Find it the way the export does, then make it convert something.

    Both halves matter. `shutil.which` alone reports "not installed" on a
    Windows box where LibreOffice is in the Start menu, because the
    installer never touches PATH. And finding it is not enough either: one
    environment had libreoffice-core, where `soffice` existed and could not
    convert a plain text file for want of the Writer module.

    This shares `soffice.py` with the export route on purpose. A doctor that
    passes while the export says "not installed" sends the reader hunting
    through the wrong half of the system.
    """
    from . import soffice

    exe = soffice.find()
    if not exe:
        r.add("pdf export", WARN, "LibreOffice not found", optional=True,
              fix="Looked on PATH and in Program Files.\n"
                  "Export DOCX and print to PDF from Word instead -- the "
                  "letter is identical either way.")
        return

    ok, why = soffice.converts()
    if ok:
        r.add("pdf export", OK, f"LibreOffice converts  ({exe})")
    elif why == "timeout":
        r.add("pdf export", WARN, "LibreOffice did not finish in time",
              optional=True,
              fix="The first conversion on a machine is much slower than "
                  "later ones. Run this again.")
    else:
        r.add("pdf export", WARN, f"found it, but it cannot convert: {why}",
              optional=True,
              fix="This is LibreOffice, not your letter.\n"
                  "Close any open LibreOffice window and try again; if it "
                  "still fails, reinstall full LibreOffice (not -core).\n"
                  "Until then export DOCX.")


def _fonts(r: Report) -> None:
    """A PDF that says Nirmala UI on a machine without it renders boxes."""
    if sys.platform == "win32":
        win = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        if (win / "Nirmala.ttf").exists() or (win / "NirmalaB.ttf").exists():
            r.add("devanagari font", OK, "Nirmala UI")
        else:
            r.add("devanagari font", WARN, "Nirmala UI not found",
                  optional=True,
                  fix="Exported PDFs may show boxes instead of Hindi.\n"
                      "Nirmala UI ships with Windows 8 and later; enable the "
                      "Hindi language pack.")
        return
    try:
        out = subprocess.run(["fc-list", ":lang=hi"], capture_output=True,
                             text=True, timeout=30).stdout
        if out.strip():
            r.add("devanagari font", OK, f"{len(out.splitlines())} installed")
        else:
            r.add("devanagari font", WARN, "none", optional=True,
                  fix="Exported PDFs will show boxes. Install "
                      "fonts-lohit-deva or similar.")
    except (OSError, subprocess.SubprocessError):
        r.add("devanagari font", WARN, "could not check", optional=True)


def run(*, db: str = "corpus.db", skeletons: str = "skeletons",
        model: str = "gemma3:1b", host: str = "http://127.0.0.1:11434",
        stub: bool = False) -> Report:
    r = Report()
    _python(r)
    _console_encoding(r)
    _disk_and_memory(r, Path(db))
    _corpus(r, Path(db))
    _skeletons(r, Path(skeletons))
    _ollama(r, model, host, stub=stub)
    _ocr(r)
    _libreoffice(r)
    _fonts(r)
    return r
