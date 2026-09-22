"""Finding LibreOffice, and getting a PDF out of it, on Windows.

This module exists because of one reported failure: DOCX and TXT export
worked and PDF did not. Three separate Windows-only reasons can produce
exactly that, and the code had none of them covered.

**LibreOffice does not put itself on PATH.** The Windows installer writes
``soffice.exe`` into ``C:\\Program Files\\LibreOffice\\program\\`` and does
not touch PATH, so ``shutil.which("soffice")`` returns ``None`` on a machine
where LibreOffice is installed and working. The application then reported
"not installed" to somebody looking at its Start-menu entry.

**A running LibreOffice blocks a headless one.** They share one user
profile, and the second process to want it exits rather than waiting. A
clerk who has a letter open in Writer gets a PDF export failure with no
visible cause. Passing ``-env:UserInstallation`` gives the headless run a
profile of its own, so the two never meet.

**``soffice.exe`` can return before the PDF is written.** It is a launcher
for ``soffice.bin``; in some installations it hands off and exits. Checking
for the file the instant the process returns then finds nothing and blames
the document. So wait for it, briefly.

Keeping this in one module rather than in both callers is deliberate:
`doctor` reporting "LibreOffice converts" while the export says "not
installed" would send someone hunting through the wrong half of the system.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

#: Where the Windows installer actually puts it. Ordered most-likely first.
_WINDOWS_DIRS = (
    r"C:\Program Files\LibreOffice\program",
    r"C:\Program Files (x86)\LibreOffice\program",
    r"C:\Program Files\LibreOffice 7\program",
    r"C:\Program Files (x86)\LibreOffice 7\program",
)

#: soffice.exe may exit before soffice.bin has finished writing. Poll for
#: the output rather than trusting the return. Two seconds is far longer
#: than the handoff and still imperceptible when the file is already there.
_SETTLE_SECONDS = 2.0
_SETTLE_POLL = 0.05

DEFAULT_TIMEOUT = 180


def find() -> str | None:
    """The LibreOffice executable, or None.

    PATH first -- a user who put it there meant it. Then the standard
    install locations, then %ProgramFiles% and %LOCALAPPDATA% in case the
    machine uses a non-English or relocated Program Files.
    """
    for name in ("soffice", "libreoffice"):
        found = shutil.which(name)
        if found:
            return found

    if sys.platform != "win32":
        return None

    candidates = [Path(d) / "soffice.exe" for d in _WINDOWS_DIRS]
    for var in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        base = os.environ.get(var)
        if base:
            candidates.append(Path(base) / "LibreOffice" / "program" / "soffice.exe")
            candidates.append(
                Path(base) / "Programs" / "LibreOffice" / "program" / "soffice.exe")
    for path in candidates:
        try:
            if path.is_file():
                return str(path)
        except OSError:
            continue
    return None


def _private_profile(tmp: Path) -> str:
    """A profile of this conversion's own, as a file:// URL.

    Without it, a LibreOffice already open on the desktop owns the shared
    profile and the headless run exits instead of converting. That failure
    is invisible: no error, no PDF.
    """
    return "-env:UserInstallation=" + (tmp / "profile").resolve().as_uri()


def to_pdf(docx: Path, outdir: Path, *,
           timeout: float = DEFAULT_TIMEOUT) -> tuple[Path | None, str]:
    """Convert to PDF. Returns (path or None, a reason when it is None)."""
    exe = find()
    if not exe:
        return None, "not installed"

    with tempfile.TemporaryDirectory(prefix="latters-soffice-") as tmp:
        cmd = [exe, _private_profile(Path(tmp)), "--headless",
               "--convert-to", "pdf", "--outdir", str(outdir), str(docx)]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, "timeout"
        except OSError as exc:
            return None, str(exc)

        pdf = outdir / (docx.stem + ".pdf")
        deadline = time.monotonic() + _SETTLE_SECONDS
        while not pdf.exists() and time.monotonic() < deadline:
            time.sleep(_SETTLE_POLL)
        if pdf.exists():
            return pdf, ""

    # javaldx warns about a missing JRE on nearly every machine and has
    # nothing to do with converting a Writer document. Dropping it keeps
    # the real message visible.
    err = (proc.stderr or b"").decode("utf-8", "replace").strip()
    reason = " / ".join(l for l in err.splitlines()
                        if l.strip() and "javaldx" not in l)
    return None, (reason or "produced no PDF and said nothing")[:300]


def converts() -> tuple[bool, str]:
    """Actually convert something. Finding the binary is not enough.

    One environment had libreoffice-core installed: `soffice` existed, ran,
    and could not convert a plain text file because the Writer module was
    absent. That surfaced as a PDF export failure blaming the letter.
    """
    if not find():
        return False, "not installed"
    try:
        with tempfile.TemporaryDirectory(prefix="latters-probe-") as tmp:
            probe = Path(tmp) / "probe.txt"
            probe.write_text("probe", encoding="utf-8")
            pdf, why = to_pdf(probe, Path(tmp))
            return (pdf is not None), why
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
