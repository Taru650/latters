"""The reported failure: DOCX and TXT exported, PDF did not.

Four separate causes could produce exactly that on Windows, and none of
them was covered. Each gets a test here, because each one is invisible in
a Linux CI run -- the container's LibreOffice is on PATH, nobody has Writer
open, and `soffice` is not a launcher stub.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latters import soffice  # noqa: E402


# --- 1. LibreOffice is not on PATH on Windows -----------------------------
def test_windows_program_files_is_searched_when_path_has_nothing(monkeypatch,
                                                                 tmp_path):
    """The Windows installer never touches PATH, so `shutil.which` says
    "not installed" about a machine with LibreOffice in the Start menu."""
    exe = tmp_path / "LibreOffice" / "program" / "soffice.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("")

    monkeypatch.setattr(soffice.shutil, "which", lambda name: None)
    monkeypatch.setattr(soffice.sys, "platform", "win32")
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    assert soffice.find() == str(exe)


def test_path_still_wins_when_it_has_one(monkeypatch):
    monkeypatch.setattr(soffice.shutil, "which",
                        lambda name: "/usr/bin/soffice" if name == "soffice"
                        else None)
    assert soffice.find() == "/usr/bin/soffice"


def test_not_found_is_reported_as_such_not_crashed(monkeypatch):
    monkeypatch.setattr(soffice.shutil, "which", lambda name: None)
    monkeypatch.setattr(soffice.sys, "platform", "linux")
    assert soffice.find() is None
    pdf, why = soffice.to_pdf(Path("nonexistent.docx"), Path("."))
    assert pdf is None and why == "not installed"


# --- 2. an open LibreOffice blocks the headless one -----------------------
def test_the_conversion_gets_its_own_profile(monkeypatch, tmp_path):
    """Two LibreOffice processes share one user profile and the second
    exits rather than waiting. A clerk with a letter open in Writer got a
    PDF failure with no cause on screen."""
    seen: list[list[str]] = []

    class _Proc:
        stderr = b""

    def fake_run(cmd, **kw):
        seen.append(cmd)
        (Path(kw.get("cwd") or ".") / "x").unlink(missing_ok=True)
        # Write the output the real thing would.
        out = Path(cmd[cmd.index("--outdir") + 1])
        (out / (Path(cmd[-1]).stem + ".pdf")).write_bytes(b"%PDF-1.4\n")
        return _Proc()

    monkeypatch.setattr(soffice.shutil, "which", lambda n: "/usr/bin/soffice")
    monkeypatch.setattr(soffice.subprocess, "run", fake_run)

    src = tmp_path / "letter.docx"
    src.write_bytes(b"")
    pdf, why = soffice.to_pdf(src, tmp_path)
    assert pdf is not None and why == ""

    profile = [a for a in seen[0] if a.startswith("-env:UserInstallation=")]
    assert profile, "no private profile: an open LibreOffice will block this"
    assert profile[0].startswith("-env:UserInstallation=file://")


# --- 3. soffice.exe can return before the PDF is written ------------------
def test_a_pdf_that_appears_late_is_still_found(monkeypatch, tmp_path):
    """soffice.exe is a launcher for soffice.bin and in some installations
    hands off and exits. Checking the instant it returns finds nothing and
    blames the document."""
    import threading

    class _Proc:
        stderr = b""

    def fake_run(cmd, **kw):
        out = Path(cmd[cmd.index("--outdir") + 1])
        target = out / (Path(cmd[-1]).stem + ".pdf")
        threading.Timer(0.3, lambda: target.write_bytes(b"%PDF-1.4\n")).start()
        return _Proc()

    monkeypatch.setattr(soffice.shutil, "which", lambda n: "/usr/bin/soffice")
    monkeypatch.setattr(soffice.subprocess, "run", fake_run)

    src = tmp_path / "letter.docx"
    src.write_bytes(b"")
    pdf, why = soffice.to_pdf(src, tmp_path)
    assert pdf is not None, f"gave up before the PDF landed: {why}"


def test_waiting_is_bounded_and_the_reason_survives(monkeypatch, tmp_path):
    """No PDF ever: say what LibreOffice said, minus the javaldx noise it
    prints on every machine without a JRE."""
    class _Proc:
        stderr = (b"javaldx: Could not find a Java Runtime Environment!\n"
                  b"Error: source file could not be loaded\n")

    monkeypatch.setattr(soffice.shutil, "which", lambda n: "/usr/bin/soffice")
    monkeypatch.setattr(soffice.subprocess, "run", lambda cmd, **kw: _Proc())
    monkeypatch.setattr(soffice, "_SETTLE_SECONDS", 0.05)

    src = tmp_path / "letter.docx"
    src.write_bytes(b"")
    pdf, why = soffice.to_pdf(src, tmp_path)
    assert pdf is None
    assert "javaldx" not in why
    assert "source file could not be loaded" in why


# --- 4. doctor and the export must agree ----------------------------------
def test_doctor_and_the_export_resolve_libreoffice_the_same_way():
    """A doctor that passes while the export says "not installed" sends
    the reader hunting through the wrong half of the system."""
    src = (Path(__file__).resolve().parent.parent
           / "src" / "latters" / "doctor.py").read_text(encoding="utf-8")
    app = (Path(__file__).resolve().parent.parent
           / "src" / "latters" / "web" / "app.py").read_text(encoding="utf-8")
    block = src[src.index("def _libreoffice"):src.index("def _fonts")]
    # A call, not a mention: both files discuss shutil.which in comments
    # explaining why they no longer use it.
    assert 'shutil.which("soffice")' not in block
    assert "shutil.which(\"libreoffice\")" not in block
    assert "soffice.find()" in block
    assert 'shutil.which("soffice")' not in app
    assert 'shutil.which("libreoffice")' not in app


def test_the_export_error_names_where_it_looked(monkeypatch, tmp_path):
    """"Not installed" is wrong and unhelpful on a machine that has it in
    Program Files but not on PATH. Say what was searched."""
    from fastapi import HTTPException

    from latters.web.app import _write_export

    monkeypatch.setattr(soffice, "find", lambda: None)
    with pytest.raises(HTTPException) as exc:
        _write_export("नमस्ते", "pdf", tmp_path / "out", None)
    assert exc.value.status_code == 501
    assert "Program Files" in exc.value.detail
    assert "DOCX" in exc.value.detail


def test_docx_export_never_depends_on_libreoffice(monkeypatch, tmp_path):
    """The fallback the error message recommends has to actually work when
    LibreOffice is the thing that is broken."""
    from latters.web.app import _write_export

    monkeypatch.setattr(soffice, "find", lambda: None)
    out = _write_export("नमस्ते", "docx", tmp_path / "out", None)
    assert out.exists() and out.suffix == ".docx"
    out = _write_export("नमस्ते", "txt", tmp_path / "out", None)
    assert out.read_text(encoding="utf-8") == "नमस्ते"
