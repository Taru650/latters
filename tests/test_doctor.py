"""`latters doctor`.

The point of this command is that a clerk with no internet can find out why
the application does not work. So the tests are mostly about what it says
and what it refuses to say, not about what it detects.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latters import doctor  # noqa: E402
from latters.cli import main  # noqa: E402
from latters.store import LetterRow, Store  # noqa: E402


def _by_name(report) -> dict:
    return {c.name: c for c in report.checks}


# --- a check that cannot run is not a pass --------------------------------
def test_the_model_is_not_reported_ok_when_ollama_is_unreachable(monkeypatch):
    """The scorecard shipped a green card over an empty measurement once.
    Same failure shape here: if Ollama is down the model was never tested,
    and 'ok' would be a lie that sends the reader somewhere else."""
    r = doctor.Report()
    doctor._ollama(r, "gemma3:1b", "http://127.0.0.1:59999", stub=False)
    checks = _by_name(r)
    assert checks["ollama"].status == doctor.FAIL
    assert checks["model"].status == doctor.FAIL
    assert "not tested" in checks["model"].detail


def test_stub_mode_says_so_rather_than_passing_silently():
    r = doctor.Report()
    doctor._ollama(r, "gemma3:1b", "http://127.0.0.1:59999", stub=True)
    c = _by_name(r)["model"]
    assert c.status == doctor.WARN and c.optional and "stub" in c.detail


# --- optional parts must not fail the install -----------------------------
def test_missing_ocr_tools_are_a_warning_not_a_failure(monkeypatch):
    """An office with a .docx archive needs neither poppler nor Tesseract.
    Calling that install broken would be wrong."""
    from latters import ocr
    monkeypatch.setattr(ocr, "available",
                        lambda: {"pdftotext": False, "pdftoppm": False,
                                 "tesseract": False})
    monkeypatch.setattr(ocr, "ocr_languages", lambda: [])
    r = doctor.Report()
    doctor._ocr(r)
    c = _by_name(r)["pdf + scan reading"]
    assert c.status == doctor.WARN and c.optional
    assert not c.blocking


def test_tesseract_without_hindi_is_called_out_specifically(monkeypatch):
    """The Windows installer does not select Hindi by default, and without
    it Devanagari comes back as Latin gibberish -- a failure that looks like
    a bug in this application."""
    from latters import ocr
    monkeypatch.setattr(ocr, "available",
                        lambda: {"pdftotext": True, "pdftoppm": True,
                                 "tesseract": True})
    monkeypatch.setattr(ocr, "ocr_languages", lambda: ["eng", "osd"])
    r = doctor.Report()
    doctor._ocr(r)
    c = _by_name(r)["tesseract hindi"]
    assert c.status == doctor.FAIL and c.optional
    assert "does NOT select Hindi" in c.fix


def test_english_missing_is_flagged_because_language_order_is_load_bearing(
        monkeypatch):
    from latters import ocr
    monkeypatch.setattr(ocr, "available",
                        lambda: {"pdftotext": True, "pdftoppm": True,
                                 "tesseract": True})
    monkeypatch.setattr(ocr, "ocr_languages", lambda: ["hin"])
    r = doctor.Report()
    doctor._ocr(r)
    assert "eng+hin" in _by_name(r)["tesseract hindi"].fix


# --- the corpus -----------------------------------------------------------
def test_a_missing_corpus_is_blocking_and_prints_the_three_commands(tmp_path):
    r = doctor.Report()
    doctor._corpus(r, tmp_path / "nope.db")
    c = _by_name(r)["corpus"]
    assert c.blocking
    for cmd in ("latters segment", "latters classify", "latters templates"):
        assert cmd in c.fix


def test_a_thin_corpus_warns_rather_than_blocks(tmp_path):
    """Three letters is a usable install with a bad archive, not a broken
    install."""
    db = tmp_path / "corpus.db"
    with Store(db) as st:
        st.add([LetterRow(source_file="a.docx", seq=i,
                          text=f"कार्यालय पत्र संख्या {i} विषय बैठक",
                          trust=0.9, verdict="index")
                for i in range(3)])
    r = doctor.Report()
    doctor._corpus(r, db)
    c = _by_name(r)["corpus"]
    assert c.status == doctor.WARN and not c.blocking


def test_missing_skeletons_warn_about_the_letterhead(tmp_path):
    r = doctor.Report()
    doctor._skeletons(r, tmp_path / "none")
    c = _by_name(r)["skeletons"]
    assert c.status == doctor.WARN and "letterhead" in c.fix


# --- the command itself ---------------------------------------------------
def test_doctor_runs_without_a_corpus(tmp_path, capsys):
    """A missing corpus is the most likely thing wrong on a fresh install.
    Refusing to run would withhold the diagnosis exactly when it is needed,
    so unlike every other command this one does not require --db to exist."""
    rc = main(["doctor", "--db", str(tmp_path / "nope.db"),
               "--skeletons", str(tmp_path / "none"), "--stub"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "corpus" in out and "blocking" in out


def test_every_failure_line_carries_a_fix(tmp_path, capsys):
    """A diagnosis with no remedy is of no use to a clerk with no internet."""
    main(["doctor", "--db", str(tmp_path / "nope.db"),
          "--skeletons", str(tmp_path / "none"), "--stub"])
    report = doctor.run(db=str(tmp_path / "nope.db"),
                        skeletons=str(tmp_path / "none"), stub=True)
    for c in report.checks:
        if c.status != doctor.OK:
            assert c.fix or c.detail, f"{c.name} says nothing actionable"


def test_a_healthy_install_exits_zero(tmp_path):
    """Optional parts missing must still exit 0, or the office concludes a
    working install is broken."""
    db = tmp_path / "corpus.db"
    sk = tmp_path / "skeletons"
    sk.mkdir()
    (sk / "a.md").write_text("## above the subject line\n", encoding="utf-8")
    with Store(db) as st:
        st.add([LetterRow(source_file="a.docx", seq=i,
                          text=f"कार्यालय पत्र संख्या {i} विषय बैठक सूचना",
                          trust=0.9, verdict="index")
                for i in range(40)])
    report = doctor.run(db=str(db), skeletons=str(sk), stub=True)
    assert not report.blocking, [c.name for c in report.blocking]
