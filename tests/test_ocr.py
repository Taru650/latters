"""PDF and scanned-image intake.

The measurements these assert on were taken against known ground truth, not
guessed, and each one decided a line of `ocr.py`. Tests that need the
binaries skip cleanly: an office archive of .docx needs neither poppler nor
Tesseract, and the core pipeline must not start depending on them.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import zlib
import struct
import random
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latters.extract import UnsupportedFormat, convert_document, read_document  # noqa: E402
from latters.ocr import (LOW_WORD_CONF, MIN_TEXT_LAYER_CHARS, OCR_LANGS,  # noqa: E402
                         OcrReport, available, ocr_languages, read_image,
                         read_pdf)

needs_poppler = pytest.mark.skipif(
    not shutil.which("pdftotext"), reason="poppler not installed")
needs_tesseract = pytest.mark.skipif(
    not shutil.which("tesseract"), reason="Tesseract not installed")
needs_hindi = pytest.mark.skipif(
    "hin" not in ocr_languages(), reason="Tesseract Hindi pack not installed")

LETTER = ("कार्यालय जिला शिक्षा अधिकारी, सारण\n\n"
          "पत्रांक / F.No. DEO/SRN/2024/1187\nदिनांक 15.03.2024\n\n"
          "सेवा में,\nसमस्त प्राचार्य, शासकीय उच्चतर माध्यमिक विद्यालय\n\n"
          "विषय: मासिक समीक्षा बैठक की सूचना।\n\nमहाशय,\n\n"
          "उपर्युक्त विषय के प्रसंग में कहना है कि दिनांक 25.03.2024 को मासिक "
          "समीक्षा बैठक आयोजित की जा रही है। कृपया आवश्यक कार्यवाही सुनिश्चित "
          "करते हुए प्रतिवेदन इस कार्यालय को उपलब्ध कराएँ।\n\n"
          "विश्वासभाजन\nहस्ताक्षर\nजिला शिक्षा अधिकारी")


@pytest.fixture(scope="module")
def letter_pdf(tmp_path_factory):
    """A real PDF with a real text layer, built the way the app builds one."""
    if not shutil.which("soffice"):
        pytest.skip("LibreOffice not installed")
    work = tmp_path_factory.mktemp("pdf")
    from latters import docx_writer as D
    docx = work / "letter.docx"
    D.write(docx, [D.para(D.run(line or " ", "Nirmala UI", size_pt=12))
                   for line in LETTER.split("\n")])
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf",
                    "--outdir", str(work), str(docx)],
                   capture_output=True, timeout=180)
    pdf = work / "letter.pdf"
    if not pdf.exists():
        pytest.skip("LibreOffice could not produce a PDF here")
    return pdf


@pytest.fixture(scope="module")
def scan_png(letter_pdf, tmp_path_factory):
    """That same letter rendered at 300 dpi -- a clean office scan."""
    if not shutil.which("pdftoppm"):
        pytest.skip("poppler not installed")
    work = tmp_path_factory.mktemp("scan")
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-f", "1", "-l", "1",
                    str(letter_pdf), str(work / "s")],
                   capture_output=True, timeout=180)
    pages = sorted(work.glob("s*.png"))
    if not pages:
        pytest.skip("could not rasterise")
    return pages[0]


# --- the two paths must not be confused -----------------------------------
@needs_poppler
def test_a_pdf_with_a_text_layer_is_not_ocrd(letter_pdf):
    """Running OCR over text that is already there throws away a perfect
    extraction and replaces it with a 98% one."""
    doc, report = read_pdf(letter_pdf)
    assert report.tier == "pdf"
    assert report.ocr_pages == 0
    assert report.confidence == 1.0
    text, _ = convert_document(doc)
    assert "समीक्षा" in text and "DEO/SRN/2024/1187" in text


@needs_poppler
def test_the_text_layer_read_warns_about_reordered_matras(letter_pdf):
    """The risk register rates PDF glyph reordering High: a legacy-font PDF
    can carry matras in visual order that look right in a viewer."""
    _, report = read_pdf(letter_pdf)
    assert any("matras" in w for w in report.warnings)


@needs_tesseract
@needs_hindi
def test_a_scan_is_read_by_ocr_and_says_so(scan_png):
    doc, report = read_image(scan_png)
    assert report.tier == "ocr" and report.ocr_pages == 1
    assert report.mean_word_confidence > LOW_WORD_CONF
    text, _ = convert_document(doc)
    assert "समीक्षा" in text


@needs_poppler
@needs_tesseract
@needs_hindi
def test_forcing_ocr_on_a_text_pdf_still_works(letter_pdf):
    _, report = read_pdf(letter_pdf, force_ocr=True)
    assert report.tier == "ocr" and report.ocr_pages >= 1


# --- the finding that decided OCR_LANGS -----------------------------------
@needs_tesseract
@needs_hindi
def test_the_letter_number_survives_ocr(scan_png):
    """THE measurement behind `OCR_LANGS = "eng+hin"`. With `-l hin` alone,
    Tesseract forces Latin and ASCII digits into Devanagari and
    `F.No. DEO/SRN/2024/1187` came out `8४0. 0£50/579/2024/787`.

    That is not 3% of characters, it is the branch code -- which gives the
    department EXACTLY, better than a classifier that cannot predict five of
    this office's departments at all.
    """
    assert OCR_LANGS.startswith("eng"), "language order is load-bearing"
    doc, _ = read_image(scan_png)
    text, _ = convert_document(doc)
    assert "DEO/SRN/2024/1187" in text


@needs_tesseract
@needs_hindi
def test_ocr_of_a_clean_scan_is_accurate_enough_to_be_worth_storing(scan_png):
    """Measured 0.9805 character accuracy on this exact input. The assertion
    is loose on purpose -- Tesseract versions differ -- but a drop below 0.90
    means something broke, not that the model got slightly worse."""
    from latters.effort import measure
    doc, _ = read_image(scan_png)
    text, _ = convert_document(doc)
    assert 1.0 - measure(LETTER, text).score > 0.90


# --- refusing to guess ----------------------------------------------------
def _noise_png(path: Path) -> Path:
    random.seed(0)
    w = h = 400
    raw = b"".join(b"\x00" + bytes(random.randint(0, 255) for _ in range(w))
                   for _ in range(h))

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return path


@needs_tesseract
def test_noise_is_reported_as_unusable_not_as_a_letter(tmp_path):
    """Measured: a clean scan averages 92-94 word confidence and pure noise
    averages 11.3. Without folding that in, an OCR'd page scores like a clean
    DOCX -- the validator only catches ILLEGAL Devanagari, and a
    misrecognised word is usually perfectly legal Devanagari that is wrong."""
    png = _noise_png(tmp_path / "noise.png")
    try:
        _, report = read_image(png)
    except UnsupportedFormat:
        return  # no text at all is also a correct answer
    assert report.mean_word_confidence < LOW_WORD_CONF
    assert report.confidence == 0.0
    assert any("unreadable" in w for w in report.warnings)


def test_confidence_maps_a_usable_scan_above_zero_and_a_bad_one_to_zero():
    assert OcrReport(tier="pdf").confidence == 1.0
    assert OcrReport(tier="ocr", mean_word_confidence=93.0).confidence > 0.8
    assert OcrReport(tier="ocr", mean_word_confidence=60.0).confidence == 0.0
    assert OcrReport(tier="ocr", mean_word_confidence=11.3).confidence == 0.0


# --- the pipeline treats OCR output as already-Unicode --------------------
@needs_tesseract
@needs_hindi
def test_ocr_text_is_not_run_through_the_legacy_font_conversion(scan_png):
    """Tesseract emits Unicode Devanagari. Passing it through the Kruti Dev
    substitution would turn correct Hindi into garbage, so every run must
    carry font=None."""
    doc, _ = read_image(scan_png)
    assert all(r.font is None for b in doc.blocks for r in b.runs)
    before = "".join(r.text for b in doc.blocks for r in b.runs)
    after, tables = convert_document(doc)
    # Every character went through untouched; no legacy table was applied.
    assert set(tables) == {"(passthrough)"}
    assert "कार्यालय" in before and "कार्यालय" in after


@needs_poppler
def test_read_document_dispatches_pdfs_and_images(letter_pdf):
    """The public entry point used to refuse PDFs outright."""
    doc = read_document(letter_pdf)
    assert doc.blocks


def test_an_unknown_extension_is_still_refused(tmp_path):
    p = tmp_path / "x.xyz"
    p.write_text("hello")
    with pytest.raises(UnsupportedFormat):
        read_document(p)


def test_availability_is_reportable_without_the_tools_installed():
    """The admin page asks before anyone uploads forty scans."""
    tools = available()
    assert set(tools) == {"pdftotext", "pdftoppm", "tesseract"}
    assert all(isinstance(v, bool) for v in tools.values())


# --- the page that never finishes -----------------------------------------
def test_a_page_that_times_out_returns_zero_not_an_exception(monkeypatch, tmp_path):
    """Found by the degradation study: Tesseract ran over 300 SECONDS on a
    heavily speckled page and would have kept going. On a batch of forty
    scans that is hours of a blocked web worker on a 15 W CPU.

    One unreadable page must not lose the other thirty-nine, so the timeout
    returns zero confidence rather than raising."""
    import subprocess as sp
    from latters import ocr

    def boom(*a, **kw):
        raise sp.TimeoutExpired(cmd="tesseract", timeout=ocr.OCR_PAGE_TIMEOUT)

    monkeypatch.setattr(ocr.shutil, "which", lambda n: "/usr/bin/" + n)
    monkeypatch.setattr(ocr.subprocess, "run", boom)
    text, mean, low = ocr.ocr_image(tmp_path / "whatever.png")
    assert text == "" and mean == 0.0 and low == 1.0


def test_the_per_page_timeout_is_short_enough_to_matter():
    """300s per page was the default that caused the hang. A page that
    cannot be read in a minute is not going to become readable."""
    from latters.ocr import OCR_PAGE_TIMEOUT
    assert OCR_PAGE_TIMEOUT <= 120


def test_a_timed_out_scan_is_reported_rather_than_stored_as_a_letter(
        monkeypatch, tmp_path):
    from latters import ocr

    monkeypatch.setattr(ocr, "ocr_image", lambda p, **kw: ("   ", 0.0, 1.0))
    with pytest.raises(UnsupportedFormat):
        ocr.read_image(tmp_path / "blank.png")


# --- what five real office letters changed --------------------------------
def test_an_unreadable_text_layer_falls_back_to_ocr(tmp_path, monkeypatch):
    """The rule used to be "a PDF with a text layer is never OCR'd", written
    from a PDF this project generated itself. Five real letters inverted it:
    four had text layers scoring 0.000 because the PDFs embed subsetted fonts
    with no ToUnicode map (विषय:- came out ftqq:-), and the fifth carried
    real Hindi with matras dropped and reordered (दिनांक as िदनांक).

    OCR beat the text layer on all five. The decision is now made on whether
    the text is READABLE, using the validator that already exists."""
    from latters import ocr

    garbage = "ftqq:- f{qiq-zzog.zozo o} srq{r{c o+:oo qd 3fft{dc qitqq t " * 6
    monkeypatch.setattr(ocr, "_pdf_text_layer", lambda p: (garbage, 1))
    monkeypatch.setattr(ocr, "_ocr_pdf",
                        lambda p, langs: ("विषय: बैठक की सूचना।", 1, [93.0], [0.0]))
    _, report = ocr.read_pdf(tmp_path / "x.pdf")
    assert report.tier == "ocr"
    assert any("text layer scored" in w for w in report.warnings)


def test_a_readable_text_layer_is_still_preferred(tmp_path, monkeypatch):
    """The inversion must not become "always OCR" -- that would throw away a
    perfect extraction for a 98% one wherever the PDF is genuinely clean."""
    from latters import ocr

    good = ("कार्यालय जिला पदाधिकारी सारण छपरा। विषय: मासिक समीक्षा बैठक की "
            "सूचना। महाशय, उपर्युक्त विषय के प्रसंग में कहना है कि आवश्यक "
            "कार्यवाही सुनिश्चित करते हुए प्रतिवेदन उपलब्ध कराएँ। ") * 3
    monkeypatch.setattr(ocr, "_pdf_text_layer", lambda p: (good, 1))
    monkeypatch.setattr(ocr, "_ocr_pdf",
                        lambda p, langs: pytest.fail("should not have OCR'd"))
    _, report = ocr.read_pdf(tmp_path / "x.pdf")
    assert report.tier == "pdf" and report.confidence == 1.0


@needs_tesseract
@needs_hindi
def test_confidence_is_the_median_so_a_letterhead_cannot_condemn_the_body(
        scan_png):
    """Five real letters: mean 85.1-91.7, median 93.3-96.0. The gap is the
    letterhead logo, the round stamp and decorative English, all of which
    recognise badly. On the mean no real letter could ever reach the index
    threshold; on the median they score what a clean synthetic page does."""
    import statistics
    from latters.ocr import ocr_image

    text, reported, _ = ocr_image(scan_png)
    assert text
    # It must be the median of the per-word confidences, not their mean.
    import csv as _csv, subprocess as _sp, tempfile as _tf
    with _tf.TemporaryDirectory() as t:
        stem = Path(t) / "p"
        _sp.run(["tesseract", str(scan_png), str(stem), "-l", OCR_LANGS,
                 "--psm", "6", "tsv"], capture_output=True, timeout=120)
        with stem.with_suffix(".tsv").open(encoding="utf-8") as fh:
            vals = [float(r["conf"]) for r in _csv.DictReader(fh, delimiter="\t")
                    if (r.get("text") or "").strip() and float(r["conf"]) >= 0]
    assert abs(reported - statistics.median(vals)) < 0.01


def test_the_noise_guard_survives_the_switch_to_the_median():
    """On pure noise every word is bad, so the median goes down with the mean
    rather than being rescued by it. Measured: median 8.8 -> multiplier 0.0."""
    assert OcrReport(tier="ocr", mean_word_confidence=8.8).confidence == 0.0
    assert OcrReport(tier="ocr", mean_word_confidence=93.3).confidence > 0.80
