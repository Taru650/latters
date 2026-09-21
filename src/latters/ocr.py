"""PDF and scanned-image intake, with OCR only where it is actually needed.

Everything here is a subprocess call to a bundled binary -- `pdftotext`,
`pdftoppm` and `tesseract`. That is the same trade LibreOffice already gets
in `web/app.py` and it is deliberate: the alternatives (PyMuPDF, pdfplumber,
pytesseract + Pillow) are tens of megabytes of wheels for a machine with 3 GB
free, a spinning disk and no internet to install them from.

Four things here were measured, not assumed, and each one changes the output.

**1. A PDF with a text layer must NOT be OCR'd.** Running OCR over text that
is already there throws away a perfect extraction and replaces it with a 98%
one. `pdftotext` first, always; OCR only when the page comes back empty.

**2. `eng+hin`, in that order, not `hin`.** Tesseract will force every glyph
into the scripts you give it. Measured on a rendered letter at 300 dpi
against known ground truth:

    -l hin        char accuracy 0.9513   letter number DESTROYED
    -l hin+eng    char accuracy 0.9781   letter number DESTROYED
    -l eng+hin    char accuracy 0.9805   letter number recovered exactly

Read those to two decimals, not four. Re-running the same page 27 times in
the degradation study gave 0.962-0.976 with nothing changed that should
have mattered, so the real figure is "about 96-98% of characters" and the
gap between the three rows is the finding, not the third decimal.

With `hin` alone, `F.No. DEO/SRN/2024/1187` came out as
`8४0. 0£50/579/2024/787` -- Latin letters and ASCII digits pushed into
Devanagari. That is not a cosmetic loss. **The letter number carries the
branch code, and the branch code is what gives the department exactly** --
better than the classifier, which cannot even predict five of this office's
departments. Destroying it costs more than the 3% of characters it is worth.

**3. OCR output is Unicode, not legacy.** Tesseract emits Devanagari
directly, so these documents must skip the Kruti Dev / DevLys conversion
entirely. Runs are emitted with `font=None`, which `convert_document` passes
through untouched.

**4. Tesseract's own confidence is worth having.** Mean per-word confidence
was 92-94 on real letters and **11.3 on pure noise**, so it separates a
usable scan from a failed one. The degradation study confirmed the property
that actually matters: confidence falls *at the same step* accuracy does.
Noise is a cliff, not a slope -- sigma 30 reads at 0.97 and sigma 50 reads
nothing at all -- and at that cliff confidence went to 0.0 too. See
`docs/OCR_DEGRADATION.md`. Without it an OCR'd letter scores like a
clean DOCX: the validator only catches *illegal* Devanagari, and a
misrecognised letter is usually perfectly legal Devanagari that happens to
be the wrong word. See `ocr_confidence`.
"""
from __future__ import annotations

import csv
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .extract import Block, Document, Run, UnsupportedFormat

#: Extensions this module handles, mapped to the source tier a successful
#: read earns before any confidence adjustment. See segment.SOURCE_TIERS.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

#: Order matters -- see the module docstring. Overridable for an office whose
#: letters are in another script.
OCR_LANGS = "eng+hin"

#: Office scanners default to 200-300 dpi and Tesseract wants ~300 for
#: Devanagari conjuncts. Rasterising higher mostly costs time on a 15 W CPU.
RASTER_DPI = 300

#: Below this many characters of text layer, a PDF page is a scan. A page
#: with a letterhead image and three words of caption would otherwise be
#: read as "has text" and the body silently lost.
MIN_TEXT_LAYER_CHARS = 120

#: Tesseract per-word confidence is 0-100. Below this a word is more likely
#: wrong than right in practice; the share of such words is what gets folded
#: into the trust score.
LOW_WORD_CONF = 60.0

#: Per PAGE, not per batch. Found by degrading a test page: Tesseract took
#: **over 300 seconds on one noisy image** and would have kept going. A page
#: that cannot be read in a minute on this hardware is not going to become
#: readable, and a batch of forty scans must not hold a web worker for
#: hours. A page that trips this is reported, not silently dropped.
OCR_PAGE_TIMEOUT = 60

#: pdftotext and pdftoppm are I/O bound and predictable; they get their own,
#: looser budget.
OCR_TIMEOUT = 180


@dataclass
class OcrReport:
    """What happened, in enough detail to argue with."""

    tier: str                      # "pdf" (text layer) or "ocr"
    pages: int = 0
    ocr_pages: int = 0
    mean_word_confidence: float | None = None
    low_confidence_share: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """A 0-1 multiplier for the conversion score of an OCR'd document.

        A text-layer PDF returns 1.0: nothing was guessed. For OCR, Tesseract's
        mean word confidence is rescaled so that 100 -> 1.0 and 60 -> 0.0,
        because a page averaging 60 is not "60% right", it is unusable. The
        noise page measured 11.3 and lands at 0.0; real letters measured 92-94
        and land near 0.8-0.85.
        """
        if self.tier != "ocr" or self.mean_word_confidence is None:
            return 1.0
        return max(0.0, min(1.0, (self.mean_word_confidence - 60.0) / 40.0))


def _tool(name: str, why: str) -> str:
    path = shutil.which(name)
    if not path:
        raise UnsupportedFormat(
            f"{name} is not installed, so {why}. It ships in the offline "
            f"bundle; see packaging/install.bat.")
    return path


def available() -> dict[str, bool]:
    """Which parts of this module can run here. The admin page shows this."""
    return {"pdftotext": bool(shutil.which("pdftotext")),
            "pdftoppm": bool(shutil.which("pdftoppm")),
            "tesseract": bool(shutil.which("tesseract"))}


def ocr_languages() -> list[str]:
    """Installed Tesseract language packs, so a missing `hin` is visible."""
    if not shutil.which("tesseract"):
        return []
    try:
        out = subprocess.run(["tesseract", "--list-langs"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    return [l.strip() for l in out.stdout.splitlines()[1:] if l.strip()]


# --------------------------------------------------------------------------
def _pdf_text_layer(path: Path) -> tuple[str, int]:
    """Extract the existing text layer. Returns (text, n_pages)."""
    exe = _tool("pdftotext", "PDFs cannot be read")
    proc = subprocess.run([exe, "-enc", "UTF-8", str(path), "-"],
                          capture_output=True, timeout=OCR_TIMEOUT)
    text = proc.stdout.decode("utf-8", "replace")
    # \f is pdftotext's page separator.
    return text, max(1, text.count("\f"))


def ocr_image(path: Path, *, langs: str = OCR_LANGS) -> tuple[str, float, float]:
    """OCR one image. Returns (text, mean_word_conf, low_conf_share).

    Two passes over the same image: `tsv` for the confidences, plain text for
    the output. Tesseract's TSV *does* carry the recognised words, but
    rebuilding line and paragraph breaks from its block/par/line columns
    reproduces work its text writer already does correctly, and getting that
    subtly wrong would corrupt the segmentation downstream.
    """
    exe = _tool("tesseract", "scanned pages cannot be read")
    with tempfile.TemporaryDirectory(prefix="latters-ocr-") as tmp:
        stem = Path(tmp) / "page"
        base = [exe, str(path), str(stem), "-l", langs, "--psm", "6"]
        try:
            subprocess.run(base, capture_output=True,
                           timeout=OCR_PAGE_TIMEOUT)
            subprocess.run(base + ["tsv"], capture_output=True,
                           timeout=OCR_PAGE_TIMEOUT)
        except subprocess.TimeoutExpired:
            # Not an error to propagate: one unreadable page in a batch of
            # forty must not lose the other thirty-nine. Zero confidence is
            # the right answer and the caller reports it.
            return "", 0.0, 1.0

        txt = stem.with_suffix(".txt")
        text = txt.read_text(encoding="utf-8", errors="replace") if txt.exists() else ""

        confs: list[float] = []
        tsv = stem.with_suffix(".tsv")
        if tsv.exists():
            with tsv.open(encoding="utf-8", errors="replace") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    if not (row.get("text") or "").strip():
                        continue
                    try:
                        c = float(row["conf"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if c >= 0:
                        confs.append(c)

    if not confs:
        return text, 0.0, 1.0
    low = sum(1 for c in confs if c < LOW_WORD_CONF) / len(confs)
    return text, sum(confs) / len(confs), low


def _ocr_pdf(path: Path, *, langs: str) -> tuple[str, int, list[float], list[float]]:
    """Rasterise every page and OCR it."""
    exe = _tool("pdftoppm", "scanned PDFs cannot be rasterised for OCR")
    pages, means, lows = [], [], []
    with tempfile.TemporaryDirectory(prefix="latters-raster-") as tmp:
        subprocess.run([exe, "-png", "-r", str(RASTER_DPI), str(path),
                        str(Path(tmp) / "p")],
                       capture_output=True, timeout=OCR_TIMEOUT)
        images = sorted(Path(tmp).glob("p-*.png")) or sorted(Path(tmp).glob("p*.png"))
        for img in images:
            text, mean, low = ocr_image(img, langs=langs)
            pages.append(text)
            means.append(mean)
            lows.append(low)
            # A page that timed out contributes 0.0, which drags the
            # document's mean down -- correctly. Half a letter read is not
            # a letter read.
    return "\n".join(pages), len(pages), means, lows


def read_pdf(path: Path, *, langs: str = OCR_LANGS,
             force_ocr: bool = False) -> tuple[Document, OcrReport]:
    """Read a PDF: text layer if it has one, OCR if it does not."""
    text, pages = ("", 0) if force_ocr else _pdf_text_layer(path)
    report = OcrReport(tier="pdf", pages=pages)

    if force_ocr or len(text.strip()) < MIN_TEXT_LAYER_CHARS * max(pages, 1):
        if not force_ocr and text.strip():
            report.warnings.append(
                f"{path.name}: a text layer exists but holds only "
                f"{len(text.strip())} characters over {pages} page(s), which "
                f"is a scan with a caption rather than a typed document. "
                f"Read by OCR instead.")
        text, n, means, lows = _ocr_pdf(path, langs=langs)
        report = OcrReport(tier="ocr", pages=n or pages, ocr_pages=n,
                           warnings=report.warnings)
        if means:
            report.mean_word_confidence = sum(means) / len(means)
            report.low_confidence_share = sum(lows) / len(lows)
    else:
        report.warnings.append(
            f"{path.name}: read from the PDF text layer. If this file was "
            f"made from a legacy Hindi font, check a line or two by eye -- a "
            f"PDF can carry reordered matras that look correct in a viewer "
            f"and convert wrongly (और typed `vkSj` can extract as `vkjS`).")

    if not text.strip():
        raise UnsupportedFormat(
            f"{path.name}: no text found, by text layer or by OCR. If it is a "
            f"photograph of a page, try a flatter, straighter scan at 300 dpi.")
    return _as_document(path, text), report


def read_image(path: Path, *, langs: str = OCR_LANGS) -> tuple[Document, OcrReport]:
    """OCR a photograph or scan of a single page."""
    text, mean, low = ocr_image(path, langs=langs)
    if not text.strip():
        raise UnsupportedFormat(
            f"{path.name}: OCR found no text. A photograph taken at an angle "
            f"or in poor light usually fails here; scan the page flat at "
            f"300 dpi instead.")
    report = OcrReport(tier="ocr", pages=1, ocr_pages=1,
                       mean_word_confidence=mean, low_confidence_share=low)
    if not text.strip() or mean == 0.0:
        report.warnings.append(
            f"{path.name}: Tesseract gave up on this page (nothing read, or "
            f"over {OCR_PAGE_TIMEOUT}s). Heavy speckle is the usual cause; "
            f"scan again in greyscale rather than colour.")
    elif mean < LOW_WORD_CONF:
        report.warnings.append(
            f"{path.name}: mean OCR confidence {mean:.0f} of 100. A clean "
            f"300 dpi scan measures 92-94 and pure noise measures 11, so this "
            f"page is closer to unreadable than to usable. Do not index it "
            f"without reading it.")
    return _as_document(path, text), report


_BLANK = re.compile(r"\n{3,}")


def _as_document(path: Path, text: str) -> Document:
    """Wrap plain text as a Document with no font on any run.

    `font=None` is what makes the rest of the pipeline treat this as already
    Unicode: `classify_font` returns no table, so `convert_document` passes
    the text through without attempting a Kruti Dev substitution -- which on
    real Devanagari would turn it into garbage.
    """
    text = _BLANK.sub("\n\n", text.replace("\f", "\n"))
    blocks = [Block(runs=[Run(text=line, font=None)])
              for line in text.split("\n")]
    return Document(path=path, blocks=blocks)
