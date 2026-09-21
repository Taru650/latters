"""How bad does a scan have to get before OCR stops being worth having?

`ocr.py` quotes 0.9805 character accuracy, and that number came from a PDF
rendered at 300 dpi -- a scan with no skew, no noise, no compression and
perfect contrast. An office scanner produces none of those things, and
quoting a best case as though it were typical is how a tool gets deployed
and then quietly distrusted.

This script degrades a known page in the ways a real scan is degraded, one
axis at a time, and measures character accuracy and Tesseract's own
confidence against ground truth. It answers three questions the office
actually has to act on:

    * what dpi should we scan at?
    * does a slightly crooked page matter?
    * will we be able to TELL when a scan was too poor to use?

The third is the important one. An OCR error is usually well-formed Hindi,
so nothing downstream can catch it; the only defence is Tesseract's
confidence, and it is only a defence if it falls when accuracy falls.

    pip install pillow          # study only -- not a runtime dependency
    python scripts/ocr_degradation.py --out docs/OCR_DEGRADATION.md

Pillow is deliberately NOT in pyproject: nothing shipped needs it, and the
target machine has 3 GB free and no internet.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


LETTER = ("कार्यालय जिला शिक्षा अधिकारी, सारण\n\n"
          "पत्रांक / F.No. DEO/SRN/2024/1187\nदिनांक 15.03.2024\n\n"
          "सेवा में,\nसमस्त प्राचार्य, शासकीय उच्चतर माध्यमिक विद्यालय\n\n"
          "विषय: मासिक समीक्षा बैठक की सूचना।\n\nमहाशय,\n\n"
          "उपर्युक्त विषय के प्रसंग में कहना है कि दिनांक 25.03.2024 को मासिक "
          "समीक्षा बैठक आयोजित की जा रही है। कृपया आवश्यक कार्यवाही सुनिश्चित "
          "करते हुए प्रतिवेदन इस कार्यालय को उपलब्ध कराएँ।\n\n"
          "विश्वासभाजन\nहस्ताक्षर\nजिला शिक्षा अधिकारी")

#: The field that matters more than its 17 characters: it carries the branch
#: code, which gives the department exactly.
LETTER_NUMBER = "DEO/SRN/2024/1187"


def build_page(work: Path) -> Path:
    """Render the known letter to a PDF the way the app exports one."""
    from latters import docx_writer as D

    docx = work / "letter.docx"
    D.write(docx, [D.para(D.run(line or " ", "Nirmala UI", size_pt=12))
                   for line in LETTER.split("\n")])
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf",
                    "--outdir", str(work), str(docx)],
                   capture_output=True, timeout=300)
    pdf = work / "letter.pdf"
    if not pdf.exists():
        raise SystemExit("LibreOffice produced no PDF; cannot run the study")
    return pdf


def rasterise(pdf: Path, out: Path, dpi: int) -> Path:
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-f", "1", "-l", "1",
                    str(pdf), str(out / f"r{dpi}")],
                   capture_output=True, timeout=300)
    pages = sorted(out.glob(f"r{dpi}*.png"))
    if not pages:
        raise SystemExit(f"could not rasterise at {dpi} dpi")
    return pages[0]


def measure(png: Path) -> tuple[float, float, bool]:
    """Returns (char_accuracy, mean_word_confidence, letter_number_intact)."""
    from latters.effort import measure as edit
    from latters.ocr import ocr_image

    text, conf, _ = ocr_image(png)
    return (1.0 - edit(LETTER, text).score, conf, LETTER_NUMBER in text)


# --- the degradations -----------------------------------------------------
def skew(src: Path, dst: Path, degrees: float) -> Path:
    from PIL import Image
    im = Image.open(src).convert("L")
    im.rotate(degrees, resample=Image.BICUBIC, expand=True,
              fillcolor=255).save(dst)
    return dst


def noise(src: Path, dst: Path, sigma: float) -> Path:
    import random
    from PIL import Image
    im = Image.open(src).convert("L")
    rng = random.Random(0)
    px = bytearray(im.tobytes())
    for i in range(len(px)):
        px[i] = max(0, min(255, px[i] + int(rng.gauss(0, sigma))))
    Image.frombytes("L", im.size, bytes(px)).save(dst)
    return dst


def jpeg(src: Path, dst: Path, quality: int) -> Path:
    from PIL import Image
    Image.open(src).convert("L").save(dst, "JPEG", quality=quality)
    return dst


def faded(src: Path, dst: Path, factor: float) -> Path:
    """A photocopy of a photocopy: black goes grey, contrast collapses."""
    from PIL import Image, ImageEnhance
    im = Image.open(src).convert("L")
    ImageEnhance.Contrast(im).enhance(factor).save(dst)
    return dst


def blur(src: Path, dst: Path, radius: float) -> Path:
    from PIL import Image, ImageFilter
    Image.open(src).convert("L").filter(
        ImageFilter.GaussianBlur(radius)).save(dst)
    return dst


def main() -> int:
    _utf8_console()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default=None, help="write a Markdown table")
    args = ap.parse_args()

    for tool in ("soffice", "pdftoppm", "tesseract"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} is not installed; cannot run the study")
    try:
        import PIL  # noqa: F401
    except ImportError:
        raise SystemExit("pip install pillow  (study only)")

    rows: list[tuple[str, str, float, float, bool]] = []
    with tempfile.TemporaryDirectory(prefix="ocrdeg-") as tmp:
        work = Path(tmp)
        pdf = build_page(work)

        print("resolution")
        for dpi in (100, 150, 200, 300, 400, 600):
            png = rasterise(pdf, work, dpi)
            acc, conf, num = measure(png)
            rows.append(("resolution", f"{dpi} dpi", acc, conf, num))
            print(f"  {dpi:4} dpi   acc {acc:.4f}  conf {conf:5.1f}  "
                  f"number {'ok' if num else 'LOST'}")

        base = rasterise(pdf, work, 300)
        for name, fn, values, fmt in (
            ("skew", skew, (0.5, 1.0, 2.0, 5.0), "{} deg"),
            ("noise", noise, (5, 15, 30, 50, 80), "sigma {}"),
            ("jpeg", jpeg, (80, 50, 30, 15), "quality {}"),
            ("fading", faded, (0.7, 0.5, 0.35, 0.2), "contrast x{}"),
            ("blur", blur, (0.5, 1.0, 2.0, 3.0), "radius {}"),
        ):
            print(name)
            for v in values:
                dst = work / f"{name}-{v}.png"
                acc, conf, num = measure(fn(base, dst, v))
                rows.append((name, fmt.format(v), acc, conf, num))
                print(f"  {fmt.format(v):14} acc {acc:.4f}  conf {conf:5.1f}  "
                      f"number {'ok' if num else 'LOST'}")

    if args.out:
        lines = ["| axis | setting | char accuracy | Tesseract conf | "
                 "letter number |", "|---|---|---:|---:|---|"]
        for axis, setting, acc, conf, num in rows:
            lines.append(f"| {axis} | {setting} | {acc:.4f} | {conf:.1f} | "
                         f"{'intact' if num else '**lost**'} |")
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
