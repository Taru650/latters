# How bad does a scan have to be before OCR stops working?

`ocr.py` was built on one measurement: 0.9805 character accuracy on a page
rendered at 300 dpi. That is a best case with no skew, no noise, no
compression and perfect contrast, and an office scanner produces none of
those. This is the study that tested the rest.

    pip install pillow        # study only, NOT a runtime dependency
    python scripts/ocr_degradation.py -o docs/OCR_DEGRADATION.md

## Three findings, in order of how much they change what you do

### 1. Almost nothing matters, except noise — and noise is a cliff

Accuracy sits between 0.962 and 0.976 across **every** axis: 100 dpi to
600 dpi, 0° to 5° of skew, JPEG quality 15, Gaussian blur radius 3.0, and
contrast crushed to a fifth. The letter number survived all of them.

Then at Gaussian noise σ=50 it goes to **0.0000**. Not degraded — gone.
There is no middle: σ=30 reads at 0.9714 and σ=50 reads nothing at all.

The practical advice this produces is short. Scan in **greyscale, not
colour** — colour scanning of a grey document is where speckle comes from,
and speckle is the one thing that kills this. Everything else people worry
about (get the page straight, scan at 600 dpi, don't use JPEG) is not worth
the effort here.

### 2. The confidence guard works, which is the part that had to be true

An OCR error is normally well-formed Hindi, so nothing downstream can catch
it. The only defence is Tesseract's own confidence, and it is only a defence
if it falls when accuracy falls. It does: at the σ=50 cliff, confidence went
to **0.0 in the same step** as accuracy. Trust collapses with it, the letter
lands in `review`, and nobody indexes a blank page believing it is a letter.

### 3. My 0.9805 was the top of a range, quoted as though it were a point

Across 27 runs of the same page the accuracy varied between 0.9617 and
0.9763 with nothing degraded enough to explain it — that is Tesseract's own
run-to-run variation plus my whitespace normalisation. **Read the figure as
"about 96–98% of characters", not 0.9805.** Quoting a single draw to four
decimals is the exact mistake this project caught itself making with
retrieval P@5 and classifier macro-F1, and I made it again here.

## The bug this study found

Tesseract **ran for over 300 seconds** on the σ=50 page and would have kept
going. On a batch upload of forty scans that is hours of a blocked web
worker, on a 15 W CPU with a spinning disk. `OCR_PAGE_TIMEOUT = 60` now
caps it per page, a page that trips it returns zero confidence rather than
raising, and the upload reports it — one unreadable page in a batch of forty
must not lose the other thirty-nine.

## What this still does not measure

Every input here is a **synthetic degradation of a clean digital render**.
A real scan is a photograph of physical paper: it has texture, ink bleed,
show-through from the reverse, staple shadows, and a photocopier's own
artefacts. None of those are modelled, and the flatness of these results
across five axes is itself a hint that the degradations are too polite.

**One real scanned letter from the office would be worth more than this
entire table.** Until then, treat the numbers as an upper bound.

## Postscript: five real letters arrived, and they were worth more

They changed two decisions this table could not have:

1. **A PDF text layer is used only if it is readable.** OCR beat the text
   layer on all five. Four text layers scored 0.000 on the validator — the
   PDFs embed subsetted fonts with no ToUnicode map, so `विषय:-` arrives as
   `ftqq:-` and no mapping recovers it. The fifth was real Hindi with matras
   dropped and reordered (`दिनांक` → `िदनांक`), which is worse for being
   plausible. Routing now uses `MIN_TEXT_LAYER_QUALITY`, not a character
   count.

2. **Confidence is the median, not the mean.** Mean 85.1–91.7 against median
   93.3–96.0; the gap is the letterhead logo, the round stamp and decorative
   English. On the mean, no real letter could reach the index threshold.

Neither was visible in any synthetic degradation, and both were wrong in the
shipped code. The lesson generalises past OCR: **the degradations you can
imagine are not the ones the real inputs have.**

## Full results

| axis | setting | char accuracy | Tesseract conf | letter number |
|---|---|---:|---:|---|
| resolution | 100 dpi | 0.9738 | 93.6 | intact |
| resolution | 150 dpi | 0.9570 | 94.0 | intact |
| resolution | 200 dpi | 0.9691 | 93.6 | intact |
| resolution | 300 dpi | 0.9668 | 93.0 | intact |
| resolution | 400 dpi | 0.9763 | 94.4 | intact |
| resolution | 600 dpi | 0.9623 | 93.2 | intact |
| skew | 0.5 deg | 0.9667 | 91.8 | intact |
| skew | 1.0 deg | 0.9683 | 94.3 | intact |
| skew | 2.0 deg | 0.9734 | 94.1 | intact |
| skew | 5.0 deg | 0.9684 | 93.1 | intact |
| noise | sigma 5 | 0.9620 | 93.5 | intact |
| noise | sigma 15 | 0.9714 | 94.2 | intact |
| noise | sigma 30 | 0.9714 | 94.0 | intact |
| noise | sigma 50 | 0.0000 | 0.0 | **lost** |
| noise | sigma 80 | 0.0000 | 0.0 | **lost** |
| jpeg | quality 80 | 0.9617 | 93.4 | intact |
| jpeg | quality 50 | 0.9619 | 93.5 | intact |
| jpeg | quality 30 | 0.9620 | 93.1 | intact |
| jpeg | quality 15 | 0.9691 | 93.7 | intact |
| fading | contrast x0.7 | 0.9620 | 93.0 | intact |
| fading | contrast x0.5 | 0.9763 | 93.9 | intact |
| fading | contrast x0.35 | 0.9763 | 93.9 | intact |
| fading | contrast x0.2 | 0.9714 | 94.4 | intact |
| blur | radius 0.5 | 0.9619 | 93.2 | intact |
| blur | radius 1.0 | 0.9645 | 92.2 | intact |
| blur | radius 2.0 | 0.9714 | 94.5 | intact |
| blur | radius 3.0 | 0.9714 | 93.6 | intact |
