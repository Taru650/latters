# What is finished, what is not, and what only you can do

All eight phases of `IMPLEMENTATION_PLAN.md` are built. **The project is not
finished**, and the gap is not code. This document says exactly where the
line is, so nobody has to infer it from a commit log.

## The one-sentence status

Every part of the pipeline works and is tested; **nothing about the Hindi it
produces has ever been checked by a person who reads Hindi**, no model has
ever run in any environment where this was developed, and the Windows
installer has never run on Windows.

## Built and verified

| | evidence |
|---|---|
| Legacy font conversion | 4 tables, 171 slots, 51-pair seed regression |
| Archive triage, segmentation | 475 letters from 5 real files, form-aware |
| Illegal-sequence validator | 11 rules; `latin_inside_word` found 2 missing slots |
| PDF + scan intake with OCR | 5 real letters; routing on readability |
| Field extraction | three-state FOUND/BLANK/ABSENT |
| Classification | department macro-F1 0.934 behind a real 0.35 gate |
| Skeleton mining | 13 cells; best supplies 88% of a letter |
| Retrieval | R@5 0.943, cellP@5 0.465 ± 0.028 vs 0.062 random, 1 ms |
| Drafting + verification surface | sources, removals, invented numbers |
| Web app | 14 routes, DOCX/PDF/TXT export, all verified over HTTP |
| Editing-effort metric | accrues from real use, no study required |
| Three scorecards | refuse to report a number they do not have |
| Backup | SQLite backup API + integrity check, not a file copy |
| `latters doctor` | every external dependency, with the fix for each |

411 tests. That is a statement about the code, not about the Hindi.

## Not done, and only you can do it

### 1. The gold set — the one that can invalidate everything above

Nothing in this project has ever checked the converted Hindi against someone
who reads Hindi. **87.6% of your archive is DevLys 040 or Kruti Dev 041, and
both tables inherit Kruti Dev 010 with zero overrides** — not proven
equivalent, just unexamined.

This is not a theoretical worry. A wrong mapping produces *well-formed Hindi*
that every automatic check in this system accepts. The validator cannot see
it, the retriever scores the same, the classifier is unaffected. Only a
reader can.

```
latters gold extract archive -o review -n 200 --blind-fraction 0.2
#  open review.docx on a machine with Kruti Dev 010/041 and DevLys 040
#  installed -- if the left column reads as Latin gibberish there, the font
#  is missing and the sheet is useless. Check that FIRST.
#  A Hindi reader fills verdict_or_correction in review.tsv.
latters gold collect review/review.tsv -o gold/office_saran.tsv
latters scorecard --db corpus.db
```

Cost: about two hours of one person who reads Hindi. There is nothing else
on this list of comparable value.

### 2. Run a model, once

No model has ever run in any environment where this was developed. Every
claim about the quality of the Hindi the system *writes* is architectural —
the skeleton supplies 88% of the letter, so the model only writes the body —
and not one word of generated Hindi has been read by anyone.

```
ollama pull gemma3:1b
latters draft "अंचल अधिकारी से दाखिल-खारिज में विलंब पर प्रतिवेदन मांगना है" ^
  --db corpus.db --skeletons skeletons
```

If the body is unusable, the fix is a bigger model after the RAM upgrade,
not more code. Say so early rather than late.

### 3. Install it on the target machine

`packaging/` has never run on Windows. Treat the first install as a
debugging session — but `latters doctor` now turns that from an
investigation into a checklist.

### 4. Ten dispatched drafts

The drafting scorecard fills itself from ordinary work. Ten letters drafted,
edited and exported give the median editing effort, which is **the number
that decides whether this project was worth building**. Above 0.70, typing
was faster.

## The two cheap hardware fixes, in order

1. **A second 8 GB SODIMM into the empty slot A (~₹2,000).** The machine is
   running single-channel, which costs roughly half its memory bandwidth,
   and memory bandwidth is what limits token generation. This is the single
   highest-value rupee in the project.
2. **An SSD (~₹2,500).** The disk measured 11–27 MB/s. Model load and first
   draft are dominated by it.

Only after the RAM is `gemma3:4b-it-qat` worth benchmarking.

## What this project got wrong, and what that predicts

Worth reading before trusting any number in these documents, because the
same mistake will happen again:

- **Three figures were published to three or four decimals from a single
  draw**, then moved when re-run: retrieval cellP@5 (0.500 → 0.465),
  department macro-F1 (0.812 → 0.787 → 0.934 once the *shipped* model was
  measured), OCR accuracy (0.9805 → "about 96–98%"). Every one was inside
  its error bar. Re-run before quoting.
- **A confidence gate that had never rejected anything** sat behind three
  documents claiming the department was applied behind a gate. 445 of 445
  predictions came back at exactly 1.0000.
- **A scorecard reported "usable, 100%"** from a seed set written by whoever
  wrote the mapping table.
- **Two OCR decisions blessed by a 27-run synthetic study were both wrong**,
  and five real letters overturned them in twenty minutes.

The pattern: *synthetic validation of a component you also designed tells you
almost nothing.* The gold set is the last place in this project where that
pattern is still unbroken — which is exactly why it is item 1.

## Where to look

| | |
|---|---|
| Why the system is shaped this way | `IMPLEMENTATION_PLAN.md` |
| Current measured numbers | `REBUILD.md` |
| For whoever inherits the code | `MAINTENANCE.md` |
| For the office, in Hindi | `HANDOVER_HI.md` |
| Step by step on the target machine | `LOCAL_SETUP.md` |
| What each phase found | `PHASE0`–`PHASE7_FINDINGS.md` |
| Scan quality limits | `OCR_DEGRADATION.md` |
