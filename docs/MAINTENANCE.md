# Maintenance — for whoever inherits this

Read `docs/IMPLEMENTATION_PLAN.md` for why the system is shaped the way it
is, and the `PHASE*_FINDINGS.md` files for what was measured. This document
is the short version: what runs, what breaks, and what to check before
believing any claim about how well it works.

## The one-paragraph summary

A Python package (`latters`) converts an office's legacy-font Word archive to
Unicode Hindi, splits it into letters, scores each one, indexes them in
SQLite, mines per-(department, letter-type) skeletons, and serves a two-page
web app that drafts a new letter from a retrieved handful of old ones using a
small local LLM through Ollama. It is offline by construction: no CDN, no
API, no telemetry. All state is `corpus.db` plus `skeletons/`.

## Where the state lives

| Path | What it is | Reconstructible? |
|---|---|---|
| `corpus.db` | letters, labels, drafts, effort scores | from the archive, except the corrections and the draft log |
| `skeletons/*.md` | per-cell letterheads, hand-corrected | **no** — this is the irreplaceable part |
| `archive/` | the original .docx files | no |
| `backups/` | `latters backup` output | — |

`latters backup` uses SQLite's backup API, not a file copy. The database runs
in WAL mode; copying the file while the server is writing silently loses the
most recent transactions.

## Running the tests

```
pip install -e ".[dev,web]"
python -m pytest -q
```

Everything is stdlib plus numpy, FastAPI and Jinja2. There is no sklearn, no
torch, no bundler, no node. That was a deliberate constraint — the target
machine has 8 GB of RAM, a spinning disk, and no internet to install wheels
with.

## The three scorecards

```
latters scorecard --db corpus.db
```

Exits non-zero while any card is unmeasured or failing, so it can gate a
release. Read the cards separately; they fail for different reasons and a
combined number would hide that.

**Do not trust a green CONVERSION card that only shows the seed regression
row.** The packaged `seed_krutidev010.tsv` was written by the same person who
wrote the mapping table, from the same assumptions. It proves the converter
still does what it did last week, and nothing at all about whether the Hindi
is right. Only a `gold/office_*.tsv` produced by a Hindi reader does that.

## What is actually known, as of the end of Phase 7

Measured on a real Saran district archive (5 files, ~508k legacy characters,
475 letters):

- Corpus: mean trust 0.904; 465 indexed, 10 for review.
- Retrieval: tfidf + department filter, same-cell P@5 0.500 ± 0.028 degraded.
- Classification: department macro-F1 0.812 (usable); letter type 0.636
  (suggest-only, and the code hard-codes `letter_type_confident = False`).
- Templates: the बैंकिंग/जाँच skeleton supplies 87% of a letter.
- DOCX and PDF export verified end to end over HTTP.

**Not known, and material:**

1. **No gold set exists.** The converted Hindi has never been read by a Hindi
   reader. 87.6% of the archive is DevLys 040 or Kruti Dev 041, and both
   tables inherit Kruti Dev 010 with zero overrides — not proven equivalent,
   just unexamined. This is the largest untested risk in the project.
2. **No generated Hindi has ever been assessed.** No model has run in any
   environment where this was developed. Every draft-quality claim is
   architectural, not empirical.
3. **Editing effort has no samples yet.** The mechanism is built (see below)
   but the office has to use the tool for the number to exist.

## Editing effort — the metric that decides success

`src/latters/effort.py`. Every generation is logged to the `drafts` table;
every export attaches the dispatched text and scores
`levenshtein(draft, dispatched) / len(dispatched)`.

Above **0.70**, more of the letter was rewritten than kept and the person
would have been faster typing it. That is a failure line, not a target to be
negotiated down when the number comes in high.

Watch `abandoned` alongside the median. A low median over three exports out
of ninety drafts means people are generating, giving up, and typing the
letter by hand — which reads as success in the median alone.

## Adding a font mapping

`src/latters/data/fonts/*.tsv`, one `legacy<TAB>unicode<TAB>note` per line.
`#inherit <table>` pulls in a base. A comment is `#` *not* followed by a tab,
because `#` is itself the `रु` slot.

To find missing slots without a Hindi reader, look for characters that appear
in a legacy-font run and have no table entry — they pass through raw. The
`latin_inside_word` validator rule catches the resulting output
automatically. That is how `Î` → ट्ट and `Ök` → झ were found; see
`PHASE2_FINDINGS.md`.

This cannot find a character mapped to the *wrong* Devanagari, because the
output is well-formed Hindi either way. Only the gold set does that.

## Things that will bite you

- **FTS5's default tokenizer destroys Devanagari.** The `categories 'L* N* Mn
  Mc Co'` in `store.py` is load-bearing; without it `समीक्षा` indexes as
  `["सम","ष"]`. There is a comment and a test.
- **Word splits runs mid-word.** `extract.py:_coalesce()` merges adjacent
  runs that convert with the same table; without it `जिला` came out `िजला`.
- **Two word-initial validator rules need `re.MULTILINE`.** Without it `^`
  anchors to the whole document and the most important corruption signal
  never fires.
- **Windows consoles are cp1252 by default.** `cli.py:_prepare_streams()`
  reconfigures stdout/stderr to UTF-8. Removing it makes every command crash
  on the first Hindi character.
- **The DOCX writer must ship `word/styles.xml` and
  `word/_rels/document.xml.rels`.** Word tolerates their absence;
  LibreOffice refuses the package, which surfaces as a failed PDF export that
  never mentions styles.
- **SQLite thread affinity.** `Store` opens with `check_same_thread=False`
  and takes a write lock, because the web app runs blocking work in threads.

## Hardware

Measured, not assumed (`PHASE0_FINDINGS.md`): i7-8550U, **8 GB
single-channel** DDR4-2400, spinning HDD at 11–27 MB/s. The AMD R7 M4xx GPU
**loses to the CPU** through DirectML — do not route generation to it.

Two cheap upgrades, in order: a second 8 GB SODIMM into the empty slot A
(~₹2,000; single-channel is costing roughly half the memory bandwidth, and
memory bandwidth is what token generation is limited by) and an SSD
(~₹2,500). Only after the RAM does `gemma3:4b-it-qat` become worth
benchmarking.

Model choice is by **Hindi words/second**, not tokens/second: Gemma 3's
tokenizer needs 1.97 tokens per Devanagari word against Qwen3's 6.03, which
is why Gemma 1B beats Qwen3 1.7B by roughly 5× on this hardware despite a
similar raw token rate.

## Unfinished

- Gold set (blocking a real conversion score).
- `num_thread` sweep from Phase 0 never reported.
- DirectML device-I/O rows never re-run after the IR-version fix.
- `gemma3:4b-it-qat` never benchmarked.
- `make_bundle.py`, `install.bat`, `start.bat`, `backup.bat` have **never
  run on Windows** — they were written on Linux and cannot be tested here.
  Treat the first install as a debugging session, not a deployment.
