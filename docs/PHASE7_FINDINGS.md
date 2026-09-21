# Phase 7 — evaluation, packaging, handover

The plan's Phase 7 has five deliverables. Four are built and tested. The
fifth — the Windows install bundle — is written and has **never run on
Windows**, and this document says so rather than implying otherwise.

The theme of the phase turned out to be one idea: *a measurement that
quietly reports success when it has no data is worse than no measurement.*
Two of the three findings below are about that, and one of them is a flaw in
code written earlier in this same phase.

---

## 1. The success metric now measures itself

The plan says: "the metric that decides success is editing effort, measured
as edit distance between draft and the letter actually dispatched", over
30 real requests.

**The 30-pair study was the wrong shape.** Its failure mode is not that the
number comes out badly — it is that nobody ever collects the 30 pairs, so
the number never exists, and the project ships on the strength of a demo.
Every draft that leaves through the export button is already one half of a
pair and the exported text is the other half.

So: `drafts` table (schema v3), `record_draft` on every generation,
`record_dispatch` on every export, `effort = levenshtein(draft, dispatched)
/ len(dispatched)`. Verified over real HTTP against a live server:
draft → edit → export → the score appears on the scorecard.

Four decisions in `effort.py` worth knowing:

- **Median, not mean.** One pasted-over draft scoring 4.0 drags a mean above
  the 0.70 failure line while four letters out of five went out untouched.
  The test asserts exactly that data set.
- **Normalise by the *dispatched* length**, not the longer of the two. A
  draft that was mostly deleted must score high; dividing by the longer text
  would flatter it.
- **Three things are folded out before measuring**, because they cost the
  writer no keystrokes: Unicode normal form (the browser submits NFC, the
  converter emits decomposed conjuncts — identical Hindi would otherwise
  score as a full rewrite of every affected word), whitespace runs, and
  trailing spaces per line (invisible, and Word adds them).
- **Abandoned drafts are counted.** A table holding only exported drafts
  would report the tool as flawless while people quietly stopped using it.
  `abandoned` sits next to the median on the card for exactly that reason.

Levenshtein and not a semantic similarity: keystrokes are the cost, and
keystrokes are edit distance. We want the pessimistic measure.

---

## 2. The scorecard's first version lied, and it was my own code

`latters scorecard` printed:

```
CONVERSION
    gold pairs                       51
    exact-line matches               51/51 (100.0%)
    character accuracy               1.0000
  verdict: usable
```

Every word of that is true and the conclusion is worthless. Those 51 pairs
are `data/gold/seed_krutidev010.tsv` — **written by the same person who
wrote the mapping table, from the same assumptions.** They prove the
converter still does what it did last week. They say nothing about whether
the Hindi is right, and the office's archive is 87.6% DevLys 040 / Kruti Dev
041 while the seed set is Kruti Dev 010.

A module whose docstring says *"a missing scorecard is printed as loudly as
a failing one"* had shipped a green card over an empty measurement, two
hundred lines below that sentence.

Fixed: seed pairs and office pairs are counted separately, the seed row is
still shown, and **only a `gold/office_*.tsv` can clear the card.** There is
a test named after the mistake.

`latters scorecard` exits non-zero while any card is unmeasured or failing,
so it can gate a release rather than decorate one. On the current state it
exits 1 and ends with:

```
3 of 3 scorecards have no data: conversion, retrieval, drafting.
This system has not been evaluated. Do not present it as though it has.
```

---

## 3. Backup is not a file copy

`copy corpus.db backups\` is what an office will do, and on a WAL database
being written by the server it produces a file missing the most recent
transactions **with no error anywhere**. That is the worst possible failure
shape for a backup: it looks like it worked.

`latters backup` uses SQLite's own backup API, which takes a consistent
snapshot of a live database, then reopens the result and runs
`PRAGMA integrity_check` — a backup nobody has opened is a hope. Tested
against a deliberately still-open `Store`.

`--keep N` prunes older copies. `backup.bat` also mirrors `skeletons/`,
which is the one piece of state nobody can reconstruct from the archive
because it holds corrections a person made by hand.

---

## 4. Packaging — written, not verified

`packaging/` holds `make_bundle.py` (run on a machine with internet),
`install.bat`, `start.bat`, `backup.bat`.

Three things in there are deliberate and would be easy to lose:

- `pip install --no-index --find-links` — without `--no-index` pip reaches
  for PyPI, hangs on a machine with no route out, and eventually fails with
  a timeout that reads like a broken package.
- `pip download --platform win_amd64` in the bundler — a Linux build machine
  otherwise silently collects Linux wheels that fail at the office, with no
  internet to fix them.
- `chcp 65001` at the top of every `.bat` — every message contains a path
  that may hold Hindi, and the default console codepage mangles it. This is
  the same class of bug as the cp1252 crash that `_prepare_streams()` fixes
  in the CLI.

`start.bat` lives in the **data** folder, not the install folder, so `%~dp0`
is the corpus directory. The eight errors in this project's own setup notes
were all one mistake: running the commands from the wrong directory.

**None of this has run on Windows.** It was written on Linux and cannot be
tested here. Treat the first install as a debugging session.

---

## 5. Handover

`docs/HANDOVER_HI.md` — Hindi, for the office. Opens with the warning that
the tool produces a draft and not a letter, and that invented numbers look
exactly like real ones. Ends with a section on what the software explicitly
does *not* do, so nobody is misled into thinking it knows the rules.

`docs/MAINTENANCE.md` — English, for whoever inherits the code. Contains the
list of things that will bite you (FTS5's tokenizer, Word's mid-word run
splits, the two validator rules that need `re.MULTILINE`, cp1252, the DOCX
parts LibreOffice requires, SQLite thread affinity) and an explicit "what is
actually known" section separating measured results from architecture.

---

## Where the project actually stands

**Built and verified:** conversion, segmentation, field extraction,
classification, skeleton mining, retrieval, drafting, the web app, DOCX and
PDF export, the effort metric, the scorecards, backup.

**Never verified, and material:**

1. **No gold set.** The converted Hindi has never been read by a Hindi
   reader. 87.6% of the archive is converted by tables that inherit Kruti
   Dev 010 with zero overrides — not proven equivalent, just unexamined.
   This is the largest untested risk in the project and nothing in Phase 7
   reduced it.
2. **No generated Hindi has ever been assessed.** No model has run in any
   environment where this was developed. Every claim about draft quality is
   architectural.
3. **Editing effort has no samples.** The mechanism works; the office has to
   use the tool for the number to exist.
4. **The installer has never run on Windows.**

371 tests pass. That is a statement about the code, not about the Hindi.
