# latters — offline departmental letter drafting assistant

Fully offline. No cloud API, no telemetry, nothing leaves the machine.

Target hardware and the full phase plan: [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

**Status: Phase 0 measured, Phases 1-6 (conversion, segmentation,
classification, retrieval, generation, web app) implemented and validated against a real district archive** — see
[`docs/PHASE0_FINDINGS.md`](docs/PHASE0_FINDINGS.md) and
[`docs/PHASE2_FINDINGS.md`](docs/PHASE2_FINDINGS.md) and
[`docs/PHASE3_FINDINGS.md`](docs/PHASE3_FINDINGS.md) and
[`docs/PHASE4_FINDINGS.md`](docs/PHASE4_FINDINGS.md) and
[`docs/PHASE5_FINDINGS.md`](docs/PHASE5_FINDINGS.md) and
[`docs/PHASE6_FINDINGS.md`](docs/PHASE6_FINDINGS.md).

---

## Phase 0 — measure the machine before writing anything else

These run on the target laptop (Windows), not in CI. Every later decision
depends on their output.

```powershell
# 1. Hardware truth: memory channels, HDD vs SSD, GPUs, DX12 feature levels
powershell -ExecutionPolicy Bypass -File scripts\phase0_probe.ps1 > docs\probe.txt

# 2. Can ONNX Runtime DirectML actually reach both GPUs?
pip install onnxruntime-directml numpy onnx
python scripts\phase0_directml.py

# 3. Benchmark the candidate models and write docs\BASELINE.md
ollama pull gemma3:1b
ollama pull qwen3:1.7b
python scripts\phase0_bench.py --models gemma3:1b qwen3:1.7b --out docs\BASELINE.md
```

The two questions worth running these for:

- **Is RAM single-channel?** If `phase0_probe.ps1` reports one memory module,
  a second matched SODIMM is likely the largest speedup available — token
  generation is memory-bandwidth-bound, and one module halves the bandwidth.
- **Does the discrete GPU beat the CPU on batched matmul?** If yes, bulk
  embedding at ingest goes there (Phase 4). It says nothing about token
  generation, which is bandwidth-bound and where that card is *slower* than
  system RAM.

`phase0_bench.py` also reports **token fertility** — tokens per Devanagari
word. It is a direct multiplier on draft latency and on how many exemplars fit
in the context window, and it is the number most likely to overturn a model
choice made from benchmark tables.

---

## Phase 1 — legacy font corpus → Unicode

```bash
pip install -e .            # no runtime dependencies; stdlib only

latters fonts tables                          # what mapping tables exist
latters inventory /path/to/archive            # 1.1 triage: how much is legacy?
latters fonts gold                            # regression: is the converter right?
latters ingest /path/to/archive --repair --rescue-latin -o ./converted
```

### What is implemented

| Module | Does |
|---|---|
| `fonts/tables.py` | TSV mapping tables with `#inherit`, so DevLys overrides Kruti Dev instead of forking it |
| `fonts/convert.py` | Maximal-munch substitution + **chhoti-i reordering** + **reph reordering** + nukta normalisation |
| `extract.py` | DOCX parsing (stdlib zip+XML), **per-run font resolution**, mis-fonted-Latin detection |
| `validate.py` | Deterministic illegal-sequence detector → `conversion_confidence` |
| `repair.py` | Visarga-typed-as-colon and punctuation spacing |
| `gold.py` | Character-level regression harness against hand-verified pairs |
| `goldbuild.py` | Mines review sheets from the archive, coverage-greedy selection, blind control, slot-coverage report |
| `docx_writer.py` | Minimal stdlib DOCX writer (per-cell fonts) for the review sheet |
| `anchors.py` | Structural anchors (letter number, subject, closing, …) as tunable data |
| `segment.py` | One file → N letters; completeness and trust scoring |
| `store.py` | SQLite corpus + FTS5 index (tokenizer configured for Devanagari) |
| `fields.py` | Field extraction with a found / **blank** / absent distinction |
| `classify.py` | Label bootstrapping + char-ngram Naive Bayes + honest cross-validation |
| `template.py` | Per-cell skeleton mining — the reason a 1B model is viable |
| `retrieve.py` | BM25 + inverted-index TF-IDF + optional dense, fused with RRF |
| `evaluate.py` | Retrieval evaluation with standard errors and a random baseline |
| `encoders.py` | Optional dense encoders (not on the default path — see below) |
| `llm.py` | Ollama client, defaults traced to the Phase 0 measurements |
| `draft.py` | Prompt budgeting, output sanitising, skeleton assembly |
| `web/` | FastAPI drafting and admin pages — no framework, no CDN |
| `cli.py` | `inventory`, `ingest`, `fonts convert/gold/tables` |

### The three things that make this non-trivial

**1. Conversion is not a `str.replace` loop.** Kruti Dev stores `ि` *before*
its consonant cluster and the reph `र्` *after* it; Unicode does the opposite
for both. Worked example:

```
dhfrZ  →  क ी ि त REPH     substitution
       →  क ी त ि REPH     chhoti-i moves after its cluster
       →  क ी र् त ि       reph becomes र् before the cluster   =  कीर्ति
```

**2. Conversion must be per-run, not per-document.** A real letter has a Hindi
body in Kruti Dev and a letter number in Times New Roman *in the same
paragraph*. Almost every ASCII character is a valid Kruti Dev slot, so a
whole-document conversion turns `DEO/RPR/2024/1187` into Devanagari noise.
`extract.py` resolves the font per run (run → run style → paragraph style →
document default) and only converts runs in a legacy face.

**3. PDF is a different problem from DOCX.** In DOCX the character order is
already logical — the font is only a rendering instruction, so conversion is a
pure ordered substitution. In PDF the extractor sorts glyphs by x-position,
which scrambles matras: `और` typed `vkSj` extracts as `vkjS` and converts to
`आरै`. `read_document` refuses PDFs with that explanation rather than
producing plausible-looking corruption.

### What the validator does *not* do

It catches **illegal** Devanagari, not **wrong but legal** Devanagari. A table
that maps one valid consonant to a different valid consonant produces
well-formed text that means something else, and scores 1.0. Measured against a
corruption that deletes every `ि`, the score barely moves.

So it is a tripwire, not a proof. Correctness comes from the **gold set**, in
this order of strength: hand-verified pairs > corpus-derived vocabulary hit
rate > sequence rules.

### The gold set is the deliverable that matters

`tests/gold/seed_krutidev010.tsv` has 51 pairs passing at 100% character
accuracy. Two reasons that number means less than it looks:

- The pairs were derived from the mapping table's own logic, so they prove the
  **engine** works and say nothing about whether the **table** is right.
- `latters gold coverage` reports that they exercise **63 of 129 mapping slots
  (49%)**. Sixty-six slots — including `ढ`, `झ्`, `ष्`, `रु`, `रू`, the nukta
  and half the digits — are tested by nothing at all. A wrong mapping in any
  of them passes the regression silently.

#### Building the real one

```bash
# 1. Mine a review sheet from the archive
latters gold extract /path/to/archive -o review/ -n 200 --blind-fraction 0.2

# 2. A Hindi reader fills review/review.tsv (see below)

# 3. Turn it into a gold file
latters gold collect review/review.tsv -o tests/gold/office_raipur.tsv

# 4. Check it
latters gold run
latters gold coverage
```

`gold extract` writes two files:

| File | Role |
|---|---|
| `review.docx` | Two columns: the line in its **original legacy font** (the ground truth — the font must be installed to read it) beside our Unicode conversion. The reviewer compares visually. |
| `review.tsv` | The form, UTF-8-BOM so Excel opens Devanagari correctly. Per row, write `ok` or the corrected Hindi. |

Two things it does that a hand-built set would not:

**Coverage-greedy selection, not random sampling.** 200 random lines are mostly
the same boilerplate and leave rare conjuncts and ligature slots untested.
Selection repeatedly takes the line covering the most as-yet-uncovered mapping
slots, so the set is smaller *and* tests more. `gold extract` reports how many
of the slots your archive actually uses are covered.

**A blind control against rubber-stamping.** Showing the reviewer our
conversion makes the work fast but invites approving plausible-looking wrong
output. So a random 20% of rows show no suggestion in either file — the reader
transcribes those from the legacy rendering alone. `gold collect` then compares
disagreement rates:

```
disagreed with the converter:
  sighted  0/7 (0.0%)
  blind    1/3 (33.3%)

!! The blind rows were corrected far more often than the sighted ones.
!! That is the signature of rubber-stamping...
```

If the blind rows disagree far more often than the sighted ones, the sighted
rows were being approved rather than checked, and the set has to be redone.

Ship nothing below 98% character accuracy **and** high slot coverage. The two
are independent: accuracy says the tested slots are right, coverage says how
many were tested.

```bash
python -m pytest tests/ -q      # 44 tests
```

### Known limitations, stated plainly

- The Kruti Dev table is the common Remington 010 layout and **needs
  verification against your archive**. Its ligature slots are the least
  certain part; the DevLys override list is the least certain part of that.
- Mis-fonted-Latin rescue is a heuristic and is imperfect by construction:
  `F.No.` breaks at the lowercase `o`. It is off by default and always warns.
- `.doc`, `.rtf` and `.pdf` are refused with the remedy in the error message.
  Convert with `soffice --headless --convert-to docx` (which preserves
  run-level fonts; `antiword` and `catdoc` do not), or, for PDFs, a
  logical-stream extractor such as `krutiextract`.
- Digit mapping assumes Kruti Dev renders ASCII digits as Devanagari digits.
  Use `--latin-digits` if your archive was typed otherwise.

### Phase 0 outputs are tracked, not ignored

`docs/probe.txt` and `docs/BASELINE.md` are committed deliberately. They are
the evidence behind every hardware-dependent decision in the plan — model
tier, `num_thread`, whether the discrete GPU takes batched embedding, whether
RAM and an SSD get bought. Re-run and re-commit them after any hardware,
driver or Ollama change, so a later disagreement about "why did we pick this
model" is settled by a file rather than by memory.

To get them off the office machine and into the repo:

```
git add -f docs/probe.txt docs/BASELINE.md
git commit -m "Phase 0: baseline measurements from <machine>"
git push
```


---

## Phase 2 — one file → N letters

```bash
latters audit   /path/to/archive              # are the anchors right for THIS office?
latters segment /path/to/archive --db corpus.db
latters stats   --db corpus.db
latters search  "अनुशासनिक कार्यवाही" --db corpus.db
```

On a real five-file district archive: **589 letters**, mean trust 0.87,
42 exact duplicates deduped by content hash.

### Run `latters audit` before trusting anything

Segmentation keys off structural anchors — `पत्रांक`, `विषय:`, `प्रसंग:`,
`विश्वासभाजन`. Every office has house style, and the patterns shipped here
were written from general knowledge. Measured against the first real archive:

| Anchor | Assumed | Actually used | Count |
|---|---|---|---:|
| salutation | महोदय | **महाशय** | 356 (महोदय: **0**) |
| reference | संदर्भ | **प्रसंग** | 270 |
| closing | आपका विश्वासभाजन | **विश्वासभाजन**, bare | 382 |

Two anchors never fired. After tuning, letters found went 437 → 589 and the
`repeated-subject` *fallback* went from **45% of boundaries to 2%** — a
segmenter leaning on its safety net for half its decisions is guessing, not
working. `latters audit` reports that ratio and warns above 15%.

Extend the patterns without touching code:

```python
load_anchors({"closing": [r"इति\s+शुभम्"]})
```

### Two things the real archive taught us

**Word splits words across runs.** A real paragraph arrived as run `'f'`
(the pre-base i-matra, alone) then run `'tyk'`. Per-run conversion gave
`िजला`, not `जिला`. Runs are coalesced before conversion now — reordering is
a property of the text, not of the formatting runs it is stored in.

**FTS5's default tokenizer destroys Devanagari.** `unicode61` counts only
`L* N* Co` as token characters, so matras and virama are dropped as
separators: `समीक्षा` indexes as `["सम","ष"]` and a search for `की` matches
`कार्यवाही`. `remove_diacritics 0` does not help. The fix is
`categories 'L* N* Mn Mc Co'`, pinned by a test.

```bash
python -m pytest tests/ -q      # 128 tests
```


---

## Phase 3 — fields, labels, skeletons

```bash
latters fields    --db corpus.db
latters classify  --db corpus.db --write
latters templates --db corpus.db -o skeletons/ --show
```

### Blank is not absent

Only **35 of 547 real letters carry a dispatch number**; 498 read
`पत्रांक- --------/रा०, दिनांक--------`. The field is present and
deliberately empty. Every field therefore has three states — `FOUND`,
`BLANK`, `ABSENT` — because a template awaiting a number is usable and a
letter missing one is a defect, and collapsing them loses both.

### Classification: department yes, letter type no

| | accuracy | baseline | lift | macro F1 | verdict |
|---|---:|---:|---:|---:|---|
| department | 0.936 | 0.662 | +0.274 | **0.812** | apply with a confidence gate |
| letter type | 0.659 | 0.182 | +0.477 | 0.636 | **suggest only, user confirms** |

Labels are bootstrapped from branch codes (`पत्रांक-----/रा०` → राजस्व) and
office lines, so 95% of the corpus is labelled with no human effort.

No sklearn — scipy + sklearn is ~100 MB of wheels for a machine with 3 GB
free and no internet to install from, and char-ngram Naive Bayes is eighty
lines of stdlib. No LLM — classifying a closed label set is the one task
where a 1B model is strictly worse: slower, and not auditable.

**Every report prints the majority baseline.** 93.6% accuracy on a label set
that is 66% one class is not evidence of anything on its own.

### Why a 1B model can do this job

545 of 547 letters have a unique body, so there are no whole-letter templates.
But line-level boilerplate is enormous, and mining it per
(department, letter-type) cell gives:

| department | type | letters | skeleton covers |
|---|---|---:|---:|
| बैंकिंग | जाँच | 8 | **87%** |
| विकास | सामान्य पत्राचार | 24 | 62% |
| राजस्व | भूमि | 79 | 36% |

"Skeleton covers" is the share of the letter the model does **not** write —
letterhead, addressee block, closing and distribution list are copied, so
they are exactly right rather than approximately right. At 9 Hindi words per
second, 87% coverage is a 12-second draft instead of a 90-second one.

```bash
python -m pytest tests/ -q      # 173 tests
```


---

## Phase 4 — retrieval

```bash
latters retrieve "अनुशासनिक कार्यवाही हेतु कारण बताओ पत्र" --db corpus.db
latters eval --db corpus.db
```

### Only the department filter measurably helps

Same-cell precision@5 on paraphrased queries, 313 queries, **1 SE = 0.028**:

| configuration | cellP@5 | ms |
|---|---:|---:|
| random | 0.068 | 0.0 |
| bm25 | 0.357 | 1.1 |
| bm25 + dept filter | 0.458 | 1.0 |
| tfidf | 0.390 | 0.4 |
| **tfidf + dept filter** | **0.480** | **0.4** |
| rrf(bm25,tfidf) + dept | 0.489 | 1.9 |

The department filter is worth **+0.09 to +0.10** across every scorer — well
beyond two standard errors. The scorers are **not distinguishable from each
other**; bm25, tfidf and rrf all sit inside one another's error bars. Every
report prints the noise floor so nobody tunes on differences that aren't real.

Default is `tfidf + dept filter`: tied with RRF, 4× faster, one fewer moving
part.

### The dense encoder was not adopted

The plan assumed a neural encoder was needed. The evidence does not support
it — and the lexical methods lost only 0.06–0.08 when half the query's words
were dropped, so they do not collapse under the vocabulary mismatch a
semantic model is bought to fix.

It also **could not be tested here** (no network access to model weights), so
this is "not justified by available evidence", not "measured and rejected".
The plumbing ships with the decision rule fixed in advance:

```bash
python scripts/phase4_encoder_eval.py --db corpus.db --model ./granite-97m
```

> Adopt it only if it beats `tfidf + dept filter` on same-cell precision by
> more than two standard errors, in the degraded condition.

### A trap worth naming

Adding the letter-type filter made same-cell precision read **1.000** — a
very flattering number that is true by construction, since filtering to the
cell guarantees every hit is in the cell. The renderer now prints `n/a` with
the reason instead.

### Both metrics are proxies

Known-item recall (0.90–0.99) is inflated: the query is verbatim text from
the target. Same-cell precision uses the Phase 3 classifier's labels as
ground truth, which cross-validated at 0.81 / 0.64 macro-F1 — so its ceiling
is well under 1.0. The real evaluation, forty clerk-written requests with the
letters they'd have wanted, still does not exist.

```bash
python -m pytest tests/ -q      # 205 tests
```


---

## Phase 5 — drafting

```bash
latters draft "दाखिल-खारिज में विलंब की जाँच हेतु प्रतिवेदन मांगना है" \
    --db corpus.db --model gemma3:1b --skeletons skeletons/
latters draft "..." --db corpus.db --stub     # everything except the LLM
```

### The model writes the body. Nothing else.

The letterhead belongs to the office, the number is next in a series, the
date is today, the closing is fixed. Copying those is instant and exactly
right; generating them spends tokens at ~9 Hindi words/second on something
that might be wrong. The mined skeleton already supplies **87%** of a
बैंकिंग/जाँच letter.

### Everything the model writes is untrusted

| removed from the output | replaced with |
|---|---|
| `पत्रांक` / `ज्ञापांक` lines | next in series, or a blank slot |
| `दिनांक` lines | today |
| duplicate subject, closing block, code fences | the skeleton's own |

**A blank letter number beats an invented one** — 95% of the real archive
leaves it blank, so a blank is obviously unfinished and a wrong number looks
finished. Every removal is reported, never applied silently.

Invented numbers are flagged: any numeric token in the draft that appears
nowhere in the request or the source letters, with Devanagari and Latin
digits normalised. Over-sensitive on purpose — a false flag costs a glance, a
missed one costs a letter with an invented case number in it.

The Phase 1 illegal-sequence validator runs on the model's own output too,
because small models produce malformed Devanagari.

### Skeletons are mined, then fixed by a person

Mining cannot recover line order from a heterogeneous cell — an addressee's
designation has no anchor and sometimes lands on the wrong side of the
salutation. That is a five-minute edit, once per category:

```bash
latters templates --db corpus.db -o skeletons/   # mine
$EDITOR skeletons/राजस्व_जाँच.md                  # fix
latters draft "..." --skeletons skeletons/       # the edit wins, and persists
```

### What is NOT measured

**No generated Hindi has been assessed** — there is no model in this
repository's environment. The metric that decides the project is editing
effort: edit distance between the draft and the letter actually dispatched.

```bash
python scripts/phase5_draft_eval.py --db corpus.db --model gemma3:1b --pairs pairs.tsv
```

> < 0.15 essentially usable · 0.15–0.40 faster than starting blank ·
> 0.40–0.70 arguable · **> 0.70 slower than typing it**

Collect 30 pairs of (request, dispatched letter). Nothing else here tells you
whether the drafts are worth using.

```bash
python -m pytest tests/ -q      # 241 tests
```


---

## Phase 6 — the web pages

```bash
pip install -e ".[web]"
latters serve --db corpus.db --skeletons skeletons --model gemma3:1b
latters serve --db corpus.db --stub          # everything except the model
```

| page | does |
|---|---|
| `/` | describe the letter → draft → edit → export DOCX/PDF/TXT |
| `/admin` | upload, browse lowest-trust-first, correct in place, delete, edit skeletons |

**No framework, no build step, no CDN.** The office machine has no internet,
so a `<script src="https://...">` tag is a page that renders and then does
nothing when clicked. FastAPI serves HTML; ~260 lines of plain JavaScript do
the rest. A test asserts no template references an external URL.

**Every draft shows its provenance** — the letters it was built from with
trust scores, what was stripped from the model's output, and any number the
model invented. Those are fields in the response schema with tests asserting
they are present, because a UI that renders the letter and hides them would
undo the whole Phase 5 design.

### Four bugs found by running it, with 306 unit tests passing

| bug | symptom |
|---|---|
| Starlette flipped `TemplateResponse`'s signature | `TypeError: unhashable type: 'dict'` from inside Jinja |
| SQLite connections are thread-bound | first browser draft raised `ProgrammingError` |
| DOCX declared `styles.xml` and never shipped it | Word tolerated it since Phase 4; LibreOffice refuses the package |
| no labels in a browser-built corpus | `classify --write` has no web equivalent, so drafts lost their filter and skeleton |

Loopback by default: there is no authentication and the corpus is official
correspondence.

```bash
python -m pytest tests/ -q      # 333 tests
```
