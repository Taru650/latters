# latters — offline departmental letter drafting assistant

Fully offline. No cloud API, no telemetry, nothing leaves the machine.

Target hardware and the full phase plan: [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

**Status: Phase 0 tooling ready to run; Phase 1 (conversion) implemented and tested.**

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

`tests/gold/seed_krutidev010.tsv` has 51 pairs, and they pass at 100%
character accuracy — but they were derived from the mapping table's own logic,
so they prove the **engine** works and say nothing about whether the **table**
is right for your archive.

Before Phase 2, build the real one:

1. Open a real letter in Word with the legacy font installed.
2. Copy the raw ASCII (switch the font to Courier to see it) → `legacy` column.
3. A Hindi reader types what the page actually says → `expected` column.
4. 200 lines, spanning every department and decade, weighted toward conjuncts,
   reph, chhoti-i, numerals and file numbers.
5. Save as `tests/gold/office_<name>.tsv`; `latters fonts gold` picks it up.

Ship nothing below 98% character accuracy.

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
