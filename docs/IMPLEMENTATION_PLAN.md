# Offline Departmental Letter Drafting Assistant — Hardware-Constrained Implementation Plan

Target machine (from supplied screenshots):

| Component | Value | Consequence |
|---|---|---|
| CPU | Intel Core i7-8550U, 4C/8T, base 1.99 GHz, boost 3.93 GHz, 8 MB L3, 15 W cTDP | AVX2 only (no AVX-512, no AMX). Sustained clocks collapse under load. |
| RAM | 8.00 GB DDR4-2400 | **The binding constraint.** 4.9/7.9 GB already in use at idle → ~3.0 GB free. |
| Disk 0 (C:) | 1.82 TB **HDD (RAID)**, 56 GB used, sitting at **100 % active time** | Model load, mmap page-in, and any swapping are catastrophic here. |
| GPU 0 | Intel UHD 620 (iGPU) | Shares the same DDR4-2400 pool. No bandwidth advantage, some FP32 compute advantage. |
| GPU 1 | AMD Radeon R7 M4xx, 4 GB **DDR3**, 64-bit bus | ≈ **14.4 GB/s** bandwidth, 320 shaders, ≈ 650 GFLOPS FP32. GCN, ROCm-unsupported. |
| OS | Windows 11 | ROCm is not an option. DirectML and Vulkan are. |

---

## 0. The premise that has to be corrected first

**Splitting one LLM's layers across CPU and the 4 GB GPU will make generation slower, not faster.**

Token generation is *memory-bandwidth-bound*, not compute-bound: every decoded token reads the
entire active weight set once.

- System RAM: DDR4-2400 dual-channel ≈ **38.4 GB/s** (≈ 19.2 GB/s if only one SODIMM is fitted — verify this first, see Phase 0).
- R7 M4xx VRAM: DDR3, 64-bit, ~900 MHz ≈ **14.4 GB/s**.

The discrete GPU's memory is **~2.7× slower than the CPU's**. Any layer offloaded there decodes
slower than it would have on the CPU, and you additionally pay PCIe round-trips per token.
`--n-gpu-layers 20` on this laptop is a downgrade. This is the well-documented failure mode where
partial offload to a weak card underperforms CPU-only.

**So the correct reading of "distribute the load" is: distribute the *pipeline stages*, not the
model's layers.** Different stages of this system have genuinely different bottlenecks, and each
piece of silicon should get the stage it is actually good at:

| Stage | Bottleneck | Assign to | Rationale |
|---|---|---|---|
| Legacy-font conversion, segmentation | Single-thread string ops | **CPU, 4 worker processes** | Embarrassingly parallel across files |
| OCR (scanned pages only) | Conv/GEMM, batched | **GPU 1 (AMD) via DirectML** | Compute-bound, fits in 4 GB easily |
| **Bulk embedding at ingest** (10k–100k chunks) | GEMM, large batch | **GPU 1 (AMD) via ONNX Runtime DirectML** | ≈650 GFLOPS vs ≈190 GFLOPS AVX2 → real ~3× win; 97M encoder is ~200 MB in VRAM |
| Query-time embedding (batch = 1) | Dispatch latency | **CPU** | GPU kick-off cost exceeds the compute |
| Cross-encoder rerank (5–20 pairs) | Small GEMM | **GPU 0 (Intel UHD 620) via DirectML** | Keeps the AMD card and CPU free; overlaps with LLM prefill |
| BM25 / FTS5 lexical search | Disk + CPU | **CPU + SSD-cached index** | Must not touch the HDD hot path |
| LLM prefill (prompt processing) | GEMM, batched | **CPU** (optionally Vulkan — measure) | Only stage where GPU offload *might* pay; prove it with `llama-bench` |
| LLM decode (token generation) | RAM bandwidth | **CPU, 4 threads** | GPU is strictly worse here |
| DOCX/PDF export | CPU | **CPU** | Trivial |

This gives you real concurrency: while the LLM is decoding a draft on 4 CPU threads, the AMD card
can be re-embedding a freshly uploaded batch of archive files and the Intel iGPU can be reranking
for the next query. That is a genuinely distributed load — and none of it degrades the thing
users actually feel (time-to-first-token).

### The bigger constraint nobody mentioned

**Disk 0 is an HDD at 100 % active time.** On a spinning disk, Ollama mmap-ing a 2.5 GB model is
40–90 seconds of cold-start, and the first swap under memory pressure will make the app appear
hung. Before any of this software matters:

1. Fit a SATA SSD or NVMe (this chassis generation almost always has an M.2 slot) — **~₹2,500, and
   it is the single highest-ROI change available.**
2. Upgrade RAM 8 GB → 16 GB (~₹2,000). This moves you from "a 4 B model is marginal" to "a 4 B model
   is comfortable and you can keep it resident."

If the answer is "no hardware purchases allowed," the plan below still works, but you are locked to
the 1.7 B tier and you must pin the model in RAM permanently (`keep_alive: -1`) so the HDD is
touched exactly once per boot.

---

## 1. Model selection

### Verdict on your current choice

Qwen3 1.7B is a defensible *fit* but a poor *match*. Two specific problems:

1. **Tokenizer fertility on Devanagari.** Qwen3 uses a 151 k byte-level BPE vocabulary tuned for
   English/Chinese/code. Devanagari fragments badly on it. Gemma 3's ~262 k SentencePiece
   vocabulary was explicitly built for 140+ languages and measures materially lower tokens-per-word
   on Indic scripts. At ~5 tok/s, a 1.5× fertility penalty is a 1.5× penalty on *wall-clock draft
   time* and on how much retrieved context fits in your KV budget. This is the most
   underappreciated variable in Indic local-LLM work.
2. **Thinking mode.** Base `qwen3:1.7b` in Ollama emits `<think>` blocks by default. On this CPU
   that burns 20–60 seconds producing tokens the user never sees. If you stay on Qwen3, you must
   suppress it (`/no_think` or the `enable_thinking: false` template) — or switch to an
   instruct-only build.

### Recommended tiering

| Tier | Model | Ollama tag | Q4 size | Fits in 3 GB free? | Use |
|---|---|---|---|---|---|
| **Primary (recommended)** | Gemma 3 4B IT, Google QAT int4 | `gemma3:4b-it-qat` | ~2.6 GB | Only after RAM discipline (Phase 0) or a 16 GB upgrade | Best Hindi quality per byte on this box. QAT quants retain near-bf16 quality unlike naive Q4. |
| **Safe default today** | Gemma 3 1B IT | `gemma3:1b` | ~0.8 GB | Yes, easily | Same Indic-friendly tokenizer, fits with room to spare. Weaker reasoning — acceptable because the architecture below deliberately removes reasoning from the LLM's job. |
| **Your current** | Qwen3 1.7B | `qwen3:1.7b` | ~1.1 GB | Yes | Keep as the A/B baseline. Must disable thinking. |
| **Reach (16 GB RAM only)** | Qwen3 4B Instruct 2507 | `qwen3:4b-instruct` | ~2.5 GB | No, not at 8 GB | Strongest instruction-following of the 4 B class; non-thinking by design. |

**Decision rule:** do not pick from this table on my say-so. Phase 1 builds a 40-prompt Hindi
letter-drafting eval set from your own archive; pick the model that wins it at acceptable latency.
Benchmark claims about multilingual ability routinely fail to survive contact with real
departmental Hindi.

### Embedding model

**`ibm-granite/granite-embedding-97m-multilingual-r2`** — ONNX weights shipped, Apache 2.0,
scores 60.3 on Multilingual MTEB Retrieval, the best open multilingual retriever under 100 M
parameters, and ~+9 points over `multilingual-e5-small` (50.9) which is the usual default.
Hindi-specific retrieval ≈ 47.1. ~200 MB in VRAM as ONNX fp16 → loads onto the AMD card trivially
and leaves 3.8 GB spare.

Fallback if Granite misbehaves on your text: `intfloat/multilingual-e5-small` (384-dim, very light).
If you have RAM headroom after the SSD/RAM upgrade, `granite-embedding-311m-multilingual-r2`
(Hindi 51.8) is a meaningful step up and still runs on the AMD card.

### Reranker

`BAAI/bge-reranker-v2-m3` is too heavy (568 M) for this box at query time. Use
`jinaai/jina-reranker-v2-base-multilingual` (278 M) as ONNX int8 on the Intel iGPU, or skip
reranking entirely in v1 — with only 3–5 results needed from a corpus of a few thousand letters,
hybrid BM25+dense with RRF is usually sufficient. **Add the reranker only if Phase 4 evaluation
shows it is needed.**

---

## 2. Architectural principle: the LLM is the smallest part of the system

Given ~5 tok/s, every LLM call you can delete is a win. Push work down the stack:

- **Department / letter-type classification** → *not* the LLM. TF-IDF char-ngrams + logistic
  regression, or centroid-kNN over the Granite embeddings. Trains in seconds, runs in
  milliseconds, is auditable, and beats a 1.7 B model on a closed label set.
- **Letter boundary segmentation** → *not* the LLM. Rule-based scoring over structural anchors
  (see Phase 2). Government letters are extraordinarily formulaic; this is the single biggest
  reason the LLM does not need to see it.
- **Conversion trust scoring** → *not* the LLM. Deterministic linguistic validators.
- **Field extraction** (letter number, date, subject, addressee, signatory) → regex-first, LLM only
  as fallback on the ~10 % that fail.
- **Generation** → the *only* LLM call. One call per draft.

This is what makes 5 tok/s livable: the user waits once, for one thing, and sees streamed output.

---

## 3. Phase-wise implementation

### Phase 0 — Hardware truth-finding and baseline (1 day)

Do not write application code until these numbers exist. Every later decision depends on them.

1. **Verify memory channel configuration.** Task Manager → Performance → Memory → "Slots used".
   If it reads `1 of 2`, you are on single-channel ≈19.2 GB/s and *adding a second matched 8 GB
   SODIMM roughly doubles LLM decode speed* — a bigger win than any software change in this document.
2. **Reclaim RAM.** Target ≥ 5.0 GB free before launching the assistant. Disable startup apps,
   set browser to run with ≤ 2 tabs during drafting. Record the achievable floor.
3. **Baseline the models.** For each candidate, on a fixed 800-token Hindi prompt:
   ```
   ollama run gemma3:1b     --verbose   # record prompt eval t/s, eval t/s, load duration
   ollama run qwen3:1.7b    --verbose
   ollama run gemma3:4b-it-qat --verbose
   ```
   Record: cold-load seconds (HDD!), prefill tok/s, decode tok/s, peak RSS.
4. **Settle the Vulkan question empirically.** Ollama 0.12.6+ ships experimental Vulkan, which is
   exactly the escape hatch for ROCm-unsupported AMD cards. Install the Vulkan backend, then
   compare `OLLAMA_VULKAN=1` with `num_gpu` at 0, 8, and 999 against CPU-only.
   **Expected result: CPU-only wins on decode; Vulkan may win on prefill.** If prefill improves
   ≥30 % and decode does not regress, keep it. If not, disable Vulkan for Ollama and reserve both
   GPUs for the ONNX/DirectML stages, which is where they are unambiguously useful.
5. **Confirm DirectML reaches both GPUs.**
   ```python
   import onnxruntime as ort
   print(ort.get_available_providers())   # expect DmlExecutionProvider
   # device_id 0 and 1 select between UHD 620 and R7 M4xx
   ```
   DirectML needs only a DirectX 12 device — no ROCm, no CUDA — which is precisely why it is the
   right GPU path on this machine. GCN-era R7 cards are DX12 FL 11_1, so this should work; if the
   AMD driver is too old, update to the last legacy Adrenalin release for the M4xx series.

**Deliverable:** `docs/BASELINE.md` with a filled-in numbers table. This becomes the regression
baseline for the rest of the project.

### Phase 1 — Corpus conversion: legacy fonts → Unicode (1 week)

This is the phase that decides whether the project works at all. Budget accordingly.

**1.1 Inventory and triage.** Walk the archive; for each file record extension, and for
`.doc`/`.docx`/`.rtf` the set of font names actually used in runs. Bucket into:
`unicode` / `krutidev` / `devlys` / `other-legacy` / `scanned-image` / `english`.
A font-name histogram tells you in an hour how big the problem really is.

**1.2 Understand the two very different conversion problems.**

- **DOC/DOCX/RTF:** the character *order in the file is already logical* (it is what the typist
  keyed). The font is only a rendering instruction. Conversion here is a **pure ordered
  byte-substitution** using the font's mapping table. High accuracy is achievable.
- **PDF:** the extractor sorts glyphs by *x-position*, which scrambles matras. `और` typed as
  `vkSj` extracts as `vkjS` and converts to the wrong word `आरै`. **Any PDF pipeline that ignores
  this will silently corrupt a large fraction of your corpus while looking fine.**

**1.3 Tooling.** Evaluate `krutiextract` (PyPI) first — it exists specifically to solve 1.2 by
reading the logical character stream rather than visual order, and is built on SIL's open
KrutiDev 011 mapping table. If it does not cover your DevLys variants, implement your own converter:

- Encode the mapping as an **ordered list of (legacy, unicode) pairs, longest-match-first** —
  conjuncts and half-forms must be substituted before single characters, or you will mangle them.
- Handle the three reordering rules explicitly:
  - **chhoti-i matra (ि)** is stored *before* its consonant cluster in KrutiDev, must move *after* in Unicode.
  - **reph (र्)** is stored as a trailing glyph, must move to the start of the cluster with a virama.
  - **Split matras** (e.g. ो, ौ) are stored as two glyphs flanking the consonant.
- Keep the table in a data file (`data/fonts/krutidev010.tsv`), never in code. Different offices
  use different variants; the admin UI must allow selecting the variant per upload.

**1.4 Validation is mandatory, not optional.** Build a 200-line hand-verified gold set: real
sentences from your archive, legacy source paired with correct Unicode. Compute character-level
accuracy on every converter change. Refuse to ship below 98 %.

**1.5 Scanned pages.** Tesseract `hin` traineddata, or PaddleOCR/Surya on DirectML. Mark OCR'd
letters at a permanently lower trust tier — they are useful for retrieval, never for verbatim quotation.

**Deliverable:** CLI `latters ingest --path <dir>` producing raw Unicode text + per-file provenance.

### Phase 2 — Segmentation: one file → N letters (1 week)

**2.1 Anchor-based rule segmenter.** Departmental Hindi letters carry strong, reliable anchors:

| Anchor class | Example patterns |
|---|---|
| Letter number | `पत्र संख्या`, `क्रमांक`, `सं॰`, `F.No.`, `फा.सं.` |
| Header/seal block | office name lines, `कार्यालय`, `भारत सरकार`, `राज्य सरकार` |
| Addressee | `प्रति,`, `सेवा में,`, `To,` |
| Subject | `विषय:`, `विषय -`, `Sub:` |
| Reference | `संदर्भ:`, `कृपया संदर्भ लें`, `Ref:` |
| Closing | `भवदीय`, `आपका विश्वासभाजन`, `हस्ताक्षर`, `(हस्ता॰)` |
| Distribution | `प्रतिलिपि:`, `प्रतिलिपि सूचनार्थ`, `Copy to:` |
| Date | Devanagari and Latin numeral date forms |

Algorithm: score each line for anchor membership → a candidate boundary is a **letter-number or
header anchor that follows a closing/distribution anchor**. Enforce sanity constraints (min/max
letter length, monotonically plausible dates, exactly one subject per segment).

**2.2 Completeness score per segment (0–1).** Weighted presence of: letter number, date, addressee,
subject, body ≥ N chars, closing, signatory. A segment scoring < 0.5 is quarantined for admin review,
not silently indexed.

**2.3 Combined trust score.** `trust = w1·conversion_confidence + w2·completeness + w3·source_tier`, where
`conversion_confidence` comes from deterministic validators:
- ratio of characters in the Devanagari Unicode block,
- hit rate against a Hindi wordlist / the corpus's own high-frequency vocabulary,
- **illegal-sequence count**: matra in word-initial position, two consecutive matras, virama at
  word end, orphaned nukta. These are near-perfect corruption detectors and cost nothing.

`source_tier`: native Unicode 1.0 > converted DOCX 0.85 > converted PDF 0.6 > OCR 0.4.

**Retrieval must filter on trust, and the UI must display it.** A retrieved letter that shows
"trust 0.42 — OCR, unverified" is honest; one that doesn't is how the office ends up sending out a
letter with a garbled name in it.

**Deliverable:** SQLite `letters` table with `id, source_file, span, text, trust, conv_conf, completeness, tier`.

### Phase 3 — Structure extraction and classification (4–5 days)

**3.1 Field extraction.** Regex-first for letter_number, date, subject, addressee, signatory,
designation, copy-to list. Measure coverage; route only failures to the LLM with a strict JSON
schema and a hard `num_predict` cap. Expect regex to handle 85–90 %.

**3.2 Department + letter-type classification.**
- Bootstrap labels from directory structure and letter-number prefixes (departmental letter numbers
  are usually department-encoded — free labels).
- Train TF-IDF(char 3–5 grams) + LogisticRegression on subject + first 300 chars of body.
- Evaluate with stratified 5-fold. Anything under ~0.85 macro-F1 means your label taxonomy is wrong,
  not your model.
- Persist the label taxonomy in the DB, editable from the admin page.

**3.3 Template mining.** For each (department, letter_type) cell with ≥ 10 examples, diff the
letters to separate **invariant skeleton** from **variable slots**. This is the highest-leverage
artifact in the whole project: with a mined skeleton, generation becomes slot-filling, which a
1 B model does reliably and fast, instead of free composition, which it does not.

### Phase 4 — Retrieval (4–5 days)

**4.1 Chunking.** Do *not* chunk letters into fragments. **A letter is the retrieval unit** — the
point is stylistic imitation, and style lives in the whole document. Embed the subject line and a
truncated body separately, store both vectors, and score `0.4·subject_sim + 0.6·body_sim`.

**4.2 Index.**
- Dense: embeddings in a single `.npy` memmap + brute-force cosine. **At < 50 k letters, do not
  install FAISS.** NumPy brute force over 50 k × 384 floats is ~20 ms and removes a dependency,
  a build step, and an index-corruption failure mode. Move to `faiss-cpu` IVF only if you cross
  ~200 k.
- Lexical: **SQLite FTS5** with a Unicode-aware tokenizer. Same file as the metadata DB, zero extra
  processes, survives being copied on a USB stick to the next office.
- Fusion: **Reciprocal Rank Fusion** (`1/(60+rank)`) over the two lists. RRF needs no score
  calibration, which matters because dense and BM25 scores are not comparable.

**4.3 Hard filters before fusion:** department, letter_type, `trust >= threshold`, and optionally a
recency window. In a departmental corpus, metadata filters do more for precision than any
embedding upgrade.

**4.4 Ingest-time GPU path.** Batch-embed on the AMD card via DirectML, batch size 32, fp16, with a
CPU fallback that is automatically used if DirectML init fails. Show progress in the admin UI —
first full ingest of a 5 000-letter archive will take 10–30 minutes and users must not think it hung.

**4.5 Build the eval set now.** 40 real requests → the letters a human says should be retrieved.
Measure Recall@5 and MRR. **Tune retrieval against this, not against how the drafts feel.**

### Phase 5 — Generation (1 week)

**5.1 Prompt construction — the whole game at 5 tok/s is token economy.**
- Send **2–3** exemplars, not 5. Each Hindi letter is 400–900 tokens; five of them blows past a
  4 096 context and forces prefill you cannot afford.
- Send the **mined skeleton** plus **2 exemplars**, not 5 raw letters. Skeleton + slots is far more
  token-efficient than repeated near-duplicates.
- Truncate exemplars: full header/subject/closing (that is the style), body capped at ~250 tokens.
- System prompt in **English**, output instruction in Hindi. English system prompts tokenize ~3×
  cheaper and small models follow them at least as well.

**5.2 Ollama options — these matter more than the model choice:**
```json
{
  "num_ctx": 4096,
  "num_thread": 4,
  "num_batch": 256,
  "num_predict": 900,
  "temperature": 0.3,
  "repeat_penalty": 1.05,
  "keep_alive": -1
}
```
- `num_thread: 4`, **not 8** — on a memory-bandwidth-bound decode, hyperthread siblings contend for
  the same L3 and memory controller and typically cost 5–15 %. Measure both.
- `keep_alive: -1` pins the model so the HDD is hit once per boot, not once per draft. This is the
  single biggest perceived-latency fix on this machine.
- Environment: `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_FLASH_ATTENTION=1`,
  `OLLAMA_KV_CACHE_TYPE=q8_0` (halves KV memory at negligible quality cost — directly buys context).

**5.3 Stream always.** Server-sent events to the browser. At 5 tok/s, streaming is the difference
between "slow but working" and "broken".

**5.4 Post-generation deterministic repair.** Do not ask the model to get the boilerplate right —
fix it in code: today's date, correct letter-number series (next in sequence from the DB), correct
office header, correct signatory block, `प्रतिलिपि` list. These are database lookups, not
generation problems, and every one you hand to the LLM is tokens spent to produce something
that might be wrong.

**5.5 Hallucination containment.** Show the 3–5 source letters alongside the draft with similarity
and trust scores. **Frame the product as "drafting assistant, output must be verified"** — in a
government context this framing is not a disclaimer, it is the thing that makes adoption possible.

### Phase 6 — Application (1 week)

**Stack:** FastAPI + SQLite + Jinja2 templates + htmx. Not React.
Reasons specific to this machine and this deployment: no Node toolchain, no build step, no
bundle, the whole app is `python -m latters` against a single `.db` file, and the ~400 MB of RAM
a dev server and browser bundle would consume is RAM the model needs. htmx handles the
progressive-render and streaming needs completely.

**Drafting page:** request textarea → detected department/type (editable dropdowns) → retrieved
exemplars panel with trust badges → streamed draft → rich-text editor → export.

**Admin page:** upload with font-variant selector, ingest progress, quarantine queue for
low-trust segments with a side-by-side legacy/Unicode corrector, search/browse/delete, label
taxonomy editor, re-embed trigger, corpus health dashboard (counts by trust tier, by department,
conversion accuracy trend).

**Export:** `python-docx` from a DOCX template carrying the office letterhead and the correct
Devanagari font; PDF via LibreOffice headless `--convert-to pdf` (bundled, offline). Do not
generate PDFs with ReportLab — Devanagari shaping will bite you.

**Concurrency:** one `asyncio` event loop; LLM calls, DirectML embedding, and ingest run in
separate process pools with an explicit **global semaphore of 1 on LLM generation**. Two concurrent
generations on 8 GB will swap to an HDD and hang the machine. Queue the second request and say so.

### Phase 7 — Evaluation, packaging, handover (4–5 days)

- **Three scorecards, tracked separately:** conversion accuracy (char-level vs gold set), retrieval
  (Recall@5 / MRR vs the 40-query set), drafting (human 1–5 on format fidelity, factual
  correctness, editing effort, over 30 real requests).
- **The metric that decides success is editing effort**, measured as edit distance between draft
  and the letter actually dispatched. Track it in production; it is the number that justifies the
  project to the office.
- **Packaging:** PyInstaller or a folder-install with an embedded Python, an `install.bat` that
  installs Ollama and pulls the model from a local file (`ollama create` from a bundled GGUF — the
  target office has no internet), and a `start.bat`. Ship the model GGUF on the USB stick.
- **Backup:** the entire state is one `.db` file plus one `.npy` file. Daily copy to a second drive.
  Document the restore in one page.
- **Handover doc in Hindi** for office staff; English technical doc for whoever maintains it.

---

## 4. RAM budget (8 GB, must be enforced, not hoped for)

| Consumer | Budget |
|---|---|
| Windows 11 + baseline services | 2.6 GB |
| Ollama server + Gemma 3 1B Q4 resident | 1.1 GB |
| KV cache @ 4096 ctx, q8_0 | 0.15 GB |
| Python app (FastAPI, SQLite, NumPy) | 0.5 GB |
| Embedding index memmap (50 k × 384 fp32) | 0.08 GB (paged) |
| ONNX Runtime + Granite 97M (CPU path) | 0.35 GB |
| Browser, 2 tabs | 0.8 GB |
| **Headroom** | **~2.4 GB** |

Swap the 1 B for the 4 B QAT model and headroom drops to ~0.9 GB — survivable only with browser
discipline and nothing else running. **This is exactly why the 16 GB upgrade is recommended before
committing to the 4 B tier.**

---

## 5. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| PDF glyph-reordering silently corrupts corpus | **High** | Gold-set validation in Phase 1; logical-stream extraction (`krutiextract`); illegal-sequence detector as a tripwire |
| Font variant in the archive is not KrutiDev 010/011 | Medium | Font-name histogram in Phase 1.1; pluggable per-variant mapping tables; admin-selectable variant per upload |
| HDD makes the app feel broken | **High** | SSD (strongly recommended) or `keep_alive: -1` + a visible warm-up screen on first launch |
| Model too weak for coherent departmental Hindi | Medium | Skeleton + slot-fill architecture removes most of the composition burden; escalate to Gemma 3 4B QAT after the RAM upgrade |
| Segmentation misses boundaries → two letters merged | Medium | Completeness score quarantines them; admin split tool in the UI |
| Staff trust a wrong draft | Medium | **Impact is high** — mandatory source-panel display, trust badges, product framing as assistant-not-author, deterministic repair of all factual boilerplate |
| Office has no internet for setup | Certain | Fully offline installer with bundled GGUF and ONNX weights |

---

## 6. Suggested order of attack

If you take one thing from this plan: **Phases 1 and 2 are the project.** Conversion and
segmentation are where the difficulty, the risk, and the differentiation actually live. Retrieval
and generation are, by 2026, close to solved commodity work once the corpus is clean.

A common failure mode on projects shaped like this one is to build the chat UI and the RAG loop
first, because those are visible and satisfying, and to discover in month three that 40 % of the
corpus is silently garbled. Build the converter, the validator, and the gold set first. The draft
quality ceiling is set entirely by corpus quality, and no model upgrade will raise it.

---

## Sources

- [Ollama experimental Vulkan support (Phoronix)](https://www.phoronix.com/news/ollama-Experimental-Vulkan)
- [Ollama hardware support docs](https://docs.ollama.com/gpu)
- [Add Vulkan GPU Backend for AMD/Intel Support — ollama#11247](https://github.com/ollama/ollama/issues/11247)
- [ROCm vs Vulkan for AMD Local LLM Hosting (2026)](https://dev.to/rosgluk/rocm-vs-vulkan-for-amd-local-llm-hosting-2026-guide-5c70)
- [AMD Radeon R7 M440 (4GB DDR3) specs — LaptopMedia](https://laptopmedia.com/video-card/amd-radeon-r7-m440-4gb-ddr3/)
- [Radeon R7 M440 — Notebookcheck](https://www.notebookcheck.net/AMD-Radeon-R7-M440-Benchmarks-and-Specs.169455.0.html)
- [llama.cpp: AMD GPU slower than CPU — issue #3422](https://github.com/ggml-org/llama.cpp/issues/3422)
- [llama.cpp Vulkan performance discussion #10879](https://github.com/ggml-org/llama.cpp/discussions/10879)
- [ONNX Runtime DirectML execution provider](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
- [Microsoft DirectML](https://github.com/microsoft/DirectML)
- [Granite Embedding Multilingual R2 — IBM](https://huggingface.co/blog/ibm-granite/granite-embedding-multilingual-r2)
- [granite-embedding-97m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2)
- [Granite Embedding Multilingual R2 paper](https://arxiv.org/html/2605.13521v2)
- [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
- [Multilingual E5 Text Embeddings: A Technical Report](https://arxiv.org/pdf/2402.05672)
- [krutiextract — PyPI](https://pypi.org/project/krutiextract/1.6.1/)
- [Python class to convert Krutidev to Unicode (gist)](https://gist.github.com/trinopoty/b83b136ff25cdfda0cbe774dc17e43ca)
- [Python3 Unicode ⇄ KrutiDev converter (gist)](https://gist.github.com/tripleee/b82a79f5b3e57dc6a487ae45077cdbd3)
- [Kruti Dev — Wikipedia](https://en.wikipedia.org/wiki/Kruti_Dev)
- [Tokenization is Killing our Multilingual LLM Dream](https://huggingface.co/blog/omarkamali/tokenization)
- [Understanding Token Fertility for Multilingual LLMs](https://medium.com/@sauryaaa/understanding-token-fertility-why-it-matters-for-multilingual-llms-3b39bb9b2fdf)
- [MorphTok: Morphologically Grounded Tokenization for Indian Languages](https://arxiv.org/pdf/2504.10335)
- [Qwen3-4B-Instruct-2507 GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF)
- [Best Open Source LLMs to Run Locally in 2026](https://huggingface.co/blog/daya-shankar/open-source-llm-models-to-run-locally)
- [local-rag-lab: FAISS + SQLite FTS5 + RRF](https://github.com/slavamirgit/local-rag-lab)
- [RagLLM: hybrid FAISS + BM25 + RRF with Ollama](https://github.com/M-Taghizadeh/RagLLM)
- [ollama-python: zero-cloud local hybrid RAG with SQLite FTS5 (PR #721)](https://github.com/ollama/ollama-python/pull/721)
