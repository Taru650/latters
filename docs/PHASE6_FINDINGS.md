# Phase 6 findings — the web application

Two pages: drafting at `/`, corpus administration at `/admin`.

```bash
pip install -e ".[web]"
latters serve --db corpus.db --skeletons skeletons --model gemma3:1b
latters serve --db corpus.db --stub      # everything except the model
```

---

## 1. No framework, no build step, no CDN

FastAPI serves server-rendered HTML; about 260 lines of plain JavaScript
handle the two interactions that need it. There is no React, no bundler and
no `<script src="https://...">`.

That last one is not a preference. **The office machine has no internet**, so
a CDN tag is a page that silently fails to work there — and it fails in the
worst way, by rendering and then doing nothing when clicked. Vendoring htmx
would fix that and still cost more code than writing two fetch calls.

The rest follows from the Phase 0 measurements: no Node toolchain on the
target, and ~3 GB of free RAM that the model is already using.

A test asserts no template references an external URL.

---

## 2. The verification surface is part of the response, not a later feature

Phase 5 made a draft's provenance available: which letters it was built from
and their trust scores, what was stripped from the model's output, and every
number the model produced that appeared nowhere in the request or the
sources. A UI that renders the letter and hides those would undo the entire
design.

So they are fields in the `/api/draft` response schema, the drafting page
renders them above the letter, and a test asserts each key is present. The
banner is red whenever `needs_review` is true, and invented numbers are
listed individually.

---

## 3. Four bugs found by running it

None were visible by reading the code, and 306 unit tests passed while all
four were live.

### Starlette flipped `TemplateResponse`

`TemplateResponse(name, context)` became `TemplateResponse(request, name,
context)`. The old form makes the context dict arrive as the template name
and fails inside Jinja's cache with `TypeError: unhashable type: 'dict'` —
an error that names neither the template nor the call.

### SQLite connections are thread-bound

Conversion, segmentation and generation are slow and synchronous, so they
run in worker threads and an upload cannot freeze the page someone else is
drafting on. sqlite3 refuses cross-thread use by default, so the first draft
requested through a browser raised `ProgrammingError`.

`check_same_thread=False` is safe here — CPython links SQLite in serialized
mode, `sqlite3.threadsafety == 3` — and writes additionally take a lock,
because SQLite permits one writer and without it a concurrent upload and
correction produce "database is locked" instead of waiting.

### The DOCX writer shipped an invalid package

`[Content_Types].xml` declared an Override for `/word/styles.xml` and the
writer never wrote that part. Word tolerates the dangling declaration, which
is why it went unnoticed since Phase 4; **LibreOffice rejects the package
outright** with "source file could not be loaded", surfacing as a failed PDF
export that never mentions styles. `word/_rels/document.xml.rels` was
missing too.

### A corpus built through the browser had no labels

`latters classify --write` is a command-line step with no equivalent in the
browser, so an office that only ever used the upload page had a corpus where
every `department` was NULL — and without a department the drafting page
silently loses its retrieval filter (worth +0.10 precision, Phase 4) and its
skeleton. Labels are bootstrapped on ingest now, and re-checked whenever the
indexes rebuild.

---

## 4. Design decisions worth stating

**One generation at a time.** An `asyncio.Semaphore(1)` guards the model. Two
concurrent 2.5 GB generations on an 8 GB machine swap to a spinning disk and
hang it. The second request waits and is told it is waiting.

**Correcting a letter re-scores it.** The reason to correct a letter is
usually that its conversion was wrong, which means its trust, completeness,
form and violations were all computed from the wrong text. Saving re-runs the
validator, the segmenter and the field extractor.

**The admin list is lowest-trust-first.** Browsing a corpus alphabetically
shows you the letters that are fine. The ones worth a human's time are the
ones the pipeline is least sure about.

**Loopback by default.** There is no authentication and the corpus is
official correspondence. `--bind` on a non-loopback address prints a warning
saying so.

**Skeleton names are validated against path traversal.** The name arrives
from a URL, so `../../etc/passwd` has to be impossible rather than unlikely.

---

## 5. What is still not verified

- **PDF export could not be tested here.** LibreOffice in this environment
  cannot convert anything, including a plain text file, so the failure path
  is exercised and the success path is not. The DOCX export is fully tested
  and round-trips through this project's own parser. The error message now
  carries LibreOffice's own words and tells the user to export DOCX and
  print from Word.
- **No generated Hindi has been assessed** — still the open question from
  Phase 5, and now it has a UI in front of it.
- **The gold set still does not exist.** The web pages make the corpus easier
  to correct, which helps, but nothing here verifies that the converted Hindi
  was right in the first place.

333 tests pass.
