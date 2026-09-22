# Running this on your laptop

For the Windows machine measured in Phase 0: i7-8550U, 8 GB single-channel,
spinning disk, Windows 11. Commands are PowerShell.

There are two ways to use it: the command line, and a local web page for
office staff. The web page needs the optional extras:

```powershell
pip install -e ".[dev,web]"
latters serve --db corpus.db --skeletons skeletons --model gemma3:1b
```

### Step 8b — if anything at all goes wrong, run this first

```powershell
latters doctor
```

It checks every external thing the application needs — Python, free disk and
RAM, the corpus, the skeletons, Ollama, the model, poppler, Tesseract and its
Hindi pack, LibreOffice, and a Devanagari font — and prints the exact command
that fixes each failure. It exits non-zero only for problems that actually
stop the application; a missing Tesseract is a warning, because a `.docx`
archive does not need one.

It runs without a corpus on purpose. A missing `corpus.db` is the most likely
thing to be wrong on a fresh install, and refusing to run would withhold the
diagnosis exactly when it is needed. `start.bat` runs it automatically if the
server exits.

### Step 9 — check what has and has not been measured

```powershell
latters scorecard --db corpus.db
latters backup --db corpus.db -o backups --keep 14
```

`scorecard` exits non-zero while any of the three cards has no data, so it
can gate a release instead of decorating one. Expect it to exit 1 today:
the gold set does not exist and nobody has dispatched a draft yet.

`backup` is not a file copy -- the database runs in WAL mode and copying it
while the server writes silently loses the newest transactions. Copy the
`backups\` folder to a second drive afterwards; a backup on the same
spinning disk protects against a mistake, not against the disk.

Then open <http://127.0.0.1:8765/>. It binds to loopback only — **there is no
authentication and the corpus is official correspondence**, so do not put it
on a network without thinking about that first.

Use `--stub` to run the pages with no language model at all: retrieval, the
skeleton, the safety checks and export all work, and only the body text is
canned. It is the fastest way to show someone what the tool does.

---

## Step 0 — the two parts to buy (before anything else)

Measured, not guessed:

| part | why | cost |
|---|---|---|
| 1 × 8 GB DDR4-2400 SODIMM, slot A | RAM is **single-channel** (confirmed: one module in DIMM B). Doubles memory bandwidth — token generation is bandwidth-bound — *and* doubles capacity to 16 GB, which unblocks the 4 B model tier. | ~₹2,000 |
| SATA or NVMe SSD | Model cold-load measured **100 s** for a 1.7 B model at ~14 MB/s. No SSD does that. | ~₹2,500 |

Everything below works without them. It will be slower, and you will be
stuck on the 1 B model.

---

## Step 1 — Python

```powershell
py --version        # need 3.10 or newer
```

No output → install from python.org, ticking **"Add python.exe to PATH"**.
The Microsoft Store build works too but puts things in odd places.

## Step 2 — install the tool

```powershell
cd C:\Projects\latters\latters
git pull

py -m venv .venv
.\.venv\Scripts\Activate.ps1
# If PowerShell refuses:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

pip install -e ".[dev]"
```

Verify — all three must work:

```powershell
latters --version
latters fonts tables                          # 4 mapping tables
latters fonts convert --text "Lkkj.k lekgj.kky;"   # सारण समाहरणालय
python -m pytest tests/ -q                    # 284 passed
```

If `fonts convert` prints boxes rather than Devanagari, that is the console
font, not a bug — the tool forces UTF-8 output. Right-click the title bar →
Properties → Font → pick **NSimSun** or **Consolas**, or use Windows Terminal.

## Step 3 — configure Ollama

Every value here comes from a Phase 0 measurement on this machine.

```powershell
setx OLLAMA_KEEP_ALIVE       "-1"     # cold load is 30-100s on a spinning disk
setx OLLAMA_MAX_LOADED_MODELS "1"     # 8 GB cannot hold two
setx OLLAMA_NUM_PARALLEL     "1"
setx OLLAMA_FLASH_ATTENTION  "1"
setx OLLAMA_KV_CACHE_TYPE    "q8_0"   # halves KV memory, buys context
```

Close and reopen PowerShell (setx only affects new sessions), then:

```powershell
ollama pull gemma3:1b
ollama serve          # leave running in its own window
```

**Do not pull an embedding model into Ollama.** Measured on this machine: a
draft following a retrieval went from 0.92 s to 98 s because the embedder
evicted the drafting model. Retrieval here needs no model at all.

## Step 4 — install the legacy fonts

Needed only for Step 8, but do it now — without them the review sheet shows
Latin gibberish instead of Hindi.

`latters inventory` (next step) names the exact fonts. The sample archive
needs **Kruti Dev 010**, **Kruti Dev 041**, **DevLys 040**. Copy the `.ttf`
files into `C:\Windows\Fonts` (drag and drop installs them).

## Step 5 — point it at the archive

**Do this before Step 6, and note the `cd`.** Every path in Step 6 is
relative to `C:\latters-data`, not to the repository.

```powershell
mkdir C:\latters-data\archive
cd C:\latters-data
# Now copy the office's .docx letters into C:\latters-data\archive
explorer C:\latters-data\archive
```

Check they arrived before going on:

```powershell
dir archive\*.docx
```

`.doc`, `.rtf` and `.pdf` are refused with the reason. Convert the first two
with LibreOffice — it preserves the per-run font information this pipeline
depends on, which `antiword` and `catdoc` do not:

```powershell
& "C:\Program Files\LibreOffice\program\soffice.exe" --headless `
    --convert-to docx --outdir C:\latters-data\archive C:\latters-data\old\*.doc
```

---

## Step 6 — first run, in order

Run these **one at a time** and read the output between them. Pasting the
whole block into PowerShell runs every line even after one fails, so a
single missing folder produces eight confusing errors instead of one clear
one.

Each is fast — seconds to a minute — except the first ingest of a large
archive.

```powershell
cd C:\latters-data      # if you are not already there

# 6.1  What is in the archive?
latters inventory archive
```
Look at the font histogram. Any family showing `unknown-font` with a large
character count needs a mapping table before you go further.

```powershell
# 6.2  Are the anchors right for THIS office?
latters audit archive
```
**This is the step people skip and regret.** It reports which structural
anchors never fired. On the sample archive two never fired, because that
office writes `महाशय` where the defaults assumed `महोदय` and `प्रसंग` where
they assumed `संदर्भ`. If `repeated-subject` is over 15% of boundaries, the
closing anchors do not match your sign-off formula — fix
`src/latters/anchors.py` before trusting any segmentation.

```powershell
# 6.3  Split into individual letters and score them
latters segment archive --db corpus.db
latters stats --db corpus.db
```

```powershell
# 6.4  Labels, cross-validated
latters classify --db corpus.db --write
```
Read the **majority baseline** line, not the accuracy. Accuracy above a
baseline by less than 0.05 means the classifier learned nothing.

```powershell
# 6.5  Mine the letter skeletons
latters templates --db corpus.db -o skeletons --show
```

```powershell
# 6.6  Retrieval sanity check
latters eval --db corpus.db
latters search "अनुशासनिक कार्यवाही" --db corpus.db
```

```powershell
# 6.7  A draft, with no model involved
latters draft "अंचल अधिकारी से प्रतिवेदन मांगना है" --db corpus.db --stub --skeletons skeletons
```
`--stub` exercises retrieval, the skeleton and every safety check with a
canned body. If the letter structure looks right here, the only remaining
variable is the model.

```powershell
# 6.8  A real draft
latters draft "अंचल अधिकारी से दाखिल-खारिज में विलंब पर प्रतिवेदन मांगना है" `
    --db corpus.db --model gemma3:1b --skeletons skeletons
```

Expect roughly 45 seconds for a 400-word letter on the current hardware.

## Step 6b — the web pages

```powershell
latters serve --db corpus.db --skeletons skeletons --model gemma3:1b
```

| page | for |
|---|---|
| `/` | drafting: describe the letter, get a draft, edit it, export DOCX/PDF/TXT |
| `/admin` | upload more letters, browse lowest-trust-first, correct text in place, delete, edit skeletons |

Every draft shows the past letters it was built from with their trust scores,
what was stripped from the model's output, and any number the model invented.
That panel is not decoration — it is the only thing standing between a
plausible draft and a wrong letter leaving the office.

PDF export needs LibreOffice. If it fails, export DOCX and print to PDF from
Word; the letter is identical either way.

## Step 7 — correct the skeletons (30 minutes, once)

Open each file in `skeletons\`. They have two sections, `## above the
subject line` and `## below the body`. Fix the order, delete lines that do
not belong, add ones the archive missed. Your version wins from then on and
survives re-ingest.

Mining cannot get this fully right — an addressee's designation and a
signatory's post look identical out of context — so this half-hour is the
cheapest quality improvement available.

## Step 8 — the gold set (the blocking task)

**Nothing in this repository has verified that the Hindi is correct.** The
converter no longer produces *illegal* Devanagari; whether it produces the
*right* Devanagari is unproven, because the mapping table is the common
Kruti Dev 010 layout and your archive is 88% DevLys 040 and Kruti Dev 041.

```powershell
latters gold extract archive -o review -n 200 --blind-fraction 0.2
```

1. Open `review\review.docx` **on a machine with the legacy fonts installed**.
   Left column = the original as it appeared; right = our conversion.
   *If the left column reads as Latin gibberish, stop — the font is missing
   and the sheet is useless.*
2. A Hindi reader fills the `verdict_or_correction` column of
   `review\review.tsv` in Excel: `ok`, or the correct Hindi.
   20% of rows show no suggestion; those are the control against
   rubber-stamping and must be typed from the left column alone.
3. Collect and check:

```powershell
latters gold collect review\review.tsv -o gold\office_saran.tsv
latters gold run
latters gold coverage
```

Ship nothing below **98% character accuracy**. `gold coverage` separately
reports how many mapping slots no pair exercises — accuracy and coverage are
independent, and the seed set exercises only 38%.

---

## The two things that decide whether this works

Neither is code, and neither exists yet.

| | what it settles |
|---|---|
| **The gold set** (Step 8) | whether the converted Hindi is correct at all |
| **30 (request, dispatched letter) pairs** | whether the drafts save time |

For the second, save 30 real requests alongside the letter that actually went
out, as a TSV, then:

```powershell
python scripts\phase5_draft_eval.py --db corpus.db --model gemma3:1b --pairs pairs.tsv
```

It reports **editing effort** — how much of the draft has to change before
dispatch:

| | |
|---|---|
| < 0.15 | essentially usable |
| 0.15–0.40 | faster than starting blank |
| 0.40–0.70 | arguable |
| > 0.70 | **slower than typing it** |

Above 0.70 the project has failed its purpose and should be stopped rather
than tuned.

---

## While you are there: the unfinished Phase 0 items

```powershell
python scripts\phase0_bench.py --models gemma3:1b gemma3:4b-it-qat --out docs\BASELINE.md
python scripts\phase0_directml.py        # the device-I/O rows
```

The first settles whether the 4 B model is usable after the RAM upgrade and
reports the `num_thread` sweep, which never came back. The second confirms
whether the GPU deficit was PCIe transfer or real.

## If something breaks

| symptom | cause |
|---|---|
| `no such path:` | typo in the archive path — deliberately an error, not "0 files" |
| `no corpus database at` | run `latters segment` first; nothing else creates it |
| `nothing to evaluate` | run `latters classify --write`, or lower `--min-cell` |
| `cannot reach Ollama` | `ollama serve` is not running |
| boxes instead of Hindi | console font — see Step 2 |
| everything slow, disk at 100% | the spinning disk; see Step 0 |
| `the web pages need the optional extras` | `pip install -e ".[web]"` |
| `scorecard` exits 1 | correct: something is unmeasured or failing. Read which card says NO DATA |
| PDF export fails, DOCX works | LibreOffice; export DOCX and print from Word |
| `source file could not be loaded` from LibreOffice | the Writer module is missing, not your letter — it fails on a plain `.txt` too. Install the full LibreOffice, not `-core` |
| PDF shows boxes where the Hindi should be | the PDF says Nirmala UI and that font is not installed. It ships with Windows 8 and later; on Linux install `fonts-lohit-deva` or another Devanagari font |
| page loads but nothing happens on click | open the browser console; the JavaScript is served from `/static/app.js`, not a CDN |
