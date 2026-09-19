# Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q          # 283 tests
```

| file | what it covers |
|---|---|
| `test_convert.py` | Phase 1: tables, conversion, validation, repair, gold tooling |
| `test_segment.py` | Phase 2: anchors, segmentation, trust, SQLite store |
| `test_phase3.py` | Phase 3: fields, classification, cross-validation, templates |
| `test_retrieve.py` | Phase 4: TF-IDF index, RRF, filters, evaluation harness |
| `test_draft.py` | Phase 5: budgeting, output sanitising, assembly, orchestration |
| `test_integration.py` | **The CLI, in sequence, as an office would run it** |

---

## Why the integration tests exist

The unit tests exercise functions and all 246 of them passed while the
application had six bugs in it, two of them showstoppers. Every one was found
by running the pipeline from a clean directory, installed as a wheel, the way
a user would — not by reading code and not by adding more unit tests.

### Bugs found this way

| bug | symptom | why unit tests missed it |
|---|---|---|
| **Devanagari crashed on a Windows console** | `UnicodeEncodeError` on the first Hindi character | pytest captures stdout as UTF-8; the target machine's cmd.exe is cp1252 |
| **`\| head` raised BrokenPipeError** | traceback on every piped command | nothing pipes in a unit test |
| **`segment` never stored the subject** | FTS subject column and the whole retrieval query set silently inert | Phase 4 was developed against a database whose subjects had been backfilled by hand |
| **`fonts gold` found nothing when installed** | deployed office had no regression harness | `tests/` is not in the wheel, and the dev checkout always had it |
| **A mistyped `--db` created an empty database** | every later command "worked" on nothing | tests always passed a real path |
| **A mistyped archive path reported "0 files", rc=0** | a typo looks identical to an empty archive | same |
| **`eval` with nothing to evaluate printed empty tables, rc=0** | reads as an evaluation that found nothing | same |
| **A mistyped `--skeletons` dir was ignored** | the clerk's corrections appeared to do nothing | same |
| **Body sentences mined into the letterhead** | assembled letters were structurally wrong | only visible by reading a whole assembled letter |
| **Salutation emitted above the subject** | wrong to anyone who writes these letters | the code had no opinion about which was right |

All are now covered by tests in `test_integration.py`, which asserts exit
codes and stderr text rather than just "it did not crash".

### Exit-code convention

| code | meaning |
|---|---|
| 0 | worked |
| 1 | worked, but the output needs a human (quarantine, `needs_review`) |
| 2 | the invocation was wrong (bad path, missing corpus, unusable input) |

The 1-vs-2 split matters: a draft that needs review is not the same as a
command that could not run, and a script that only checks `rc != 0` should
still be able to tell them apart.

---

## Testing the installed package, not the checkout

Everything in development runs against `src/`. That is not what the office
gets, and two bugs lived in the gap.

```bash
pip wheel . -w /tmp/wheel --no-deps
python -m venv /tmp/v && /tmp/v/bin/pip install /tmp/wheel/*.whl numpy
cd /tmp && /tmp/v/bin/latters fonts gold      # must find the packaged gold set
```

## Full pipeline against a real archive

```bash
latters inventory  ARCHIVE                    # what fonts, how much is legacy
latters audit      ARCHIVE                    # are the anchors right for THIS office
latters segment    ARCHIVE --db corpus.db     # split into letters, score, store
latters classify   --db corpus.db --write     # bootstrap labels, cross-validate
latters templates  --db corpus.db -o sk/      # mine skeletons
latters eval       --db corpus.db             # retrieval against baselines
latters draft      "..." --db corpus.db --stub --skeletons sk/
```

`--stub` runs everything except the LLM, so the whole pipeline is exercisable
on a machine with no model — including in CI.

## What testing still cannot reach

- **Generated Hindi quality.** No model here. `scripts/phase5_draft_eval.py`
  measures editing effort on the target machine, and needs 30 pairs of
  (request, dispatched letter) that do not exist yet.
- **Conversion correctness.** The gold set proves the *engine*; only
  hand-verified pairs from a Hindi reader prove the *table*.
- **Real hardware.** Phases 0's benchmarks run on the laptop, not here.
