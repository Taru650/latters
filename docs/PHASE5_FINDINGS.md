# Phase 5 findings — generation

The one phase whose central question could not be answered here. There is no
Ollama and no model in this environment, so **no generated Hindi has been
assessed**. What follows is what the design does about that.

---

## 1. Shrink the untestable part

If generation quality cannot be tested, the right response is to make
generation responsible for as little as possible.

**The model writes the body paragraphs. Nothing else.** Everything a
departmental letter contains except its argument is a lookup: the letterhead
belongs to the office, the dispatch number is next in a series, the date is
today, the closing formula is fixed. Copying those is instant and exactly
right; generating them spends tokens at ~9 Hindi words per second to produce
something that might be wrong.

```
request → classify → retrieve exemplars → pick skeleton
        → LLM writes ONLY the body
        → assemble skeleton + body + looked-up fields
        → validate output, flag anything invented
```

Measured share the skeleton already supplies (Phase 3): **87%** for
बैंकिंग/जाँच, 62% for विकास/सामान्य पत्राचार. That is the difference between
a 12-second draft and a 90-second one, and between one testable component
and none.

---

## 2. Everything the model writes is treated as untrusted

### Stripped, and reported

A 1B model's instruction-following is a suggestion. Each of these is removed
from the output if it appears, and the removal is shown to the user rather
than applied silently:

| removed | replaced with |
|---|---|
| `पत्रांक` / `ज्ञापांक` lines | the next number in the series, or a blank slot |
| `दिनांक` lines | today |
| duplicate `विषय` line | the one the user supplied |
| closing / distribution block | the skeleton's own |
| markdown code fences | — |

**A blank letter number beats an invented one.** 95% of the real archive
leaves it blank, so a blank is normal and obviously unfinished; a wrong
number looks finished.

### Invented numbers are flagged

The dangerous failure is not bad Hindi — a clerk sees that immediately. It is
a plausible, wrong number: a file reference, an amount, a section of an Act.
`unsupported_facts()` reports every numeric token in the draft that appears
nowhere in the request or the retrieved letters, normalising Devanagari and
Latin digits so ८८७ and 887 are one token.

Deliberately over-sensitive. A false flag costs a glance; a missed one costs
a letter going out with an invented case number.

### The Phase 1 validator is turned on the model's own output

Small models produce malformed Devanagari. The same illegal-sequence
detector built for the legacy-font conversion runs on the generated text, so
a draft with a word-initial matra or a virama-then-matra is flagged before
anyone reads it.

---

## 3. A real gap the stub found

`bootstrap()` reads *structure* — branch codes in the letter number, the
office line in the letterhead. A clerk's free-text request has neither, so
every request returned `UNLABELLED`, and the pipeline silently lost both its
department filter (worth +0.10 precision, Phase 4) and its skeleton.

The Phase 3 classifier was trained for exactly this and had never been wired
to the request path. `TrainedClassifier` now does it, with the gates from the
Phase 3 measurements: department (0.812 macro-F1) is applied above a
confidence threshold; letter type (0.636) is *always* reported as a guess to
confirm, whether it came from the keyword rules or the model — the rules
generated the labels the model was measured on, so 0.64 bounds both.

---

## 4. Skeleton ordering cannot be fully mined, so people fix it

Mining produced letters with the salutation above the letterhead. Three
rounds of fixes:

1. **Frequency order, not document order.** `Counter.most_common()` is the
   obvious call and it is wrong: boilerplate has to come out in the order a
   letter reads.
2. **Median position is too weak on a heterogeneous cell.** 53 letters
   contain several layouts, so the medians interleave. Sorting by *role*
   first — using the Phase 2 anchors, which already know what each line is —
   and by position only within a role, fixed the landmarks.
3. **Unanchored lines still land imprecisely.** An addressee's designation or
   a district name has no anchor and can only be placed relative to its
   neighbours; sometimes it lands on the wrong side of the salutation.

The third is not solvable by inference on this data. It is a five-minute
editing task, once per category, by someone who knows the office's letters:

```bash
latters templates --db corpus.db -o skeletons/     # mine
$EDITOR skeletons/राजस्व_जाँच.md                    # fix the order
latters draft "…" --db corpus.db --skeletons skeletons/
```

Machine-mined is the starting point; human-corrected is the artefact, and it
survives the next ingest.

### Bugs found while getting there

- **Duplicate date line.** `दिनांक ░` and `दिनांक░` were different canonical
  forms, so both survived deduplication.
- **`दिनांक19.09.2026`.** The separator character class swallowed the space.
- **The date anchor required a digit**, so the blank template dates that are
  95% of the archive were never recognised as dates at all — which also cost
  the Phase 2 completeness score.
- **Boilerplate emitted twice.** A sentence common to every letter in a cell
  is mined as boilerplate; if the model also writes it, the letter says it
  twice.

---

## 5. What is still unmeasured, and how to measure it

**No generated Hindi has been assessed.** Not fluency, not correctness, not
whether a 1B model can write a departmental letter at all. That is the
central question of this phase and it is open.

`scripts/phase5_draft_eval.py` runs on the target machine. Without reference
letters it reports latency, the safety checks, and how much of each draft the
skeleton supplied. With them it reports the number that actually matters:

> **Editing effort** — character-level edit distance between the draft and
> the letter actually dispatched, as a fraction of the dispatched letter's
> length.
>
> | | |
> |---|---|
> | < 0.15 | essentially usable |
> | 0.15–0.40 | faster than starting blank |
> | 0.40–0.70 | arguable |
> | > 0.70 | **slower than typing it — the project has failed its purpose** |

Collect 30 pairs of (request, dispatched letter) into a TSV and pass
`--pairs`. Nothing else in this repository tells you whether the drafts are
worth using.

```bash
ollama pull gemma3:1b
python scripts/phase5_draft_eval.py --db corpus.db --model gemma3:1b --pairs pairs.tsv
```

---

## 6. The standing caveat, now at its most serious

Five phases of numbers rest on rules nobody has checked. Until now that was a
measurement problem. It stops being one here: this phase produces letters
that a government office would send.

- No gold set: the underlying Hindi text has never been read by a Hindi
  reader, so the corpus the drafts imitate may itself be subtly wrong.
- No draft evaluation: no generated letter has been judged by anyone.
- The labels the retrieval and skeleton selection depend on are rule-derived.

Every draft prints its sources with trust scores, lists what was stripped,
flags invented numbers, and ends with the line that this is a draft to be
read and corrected before dispatch. That framing is not a disclaimer. In a
government context it is the thing that makes the tool adoptable at all.
