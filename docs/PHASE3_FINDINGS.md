# Phase 3 findings — field extraction, classification, template mining


> **Numbers here are from the build current when this phase was written.** Several have since been re-measured on a clean rebuild; `docs/REBUILD.md` is the current record and says which moved.

Measured on the 547-letter corpus built in Phase 2 from a real Saran district
archive. Every number below is 5-fold cross-validated or a direct count.

---

## 1. The archive is mostly *unsent templates*

Only **35 of 547 letters carry a real dispatch number.** The other 498 read:

```
पत्रांक- -------------------/रा०,     दिनांक------------------
```

The field is present and deliberately empty — the office fills it at dispatch.
That is a product finding, not a defect, and it drove the central design
decision here: **every field has three states, not two.**

| | meaning | what the UI does |
|---|---|---|
| `FOUND` | a value is present | show it |
| `BLANK` | the label is there, the value is left to fill | render as a slot |
| `ABSENT` | the field is missing entirely | flag as a defect |

Collapsing BLANK into ABSENT would have marked 91% of the archive broken.
Collapsing it into FOUND — which the first implementation did — returned the
*next field's label* as the value: every template reported a letter number of
`/दिनांक`, then of `/रा०`.

### Coverage over 547 real letters

| field | found | blank | absent | usable |
|---|---:|---:|---:|---:|
| letter_number | 35 | 498 | 14 | **97%** |
| date | 85 | 452 | 10 | **98%** |
| office | 458 | 0 | 89 | 84% |
| signatory | 399 | 0 | 148 | 73% |
| branch | 382 | 0 | 165 | 70% |
| subject | 358 | 19 | 170 | 69% |
| addressee | 358 | 2 | 187 | 66% |
| reference | 223 | 1 | 323 | 41% |
| copy_to | 159 | 0 | 388 | 29% |

`office` rose from 41% to 84% by falling back to the signature block: when a
segment starts below the letterhead — which happens whenever the previous
letter ran long — the office name survives only down there.

**The branch code is the most valuable field.** `पत्रांक-----/रा०` still
carries `रा०` when the number itself is blank, which makes it the one reliable
department signal in a template-heavy archive.

---

## 2. Classification: department works, letter type does not

No sklearn. scipy plus sklearn is ~100 MB of wheels for a machine with 3 GB
free and no internet to install from; a character n-gram multinomial Naive
Bayes is eighty lines of standard library and trains 547 letters in 3.6
seconds. Character n-grams rather than words because this corpus spells
स्थानान्तरण and स्थानांतरण, संदर्भ and सन्दर्भ interchangeably.

No LLM, either. Classifying into a closed label set is the one task where a
1B model is strictly worse: an hour of compute at 9 Hindi words per second,
for an answer nobody can audit.

### Labels are bootstrapped, not hand-labelled

Departmental letter numbers are branch-encoded, so the labels are already in
the data. 95% of letters get a department from the branch code or the office
line, with no human labelling at all.

### Department — usable

```
n=518  classes=5 (after folding)
accuracy          0.936
majority baseline 0.662   (always predict 'राजस्व')
lift over baseline+0.274
macro F1          0.812

  class          support   prec  recall     f1
  राजस्व             343   0.98    0.99   0.98
  विकास               85   0.86    0.94   0.90
  बैंकिंग             50   0.87    0.96   0.91
  स्थापना             21   0.95    0.86   0.90
  अन्य                19   0.40    0.11   0.17
```

### Letter type — a suggestion, not a decision

```
accuracy 0.659   baseline 0.182   lift +0.477   macro F1 0.636
```

The plan set 0.85 macro-F1 as the bar. **Letter type does not clear it and
should not be auto-applied** — surface it as a pre-filled dropdown the user
confirms. Department, at 0.812, can be applied with a confidence gate.

### Three things that moved the numbers

**Reporting the majority baseline.** 93.6% accuracy on a label set that is
66% one class is not, by itself, evidence of anything. Every report prints
the baseline and the lift, and warns when the lift is under 0.05.

**Folding the tail.** Six department classes had under 10 examples and scored
0.00 each, dragging macro-F1 to 0.432. Folding them into a named `अन्य`
bucket took it to 0.812. That is not hiding the problem — the folded classes
are listed — it is refusing to average over numbers that mean nothing. A
folded class needs more data, it is not a model failure.

**Choosing the input slice.** Worth more than the model:

| task | full text | narrowed | chosen |
|---|---:|---:|---|
| department | 0.773 | **0.812** | first 400 chars — the department is in the letterhead, the body is noise |
| letter type | 0.570 | **0.630** | subject ×3 + first 300 chars — the type is in the subject line |

### A precision bug the tests caught

`वाद\s*सं` matched *inside* `परिवाद संख्या`, so every complaint was filed as
a court case. Fixed with a Devanagari negative lookbehind; परिवाद F1 went
0.45 → 0.66 and न्यायालय वाद 0.71 → 0.76.

### 46% of letters had no type at all, until they did

251 letters matched no keyword rule. Reading them showed 167 were the generic
`... के संबंध में` construction — not a missing type, but a real category:
general correspondence. Naming it `सामान्य पत्राचार` took type coverage from
54% to 94%. The rest yielded four new rules the first list missed entirely
(भूमि, परिवाद, न्यायालय वाद, योजना), all found by reading the data rather
than by guessing.

---

## 3. Template mining — why a 1B model is viable

545 of 547 letters have a unique body, so there are no whole-letter templates
to reuse. **Document-level mining finds nothing; line-level mining finds a
great deal.** `प्रभारी पदाधिकारी,` appears in 204 letters, `जिला राजस्व शाखा,`
in 125.

For each (department, letter-type) cell with 8+ letters, lines appearing in
30%+ of the cell are the skeleton; everything else is body. Numbers, dates and
fill-in dash runs are normalised away first, or every letterhead looks unique
because its date differs.

| department | type | letters | skeleton covers |
|---|---|---:|---:|
| बैंकिंग | जाँच | 8 | **87%** |
| बैंकिंग | सामान्य सूचना | 10 | **84%** |
| बैंकिंग | सामान्य पत्राचार | 8 | 76% |
| बैंकिंग | योजना | 10 | 74% |
| विकास | प्रतिवेदन | 9 | 68% |
| विकास | सामान्य पत्राचार | 24 | 62% |
| राजस्व | परिवाद | 24 | 48% |
| राजस्व | जाँच | 53 | 46% |
| राजस्व | भूमि | 79 | 36% |

**"Skeleton covers" is the share of a letter the model does not have to
write.** At 87%, the letterhead, addressee block, closing and distribution
list are *copied* — so they are exactly right rather than approximately
right, and cost zero generation tokens. At 9 Hindi words per second that is
the difference between a 12-second draft and a 90-second one.

Cells under 8 letters are skipped. A skeleton mined from three letters is
three letters' idiosyncrasies, and using it would make every future draft in
that category look like those three.

---

## 4. What this does not establish

- Labels are **rule-derived, not human-verified.** Cross-validation measures
  agreement with the rules, not with the truth. If a rule is wrong, CV
  confirms the classifier learned the wrong thing accurately.
- One district, one archive. `रा०` is 66% of it. Another office will have a
  different branch vocabulary and probably different letter types.
- Letter-type macro-F1 of 0.636 is not good enough to act on unconfirmed.
- The gold set still has not come back, so the Phase 1 caveat stands
  underneath all of this: none of the underlying text has been checked by a
  Hindi reader.

```
latters fields    --db corpus.db
latters classify  --db corpus.db --write
latters templates --db corpus.db -o skeletons/ --show
```
