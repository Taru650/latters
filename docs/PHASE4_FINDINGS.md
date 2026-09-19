# Phase 4 findings — retrieval

Measured on the 547-letter corpus, 313 evaluable queries. Phase 3 closed by
asking whether the metadata filter would do most of the work and whether a
neural encoder was needed at all. Both questions now have answers.

---

## 1. The headline: only the department filter measurably helps

Same-cell precision@5 under degraded (paraphrased) queries — the condition
closest to the real task:

| configuration | cellP@5 | R@5 | ms |
|---|---:|---:|---:|
| random | 0.068 ± 0.014 | 0.013 | 0.0 |
| bm25 | 0.384 ± 0.027 | 0.917 | 1.1 |
| **bm25 + dept filter** | **0.479** | 0.952 | 1.2 |
| tfidf | 0.417 ± 0.028 | 0.911 | 0.5 |
| **tfidf + dept filter** | **0.500** | 0.936 | 0.5 |
| rrf(bm25,tfidf) | 0.407 ± 0.028 | 0.914 | 1.8 |
| **rrf + dept filter** | **0.500** | 0.949 | 1.8 |

*(Numbers refreshed after end-to-end testing. They rose slightly because
`latters segment` now stores each letter's subject — it previously did not,
and Phase 4's original figures were obtained only because the column had been
backfilled by hand. The conclusions are unchanged: the department filter is
significant, the scorers are not distinguishable, and rrf and tfidf are now
exactly tied.)*

**One standard error is 0.028, so two rows differing by less than ~0.06 are
not distinguishable at this sample size.** That single line changes how the
whole table reads:

- **The department filter is significant**: +0.08 to +0.10 across every
  scorer, well beyond 2 SE, and consistent in both conditions.
- **The scorers are not distinguishable from each other.** bm25 0.479, tfidf
  0.500, rrf 0.500 all sit inside one another's error bars. Picking a
  "winner" from this table would be reporting noise.

There is a principled reason the fusion adds little: RRF pays off when its
inputs fail *differently*, and BM25 and character n-gram TF-IDF are both
lexical, so they fail on the same queries. Fusion earns its keep against a
genuinely complementary signal — which is what a dense encoder would have
been.

**Chosen default: `tfidf + dept filter`.** Statistically tied with RRF, four
times faster, and one fewer moving part. RRF stays available and is what to
reach for if a dense encoder is ever added.

---

## 2. The dense encoder was NOT adopted, and the decision is reproducible

The plan assumed a neural encoder (Granite 97M) was needed. On this evidence
it is not justified: ~200 MB of weights the target office cannot download, an
onnxruntime dependency, and resident RAM the drafting model needs — to
improve on a configuration whose competitors it cannot be shown to beat.

Two honest caveats:

- **It could not be tested here.** The development environment cannot reach
  huggingface.co, so no trained encoder has ever run against this corpus.
  The finding is "not justified by available evidence", not "measured and
  rejected".
- **The degraded condition is where it would show.** Lexical methods lost
  only 0.06–0.08 cellP@5 when half the query's words were dropped. They do
  not collapse under vocabulary mismatch, which is the failure mode a
  semantic model is bought to fix.

So the plumbing ships and the question is left decidable on the office's own
machine, with the rule fixed in advance so the result cannot be rationalised
afterwards:

```
python scripts/phase4_encoder_eval.py --db corpus.db --model ./granite-97m
```

> **Adopt a dense encoder only if it beats `tfidf + dept filter` on same-cell
> precision@5 by more than two standard errors (~0.06 at 300 queries), in the
> DEGRADED condition.**

A `RandomProjectionEncoder` is included as a no-download floor: it implements
the same protocol, exercises the whole dense path in tests, and gives a
sanity bar — a trained model that cannot beat a random projection is
misconfigured. It scores 0.415 degraded, below the lexical incumbent, exactly
as a surface-form-only encoder should.

---

## 3. A measurement trap worth naming

Adding the letter-type filter made same-cell precision read **1.000**. That
is not a result: filtering to the cell guarantees every hit is in the cell.
The metric is true by construction and means nothing.

It would have been an easy and very flattering number to report. `Result`
now carries `cell_metric_degenerate` and the renderer prints `n/a` instead,
with the reason. The known-item metrics under that filter remain meaningful
(R@5 0.946 → 0.978), and those are reported.

---

## 4. Engineering findings

### The index data structure was the difference between usable and not

A dense *documents × buckets* TF-IDF matrix is **71.7 MB for 547 letters**,
and would be **655 MB at 5,000** — unusable on a machine with 3 GB free. The
same information as an inverted index (parallel posting arrays plus a
per-bucket offset, i.e. CSC without the scipy dependency) is **6.8 MB**, a
projected 62 MB at 5,000, and queries got faster too: scoring touches only
the buckets the query contains rather than all 32,768.

### Devanagari punctuation is inside the Devanagari block

`[ऀ-ॿ]` as a word class makes danda `।` (U+0964) a searchable term
that appears in every document. Also affected: `॥` U+0965, `॰` U+0970.
Excluded now, in both the retriever and the evaluator.

### Unquoted user text is an FTS5 syntax error waiting to happen

A stray colon, quote, hyphen or the bare word `OR` makes `MATCH` raise, and
Devanagari official prose is full of them (`अनु०ः-यथोपरि।`). Every term is
extracted and quoted before it reaches SQLite, with a test that feeds the
nastiest real strings through.

### numpy `where` misuse produced NaN scores

`np.where(x > 0, 1 + np.log(x, where=x > 0), 0)` reads uninitialised memory
for the masked-out elements. Caught by running the test suite with
`-W error::RuntimeWarning`; there is now a test asserting no NaN or inf for
empty, punctuation-only and very long queries.

---

## 5. What these numbers are not

Both tasks are proxies and neither is the real evaluation.

**Known-item recall (R@5 0.90–0.99) is inflated** and should not be quoted as
a quality figure: the query is verbatim text from the target document, which
is precisely the case BM25 is built for and precisely what a clerk's
paraphrase is not.

**Same-cell precision uses the Phase 3 classifier's labels as ground truth**,
and those cross-validated at 0.81 macro-F1 for department and 0.64 for letter
type. The ceiling is therefore well below 1.0, and some of what counts as a
miss is a mislabel rather than a bad retrieval.

**The real evaluation is the one the plan specified and that still does not
exist:** forty requests written by a clerk, each with the past letters they
would actually have wanted. Until that exists, read this table as relative
comparison between configurations, never as absolute quality.

```
latters eval     --db corpus.db
latters retrieve "अनुशासनिक कार्यवाही हेतु कारण बताओ पत्र" --db corpus.db
```
