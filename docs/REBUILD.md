# Clean rebuild on the corrected mapping tables

The per-phase findings documents each record the state of the corpus at the
moment that phase was written. Six phases and two mapping fixes later, several
of those numbers no longer describe anything that exists. This is the single
current record: one rebuild, from the archive, on the tables as they stand.

**The corpus database is not in this repository and must not be.** It holds
455 letters of real district correspondence — names, case numbers,
complainants, land disputes. Committing it would put that in git history
permanently, where deleting the file does not remove it. Only the numbers
belong here.

## Reproducing it

```
latters segment archive --db corpus.db
latters classify --db corpus.db --write
latters templates --db corpus.db -o skeletons
latters scorecard --db corpus.db
```

Archive: the five `.docx` files from a Saran district (Bihar) office,
~508k legacy characters, 87.6% DevLys 040 / Kruti Dev 041.

## Corpus

| | |
|---|---|
| letters found | 475 |
| letters stored | 455 (20 rejected as content-identical duplicates) |
| by form | 360 letter, 83 order, 12 fragment |
| mean trust | 0.925 |
| mean completeness | 0.949 |
| mean conversion confidence | 0.939 |
| verdicts | 446 index, 9 review, **0 quarantine** |
| vocabulary | 806 words seen in 3+ letters |

Per file: SDC_NEW_Zeba_arshi 178, mukesh_sir_letter_office_order 128,
manzoor_bhaiya 104, forwading 19, Letter 46.

## Scorecards

```
latters scorecard --db corpus.db   →  exit 1
```

### CONVERSION — no data

```
seed regression (our own pairs)  51/51 exact, 1.0000 char
```

That row is `data/gold/seed_krutidev010.tsv`, written by whoever wrote the
mapping table, from the same assumptions, in Kruti Dev 010. It proves the
converter still does what it did last week. It is not evidence about this
archive and the scorecard refuses to let it clear the card.

### RETRIEVAL — measured

| | |
|---|---|
| queries | 315 |
| known-item recall@5 | **0.943 ± 0.013** |
| MRR | 0.830 |
| same-cell P@5 (degraded) | **0.465 ± 0.028** |
| random baseline, same-cell P@5 | 0.062 |
| median latency | 1 ms |

**verdict: beats the random baseline.** 0.465 against 0.062 with 2 SE ≈
0.056 is not a close call. Known-item recall@5 of 0.943 means that for 297
of 315 queries the letter a drafter would have pulled is in the top five.
At 1 ms the retriever will never be the bottleneck.

The full configuration sweep, degraded (paraphrased) queries, 1 SE = 0.028:

| configuration | R@1 | R@5 | MRR | cellP@5 | ms |
|---|---:|---:|---:|---:|---:|
| random | 0.000 | 0.006 | 0.000 | 0.062 | 0.0 |
| bm25 | 0.752 | 0.892 | 0.811 | 0.363 | 1.4 |
| bm25 + dept | 0.768 | 0.933 | 0.838 | 0.463 | 1.4 |
| tfidf | 0.727 | 0.917 | 0.807 | 0.391 | 0.7 |
| **tfidf + dept** | 0.749 | **0.943** | 0.830 | **0.465** | **0.7** |
| rrf(bm25,tfidf) | 0.743 | 0.914 | 0.816 | 0.402 | 2.3 |
| rrf + dept | 0.765 | 0.949 | 0.842 | 0.490 | 2.2 |
| rrf + dept + type | 0.781 | 0.962 | 0.859 | n/a | 2.3 |

The conclusion from Phase 4 survives the rebuild unchanged: **the department
filter is the only component that measurably helps** (+0.07 to +0.10 on every
scorer), and the scorers are not distinguishable from each other — bm25,
tfidf and rrf all sit inside one another's error bars. `tfidf + dept` stays
the default: tied with rrf, 3× faster, one fewer moving part.

`rrf + dept + type` reads n/a because filtering by letter type makes
same-cell precision true by construction.

### DRAFTING — no data

0 dispatched drafts. The mechanism is live (every export records what
actually went out) and fills itself once the office uses the tool. Ten
dispatched drafts make the median meaningful.

## Classification

```
DEPARTMENT   445/455 labelled (98%)   n=445  classes=5
  accuracy          0.942
  majority baseline 0.645  (always predict 'राजस्व')
  macro F1          0.787

  राजस्व      287   0.97 / 1.00 / 0.98
  विकास        86   0.91 / 0.87 / 0.89
  बैंकिंग      38   0.84 / 1.00 / 0.92
  स्थापना      20   0.95 / 0.90 / 0.92
  अन्य         14   0.50 / 0.14 / 0.22

LETTER TYPE  426/455 labelled (94%)   n=426  classes=13
  accuracy          0.662
  majority baseline 0.218  (always predict 'सामान्य पत्राचार')
  macro F1          0.672
```

Department is usable behind a confidence gate. Letter type is not, and
`letter_type_confident = False` stays hard-coded.

The `अन्य` row is the interesting failure: F1 0.22 on 14 examples, recall
0.14. "Other" is being predicted almost never, which means it is acting as a
dumping ground in the bootstrapped labels rather than as a class. That is a
labelling artefact, not a model problem, and no amount of tuning fixes it.

## Skeletons

13 mined. The best cells:

| department | type | letters | skeleton covers |
|---|---|---|---|
| बैंकिंग | जाँच | 8 | 88% |
| बैंकिंग | सामान्य पत्राचार | 8 | 77% |

"Covers" is the share of a typical letter the office's own boilerplate
already supplies, so the model only writes the remainder. That is what makes
a 1B model viable at ~9 Hindi words per second.

## What changed against the per-phase numbers

| | earlier | now | reading |
|---|---|---|---|
| mean trust | 0.904 | 0.925 | different denominator (455 stored vs 475 found); not comparable as stated |
| same-cell P@5 | 0.500 ± 0.028 | 0.465 ± 0.028 | **1.3 SE apart — the same measurement, not a regression** |
| department macro-F1 | 0.812 | 0.787 | within the folding threshold's sensitivity; both usable |
| letter-type macro-F1 | 0.636 | 0.672 | improved, still not actionable |
| skeleton coverage | 87% | 88% | unchanged in substance |

Nothing here moved outside the error bars, which is the honest reading:
**the `Î` and `Ök` mapping fixes corrected two characters in ~508k and were
never going to show up in an aggregate.** They were worth making because
`छुÎी` is wrong, not because the corpus statistics needed them.

The earlier figures were quoted to three decimals in several documents as
though they were stable. They were single draws. Where a per-phase document
still shows the old number, this file is the current one.

## Still unmeasured

1. **Conversion.** No Hindi reader has checked any of it. The review pack
   exists (`latters gold extract`); nobody has filled it in.
2. **Draft quality.** No model has ever run in an environment where this was
   developed. Every claim about the Hindi the system writes is
   architectural, not empirical.
3. **Editing effort.** Zero samples.

Retrieval scoring 0.943 says we find the right *past letter*. It says
nothing about whether the Hindi in that letter is what was typed — a wrong
mapping produces well-formed Hindi that scores exactly the same.
