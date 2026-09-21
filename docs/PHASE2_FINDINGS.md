# Phase 2 findings — measured against a real district archive

Five DOCX files from Saran district, Bihar (~500,000 legacy characters).
The first real data this project has seen. Everything below is measured.

---

## 1. The archive is 96% legacy, in font variants the plan did not anticipate

| Font | Characters | Share | Table used |
|---|---:|---:|---|
| DevLys 040 | 289,776 | 56.9% | `devlys040` (inherits base) |
| Kruti Dev 041 | 156,475 | 30.7% | `krutidev041` (inherits base) |
| Kruti Dev 010 | 42,076 | 8.3% | `krutidev010` |
| Times New Roman / Arial | 11,534 | 2.2% | none (pass through) |
| Kruti Dev 045 | 601 | 0.1% | `krutidev041` |

The plan built tables for the **010** variants. The archive is mostly **040**
and **041**. No slot has yet been *proven* to differ, so those tables inherit
the base and carry no overrides — that is a statement of ignorance, not of
equivalence, and only a gold set can settle it.

Junk font names also appear (`Calibari`, `ti0`, `inherit`, `Berlin Sans FB`),
which is normal in a two-decade-old archive and is why unknown fonts pass
through unconverted rather than being guessed at.

---

## 2. Bugs the real data found that synthetic fixtures never would

### Word splits words across runs — the worst one

A real paragraph arrived as:

```
run 1: 'f'        <- the pre-base chhoti-i matra, alone
run 2: 'tyk'      <- the rest of जिला
```

Per-run conversion produced `िजला`. Word splits runs at revision and proofing
boundaries with no regard for word boundaries, so this affects a large share
of the corpus. **Fix:** coalesce adjacent runs that share a mapping table
*before* converting. Reordering is a property of the text, not of the
formatting runs it happens to be stored in.

### The validator that should have caught it was disabled by a missing flag

`matra_word_initial` compiled without `re.MULTILINE`, so `^` anchored to the
start of the whole document. The single most important corruption signal
fired essentially never. A file containing `िजला` scored **0.93 "clean"**.

### `<half-form key> + k` is the FULL consonant

`Ik=kad` (101 occurrences) converted to `प्ात्रांक` instead of `पत्रांक`.
`I` is प् and `Ik` is प — half form plus aa-matra is illegal Devanagari, so
this is unambiguous. The table had this for `Hk`, `Fk`, `'k` and friends but
was missing `Dk Xk Pk Tk Rk Uk Ik Ck Ek Yk Ok Lk ºk ¶k`.

### Word's autocorrect rewrote the legacy text

Straight quotes in the source became curly ones, so the sha and ssa half-form
slots arrive as `’ ‘ ” “`. 3,100 characters affected — and they sit inside
high-frequency words (`egk’k;` → महाशय, `funs’k` → निदेश).

### `kW` is one unit

`vkWuykbZu` is ऑनलाईन, not आॉनलाईन. 215 occurrences. `vkW` additionally needs
its own three-character slot, because maximal munch matches `vk` at position 0
before `kW` can be seen at position 1.

### The mis-fonted-Latin rescue ate real Devanagari

`mi;qZDRk` (उपर्युक्त) contains the all-caps run `ZDR`, which the heuristic
"rescued" into the output as Latin. **Fix:** a rescued span must also contain
a digit or a separator. Real file numbers always do; legacy capital runs do not.

### Slots decoded from unambiguous context

| Legacy | Unicode | Evidence |
|---|---|---|
| `J` | श्र | `Jh` → श्री, 494× |
| `@` `$` | `/` | `i=kad&75@fnukad` → पत्रांक-७५/दिनांक |
| `)` | द्ध | `o`f)` → वृद्धि, `fo:)` → विरूद्ध |
| `Ÿk` / `Ÿ` | त / त् | `lekgŸkkZ` → समाहर्ता |
| `Ùk` / `Ù` | त्त / त्त् | `foÙk` → वित्त |
| `í` `Ì` | द्द | `jíhdj.k` → रद्दीकरण, `mÌs’;` → उद्देश्य |
| `Í` | ट्ठ | `dÍk` → कट्ठा (a Bihar land measure) |
| `¶` | फ् | `¶ykoj` → फ्लावर |
| `¾` | `=` | `84]000¾00` |
| `ß` `Þ` | `“` `”` | `lewg ßdÞ` → समूह "क" |

Net effect: total sequence violations across the five files fell from **~700
to 254**, and `पत्रांक` went from 104 broken to **778 correct**.

---

## 3. The anchors were tuned for the wrong office

This is the finding that generalises. My anchor patterns were written from
general knowledge of Indian government correspondence. Measured against this
archive:

| Anchor | I assumed | The archive uses | Count |
|---|---|---|---:|
| salutation | महोदय | **महाशय** | 356 (महोदय: **0**) |
| reference | संदर्भ | **प्रसंग** | 270 (संदर्भ: 0) |
| closing | आपका विश्वासभाजन | **विश्वासभाजन**, bare | 382 |
| closing | — | **हस्ताक्षर**, **अनु०** | 98, 93 |
| letter_number | पत्र संख्या | **पत्रांक**, **ज्ञापांक** | 769, 228 |

Two anchors never fired at all. The effect on segmentation:

| | before tuning | after |
|---|---:|---:|
| letters found | 437 | **589** |
| `letter_number-after-close` | 166 (38%) | **462 (78%)** |
| `header-after-close` | 75 (17%) | 110 (19%) |
| `repeated-subject` *(fallback)* | **191 (45%)** | **12 (2%)** |

The fallback going from 45% of boundaries to 2% is the real result. A
segmenter leaning on its safety net for nearly half its decisions is not
working; it is guessing. `latters audit` now reports this ratio and warns
above 15%.

**Generalisation: every office needs this tuning pass.** `latters audit`
exists to make it a ten-minute job rather than a discovery made in month three.

---

## 4. Current state of the corpus

```
589 letters from 5 files
  index       440
  review      149
  quarantine    0
547 stored (42 exact duplicates deduped by content hash)
mean trust 0.874   mean completeness 0.796   mean conversion 0.945
letter length  p05=195  median=756  p95=1713  max=9148
```

The 42 deduplicates are real: departmental boilerplate repeats verbatim.

**One known defect:** a single 9,148-character segment, 5× above p95. That is
several letters that never split, in a file whose first block is a letterhead
with no closing anywhere. `latters audit` flags it automatically.

---

## 5. What this does NOT establish

Everything above is *self-consistency*: the converter no longer produces
illegal Devanagari, and the segmenter no longer leans on its fallback.
Neither proves the output is **correct**.

- No pair in this archive has been checked by a Hindi reader.
- The 040 and 041 tables are assumed identical to 010 with no evidence.
- 589 is the number of segments produced, not the number of letters present.
  Nobody has counted the real number.

The gold-set tooling from Phase 1 is what closes this:

```
latters gold extract <archive> -o review/ -n 200 --blind-fraction 0.2
```

Until that comes back, treat every number in this document as a measure of
internal consistency and nothing more.


---

## 6. Not every document is a letter (found on the operator's first run)

`latters audit` flagged one 9,148-character segment, 5× above p95, and said
it was "almost certainly several letters that never got split". It was not.
It is a single disciplinary **आदेश**: 70 lines, no addressee, no विषय line,
no भवदीय, ending in a signature and four `प्रतिलिपि:` lines.

It scored 0.5 completeness because the scorer assumed every document has a
letter's anatomy. Measured across the archive:

| | |
|---|---:|
| segments with no addressee at all | **211 of 589** |
| ...of those, scoring under 0.7 | **169** |
| carrying an explicit `कार्यालय आदेश` marker | 15 |

**29% of the corpus was being marked defective for being the wrong genre.**

### Form-aware completeness

Three forms, each scored against its own anatomy:

| form | identified by | weighted on |
|---|---|---|
| `letter` | `सेवा में` / `प्रति` on its own line | number, date, addressee, subject, closing |
| `order` | `आदेश` / `ज्ञापन` / `अधिसूचना` / `परिपत्र` heading, **or** a number + date + real body | number, date, header, distribution |
| `fragment` | neither | a letter's weights — so it stays penalised |

The `fragment` class is the point. The obvious failure mode of form-awareness
is laundering every truncated segment into "complete" by calling it an order,
so a document with no addressee becomes an order only if it *says* so or
*behaves* like one. Result:

| form | n | median completeness | under 0.7 |
|---|---:|---:|---:|
| letter | 378 | 1.000 | 3 |
| order | 99 | 0.850 | **0** |
| fragment | 112 | 0.482 | 112 |

Corpus mean trust rose 0.891 → 0.904 and indexed letters 458 → 488, with the
112 genuine fragments still correctly quarantined.

### The audit's own diagnosis was wrong, too

"Almost certainly several letters" was an assertion with nothing behind it. A
merged block carries several complete anatomies — two or more closings, two
or more subject lines. This one carries zero of each. The warning now counts
them and distinguishes a failed split from one long document, and says which.


---

## 7. Endorsements are not letters either

With form-awareness in place, 112 segments remained `fragment`. Reading them
showed **98 were endorsement blocks**:

```
ज्ञापांक---------------/रा०, दिनांक----------------
प्रतिलिपि:- समाहर्त्ता, सारण छपरा को सादर सूचनार्थ समर्पित।
प्रतिलिपि:- उप विकास आयुक्त, सारण को सूचनार्थ प्रेषित।
```

Median 220 characters; 99 of 112 carried a letter number and 102 a date, but
only 3 a subject line.

A पृष्ठांकन carries **its own ज्ञापांक and दिनांक**, which is precisely what
the `letter-number-after-closing` boundary rule fires on. The rule is right
in general and wrong here: an endorsement is the copy-forwarding tail of the
letter above it, part of the same dispatch, not a letter of its own.

Detected by a distribution line plus the absence of everything that starts a
letter — no subject, no salutation, no addressee — under 600 characters, and
merged back into its parent. An endorsement with no preceding letter is kept
rather than dropped: losing text is worse than an odd segment.

| | before | after |
|---|---:|---:|
| segments | 589 | **475** |
| letter | 378 | 378 |
| order | 99 | 85 |
| **fragment** | **112** | **12** |
| needing review | 76 | **10** |

475 is a far more believable count for five files than 589.

### Form precedence, written out

Two arrangements each fixed one case and broke another, so the order is now
explicit in the code:

1. **An addressee line means letter**, whatever heading it carries. `सेवा में`
   outranks everything.
2. **An explicit `आदेश` / `ज्ञापन` heading means order** — above the
   endorsement test, because a short order ending in a प्रतिलिपि line is
   indistinguishable from an endorsement by length alone.
3. A short distribution-only block is an **endorsement**.
4. Number + date + real body behaves like an **order** even unheaded.
5. Otherwise **fragment**.

### The 12 that remain

Annexure tables (`पदाधिकारी का नाम, पदनाम एवं कार्यालय का नाम`), body
continuations, and an interrogatory (`(ख) क्या यह सही है कि ...`). Genuinely
ambiguous, 2.5% of the corpus, correctly held for human review.

## Unmapped-slot audit (run before the gold review, 5 real files)

Waiting for a Hindi reader is not the only way to find conversion bugs. Every
legacy character that appears in a legacy-font run and has **no entry in the
mapping table** is passed through raw, so it can be found mechanically. Over
the five archive files (~508k legacy characters, 72,921 Devanagari words):

| unmapped character | occurrences | verdict |
|---|---|---|
| `-` U+002D | 31,618 | correct — the hyphen is a passthrough, and most are dashed fill-lines |
| `Î` U+00CE | 1 | **missing slot.** `NqÎh` in "जब सरकारी `NqÎh` रहती है" is छुट्टी, so `Î` → ट्ट |
| `Ö` U+00D6 | 1 | **missing slot.** `ek¡Ökh, lkj.kA` is माँझी, सारण — the Saran block, so `Ök` → झ and `Ö` → झ् (a second slot for `>`) |

Both slots are inferred from sentence context, not from the font. That is
weaker evidence than a gold pair and both lines are in the review sheet, but
छुÎी is wrong under any reading, so shipping the fix cannot make it worse.

### The rule that matters more than the two slots

`छुÎी` scored **clean, 0.965**. Every rule in `validate.py` inspected
Devanagari only, so a raw Latin key welded into a Hindi word — the single
least ambiguous signature of a missing mapping slot — was invisible. The new
`latin_inside_word` rule closes it, and will catch the next missing slot
without anyone auditing anything.

Its two carve-outs were measured, not guessed. Without them the rule fired 36
times on this archive; the false positives were all one of two shapes:

- `२० फीट×१० फीट` — U+00D7 and U+00F7 are signs, not letters
- `११२१२२५५८८६८८/१A` — Devanagari digits legitimately abut a case-number suffix

With both carve-outs: **8 hits in 72,921 words, zero false positives.** Four
were the two slots above; the other four are one table header, `C.D Ratio`
followed by eight `a` characters typed in DevLys 040, which converts to eight
anusvaras. That is junk padding in the source, and flagging it is correct.

### What this does not establish

87.6% of the archive is DevLys 040 or Kruti Dev 041, and both tables still
inherit Kruti Dev 010 with **zero overrides** — no slot has been proven to
differ, and none has been proven identical either. This audit finds characters
we have no mapping for. It cannot find a character we map to the *wrong*
Devanagari, because the output is well-formed Hindi either way. Only the gold
set does that.
