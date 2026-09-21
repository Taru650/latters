"""Department and letter-type classification, without an LLM and without sklearn.

Why not the LLM
---------------
Classifying into a closed label set is the one task where a 1B model is
strictly worse than a linear model: slower by orders of magnitude, not
auditable, and no better on lexical signals this strong. At 9 Hindi words per
second, classifying 589 letters would take an hour and produce answers nobody
can check. A char-ngram Naive Bayes trains in under a second and its evidence
can be printed.

Why not sklearn
---------------
scipy + sklearn is ~100 MB of wheels on a machine with 3 GB of free RAM and,
in the target deployment, no internet to install them from. Multinomial Naive
Bayes over character n-grams is about eighty lines of standard library and
performs comparably on short, formulaic text. The dependency is not worth it.

Character n-grams rather than words because Devanagari orthography in this
corpus is inconsistent -- स्थानान्तरण and स्थानांतरण, संदर्भ and सन्दर्भ,
सितम्बर and सितंबर all appear -- and character n-grams degrade gracefully
across those variants where word features do not.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

UNLABELLED = "?"

# --------------------------------------------------------------------------
# Label bootstrapping
# --------------------------------------------------------------------------

#: Branch codes seen in letter numbers, e.g. `पत्रांक-----/रा०`. This is the
#: most reliable department signal in a template-heavy archive because it
#: survives even when the number itself is blank.
BRANCH_DEPARTMENTS: dict[str, str] = {
    "रा": "राजस्व", "रा०": "राजस्व", "राज": "राजस्व",
    "स्था": "स्थापना", "स्था०": "स्थापना",
    "वि": "विकास", "वि०": "विकास", "विधि": "विधि",
    "म०नि०": "मनरेगा", "प०पा०": "पंचायती राज",
    "निर्वाचन": "निर्वाचन", "सी": "सामान्य", "सी०": "सामान्य",
    "शि": "शिक्षा", "शि०": "शिक्षा", "स्वा": "स्वास्थ्य",
}

#: Fallback: the office line in the letterhead.
OFFICE_DEPARTMENTS: list[tuple[str, str]] = [
    (r"राजस्व", "राजस्व"),
    (r"स्थापना", "स्थापना"),
    (r"प्रखण्ड\s*विकास|प्रखंड\s*विकास|विकास\s*पदाधिकारी", "विकास"),
    (r"बैंकिंग|कोषांग", "बैंकिंग"),
    (r"शिक्षा", "शिक्षा"),
    (r"स्वास्थ्य", "स्वास्थ्य"),
    (r"पंचायत", "पंचायती राज"),
    (r"निर्वाचन", "निर्वाचन"),
    (r"आपूर्ति", "आपूर्ति"),
    (r"नगर\s*निगम|नगर\s*परिषद", "नगर निकाय"),
]

#: Letter types, matched against the subject line. Ordered: first match wins,
#: so the more specific patterns come first.
LETTER_TYPES: list[tuple[str, str]] = [
    (r"कारण\s*बताओ|शोकॉज|शो\s*कॉज|स्पष्टीकरण", "स्पष्टीकरण"),
    (r"अनुशासनिक|विभागीय\s*कार्यवाही|आरोप\s*पत्र|निलंब", "अनुशासनिक"),
    (r"अग्रसारित|अग्रेषित|अग्रसरित|forward", "अग्रसारण"),
    (r"समीक्षा\s*बैठक|बैठक\s*(?:की|हेतु|का)|बैठक", "बैठक सूचना"),
    (r"स्थानान्तरण|स्थानांतरण|पदस्थापन|तबादला", "स्थानांतरण"),
    (r"सेवानिवृ|पेंशन|पेन्शन|उपादान", "सेवानिवृत्ति"),
    (r"वेतन|भुगतान|देयक|राशि\s*की\s*निकासी", "भुगतान"),
    (r"जाँच|जांच|अन्वेषण|सत्यापन", "जाँच"),
    (r"प्रतिवेदन|प्रगति\s*प्रतिवेदन|रिपोर्ट", "प्रतिवेदन"),
    (r"आवंटन|आबंटन", "आवंटन"),
    (r"निविदा|टेंडर", "निविदा"),
    (r"नियुक्ति|संविदा\s*नियुक्ति", "नियुक्ति"),
    (r"अनुमति|स्वीकृति", "अनुमति"),
    # Added from the data: these are the topics the first keyword list missed
    # entirely, found by reading the 251 subjects that matched no rule.
    # `वाद` must not match inside परिवाद -- otherwise every complaint is
    # filed as a court case. Caught by a test.
    (r"बनाम|न्यायालय|(?<![ऀ-ॿ])वाद\s*सं|रिट\s*याचिका|Cr\.?\s*W|\bWJC\b|अवमानना",
     "न्यायालय वाद"),
    (r"परिवाद|शिकायत|लोक\s*शिकायत", "परिवाद"),
    (r"दाखिल[\s-]*खारिज|जमाबंदी|भू[\s-]*लगान|अतिक्रमण|भूमि|खेसरा|खाता", "भूमि"),
    (r"योजना|अभियान|कैम्प|शिविर|Campaign", "योजना"),
    (r"सूचना|सूचित", "सामान्य सूचना"),
]

#: When a letter has a subject line but matches no rule above, it is not
#: unlabelled -- it is general correspondence, which is a real and very common
#: category. 167 of 251 such subjects in the sample archive are the generic
#: "... के संबंध में" construction. Calling that UNLABELLED discards 46% of the
#: corpus; calling it what it is keeps it usable.
DEFAULT_LETTER_TYPE = "सामान्य पत्राचार"

_BRANCH_CLEAN = re.compile(r"[०0\.\s]+$")


@dataclass
class Bootstrapped:
    department: str = UNLABELLED
    department_source: str = "none"
    letter_type: str = UNLABELLED
    letter_type_source: str = "none"


def bootstrap(fields, text: str = "") -> Bootstrapped:
    """Derive labels from structure, with no human labelling.

    Departmental letter numbers are branch-encoded, so the labels are already
    in the data. This is the difference between a labelling exercise that
    takes an afternoon and one that takes a month.
    """
    out = Bootstrapped()

    if getattr(fields, "branch", None) and fields.branch.value:
        code = fields.branch.value
        for key in (code, _BRANCH_CLEAN.sub("", code)):
            if key in BRANCH_DEPARTMENTS:
                out.department = BRANCH_DEPARTMENTS[key]
                out.department_source = "branch"
                break

    if out.department is UNLABELLED or out.department == UNLABELLED:
        haystack = " ".join(filter(None, [
            fields.office.value if fields.office else None,
            fields.signatory.value if fields.signatory else None,
        ]))
        for pattern, label in OFFICE_DEPARTMENTS:
            if re.search(pattern, haystack):
                out.department = label
                out.department_source = "office"
                break

    subject = (fields.subject.value if fields.subject else None) or ""
    probe = subject or text[:400]
    for pattern, label in LETTER_TYPES:
        if re.search(pattern, probe):
            out.letter_type = label
            out.letter_type_source = "subject" if subject else "body"
            break
    else:
        if subject:
            out.letter_type = DEFAULT_LETTER_TYPE
            out.letter_type_source = "default"
    return out


#: Classes below this many examples cannot be learned or evaluated; their
#: per-class scores are noise and they drag macro-F1 down while telling you
#: nothing.
MIN_SUPPORT = 10
RARE_LABEL = "अन्य"


def fold_rare(labels: list[str], *, min_support: int = MIN_SUPPORT,
              rare: str = RARE_LABEL) -> tuple[list[str], set[str]]:
    """Merge classes with too few examples into a single 'other' bucket.

    Reporting macro-F1 over classes with three examples each is reporting
    noise. Folding them is not hiding the problem -- the folded set is named
    and returned -- it is refusing to average over numbers that mean nothing.
    A folded class is a class you need more data for, not one the model
    failed at.
    """
    counts = Counter(labels)
    folded = {y for y, n in counts.items() if n < min_support}
    return [rare if y in folded else y for y in labels], folded


@dataclass(frozen=True)
class CoverageGap:
    """The classes the shipped classifier cannot predict, and who pays.

    `fold_rare` merges classes under `min_support` into a single 'other'
    bucket. That is right for *reporting* -- macro-F1 over classes with three
    examples is noise -- but it produced two quieter problems that were live
    in this project for four phases:

    1. **The reported model was not the shipped model.** `cross_validate` was
       run on the folded labels, so 'अन्य' appeared as a 5th class scoring
       F1 0.22, while `TrainedClassifier.fit` *drops* those rows and ships a
       4-class model. The published macro-F1 of 0.787 understated the model
       that actually runs by 0.147 (it scores 0.934), and the 0.22 row
       described a class the shipped classifier cannot emit.

    2. **Nobody said what happens to those letters.** They are not edge
       cases in the data -- they are whole departments. Measured on the real
       corpus, all 14 letters from निर्वाचन, पंचायती राज, विधि, मनरेगा and
       सामान्य were classified as one of the four big departments, **14 out
       of 14 wrong**. Their stored labels are correct (the branch code is
       exact), so retrieval over the archive is fine; the damage is at draft
       time, where a request from one of those departments silently filters
       retrieval to the wrong department's letters.

    Folding is still the right call -- one more example of निर्वाचन does not
    make it learnable -- but the gap has to be stated, not averaged away.
    """

    folded: tuple[str, ...]
    n_letters: int
    n_total: int

    @property
    def share(self) -> float:
        return self.n_letters / self.n_total if self.n_total else 0.0

    def render(self) -> str:
        if not self.folded:
            return ""
        return (
            f"!! {self.n_letters} letter(s) ({self.share:.0%}) are in "
            f"departments the classifier cannot predict:\n"
            f"!!   {', '.join(sorted(self.folded))}\n"
            f"!! Their stored labels are correct -- the branch code is exact "
            f"-- so\n"
            f"!! search and retrieval over the archive are unaffected. But a "
            f"free-text\n"
            f"!! REQUEST from one of these will be assigned one of the "
            f"trained classes,\n"
            f"!! and the confidence margin cannot detect it: measured on this "
            f"corpus,\n"
            f"!! every such letter was misclassified. Set the department by "
            f"hand for\n"
            f"!! these, or collect {MIN_SUPPORT}+ letters each and re-run."
        )


def coverage_gap(labels: list[str], *, min_support: int = MIN_SUPPORT
                 ) -> CoverageGap:
    """Which classes get dropped, and how many letters that is."""
    counts = Counter(labels)
    folded = tuple(y for y, n in counts.items() if n < min_support)
    return CoverageGap(folded=folded,
                       n_letters=sum(counts[y] for y in folded),
                       n_total=len(labels))


# --------------------------------------------------------------------------
# Featurisers
# --------------------------------------------------------------------------
# Which slice of the letter to show the classifier is worth more than the
# model. Measured by 5-fold CV on 547 real letters:
#
#   department   full text 0.773 macro-F1 | head 400 chars 0.812 | fields 0.773
#   letter type  full text 0.570          | subject only 0.577   | subject x3 0.630
#
# The department lives in the letterhead, so the body is noise for it. The
# type lives in the subject line, so the subject is repeated to weight it
# without discarding the context the body provides.


def department_features(text: str, fields=None) -> str:
    """The letterhead carries the department; the body does not."""
    return text[:400]


def letter_type_features(text: str, fields=None) -> str:
    """Weight the subject line, keep a little context."""
    subject = ""
    if fields is not None and getattr(fields, "subject", None) and fields.subject.value:
        subject = fields.subject.value
    return (subject + " ") * 3 + text[:300]


# --------------------------------------------------------------------------
# Character n-gram multinomial Naive Bayes
# --------------------------------------------------------------------------
_WS = re.compile(r"\s+")


def ngrams(text: str, lo: int = 2, hi: int = 4, *, limit: int = 1200) -> Counter[str]:
    text = _WS.sub(" ", text)[:limit]
    out: Counter[str] = Counter()
    for n in range(lo, hi + 1):
        for i in range(len(text) - n + 1):
            out[text[i:i + n]] += 1
    return out


@dataclass
class NaiveBayes:
    lo: int = 2
    hi: int = 4
    alpha: float = 0.2
    #: Features seen fewer times than this across the whole corpus are noise.
    min_df: int = 2
    log_prior: dict[str, float] = field(default_factory=dict)
    log_prob: dict[str, dict[str, float]] = field(default_factory=dict)
    default: dict[str, float] = field(default_factory=dict)
    vocab: set[str] = field(default_factory=set)
    classes: list[str] = field(default_factory=list)

    def fit(self, texts: list[str], labels: list[str]) -> "NaiveBayes":
        df: Counter[str] = Counter()
        featurised = []
        for t in texts:
            g = ngrams(t, self.lo, self.hi)
            featurised.append(g)
            df.update(g.keys())
        self.vocab = {f for f, n in df.items() if n >= self.min_df}

        per_class: dict[str, Counter[str]] = defaultdict(Counter)
        counts: Counter[str] = Counter()
        for g, y in zip(featurised, labels):
            counts[y] += 1
            for f, n in g.items():
                if f in self.vocab:
                    per_class[y][f] += n

        self.classes = sorted(counts)
        total = sum(counts.values())
        v = max(len(self.vocab), 1)
        for y in self.classes:
            self.log_prior[y] = math.log(counts[y] / total)
            denom = sum(per_class[y].values()) + self.alpha * v
            self.log_prob[y] = {f: math.log((n + self.alpha) / denom)
                                for f, n in per_class[y].items()}
            self.default[y] = math.log(self.alpha / denom)
        return self

    def scores(self, text: str) -> dict[str, float]:
        g = ngrams(text, self.lo, self.hi)
        out = {}
        for y in self.classes:
            s = self.log_prior[y]
            lp, dflt = self.log_prob[y], self.default[y]
            for f, n in g.items():
                if f in self.vocab:
                    s += n * lp.get(f, dflt)
            out[y] = s
        return out

    def predict(self, text: str) -> tuple[str, float]:
        """Returns (label, confidence).

        THE CONFIDENCE IS LENGTH-NORMALISED AND THAT IS NOT COSMETIC.

        `scores()` sums `n * log P(feature|class)` over every in-vocabulary
        n-gram, so a 400-word letter accumulates hundreds of log-probabilities
        and the winning class beats the runner-up by hundreds of nats. A plain
        softmax over those underflows: `exp(runner_up - top)` is 0.0 and the
        confidence is exactly 1.0. Measured on a real corpus, **445 of 445
        letters came back at 1.0000** and the 0.40 gate they were supposed to
        pass had never rejected anything in the project's history, while three
        documents claimed the department was "applied with a confidence gate".

        Dividing by the total feature weight turns the score into a mean
        log-probability per feature. Every class is divided by the same
        positive constant, so **the argmax and therefore the accuracy are
        unchanged**; only the margin becomes readable. Out-of-fold on 431
        letters the result separates errors properly:

            confidence 0.25-0.35    n=37    accuracy 0.757
            confidence 0.35-0.40    n=113   accuracy 0.982
            confidence 0.40+        n=281   accuracy 0.981

        **What it does NOT do is detect a department the model was never
        trained on.** A softmax margin is relative to a closed class set and
        cannot express "none of the above"; an absolute per-feature likelihood
        was tried too and separated no better (out-of-domain letters scored
        *higher*, -7.21 against -7.30, because these are all letters from one
        office and the n-grams are dominated by shared boilerplate -- the
        department lives in the branch code, which a free-text request has
        none of). See `coverage_gap()`.
        """
        s = self.scores(text)
        if not s:
            return UNLABELLED, 0.0
        g = ngrams(text, self.lo, self.hi)
        weight = sum(n for f, n in g.items() if f in self.vocab) or 1
        scaled = {y: v / weight for y, v in s.items()}
        best = max(scaled, key=scaled.get)
        top = scaled[best]
        z = sum(math.exp(v - top) for v in scaled.values())
        return best, 1.0 / z


# --------------------------------------------------------------------------
# Honest evaluation
# --------------------------------------------------------------------------
@dataclass
class CVReport:
    n: int
    classes: dict[str, int]
    accuracy: float
    macro_f1: float
    #: Accuracy of always predicting the most common class. A classifier that
    #: does not clearly beat this has learned nothing, however good the
    #: accuracy looks on a skewed label set.
    majority_baseline: float
    per_class: dict[str, dict[str, float]]

    @property
    def beats_baseline_by(self) -> float:
        return self.accuracy - self.majority_baseline

    def render(self) -> str:
        lines = [
            f"n={self.n}  classes={len(self.classes)}",
            f"accuracy          {self.accuracy:.3f}",
            f"majority baseline {self.majority_baseline:.3f}  "
            f"(always predict '{max(self.classes, key=self.classes.get)}')",
            f"lift over baseline{self.beats_baseline_by:+.3f}",
            f"macro F1          {self.macro_f1:.3f}",
            "",
            f"  {'class':22} {'support':>7} {'prec':>6} {'recall':>7} {'f1':>6}",
        ]
        for y, m in sorted(self.per_class.items(), key=lambda kv: -kv[1]["support"]):
            lines.append(f"  {y:22} {int(m['support']):7d} {m['precision']:6.2f} "
                         f"{m['recall']:7.2f} {m['f1']:6.2f}")
        if self.beats_baseline_by < 0.05:
            lines += ["", "!! The classifier barely beats always guessing the most",
                      "!! common class. On a label set this skewed, accuracy is not",
                      "!! evidence of anything. Get more letters from the other",
                      "!! departments before relying on this."]
        thin = [y for y, n in self.classes.items() if n < 10]
        if thin:
            lines += ["", f"!! {len(thin)} class(es) have under 10 examples: "
                          f"{', '.join(sorted(thin)[:8])}",
                      "!! Their per-class scores are noise."]
        return "\n".join(lines)


def cross_validate(texts: list[str], labels: list[str], *, k: int = 5,
                   seed: int = 0, **kw) -> CVReport:
    """Stratified k-fold. Reports against the majority baseline, always."""
    counts = Counter(labels)
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[y].append(i)
    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for y, idx in by_class.items():
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(i)

    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    correct = 0
    for f in range(k):
        test = folds[f]
        train = [i for g in range(k) if g != f for i in folds[g]]
        if not train or not test:
            continue
        model = NaiveBayes(**kw).fit([texts[i] for i in train], [labels[i] for i in train])
        for i in test:
            pred, _ = model.predict(texts[i])
            truth = labels[i]
            if pred == truth:
                tp[truth] += 1
                correct += 1
            else:
                fp[pred] += 1
                fn[truth] += 1

    per_class = {}
    f1s = []
    for y in counts:
        p = tp[y] / (tp[y] + fp[y]) if tp[y] + fp[y] else 0.0
        r = tp[y] / (tp[y] + fn[y]) if tp[y] + fn[y] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        f1s.append(f1)
        per_class[y] = {"support": counts[y], "precision": p, "recall": r, "f1": f1}

    return CVReport(
        n=len(labels), classes=dict(counts),
        accuracy=correct / len(labels) if labels else 0.0,
        macro_f1=sum(f1s) / len(f1s) if f1s else 0.0,
        majority_baseline=max(counts.values()) / len(labels) if labels else 0.0,
        per_class=per_class)


# --------------------------------------------------------------------------
# Applying the classifier to a REQUEST, not a letter
# --------------------------------------------------------------------------
@dataclass
class TrainedClassifier:
    """Predict department and letter type for a free-text request.

    `bootstrap` reads structure -- branch codes in the letter number, the
    office line in the letterhead. A clerk's request has neither, so
    bootstrap returns UNLABELLED for every request and the drafting pipeline
    loses its department filter and its skeleton. This is what the Phase 3
    model was trained for; it just has to be wired to the request path.

    The confidence gates come straight from the Phase 3 cross-validation:
    department reached 0.812 macro-F1 and is applied automatically above its
    threshold; letter type reached 0.636 and is only ever a *suggestion*,
    which is why `letter_type_confident` is reported separately.
    """
    department: NaiveBayes | None = None
    letter_type: NaiveBayes | None = None
    #: Below this margin the prediction is not acted on. Both numbers are
    #: swept out-of-fold on a real 455-letter corpus, not chosen by feel --
    #: the previous 0.40/0.55 pair was invented and, against the saturated
    #: softmax they were written for, rejected exactly nothing.
    #:
    #: DEPARTMENT, 5 classes, overall accuracy 0.963:
    #:     gate 0.33  keeps 409/431  accuracy 0.980   8/16 errors caught
    #:     gate 0.35  keeps 394/431  accuracy 0.982   9/16 errors caught
    #:     gate 0.38  keeps 346/431  accuracy 0.983  10/16 errors caught
    #: 0.35 buys +0.019 accuracy for 9% of requests losing their filter;
    #: 0.38 costs another 11% of requests for one more error.
    department_threshold: float = 0.35

    #: LETTER TYPE is 13 classes, so the margin is spread thin and **a gate
    #: is the wrong instrument entirely**: at 0.30, two requests out of 406
    #: survive. A non-zero value here does not make the suggestion safer, it
    #: deletes it -- and with it the (department, type) skeleton lookup, so
    #: every letter silently loses its letterhead. `letter_type_confident`
    #: below is the real protection: the type is always presented as a guess
    #: for the user to confirm, never acted on silently.
    letter_type_threshold: float = 0.0

    @classmethod
    def fit(cls, rows: list[tuple[str, str | None, str | None]], *,
            min_support: int = MIN_SUPPORT, **kw) -> "TrainedClassifier":
        """`rows` is (text, department, letter_type) from a labelled corpus."""
        out = cls()
        for attr, featurise, idx in (("department", department_features, 1),
                                     ("letter_type", letter_type_features, 2)):
            X = [featurise(r[0]) for r in rows if r[idx]]
            y = [r[idx] for r in rows if r[idx]]
            if len(set(y)) < 2:
                continue
            folded, _ = fold_rare(y, min_support=min_support)
            keep = [i for i, label in enumerate(folded) if label != RARE_LABEL]
            if len(keep) < 10 or len({folded[i] for i in keep}) < 2:
                continue
            setattr(out, attr, NaiveBayes(**kw).fit([X[i] for i in keep],
                                                    [folded[i] for i in keep]))
        return out

    def predict(self, request: str) -> tuple[str | None, float, str | None, float]:
        """Returns (department, confidence, letter_type, confidence)."""
        dept = ltype = None
        dconf = lconf = 0.0
        if self.department is not None:
            dept, dconf = self.department.predict(department_features(request))
            if dconf < self.department_threshold:
                dept = None
        if self.letter_type is not None:
            ltype, lconf = self.letter_type.predict(letter_type_features(request))
            if lconf < self.letter_type_threshold:
                ltype = None
        return dept, dconf, ltype, lconf

    @property
    def letter_type_confident(self) -> bool:
        """Always False: Phase 3 measured 0.636 macro-F1, below the 0.85 bar.

        Kept as a named property so callers cannot quietly start trusting it
        without changing this line and the measurement behind it.
        """
        return False
