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
        """Returns (label, confidence), confidence being the softmax margin."""
        s = self.scores(text)
        if not s:
            return UNLABELLED, 0.0
        best = max(s, key=s.get)
        top = s[best]
        z = sum(math.exp(v - top) for v in s.values())
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
