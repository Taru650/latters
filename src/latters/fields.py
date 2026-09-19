"""Extract the structured fields of a letter.

Regex first, by design. In a measured sample these patterns cover most of the
corpus, and every one of them is auditable by an office clerk. Sending 589
letters through a 1B model at 9 Hindi words/second to extract a date would
take hours and produce answers nobody can check.

BLANK IS NOT MISSING
--------------------
The single most important distinction here, and one a naive extractor gets
wrong. 95% of letters in the sample archive contain runs of dashes where the
number and date belong::

    पत्रांक--------------------/रा०, दिनांक------------------

The field is *present and deliberately empty* -- the office fills it in at
dispatch. That is completely different from a letter where the field is
absent, which means either a bad segmentation or a damaged source.

So every field has three states: FOUND, BLANK, ABSENT. The drafting page
needs BLANK fields to render as slots to fill; the quarantine queue needs
ABSENT fields to show as a defect. Collapsing them loses both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from enum import Enum

_D = r"[0-9०-९]"
#: A run of dashes, underscores or dotted leaders standing in for a value.
BLANK_RUN = r"[-–—_.░]{3,}"
_BLANK_RE = re.compile(BLANK_RUN)

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९",
                             "0123456789")


class State(Enum):
    FOUND = "found"
    BLANK = "blank"      #: label present, value left to be filled at dispatch
    ABSENT = "absent"


@dataclass
class Field:
    state: State
    value: str | None = None
    raw: str | None = None

    def __bool__(self) -> bool:
        return self.state is State.FOUND


@dataclass
class Fields:
    letter_number: Field = dc_field(default_factory=lambda: Field(State.ABSENT))
    date: Field = dc_field(default_factory=lambda: Field(State.ABSENT))
    subject: Field = dc_field(default_factory=lambda: Field(State.ABSENT))
    addressee: Field = dc_field(default_factory=lambda: Field(State.ABSENT))
    reference: Field = dc_field(default_factory=lambda: Field(State.ABSENT))
    signatory: Field = dc_field(default_factory=lambda: Field(State.ABSENT))
    #: The branch/section code that follows a letter number, e.g. /रा० for
    #: राजस्व (revenue) or /स्था० for स्थापना (establishment). It survives even
    #: when the number itself is blank, which makes it the most reliable
    #: department signal in a template-heavy archive.
    branch: Field = dc_field(default_factory=lambda: Field(State.ABSENT))
    office: Field = dc_field(default_factory=lambda: Field(State.ABSENT))
    copy_to: Field = dc_field(default_factory=lambda: Field(State.ABSENT))

    def as_dict(self) -> dict[str, dict]:
        return {k: {"state": v.state.value, "value": v.value}
                for k, v in self.__dict__.items()}

    def states(self) -> dict[str, str]:
        return {k: v.state.value for k, v in self.__dict__.items()}


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" \t:ः-–—।,")


def _raw(s: str) -> str:
    """Whitespace-normalise only.

    `_clean` strips dashes, which destroys exactly the information
    `_letter_number` needs: whether the value was a blank fill-in run. Keep
    the raw slice raw.
    """
    return re.sub(r"[ \t]+", " ", s).strip()


#: A value never runs past the start of the next field. Without this,
#: `पत्रांक--------/रा०, दिनांक------` yields a letter_number of "/दिनांक":
#: the dashes are stripped as a blank run, and the next field's own label is
#: then mistaken for this field's value.
_NEXT_LABEL_RE = re.compile(
    r"(?:दिनांक|दिनंाक|दिनाक|Dated?\b|विषय|प्रसंग|संदर्भ|सन्दर्भ|प्रतिलिपि"
    r"|सेवा\s*में|प्रेषक|मह(?:ोदय|ाशय)|Sub\b|Ref\b|Copy\s+to)")


def _labelled(text: str, label: str, *, same_line_only: bool = True,
              max_len: int = 300) -> Field:
    """Pull the value following a label, distinguishing blank from absent."""
    m = re.search(label, text)
    if not m:
        return Field(State.ABSENT)
    tail = text[m.end():]
    if not same_line_only:
        chunk = tail[:max_len]
    else:
        chunk = tail.split("\n", 1)[0][:max_len]
    cut = _NEXT_LABEL_RE.search(chunk)
    if cut and cut.start() > 0:
        chunk = chunk[:cut.start()]
    raw = chunk

    # Strip the separator the label is followed by, then decide.
    body = chunk.lstrip(" \t:ः–—-–—")
    stripped = _BLANK_RE.sub(" ", chunk).strip(" \t:ः-–—")
    if not stripped or not re.search(r"[ऀ-ॿA-Za-z0-9]", stripped):
        return Field(State.BLANK, None, _raw(raw))
    return Field(State.FOUND, _clean(stripped if _BLANK_RE.search(chunk) else body),
                 _raw(raw))


_NUM_LABEL = r"(?:पत्रांक|ज्ञापांक|ज्ञापंाक|स्मारांक|पत्र\s*संख्या|क्रमांक|फा\s*\.?\s*सं\s*\.?|F\s*\.?\s*No\s*\.?)"
_DATE_LABEL = r"(?:दिनांक|दिनंाक|दिनाक|Dated?)"


def _letter_number(text: str) -> Field:
    """The dispatch number, which is blank far more often than not.

    Two traps, both hit by real data:

    * The dashes are followed by the branch code and the next field's label,
      so naive blank-stripping returns "/रा०" or "/दिनांक" as the number.
    * A real number always contains a digit. Nothing that lacks one is a
      letter number, whatever survived the strip.

    So: take the text before the first blank run, and require a digit.
    """
    f = _labelled(text, _NUM_LABEL)
    if f.state is not State.FOUND:
        return f
    head = _BLANK_RE.split(f.raw or "", 1)[0]
    head = _clean(head.lstrip(" \t:ः\u2013\u2014-–—"))
    if head and re.search(_D, head):
        return Field(State.FOUND, head.rstrip("/,. "), f.raw)
    return Field(State.BLANK, None, f.raw)


def _date(text: str) -> Field:
    f = _labelled(text, _DATE_LABEL)
    if f.state is State.FOUND and f.value:
        iso = normalise_date(f.value)
        if iso:
            return Field(State.FOUND, iso, f.raw)
        # A label followed by unrelated prose is not a date.
        if not re.search(_D, f.value):
            return Field(State.BLANK, None, f.raw)
    return f


_DATE_RE = re.compile(rf"({_D}{{1,2}})\s*[-/.–]\s*({_D}{{1,2}})\s*[-/.–]\s*({_D}{{2,4}})")
_MONTHS = {m: i for i, m in enumerate(
    "जनवरी फरवरी मार्च अप्रैल मई जून जुलाई अगस्त सितम्बर अक्टूबर नवम्बर दिसम्बर".split(), 1)}
_MONTHS.update({"सितंबर": 9, "नवंबर": 11, "दिसंबर": 12, "अगस्त": 8})
_TEXT_DATE_RE = re.compile(rf"({_D}{{1,2}})\s+({'|'.join(_MONTHS)})\s+({_D}{{2,4}})")


def normalise_date(s: str) -> str | None:
    """Return ISO yyyy-mm-dd. Indian official dates are day-first, always."""
    m = _DATE_RE.search(s)
    if m:
        d, mo, y = (g.translate(_DEVA_DIGITS) for g in m.groups())
    else:
        m = _TEXT_DATE_RE.search(s)
        if not m:
            return None
        d, mon, y = m.group(1).translate(_DEVA_DIGITS), m.group(2), m.group(3).translate(_DEVA_DIGITS)
        mo = str(_MONTHS[mon])
    try:
        di, mi, yi = int(d), int(mo), int(y)
    except ValueError:
        return None
    if yi < 100:
        yi += 2000 if yi < 50 else 1900
    if not (1 <= di <= 31 and 1 <= mi <= 12 and 1900 <= yi <= 2100):
        return None
    return f"{yi:04d}-{mi:02d}-{di:02d}"


#: `पत्रांक-----------/रा०,` -- the code sits after the blank, before the comma.
_BRANCH_RE = re.compile(
    rf"{_NUM_LABEL}[^\n]{{0,60}}?/\s*([\u0900-\u097F]{{1,12}}[\u0966-\u096F0-9]*[\u0966०]?)"
)
#: Pure numerals and the date label are not branch codes.
_NOT_A_BRANCH = re.compile(r"^(?:दिनांक|दिनंाक|दिनाक)$|^[0-9\u0966-\u096F]+$")


def _branch(text: str) -> Field:
    for m in _BRANCH_RE.finditer(text):
        code = _clean(m.group(1))
        if code and not _NOT_A_BRANCH.match(code):
            return Field(State.FOUND, code, _clean(m.group(0)))
    return Field(State.ABSENT)


def _addressee(text: str) -> Field:
    """The lines between सेवा में / प्रति and the subject line."""
    m = re.search(r"^[ \t]*(?:सेवा\s*में|प्रति|प्रेषिती|To)\s*[,ः:]?[ \t]*$",
                  text, re.M)
    if not m:
        return Field(State.ABSENT)
    tail = text[m.end():]
    stop = re.search(r"^[ \t]*(?:विषय|प्रसंग|संदर्भ|सन्दर्भ|मह(?:ोदय|ाशय)|Sub)", tail, re.M)
    block = tail[:stop.start()] if stop else tail[:400]
    lines = [_clean(l) for l in block.splitlines() if _clean(l)]
    if not lines:
        return Field(State.BLANK, None, None)
    return Field(State.FOUND, " / ".join(lines[:5]), block.strip()[:300])


def _signatory(text: str) -> Field:
    """The designation lines after the closing formula."""
    m = None
    for pat in (r"विश्वासभाजन", r"भवदीय(?:ा)?", r"हस्ताक्षर(?:ित)?"):
        m = re.search(pat, text)
        if m:
            break
    if not m:
        return Field(State.ABSENT)
    tail = text[m.end():]
    stop = re.search(r"^[ \t]*(?:प्रतिलिपि|अनु\s*[०0]|Copy)", tail, re.M)
    block = tail[:stop.start()] if stop else tail[:250]
    lines = [_clean(l) for l in block.splitlines() if _clean(l)]
    if not lines:
        return Field(State.BLANK, None, None)
    return Field(State.FOUND, " / ".join(lines[:3]), block.strip()[:200])


_OFFICE_RE = re.compile(r"कार्यालय|समाहरणालय|निदेशालय|मंत्रालय|शाखा|विभाग|निगम|पंचायत|कोषांग")


def _office(text: str) -> Field:
    """The issuing office.

    Prefer the letterhead, but fall back to the signature block: when a
    segment begins below the letterhead -- which happens whenever the
    previous letter ran long -- the office name survives only down there.
    """
    lines = text.splitlines()
    for line in lines[:6]:
        c = _clean(line)
        if _OFFICE_RE.search(c):
            return Field(State.FOUND, c, c)
    sig = _signatory(text)
    if sig.state is State.FOUND and sig.value:
        for part in sig.value.split(" / "):
            if _OFFICE_RE.search(part):
                return Field(State.FOUND, _clean(part), sig.raw)
    for line in lines[6:14]:
        c = _clean(line)
        if _OFFICE_RE.search(c):
            return Field(State.FOUND, c, c)
    return Field(State.ABSENT)


def extract(text: str) -> Fields:
    f = Fields()
    f.letter_number = _letter_number(text)
    f.branch = _branch(text)
    f.date = _date(text)
    f.subject = _labelled(text, r"(?:विषय|Sub(?:ject)?)", same_line_only=False, max_len=400)
    if f.subject.state is State.FOUND and f.subject.value:
        # A subject runs to the first sentence end, not to the end of the letter.
        f.subject.value = re.split(r"।|\n\s*\n|\n(?=\s*(?:प्रसंग|संदर्भ|मह))",
                                   f.subject.value)[0].strip()[:300]
    f.reference = _labelled(text, r"(?:प्रसंग|संदर्भ|सन्दर्भ|Ref(?:erence)?)")
    f.addressee = _addressee(text)
    f.signatory = _signatory(text)
    f.office = _office(text)
    f.copy_to = _labelled(text, r"(?:प्रतिलिपि|Copy\s+to)", same_line_only=False, max_len=300)
    return f
