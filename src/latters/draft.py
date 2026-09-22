"""Draft a new letter from the office's own past letters.

THE MODEL WRITES THE BODY. NOTHING ELSE.
----------------------------------------
Everything a departmental letter contains except its argument is a lookup:
the letterhead belongs to the office, the dispatch number is the next in a
series, the date is today, the closing formula is fixed, the distribution
list comes from the letter type. Generating any of that is spending tokens
at ~9 Hindi words per second to produce something that might be wrong, when
copying it is instant and exactly right.

So the pipeline is:

    request -> classify -> retrieve exemplars -> pick skeleton
            -> LLM writes ONLY the body paragraphs
            -> assemble skeleton + body + looked-up fields
            -> validate the output, flag anything invented

That is what makes a 1B model viable. Measured in Phase 3, the skeleton for
बैंकिंग/जाँच already supplies 87% of the letter.

WHAT IS DELIBERATELY NOT GENERATED
----------------------------------
Every one of these is stripped from the model's output if it appears, and
substituted from data:

* पत्रांक / ज्ञापांक -- the next number in the office's series, or a blank
  slot to fill at dispatch (95% of the real archive leaves it blank).
* दिनांक -- today.
* letterhead, addressee block, closing formula, distribution list -- copied
  from the mined skeleton.

A small model will happily invent a plausible file number. Inventing one is
worse than leaving it blank, because a blank is obviously unfinished and a
wrong number looks finished.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .fields import BLANK_RUN, extract
from .segment import trust as trust_score
from .template import Skeleton
from .validate import assess

# --------------------------------------------------------------------------
# Token budgeting
# --------------------------------------------------------------------------
#: Below this many letters, retrieval has nothing representative to return
#: and no (department, type) cell reaches the 8 letters a skeleton needs.
MIN_USEFUL_CORPUS = 30

#: Tokens per Devanagari word, measured in Phase 0 on the target machine:
#: Gemma 3 1.97, Qwen3 6.03. The default is Gemma's; pass the measured value
#: for whatever model is actually configured, because at 4096 context the
#: difference is 2,080 Hindi words of room against 680.
DEFAULT_FERTILITY = 1.97

_DEVA_WORD = re.compile(r"[ऀ-ॣ०-९ॲ-ॿ]+")
_LATIN_WORD = re.compile(r"[A-Za-z]+")
_DIGITS = re.compile(r"[0-9]+")


def estimate_tokens(text: str, *, fertility: float = DEFAULT_FERTILITY) -> int:
    """Approximate token count without loading a tokenizer.

    Good enough to budget a prompt; not exact. Erring high is the safe
    direction, since overflowing the context silently truncates the exemplars
    the draft depends on.
    """
    deva = len(_DEVA_WORD.findall(text))
    latin = len(_LATIN_WORD.findall(text))
    digits = sum(len(d) for d in _DIGITS.findall(text))
    punct = len(re.findall(r"[^\w\s]", text))
    return int(deva * fertility + latin * 1.3 + digits / 2 + punct * 0.5) + 8


@dataclass
class Budget:
    context: int = 4096
    reserve_for_output: int = 900
    fertility: float = DEFAULT_FERTILITY

    @property
    def for_prompt(self) -> int:
        return max(256, self.context - self.reserve_for_output)

    def fits(self, text: str) -> bool:
        return estimate_tokens(text, fertility=self.fertility) <= self.for_prompt


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
#: English, deliberately. It tokenises roughly three times cheaper than the
#: same instruction in Hindi, and small models follow English instructions at
#: least as well. The *output* language is stated explicitly instead.
SYSTEM_PROMPT = """You draft official letters for an Indian government office.

Rules:
- Write ONLY the body paragraphs of the letter, in Hindi (Devanagari).
- Do NOT write the letterhead, the addressee block, the subject line, the
  closing formula, the signature, or the copy-to list. Those are added
  automatically.
- Do NOT invent a letter number (पत्रांक), a date, a file number, an amount,
  or a person's name. If the request does not supply one, omit it.
- Match the tone, vocabulary and sentence structure of the example letters.
- Be concise. Official Hindi, no English words unless the examples use them."""

_EXEMPLAR_BODY_TOKENS = 260


def _exemplar_body(text: str) -> str:
    """The argument of a letter, without its furniture.

    Sending the letterhead and closing of three exemplars would spend most of
    the context re-teaching the model boilerplate the skeleton already
    supplies verbatim.
    """
    lines = text.splitlines()
    start = 0
    for i, l in enumerate(lines):
        if re.search(r"मह(?:ोदय|ाशय)|^\s*Sir\b", l):
            start = i + 1
            break
        if re.search(r"विषय\s*[:ः\-–]", l):
            start = i + 1
    end = len(lines)
    for i in range(start, len(lines)):
        if re.search(r"विश्वासभाजन|भवदीय|हस्ताक्षर|प्रतिलिपि|अनु\s*[०0]", lines[i]):
            end = i
            break
    body = "\n".join(lines[start:end]).strip()
    return body or text.strip()


def _truncate_to_tokens(text: str, limit: int, *, fertility: float) -> str:
    if estimate_tokens(text, fertility=fertility) <= limit:
        return text
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(" ".join(words[:mid]), fertility=fertility) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo]) + " …"


@dataclass
class PromptParts:
    system: str
    prompt: str
    n_exemplars: int
    estimated_tokens: int
    dropped_exemplars: int = 0


def build_prompt(request: str, exemplars: list[str], *,
                 skeleton: Skeleton | None = None, subject: str | None = None,
                 budget: Budget | None = None,
                 max_exemplars: int = 3) -> PromptParts:
    """Assemble the prompt, dropping exemplars until it fits.

    Two or three exemplars, not five. Each Hindi letter body is 250-450
    tokens; five of them overflow a 4096 context and force a prefill this
    machine cannot afford. The skeleton carries the format, so the exemplars
    only have to carry the voice.
    """
    budget = budget or Budget()
    header = []
    if skeleton is not None:
        header.append(f"Category: {skeleton.department} / {skeleton.letter_type}")
        header.append(f"Write roughly {max(120, skeleton.median_body_chars // 2)}"
                      f"-{skeleton.median_body_chars} characters of body text.")
    if subject:
        header.append(f"विषय (already written, do not repeat it): {subject}")

    chosen = exemplars[:max_exemplars]
    dropped = 0
    while chosen:
        blocks = []
        for i, ex in enumerate(chosen, 1):
            body = _truncate_to_tokens(_exemplar_body(ex), _EXEMPLAR_BODY_TOKENS,
                                       fertility=budget.fertility)
            blocks.append(f"--- Example {i} (body only) ---\n{body}")
        prompt = "\n\n".join(header + blocks + [
            "--- Request ---", request.strip(),
            "--- Now write only the body paragraphs, in Hindi ---"])
        if budget.fits(prompt) or len(chosen) == 1:
            return PromptParts(SYSTEM_PROMPT, prompt, len(chosen),
                               estimate_tokens(prompt, fertility=budget.fertility),
                               dropped)
        chosen = chosen[:-1]
        dropped += 1
    prompt = "\n\n".join(header + ["--- Request ---", request.strip(),
                                   "--- Write the body paragraphs in Hindi ---"])
    return PromptParts(SYSTEM_PROMPT, prompt, 0,
                       estimate_tokens(prompt, fertility=budget.fertility), dropped)


# --------------------------------------------------------------------------
# Deterministic repair
# --------------------------------------------------------------------------
_NUM_LINE = re.compile(
    r"^.*(?:पत्रांक|ज्ञापांक|पत्र\s*संख्या|स्मारांक|फा\s*\.?\s*सं).*$", re.M)
_DATE_LINE = re.compile(r"^\s*दिनांक\s*[:ः\-–]?.*$", re.M)
_INLINE_NUM = re.compile(
    r"(?:पत्रांक|ज्ञापांक|पत्र\s*संख्या|स्मारांक)\s*[:ः\-–]?\s*[^\s,।]{1,30}")
_SUBJECT_LINE = re.compile(r"^\s*विषय\s*[:ः\-–].*$", re.M)
#: A trailing comma used to defeat this entirely. Gemma 3 wrote
#: "धन्यवाद,\nभवदीय," and both survived, so the letter carried the model's
#: closing AND the skeleton's -- two sign-offs, one letter. `[,।;:.\s]*`
#: now absorbs the punctuation, and धन्यवाद is in the list: it is a normal
#: way to end a message and never how this office ends a letter.
_CLOSING_LINE = re.compile(
    r"^\s*(?:विश्वासभाजन|भवदीय[ा]?|भवदीया|धन्यवाद|सधन्यवाद|आपका\s+विश्वासभाजन"
    r"|हस्ताक्षर(?:ित)?|अनु\s*[०0].*|प्रतिलिपि.*)[,।;:.\s]*$", re.M)
#: The salutation is the skeleton's job, and the model gets it wrong. The
#: first real draft opened `महोदय,` -- correct Hindi, and not this office's
#: word: the archive has महाशय 356 times and महोदय zero. Leaving both in
#: gives the letter two salutations, one of them in a register the office
#: does not use. The addressee block above it is NOT stripped: `सेवा में,
#: जिलाधिकारी, वैशाली` is content the model derived from the request, and
#: assemble() already skips a skeleton line the body repeats.
_SALUTATION_LINE = re.compile(
    r"^\s*(?:महोदय[ा]?|महाशय[ा]?|श्रीमान[्]?|प्रिय\s+\S+)[,।\s]*$", re.M)
_FENCE = re.compile(r"^\s*```.*$", re.M)

#: Markdown in a government letter. A 1B model reaches for bullets and bold
#: because that is what its training data looks like; the office's letters
#: contain neither. Seen in the first real draft:
#:     *   **कार्यक्षेत्र:** तराना कुमार के वर्तमान ...
#: The heading survives, the asterisks do not.
_MD_BULLET = re.compile(r"^\s*[*+\-]\s+", re.M)
_MD_BOLD = re.compile(r"\*{1,3}(.+?)\*{1,3}")
_MD_HEADING = re.compile(r"^\s*#{1,6}\s+", re.M)

#: Fill-in-the-blank placeholders. The same draft ended with
#: [आपका नाम] [आपका पद] [विभाग का नाम] [संपर्क नंबर] [ईमेल आईडी] --
#: a signature block the model invented, below the one the skeleton
#: supplies. Anything bracketed is a placeholder: the office's own letters
#: use ( ) for parenthetical text and never [ ].
_PLACEHOLDER = re.compile(r"^\s*[\[<][^\]>\n]{1,40}[\]>][,।\s]*$", re.M)
_INLINE_PLACEHOLDER = re.compile(r"[\[<][^\]>\n]{1,40}[\]>]")


def clean_body(text: str) -> tuple[str, list[str]]:
    """Strip everything the model was told not to write but wrote anyway.

    Instruction-following in a 1B model is a suggestion, not a guarantee.
    Every removal is reported so the UI can show what was discarded rather
    than silently editing the model's output.
    """
    removed: list[str] = []

    def _strip(pattern, label):
        nonlocal text
        hits = pattern.findall(text)
        if hits:
            removed.append(f"{label} ({len(hits)})")
            text = pattern.sub("", text)

    _strip(_FENCE, "code fence")
    _strip(_MD_HEADING, "markdown heading")
    _strip(_MD_BULLET, "markdown bullet")
    if _MD_BOLD.search(text):
        removed.append("markdown bold")
        text = _MD_BOLD.sub(r"\g<1>", text)
    _strip(_PLACEHOLDER, "placeholder line the model invented")
    if _INLINE_PLACEHOLDER.search(text):
        removed.append("inline placeholder")
        text = _INLINE_PLACEHOLDER.sub("", text)
    _strip(_NUM_LINE, "invented letter-number line")
    _strip(_DATE_LINE, "invented date line")
    _strip(_SUBJECT_LINE, "duplicate subject line")
    _strip(_SALUTATION_LINE, "salutation the skeleton supplies")
    _strip(_CLOSING_LINE, "closing block the skeleton supplies")
    if _INLINE_NUM.search(text):
        removed.append("inline letter number")
        text = _INLINE_NUM.sub("", text)

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, removed


_TOKEN_NUM = re.compile(r"[0-9०-९]{2,}")


def unsupported_facts(body: str, request: str, exemplars: list[str]) -> list[str]:
    """Numbers in the draft that appear nowhere in the request or exemplars.

    The most dangerous error this system can make is not bad Hindi -- a clerk
    sees that. It is a plausible, wrong number: a file reference, an amount, a
    section of an Act. This flags every numeric token the model produced that
    it was not given, so the UI can highlight them for checking.

    Deliberately over-sensitive. A false flag costs a glance; a missed one
    costs a letter going out with an invented case number in it.
    """
    source = request + " " + " ".join(exemplars)
    known = set(_TOKEN_NUM.findall(source))
    # Normalise Devanagari digits so ८८७ and 887 count as the same token.
    trans = str.maketrans("०१२३४५६७८९", "0123456789")
    known |= {k.translate(trans) for k in known}
    out = []
    for tok in _TOKEN_NUM.findall(body):
        if tok not in known and tok.translate(trans) not in known:
            out.append(tok)
    return sorted(set(out))


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
BLANK_SLOT = "-" * 20


@dataclass
class Draft:
    text: str
    body: str
    department: str | None = None
    letter_type: str | None = None
    subject: str | None = None
    #: (letter_id, why it was chosen) for every exemplar used.
    sources: list[tuple[int, str]] = field(default_factory=list)
    removed_from_model_output: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    quality_score: float = 0.0
    quality_verdict: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)

    #: Too few past letters for retrieval to mean anything.
    thin_corpus: bool = False

    @property
    def needs_review(self) -> bool:
        return (bool(self.unsupported_numbers) or self.truncated
                or self.quality_verdict != "clean"
                or not self.sources or self.thin_corpus)


def assemble(body: str, *, skeleton: Skeleton | None, subject: str | None,
             letter_number: str | None = None, on: date | None = None,
             fill_date: bool = True) -> str:
    """Skeleton + body + looked-up fields, in the office's own order."""
    on = on or date.today()
    out: list[str] = []
    # A sentence common to every letter in a cell is mined as boilerplate.
    # If the model also writes it, the letter says it twice. Skip any
    # boilerplate line the body already contains.
    body_lines = {l.strip() for l in body.splitlines() if l.strip()}

    if skeleton is not None:
        # Only the block above the subject line. Everything between the
        # salutation and the closing is the body's territory, and the
        # skeleton has already separated the two.
        for literal in skeleton.before_subject():
            if literal.strip() in body_lines or re.match(r"^\s*विषय", literal):
                continue
            if re.search(r"पत्रांक|ज्ञापांक|दिनांक", literal):
                line = re.sub(BLANK_RUN, BLANK_SLOT, literal)
                if letter_number:
                    line = re.sub(rf"((?:पत्रांक|ज्ञापांक))\s*[:ः–-]?\s*{BLANK_SLOT}",
                                  rf"\g<1>- {letter_number}", line)
                if fill_date:
                    # The separator class must not swallow the space, or the
                    # result reads "दिनांक19.09.2026".
                    line = re.sub(rf"(दिनांक)\s*[:ः–-]?\s*{BLANK_SLOT}",
                                  rf"\g<1> {on.strftime('%d.%m.%Y')}", line)
                out.append(line)
            else:
                out.append(literal)

    if subject:
        out.append(f"विषय:- {subject.rstrip('।')}।")
    if skeleton is not None:
        # The salutation belongs between the subject and the body.
        out.extend(l for l in skeleton.salutation() if l.strip() not in body_lines)
    out.append("")
    out.append(body.strip())
    out.append("")

    if skeleton is not None:
        out.extend(skeleton.after_body() or ["विश्वासभाजन"])
    else:
        out.append("विश्वासभाजन")
    return "\n".join(out).strip()


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
@dataclass
class DraftService:
    """Ties Phases 3, 4 and 5 together behind one call.

    The LLM is injected rather than constructed, so the whole path is
    testable with a stub and so the app can swap models without touching
    this code.
    """
    retriever: object                 #: latters.retrieve.Retriever
    llm: object                       #: latters.llm.LLM
    skeletons: dict[tuple[str, str], Skeleton] = field(default_factory=dict)
    budget: Budget = field(default_factory=Budget)
    max_exemplars: int = 3
    min_trust: float = 0.6
    #: A latters.classify.TrainedClassifier. Without one, a free-text request
    #: gets no department -- bootstrap reads branch codes and letterheads,
    #: which a request does not have -- and the pipeline silently loses both
    #: its retrieval filter and its skeleton.
    classifier: object | None = None

    def classify(self, request: str) -> tuple[str | None, str | None, dict]:
        from .classify import UNLABELLED, bootstrap

        # Structure first: if the clerk pasted a letter, believe its own
        # branch code over any model.
        b = bootstrap(extract(request), request)
        dept = None if b.department == UNLABELLED else b.department
        ltype = None if b.letter_type == UNLABELLED else b.letter_type
        info = {"department_source": b.department_source if dept else "none",
                "letter_type_source": b.letter_type_source if ltype else "none"}

        if self.classifier is not None and (dept is None or ltype is None):
            pd, dconf, pl, lconf = self.classifier.predict(request)
            if dept is None and pd:
                dept, info["department_source"] = pd, f"model ({dconf:.2f})"
            if ltype is None and pl:
                ltype, info["letter_type_source"] = pl, f"model ({lconf:.2f})"
        return dept, ltype, info

    def draft(self, request: str, *, department: str | None = None,
              letter_type: str | None = None, subject: str | None = None,
              letter_number: str | None = None, on: date | None = None,
              options: dict | None = None) -> Draft:
        from .retrieve import Filters

        # A corpus this small cannot supply a representative exemplar, and a
        # draft from it looks exactly as confident as a good one. Say so.
        n_corpus = len(getattr(self.retriever, "letters", []))
        thin = n_corpus < MIN_USEFUL_CORPUS

        guess_dept, guess_type, how = self.classify(request)
        department = department or guess_dept
        letter_type = letter_type or guess_type
        warnings: list[str] = []
        if thin:
            warnings.append(
                f"the corpus holds only {n_corpus} letter(s). Below roughly "
                f"{MIN_USEFUL_CORPUS} there is nothing representative to "
                "retrieve, no cell is large enough for a skeleton, and this "
                "draft is close to asking the model cold. Ingest more of the "
                "archive before using drafts from it.")
        if guess_type and letter_type == guess_type:
            # Warn whether the guess came from the keyword rules or the
            # model: the rules are what generated the labels the model was
            # trained and measured on, so 0.64 macro-F1 bounds both.
            warnings.append(
                f"letter type '{letter_type}' was guessed "
                f"({how['letter_type_source']}), not supplied. Phase 3 "
                "cross-validated letter type at 0.64 macro-F1, below the 0.85 "
                "bar, so confirm it before relying on the skeleton it chose.")
        if department is None:
            warnings.append(
                "no department identified; retrieval is searching the whole "
                "corpus, which measurably lowers precision (Phase 4: the "
                "department filter is worth about +0.10)")

        hits = self.retriever.search(
            request, limit=self.max_exemplars,
            filters=Filters(department=department, min_trust=self.min_trust),
            use=("bm25", "tfidf"))
        if not hits and department:
            warnings.append(f"nothing found in {department}; searched wider")
            hits = self.retriever.search(
                request, limit=self.max_exemplars,
                filters=Filters(min_trust=self.min_trust), use=("bm25", "tfidf"))

        exemplars = [h.letter.text for h in hits]
        sources = [(h.letter.id,
                    f"{h.letter.department or '?'}/{h.letter.letter_type or '?'} "
                    f"trust {h.letter.trust:.2f} score {h.score:.4f}")
                   for h in hits]
        if not exemplars:
            warnings.append(
                "no exemplars retrieved: the draft is unguided by the archive "
                "and is little better than asking the model cold")

        skeleton = self.skeletons.get((department or "", letter_type or ""))
        if skeleton is None:
            skeleton = self.skeletons.get(FALLBACK_CELL)
            if skeleton is not None:
                warnings.append(
                    "no skeleton for this category, so the office-wide "
                    "letterhead was used instead. A skeleton of its own needs "
                    "8+ past letters in the same (department, type) cell -- "
                    "check the addressee and the branch code by eye.")
            else:
                warnings.append(
                    "no skeleton at all, so this draft has NO letterhead, "
                    "letter number or date. Run `latters templates --db "
                    "corpus.db -o skeletons`; if that mines nothing, the "
                    "corpus is too small.")

        parts = build_prompt(request, exemplars, skeleton=skeleton,
                             subject=subject, budget=self.budget,
                             max_exemplars=self.max_exemplars)
        if parts.dropped_exemplars:
            warnings.append(
                f"dropped {parts.dropped_exemplars} exemplar(s) to fit the "
                f"{self.budget.context}-token context")

        completion = self.llm.generate(parts.prompt, system=parts.system,
                                       options=options)
        body, removed = clean_body(completion.text)
        invented = unsupported_facts(body, request, exemplars)
        text = assemble(body, skeleton=skeleton, subject=subject,
                        letter_number=letter_number, on=on)
        q = assess(body)

        if completion.truncated:
            warnings.append(
                "the model hit its output limit mid-letter; raise num_predict "
                "or shorten the request")
        if q.verdict != "clean":
            warnings.append(
                f"the generated Hindi has sequence problems ({q.violations}); "
                "small models do produce malformed Devanagari")

        return Draft(
            text=text, body=body, department=department, letter_type=letter_type,
            subject=subject, sources=sources, removed_from_model_output=removed,
            unsupported_numbers=invented, quality_score=q.score,
            quality_verdict=q.verdict, thin_corpus=thin,
            prompt_tokens=completion.prompt_tokens,
            output_tokens=completion.output_tokens,
            seconds=completion.wall_seconds, truncated=completion.truncated,
            warnings=warnings)


def build_service(store, llm, *, skeleton_overrides: str | None = None,
                  **kw) -> "DraftService":
    """Wire retrieval, the trained classifier and the skeletons together."""
    from .classify import TrainedClassifier
    from .retrieve import Retriever, TfidfIndex, load_letters

    letters = load_letters(store.db)
    retriever = Retriever(store.db, letters, tfidf=TfidfIndex().fit(letters))
    rows = [(r["text"], r["department"], r["letter_type"])
            for r in store.db.execute(
                "SELECT text, department, letter_type FROM letters")]
    return DraftService(retriever=retriever, llm=llm,
                        skeletons=load_skeletons(store.db,
                                                  overrides=skeleton_overrides),
                        classifier=TrainedClassifier.fit(rows), **kw)


#: Key for the office-wide skeleton used when a cell has too few letters
#: to mine one of its own. Not a real (department, type) pair, and cannot
#: collide with one: `classify` never produces an empty label.
FALLBACK_CELL = ("", "")


def load_skeletons(db, *, overrides: str | None = None
                   ) -> dict[tuple[str, str], Skeleton]:
    """Mine skeletons from a corpus database, keyed by cell.

    Human-corrected files in `overrides` replace the mined ones. Mining
    cannot recover line order reliably from a heterogeneous cell; a person
    who knows the office's letters can fix it once, in five minutes, and
    should not have their work overwritten on the next ingest.
    """
    from .classify import UNLABELLED, bootstrap
    from .template import mine_all

    rows = []
    for r in db.execute("SELECT text, department, letter_type FROM letters"):
        dept, ltype = r["department"], r["letter_type"]
        if not dept or not ltype:
            b = bootstrap(extract(r["text"]), r["text"])
            dept = dept or (None if b.department == UNLABELLED else b.department)
            ltype = ltype or (None if b.letter_type == UNLABELLED else b.letter_type)
        if dept and ltype:
            rows.append((r["text"], dept, ltype))
    mined = {(s.department, s.letter_type): s for s in mine_all(rows)}

    # An OFFICE-WIDE fallback, mined across every letter regardless of cell.
    #
    # Without it, a request in a category with fewer than 8 past letters got
    # no letterhead at all: no पत्रांक line, no दिनांक, no `सेवा में` block --
    # just a subject, a body and a closing. A departmental letter without a
    # letter number and a date is not a letter, whatever the body says.
    #
    # The warning shown in that case already claimed "the letterhead and
    # closing are generic", which was simply untrue: there was no letterhead.
    # This makes the warning honest. The office's letterhead barely varies
    # between cells -- it is the same office -- so mining it across all of
    # them is both safe and what the warning promised all along.
    if rows:
        office_wide = mine_all([(t, "", "") for t, _, _ in rows], min_cell=1)
        if office_wide:
            mined[FALLBACK_CELL] = office_wide[0]
    if overrides:
        from pathlib import Path as _Path
        from .template import load_overrides
        # A mistyped --skeletons path used to be ignored in silence, so the
        # clerk's corrections appeared to have had no effect.
        if not _Path(overrides).is_dir():
            raise FileNotFoundError(
                f"no skeleton directory at {overrides}. Create one with: "
                "latters templates --db <db> -o <dir>")
        loaded = load_overrides(overrides)
        # An EMPTY directory and a directory full of unreadable files are
        # different problems and used to raise the same error.
        #
        # Empty is the normal state of a small office: `templates` writes
        # nothing until a cell reaches 8 letters, so a corpus of 48 letters
        # spread over six categories produces zero files -- and refusing
        # here meant that office could not draft at all, even though the
        # skeletons mined from the database above, including the
        # office-wide fallback, were sitting right there unused.
        #
        # Files that exist and cannot be parsed stay an error: the clerk
        # edited them, and silently ignoring that work is the failure this
        # guard was written for.
        if not loaded:
            present = [f for f in _Path(overrides).glob("*.md")]
            if present:
                raise ValueError(
                    f"{overrides} holds {len(present)} .md file(s) and none "
                    "could be read. Each needs its `# department / type` "
                    "heading and the `## above the subject line` / "
                    "`## below the body` sections intact.")
            # Empty: say so, carry on with what was mined.
            import warnings as _w
            _w.warn(
                f"{overrides} is empty, so the letterhead comes from the "
                f"corpus rather than from any corrected file. `latters "
                f"templates -o {overrides}` writes nothing until a "
                f"(department, type) cell has 8+ letters.",
                stacklevel=2)
        mined.update(loaded)
    return mined
