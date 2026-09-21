"""Three scorecards, tracked separately, and honest about the two that are empty.

The plan asks for conversion accuracy, retrieval, and drafting to be scored
separately rather than rolled into one number. They measure different things
and they fail for different reasons: a perfect retriever cannot rescue a
garbled corpus, and a clean corpus cannot rescue a model that will not write
departmental Hindi.

The design decision here is what to print when a scorecard has no data.

The tempting answer is to omit it, and that is how a project ends up
believing it has been evaluated. The rule this module follows is that **a
missing scorecard is printed as loudly as a failing one**, with the exact
command that would fill it. Two of the three are unfillable without work only
the office can do -- a Hindi reader for conversion, real dispatched letters
for drafting -- so the report has to keep saying so until they are done.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .effort import FAILED_ITS_PURPOSE, summarise
from .store import Store


@dataclass
class Card:
    """One scorecard. `rows` empty and `blocked` set means no data yet."""

    name: str
    question: str
    rows: list[tuple[str, str]] = field(default_factory=list)
    blocked: str | None = None
    how_to_fill: list[str] = field(default_factory=list)
    verdict: str | None = None

    @property
    def ok(self) -> bool:
        return self.blocked is None


def conversion_card(gold_dir: Path | None) -> Card:
    """Character accuracy of the legacy-font conversion, against the gold set."""
    card = Card(
        "conversion",
        "Is the Hindi we extracted from the archive actually the Hindi that "
        "was typed?")
    from . import gold as goldmod

    files = goldmod.discover(gold_dir)
    # The packaged seed set is a REGRESSION TEST FOR OUR OWN CONVERTER, not
    # evidence about anybody's archive: every pair in it was written by the
    # person who wrote the mapping table, from the same assumptions. The
    # first version of this card counted it and printed "usable, 100%", which
    # is precisely the self-deception the module docstring warns about -- so
    # the two are counted separately and only office pairs can clear the card.
    office = [f for f in files if not f.name.startswith("seed_")]
    seed = [f for f in files if f.name.startswith("seed_")]

    if seed:
        r = goldmod.run(seed)
        card.rows.append(
            ("seed regression (our own pairs)",
             f"{r.exact}/{r.n} exact, {r.char_accuracy:.4f} char"))

    if not office:
        card.blocked = (
            "no office gold set. Nothing in this project has ever checked "
            "the converted Hindi against a Hindi reader who knows this "
            "office's letters, and no automatic measurement substitutes: a "
            "mapping that is wrong produces well-formed Hindi that every "
            "check here accepts."
            + (" The seed pairs above only prove the converter still does "
               "what it did last week." if seed else ""))
        card.how_to_fill = [
            "latters gold extract <archive> -o review -n 200 --blind-fraction 0.2",
            "# a Hindi reader fills the verdict column, with the fonts installed",
            "latters gold collect review/review.tsv -o gold/office.tsv",
            "latters scorecard --db corpus.db",
        ]
        return card

    report = goldmod.run(office)
    card.rows += [
        ("office gold pairs", str(report.n)),
        ("exact-line matches",
         f"{report.exact}/{report.n} ({report.exact / max(report.n, 1):.1%})"),
        ("character accuracy", f"{report.char_accuracy:.4f}"),
        ("failing lines", str(len(report.failures))),
    ]
    # 0.98 characters is roughly one wrong character per two lines of a
    # letter -- already enough to make a reader distrust the whole corpus.
    card.verdict = ("usable" if report.char_accuracy >= 0.98 else
                    "NOT USABLE -- fix the mapping before trusting the corpus")
    return card


def retrieval_card(db: str) -> Card:
    """Does the retriever find the letters a drafter would have looked up?"""
    card = Card("retrieval",
                "Given a request, do the letters we retrieve resemble the one "
                "that should be written?")
    from .evaluate import build_queries, evaluate, random_baseline
    from .retrieve import Retriever, TfidfIndex, load_letters

    with Store(db) as store:
        letters = load_letters(store.db)
        if len(letters) < 30:
            card.blocked = (
                f"{len(letters)} letters indexed. Below about 30 there is "
                "nothing representative to retrieve, and any score would be "
                "measuring the fixture rather than the archive.")
            card.how_to_fill = ["latters segment <archive> --db corpus.db",
                                "latters classify --db corpus.db --write"]
            return card
        queries = build_queries(letters, min_cell=5)
        if not queries:
            card.blocked = (
                f"{len(letters)} letters, but none qualifies as a query: a "
                "query needs a subject line AND five letters sharing its "
                "(department, letter type) cell.")
            card.how_to_fill = [f"latters classify --db {db} --write"]
            return card
        retriever = Retriever(store.db, letters, tfidf=TfidfIndex().fit(letters))
        res = evaluate(retriever, queries, name="tfidf+dept",
                       use=("tfidf",), filter_department=True,
                       degrade_keep=0.5)
        base = random_baseline(letters, queries)

    card.rows = [
        ("queries", str(res.n)),
        ("known-item recall@5",
         f"{res.recall_at_5:.3f} +/- {res.stderr(res.recall_at_5):.3f}"),
        ("MRR", f"{res.mrr:.3f}"),
        ("same-cell P@5",
         "n/a (the department filter makes this definitionally 1.0)"
         if res.cell_metric_degenerate
         else f"{res.same_cell_p5:.3f} +/- {res.stderr(res.same_cell_p5):.3f}"),
        ("random baseline, same-cell P@5", f"{base.same_cell_p5:.3f}"),
        ("median latency", f"{res.median_ms:.0f} ms"),
    ]
    # Two standard errors is the bar used everywhere else in this project:
    # anything smaller is noise being reported as a finding.
    margin = 2 * res.stderr(res.same_cell_p5)
    card.verdict = (
        "n/a -- the degenerate metric cannot be compared to a baseline"
        if res.cell_metric_degenerate else
        "beats the random baseline"
        if res.same_cell_p5 - base.same_cell_p5 > margin else
        "NOT distinguishable from picking five letters at random")
    return card


def drafting_card(db: str) -> Card:
    """Editing effort -- the number the plan says decides success."""
    card = Card("drafting",
                "Is using this faster than typing the letter?")
    with Store(db) as store:
        counts = store.draft_stats()
        scores = store.effort_scores()

    summary = summarise(scores)
    if summary is None or summary.n < 10:
        have = summary.n if summary else 0
        card.blocked = (
            f"{have} dispatched draft(s) recorded; at least 10 are needed "
            "before a median means anything. This fills itself as the office "
            "uses the tool -- every export records what actually went out.")
        card.how_to_fill = [
            "latters serve --db corpus.db --skeletons skeletons",
            "# then draft, edit and export letters as normal work",
        ]
        if counts["generated"]:
            card.rows = [("drafts generated", str(counts["generated"])),
                         ("of those, exported", str(counts["dispatched"])),
                         ("abandoned", str(counts["abandoned"]))]
        return card

    card.rows = [
        ("dispatched drafts", str(summary.n)),
        ("drafts abandoned", str(counts["abandoned"])),
        ("median editing effort", f"{summary.median:.3f}"),
        ("90th percentile", f"{summary.p90:.3f}"),
        ("went out as drafted (<=0.10)", f"{summary.as_drafted}/{summary.n}"),
        (f"failed its purpose (>{FAILED_ITS_PURPOSE:.2f})",
         f"{summary.failed}/{summary.n}  ({summary.failure_rate:.0%})"),
    ]
    card.verdict = ("saving time"
                    if summary.median <= 0.30 else
                    "marginal -- letters need heavy editing"
                    if summary.median <= FAILED_ITS_PURPOSE else
                    "FAILING -- more is rewritten than kept; typing is faster")
    return card


def build(db: str, gold_dir: Path | None) -> list[Card]:
    return [conversion_card(gold_dir), retrieval_card(db), drafting_card(db)]


def render(cards: list[Card]) -> str:
    out: list[str] = []
    for c in cards:
        out.append(f"{c.name.upper()}")
        out.append(f"  {c.question}")
        out.append("")
        if c.blocked:
            out.append(f"  NO DATA -- {c.blocked}")
            if c.how_to_fill:
                out.append("")
                out.append("  to fill it:")
                out.extend(f"      {l}" for l in c.how_to_fill)
        for label, value in c.rows:
            out.append(f"    {label:<32} {value}")
        if c.verdict:
            out.append("")
            out.append(f"  verdict: {c.verdict}")
        out.append("")
        out.append("-" * 72)
    blocked = [c.name for c in cards if not c.ok]
    if blocked:
        out.append("")
        out.append(f"{len(blocked)} of 3 scorecards have no data: "
                   + ", ".join(blocked) + ".")
        out.append("This system has not been evaluated. Do not present it as "
                   "though it has.")
    return "\n".join(out)
