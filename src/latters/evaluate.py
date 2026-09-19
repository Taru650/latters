"""Retrieval evaluation, with its own limitations stated up front.

WHAT THIS MEASURES, AND WHAT IT CANNOT
--------------------------------------
The real task is: a clerk describes a letter they need, and the system
returns 3-5 past letters worth drafting from. Evaluating that properly needs
a human to say, for forty real requests, which past letters they would have
wanted. That judgement does not exist yet, so this module measures two
proxies and is explicit that they are proxies.

**Known-item.** Query with a letter's own subject line; can we find that
letter? This measures raw matching power. It is *badly biased toward lexical
methods*, because the query is verbatim text from the target document --
exactly the case BM25 is built for and exactly the case a clerk's paraphrase
is not. A high score here is close to meaningless on its own.

**Same-cell precision@5.** Query with a letter's subject, exclude the letter
itself, and ask how many of the top 5 come from the same (department,
letter-type) cell. This is much closer to the real task, because what the
drafting page needs is letters of the same kind, not one specific letter.
Its ground truth is the Phase 3 classifier's labels, which cross-validated at
0.81 macro-F1 for department and 0.64 for letter type -- so the ceiling here
is well below 1.0 and a "miss" is sometimes a mislabel.

**Degraded queries.** Both tasks are also run with half the subject's words
dropped at random, simulating a clerk who phrases the request differently
from how the original letter was written. This is where dense embeddings are
supposed to earn their keep, so the gap between the full and degraded
conditions is the number that decides whether a neural encoder is worth its
200 MB and its RAM.

Every result is reported against a random baseline, because precision@5 on a
corpus where one cell holds 15% of the letters is not impressive by itself.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .retrieve import Filters, Letter, Retriever

#: Devanagari punctuation lives INSIDE the Devanagari block: danda
#: U+0964, double danda U+0965, the abbreviation sign U+0970, the high
#: spacing dot U+0971. A naive [\u0900-\u097F] word class therefore makes
#: "।" a searchable term that matches every document in the corpus.
_WORD = re.compile(r"[ऀ-ॣ०-९ॲ-ॿA-Za-z0-9]+")


@dataclass
class Query:
    letter_id: int
    text: str
    department: str | None
    letter_type: str | None
    #: Other letters in the same cell; the same-cell target set.
    cell_mates: frozenset[int] = frozenset()


def degrade(text: str, *, keep: float = 0.5, seed: int = 0) -> str:
    """Drop words at random to simulate a differently-phrased request."""
    words = _WORD.findall(text)
    if len(words) < 4:
        return text
    rng = random.Random(seed)
    n = max(2, int(len(words) * keep))
    picked = sorted(rng.sample(range(len(words)), n))
    return " ".join(words[i] for i in picked)


def build_queries(letters: Sequence[Letter], *, min_cell: int = 5) -> list[Query]:
    """One query per letter that has a subject and enough cell-mates."""
    cells: dict[tuple, list[int]] = defaultdict(list)
    for l in letters:
        if l.department and l.letter_type:
            cells[(l.department, l.letter_type)].append(l.id)

    out = []
    for l in letters:
        if not l.subject or len(l.subject) < 12:
            continue
        key = (l.department, l.letter_type)
        mates = cells.get(key, [])
        if len(mates) < min_cell:
            continue
        out.append(Query(l.id, l.subject, l.department, l.letter_type,
                         frozenset(m for m in mates if m != l.id)))
    return out


@dataclass
class Result:
    name: str
    n: int
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    same_cell_p5: float = 0.0
    median_ms: float = 0.0
    #: True when the cell filter makes same-cell precision definitionally 1.0.
    cell_metric_degenerate: bool = False
    extra: dict = field(default_factory=dict)

    def stderr(self, value: float) -> float:
        """Binomial standard error, so differences can be read honestly.

        With ~300 queries and a rate near 0.5, one standard error is about
        0.029, so two configurations differing by less than ~0.06 are not
        distinguishable. Without this printed, small differences get reported
        as findings and tuning chases noise.
        """
        if self.n <= 1:
            return 0.0
        return (value * (1.0 - value) / self.n) ** 0.5


def _mrr_and_recall(ranked: Sequence[int], target: int, k: int = 5) -> tuple[float, int, int]:
    for i, doc in enumerate(ranked, 1):
        if doc == target:
            return 1.0 / i, int(i == 1), int(i <= k)
    return 0.0, 0, 0


def evaluate(retriever: Retriever, queries: Sequence[Query],
             *, name: str, use: Sequence[str] = ("bm25", "tfidf"),
             filter_department: bool = False, filter_type: bool = False,
             degrade_keep: float | None = None, limit: int = 5,
             min_trust: float = 0.0) -> Result:
    import time

    r1 = r5 = 0
    mrr = 0.0
    p5_total = 0.0
    p5_count = 0
    times = []

    for qi, q in enumerate(queries):
        text = degrade(q.text, keep=degrade_keep, seed=qi) if degrade_keep else q.text

        # Known-item: the target must be reachable, so it is not excluded.
        f = Filters(department=q.department if filter_department else None,
                    letter_type=q.letter_type if filter_type else None,
                    min_trust=min_trust)
        t0 = time.perf_counter()
        hits = retriever.search(text, filters=f, limit=limit, use=use)
        times.append((time.perf_counter() - t0) * 1000)
        m, a, b = _mrr_and_recall([h.letter.id for h in hits], q.letter_id, limit)
        mrr += m
        r1 += a
        r5 += b

        # Same-cell: exclude the source letter so the task is "find others
        # like this", not "find this".
        if q.cell_mates:
            f2 = Filters(department=q.department if filter_department else None,
                         letter_type=q.letter_type if filter_type else None,
                         min_trust=min_trust, exclude_ids=frozenset({q.letter_id}))
            hits2 = retriever.search(text, filters=f2, limit=limit, use=use)
            if hits2:
                p5_total += sum(1 for h in hits2 if h.letter.id in q.cell_mates) / len(hits2)
                p5_count += 1

    n = max(len(queries), 1)
    times.sort()
    return Result(name=name, n=len(queries),
                  recall_at_1=r1 / n, recall_at_5=r5 / n, mrr=mrr / n,
                  same_cell_p5=p5_total / p5_count if p5_count else 0.0,
                  # Filtering by letter_type makes "is the hit in the same
                  # cell" true by construction. The number is 1.0 and means
                  # nothing; flag it rather than print a fake win.
                  cell_metric_degenerate=filter_type,
                  median_ms=times[len(times) // 2] if times else 0.0)


def random_baseline(letters: Sequence[Letter], queries: Sequence[Query],
                    *, limit: int = 5, seed: int = 0) -> Result:
    """What you get by returning five letters at random.

    The honest floor for same-cell precision: on a corpus where one cell
    holds a sixth of everything, a meaningless retriever already scores
    well above zero.
    """
    rng = random.Random(seed)
    ids = [l.id for l in letters]
    r5 = 0
    p5 = 0.0
    p5_count = 0
    for q in queries:
        picked = rng.sample(ids, min(limit, len(ids)))
        r5 += int(q.letter_id in picked)
        if q.cell_mates:
            p5 += sum(1 for p in picked if p in q.cell_mates) / len(picked)
            p5_count += 1
    n = max(len(queries), 1)
    return Result(name="random", n=len(queries), recall_at_5=r5 / n,
                  same_cell_p5=p5 / p5_count if p5_count else 0.0)


def render(results: Sequence[Result], *, note: str = "") -> str:
    lines = [f"  {'configuration':30} {'R@1':>6} {'R@5':>6} {'MRR':>6} "
             f"{'cellP@5':>14} {'ms':>6}",
             "  " + "-" * 74]
    best = max((r.same_cell_p5 for r in results
                if not r.cell_metric_degenerate), default=0.0)
    for r in results:
        if r.cell_metric_degenerate:
            cell = "    n/a"
        else:
            cell = f"{r.same_cell_p5:.3f}±{r.stderr(r.same_cell_p5):.3f}"
        lines.append(f"  {r.name:30} {r.recall_at_1:6.3f} {r.recall_at_5:6.3f} "
                     f"{r.mrr:6.3f} {cell:>14} {r.median_ms:6.1f}")

    real = [r for r in results if not r.cell_metric_degenerate and r.n > 1]
    if real:
        se = max(r.stderr(r.same_cell_p5) for r in real)
        lines += ["",
                  f"  One standard error on cellP@5 is about {se:.3f}, so two rows",
                  f"  differing by less than ~{2 * se:.2f} are not distinguishable at this",
                  "  sample size. Do not tune on differences smaller than that."]
    if any(r.cell_metric_degenerate for r in results):
        lines += ["",
                  "  'n/a': filtering by letter_type makes same-cell precision true",
                  "  by construction. It would read 1.000 and mean nothing."]
    if note:
        lines += ["", note]
    return "\n".join(lines)
