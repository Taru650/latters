"""``latters`` command line -- Phase 0 and Phase 1 surface.

    latters fonts tables                 list mapping tables
    latters fonts convert  [-t TABLE]    convert text on stdin / --text
    latters fonts gold     [PATHS...]    run the gold-set regression
    latters inventory PATH               Phase 1.1 archive triage
    latters ingest PATH                  convert an archive to Unicode + scores

    latters gold extract ARCHIVE -o DIR  mine review sheet from the archive
    latters gold collect REVIEW.tsv      turn a filled sheet into a gold file
    latters gold coverage                which mapping slots are untested
    latters gold run                     the regression (same as `fonts gold`)

    latters segment ARCHIVE --db X.db    convert, split into letters, score, store
    latters audit ARCHIVE                anchor firing and boundary diagnostics
    latters stats --db X.db              corpus health
    latters search QUERY --db X.db       lexical search over the corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from . import __version__
from .extract import UnsupportedFormat, classify_font, convert_document, read_document
from .fonts.convert import Converter
from .fonts.tables import available_tables, load_table
from .gold import discover, run as run_gold
from .repair import repair as repair_text
from .validate import assess, build_vocabulary

ARCHIVE_SUFFIXES = (".docx", ".doc", ".dot", ".rtf", ".pdf")


def _iter_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in ARCHIVE_SUFFIXES and not p.name.startswith("~$")
    )


# --------------------------------------------------------------------------
def cmd_tables(args: argparse.Namespace) -> int:
    for name in available_tables():
        table = Converter(name).table
        print(f"{name:16} {len(table.mapping):4d} entries   "
              f"{', '.join(p.name for p in table.source_files)}")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    text = args.text if args.text is not None else sys.stdin.read()
    conv = Converter(args.table, latin_digits=args.latin_digits).convert(text)
    out = conv.text
    repairs = {}
    if args.repair:
        out, repairs = repair_text(out)
    if args.json:
        q = assess(out)
        print(json.dumps({
            "table": conv.table, "text": out, "unmapped": conv.unmapped,
            "repairs": repairs, "score": q.score, "verdict": q.verdict,
            "violations": q.violations,
        }, ensure_ascii=False, indent=2))
    else:
        print(out)
        if conv.unmapped:
            print(f"\n[unmapped: {conv.unmapped}]", file=sys.stderr)
    return 0


def cmd_gold(args: argparse.Namespace) -> int:
    paths = [Path(p) for p in args.paths] if args.paths else discover()
    if not paths:
        print("no gold files found; see src/latters/gold.py for how to build one",
              file=sys.stderr)
        return 2
    report = run_gold(paths)
    print(report.render(show=args.show))
    if report.char_accuracy < args.threshold:
        print(f"\nFAIL: char accuracy {report.char_accuracy:.4%} "
              f"below threshold {args.threshold:.2%}", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------
def cmd_gold_extract(args: argparse.Namespace) -> int:
    from .goldbuild import (mark_blind, mine_candidates, select_by_coverage,
                            write_review_docx, write_review_tsv)

    out_dir = Path(args.out)
    candidates, problems = mine_candidates(Path(args.archive))
    if not candidates:
        print("no legacy-font lines found. Either the archive is already "
              "Unicode, or the fonts in it are not recognised -- run "
              "`latters inventory` to see the font histogram.", file=sys.stderr)
        for p in problems[:10]:
            print("  " + p, file=sys.stderr)
        return 1

    chosen = select_by_coverage(candidates, args.count, seed=args.seed)
    n_blind = mark_blind(chosen, args.blind_fraction, seed=args.seed)
    docx = write_review_docx(chosen, out_dir / "review.docx",
                             unicode_font=args.unicode_font)
    tsv = write_review_tsv(chosen, out_dir / "review.tsv")

    all_keys: set[str] = set()
    for c in candidates:
        all_keys |= c.keys
    sel_keys: set[str] = set()
    for c in chosen:
        sel_keys |= c.keys

    print(f"mined      {len(candidates)} distinct legacy lines")
    print(f"selected   {len(chosen)} covering {len(sel_keys)}/{len(all_keys)} "
          f"of the mapping slots the archive actually uses")
    print(f"blind      {n_blind} rows ({args.blind_fraction:.0%}) with no suggestion shown")
    if problems:
        print(f"skipped    {len(problems)} file(s); first few:")
        for p in problems[:5]:
            print("  " + p)
    print(f"\nwrote {docx}\nwrote {tsv}")
    print("""
Next:
  1. Open review.docx on a machine with the legacy font INSTALLED. If the left
     column reads as Latin gibberish, the font is missing and the sheet is
     useless -- that column is the ground truth.
  2. A Hindi reader fills the `verdict_or_correction` column of review.tsv:
     `ok` if our conversion matches, otherwise the correct Hindi.
  3. latters gold collect review.tsv -o tests/gold/office_<name>.tsv""")
    return 0


def cmd_gold_collect(args: argparse.Namespace) -> int:
    from .goldbuild import collect_review, coverage, write_gold

    col = collect_review(Path(args.review))
    if not col.pairs:
        print("no usable pairs collected", file=sys.stderr)
        for p in col.problems[:20]:
            print("  " + p, file=sys.stderr)
        return 1

    print(f"pairs collected   {len(col.pairs)}")
    print(f"  approved as-is  {col.n_ok}")
    print(f"  typed an answer {col.n_typed}")
    print(f"  not reviewed    {col.n_blank}")

    sighted, blind = col.sighted_disagreement, col.blind_disagreement
    if sighted is not None and blind is not None:
        print(f"\ndisagreed with the converter:")
        print(f"  sighted  {col.n_sighted_disagree}/{col.n_sighted} ({sighted:.1%})")
        print(f"  blind    {col.n_blind_disagree}/{col.n_blind} ({blind:.1%})")
        # The blind rows are the control. If they disagree with the converter
        # far more often than the sighted rows, the sighted rows were being
        # approved rather than checked, and the set cannot be trusted.
        if blind > sighted * 2 + 0.05:
            print(
                "\n!! The blind rows were corrected far more often than the "
                "sighted ones.\n"
                "!! That is the signature of rubber-stamping: when our "
                "conversion was\n"
                "!! visible it was approved, and when it was hidden the reader "
                "disagreed.\n"
                "!! Treat this gold set as unreliable and re-review it blind.",
                file=sys.stderr)
        else:
            print("blind and sighted rates are consistent -- no sign of "
                  "rubber-stamping")
    elif blind is None:
        print("\nno blind rows in this sheet -- rerun `gold extract` with "
              "--blind-fraction to get the rubber-stamping control")

    if col.problems:
        print(f"\n{len(col.problems)} row(s) excluded:")
        for p in col.problems[:15]:
            print("  " + p)

    out = Path(args.out)
    write_gold(col, out, title=args.title or out.stem)
    print(f"\nwrote {out}")

    for table_name in sorted({t for _, _, t, _ in col.pairs}):
        cov = coverage(col.pairs, table_name)
        print(f"coverage[{table_name}]  {len(cov.covered)}/{cov.total} slots "
              f"({cov.fraction:.0%})")
    print("\nRun `latters gold run` to check the converter against it.")
    return 0


def cmd_gold_coverage(args: argparse.Namespace) -> int:
    from .gold import discover, load_pairs
    from .goldbuild import coverage

    paths = [Path(p) for p in args.paths] if args.paths else discover()
    pairs: list[tuple[str, str, str, str]] = []
    for p in paths:
        pairs.extend(load_pairs(p))
    if not pairs:
        print("no gold pairs found", file=sys.stderr)
        return 2

    for table_name in sorted({t for _, _, t, _ in pairs}):
        cov = coverage(pairs, table_name)
        print(f"\n{table_name}: {len(cov.covered)}/{cov.total} slots exercised "
              f"({cov.fraction:.0%}) by {len(pairs)} pairs")
        missing = cov.uncovered
        if missing:
            print(f"  {len(missing)} slot(s) NOT tested by any pair:")
            table = load_table(table_name)
            for key in missing[: args.show]:
                print(f"    {key!r:8} -> {table.mapping[key]}")
            if len(missing) > args.show:
                print(f"    ... and {len(missing) - args.show} more")
            print("  A pair count means nothing on its own. These slots are "
                  "unproven:\n  a wrong mapping in any of them would pass the "
                  "regression silently.")
    return 0


# --------------------------------------------------------------------------
def _convert_all(root: Path, args) -> list[tuple[Path, str]]:
    out = []
    for path in _iter_files(root):
        try:
            doc = read_document(path)
        except Exception:
            continue
        text, _ = convert_document(doc, latin_digits=getattr(args, "latin_digits", False),
                                   rescue_latin=True)
        text, _ = repair_text(text)
        out.append((path, text))
    return out


def cmd_segment(args: argparse.Namespace) -> int:
    from .segment import segment as split, trust, verdict
    from .store import LetterRow, Store

    converted = _convert_all(Path(args.path), args)
    if not converted:
        print("nothing readable in that path", file=sys.stderr)
        return 1
    vocab = build_vocabulary([t for _, t in converted], min_count=args.vocab_min)

    rows: list[LetterRow] = []
    per_file: list[tuple[str, int]] = []
    for path, text in converted:
        segments = split(text)
        per_file.append((path.name, len(segments)))
        for i, seg in enumerate(segments, 1):
            q = assess(seg.text, vocabulary=vocab or None)
            comp = seg.completeness()
            score = trust(q.score, comp, args.tier)
            rows.append(LetterRow(
                source_file=path.name, seq=i, text=seg.text,
                start_line=seg.start_line, end_line=seg.end_line,
                source_tier=args.tier, conversion_confidence=q.score,
                completeness=comp, trust=score, verdict=verdict(score),
                opened_by=seg.opened_by, anchors=seg.anchors,
                violations=q.violations, missing=seg.missing()))

    for name, n in per_file:
        print(f"  {name[:44]:46} {n:4d} letters")
    buckets = Counter(r.verdict for r in rows)
    print(f"\n{len(rows)} letters from {len(converted)} file(s); "
          f"vocabulary {len(vocab)} words")
    for v in ("index", "review", "quarantine"):
        print(f"  {v:11} {buckets.get(v, 0)}")

    if args.db:
        with Store(args.db) as store:
            ins, dup = store.add(rows)
            print(f"\nstored {ins} new, {dup} already present -> {args.db}")
            print("  " + json.dumps(store.stats(), ensure_ascii=False))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Are the anchors right for THIS office? Run before trusting any output.

    Every archive has house style. The sample archive this was first tuned
    against says महाशय where the patterns assumed महोदय, and प्रसंग where they
    assumed संदर्भ -- so two anchors never fired at all and the segmenter fell
    back on a heuristic for 45% of its boundaries. This command surfaces that
    in one pass instead of leaving it to be discovered downstream.
    """
    from .anchors import ANCHORS
    from .segment import segment as split, tag_lines

    converted = _convert_all(Path(args.path), args)
    fired: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    lengths: list[int] = []
    completeness: list[float] = []
    for _, text in converted:
        for line in tag_lines(text):
            fired.update(line.anchors)
        for seg in split(text):
            reasons[seg.opened_by] += 1
            lengths.append(seg.body_chars)
            completeness.append(seg.completeness())

    print(f"files {len(converted)}   letters {sum(reasons.values())}\n")
    print("anchor firing (lines)")
    for a in ANCHORS:
        n = fired.get(a.name, 0)
        flag = "   <-- NEVER FIRED: the pattern is wrong for this office" if not n else ""
        print(f"  {a.name:14} {n:6d}{flag}")

    total = sum(reasons.values()) or 1
    print("\nboundary cause")
    for r, n in reasons.most_common():
        print(f"  {r:28} {n:5d}  {n/total:5.1%}")
    fallback = reasons.get("repeated-subject", 0) / total
    if fallback > 0.15:
        print(f"\n!! {fallback:.0%} of boundaries came from the repeated-subject "
              "fallback.\n!! That means the closing anchors are not matching this "
              "office's sign-off\n!! formula. Fix those before trusting the "
              "segmentation -- the fallback\n!! is a safety net, not a "
              "segmentation strategy.")

    if lengths:
        lengths.sort(); completeness.sort()
        def pct(xs, p): return xs[min(len(xs) - 1, int(len(xs) * p))]
        print(f"\nletter length  p05={pct(lengths,.05)} median={pct(lengths,.5)} "
              f"p95={pct(lengths,.95)} max={lengths[-1]}")
        print(f"completeness   p10={pct(completeness,.1)} median={pct(completeness,.5)} "
              f"perfect={sum(1 for c in completeness if c >= 1.0)}")
        if lengths[-1] > 5 * pct(lengths, .95):
            print(f"\n!! The longest segment ({lengths[-1]} chars) is far above p95 "
                  f"({pct(lengths,.95)}).\n!! That is almost certainly several "
                  "letters that never got split.")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from .store import Store
    with Store(args.db) as store:
        print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from .store import Store
    with Store(args.db) as store:
        hits = store.search(args.query, limit=args.limit, min_trust=args.min_trust)
    if not hits:
        print("no matches")
        return 1
    for h in hits:
        first = h["text"].splitlines()[0][:70]
        print(f"[{h['id']:5}] trust={h['trust']:.2f} {h['source_file'][:26]:28} {first}")
    return 0


# --------------------------------------------------------------------------
def cmd_inventory(args: argparse.Namespace) -> int:
    """Phase 1.1: how much of this archive is legacy, and in which fonts?

    Run this before writing any other code against the archive. The histogram
    decides how big the conversion problem actually is and which font families
    have to be supported -- guessing at that is how projects like this get
    three months in before discovering the real distribution.
    """
    root = Path(args.path)
    files = _iter_files(root)
    fonts: Counter[str] = Counter()
    per_bucket: Counter[str] = Counter()
    unreadable: list[tuple[str, str]] = []

    for path in files:
        try:
            doc = read_document(path)
        except UnsupportedFormat as exc:
            per_bucket[path.suffix.lower().lstrip(".") + "-needs-conversion"] += 1
            unreadable.append((str(path), str(exc).split("\n")[0]))
            continue
        except Exception as exc:  # corrupt zip, truncated file, etc.
            per_bucket["unreadable"] += 1
            unreadable.append((str(path), f"{type(exc).__name__}: {exc}"))
            continue

        fonts.update(doc.font_histogram)
        tables = {classify_font(f)[0] for f in doc.font_histogram}
        if not doc.blocks:
            per_bucket["no-text-probably-scanned"] += 1
        elif tables - {None}:
            per_bucket["legacy"] += 1
        else:
            per_bucket["unicode-or-latin"] += 1

    total_chars = sum(fonts.values()) or 1
    report = {
        "root": str(root),
        "files_seen": len(files),
        "buckets": dict(per_bucket.most_common()),
        "fonts": [
            {
                "font": f, "chars": n, "share": round(n / total_chars, 4),
                "table": classify_font(f)[0], "match": classify_font(f)[1],
            }
            for f, n in fonts.most_common()
        ],
        "unreadable_sample": unreadable[:20],
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print(f"archive: {root}   files: {len(files)}\n")
    print("buckets")
    for k, v in per_bucket.most_common():
        print(f"  {k:28} {v:6d}")
    print("\nfonts by characters typed")
    print(f"  {'font':34} {'chars':>9} {'share':>7}  {'table':<12} match")
    for row in report["fonts"]:
        print(f"  {row['font'][:34]:34} {row['chars']:9d} {row['share']:6.1%}  "
              f"{str(row['table']):<12} {row['match']}")
    if unreadable:
        print(f"\nneeds external conversion or is unreadable: {len(unreadable)} file(s)")
        for p, why in unreadable[:10]:
            print(f"  {Path(p).name}: {why}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    root = Path(args.path)
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_files(root)
    converted: list[tuple[Path, str, dict]] = []
    skipped: list[tuple[Path, str]] = []

    for path in files:
        try:
            doc = read_document(path)
        except (UnsupportedFormat, Exception) as exc:
            skipped.append((path, str(exc).split("\n")[0]))
            continue
        text, used = convert_document(
            doc, latin_digits=args.latin_digits,
            force_table=args.table, rescue_latin=args.rescue_latin,
        )
        if args.repair:
            text, _ = repair_text(text)
        converted.append((path, text, {"tables": used, "warnings": doc.warnings}))

    # Vocabulary comes from the corpus itself: a conversion bug makes different
    # garbage each time, so a word repeated across many letters is real.
    vocab = build_vocabulary([t for _, t, _ in converted], min_count=args.vocab_min)

    rows = []
    for path, text, meta in converted:
        q = assess(text, vocabulary=vocab or None)
        rows.append({
            "file": str(path), "verdict": q.verdict, "score": q.score,
            "devanagari_ratio": q.devanagari_ratio, "vocab_hit_rate": q.vocab_hit_rate,
            "words": q.n_devanagari_words, "violations": q.violations,
            "tables": meta["tables"], "warnings": meta["warnings"], "chars": len(text),
        })
        if out_dir:
            (out_dir / (path.stem + ".txt")).write_text(text, encoding="utf-8")

    summary = Counter(r["verdict"] for r in rows)
    if args.json:
        print(json.dumps({"summary": dict(summary), "vocabulary_size": len(vocab),
                          "skipped": [[str(p), w] for p, w in skipped],
                          "documents": rows}, ensure_ascii=False, indent=2))
        return 0

    print(f"converted {len(rows)} file(s); vocabulary {len(vocab)} words")
    for verdict in ("clean", "review", "quarantine"):
        print(f"  {verdict:11} {summary.get(verdict, 0)}")
    if skipped:
        print(f"  {'skipped':11} {len(skipped)}")
    flagged = [r for r in rows if r["verdict"] != "clean"]
    if flagged:
        print("\nneeds attention")
        for r in sorted(flagged, key=lambda r: r["score"])[:20]:
            print(f"  {r['score']:.3f} {r['verdict']:11} {Path(r['file']).name}  {r['violations']}")
    if out_dir:
        print(f"\ntext written to {out_dir}")
    return 1 if summary.get("quarantine") else 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="latters", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    fonts = sub.add_parser("fonts", help="mapping tables and conversion")
    fsub = fonts.add_subparsers(dest="subcommand", required=True)

    fsub.add_parser("tables", help="list mapping tables").set_defaults(func=cmd_tables)

    c = fsub.add_parser("convert", help="convert legacy text to Unicode")
    c.add_argument("--text"); c.add_argument("-t", "--table", default="krutidev010")
    c.add_argument("--latin-digits", action="store_true",
                   help="leave ASCII digits alone instead of mapping to Devanagari")
    c.add_argument("--repair", action="store_true", help="apply post-conversion repairs")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_convert)

    g = fsub.add_parser("gold", help="run the gold-set regression")
    g.add_argument("paths", nargs="*")
    g.add_argument("--threshold", type=float, default=0.98)
    g.add_argument("--show", type=int, default=20)
    g.set_defaults(func=cmd_gold)

    gold = sub.add_parser("gold", help="build and audit the gold set")
    gsub = gold.add_subparsers(dest="subcommand", required=True)

    ge = gsub.add_parser("extract", help="mine a review sheet from the archive")
    ge.add_argument("archive")
    ge.add_argument("-o", "--out", default="review", help="output directory")
    ge.add_argument("-n", "--count", type=int, default=200,
                    help="how many lines to select (default 200)")
    ge.add_argument("--blind-fraction", type=float, default=0.2,
                    help="fraction of rows shown with no suggestion, as a "
                         "control against rubber-stamping (default 0.2)")
    ge.add_argument("--unicode-font", default="Mangal",
                    help="font for the Unicode column of review.docx")
    ge.add_argument("--seed", type=int, default=0)
    ge.set_defaults(func=cmd_gold_extract)

    gc = gsub.add_parser("collect", help="turn a filled review sheet into a gold file")
    gc.add_argument("review")
    gc.add_argument("-o", "--out", required=True)
    gc.add_argument("--title", default=None)
    gc.set_defaults(func=cmd_gold_collect)

    gv = gsub.add_parser("coverage", help="which mapping slots no pair tests")
    gv.add_argument("paths", nargs="*")
    gv.add_argument("--show", type=int, default=25)
    gv.set_defaults(func=cmd_gold_coverage)

    gr = gsub.add_parser("run", help="run the gold-set regression")
    gr.add_argument("paths", nargs="*")
    gr.add_argument("--threshold", type=float, default=0.98)
    gr.add_argument("--show", type=int, default=20)
    gr.set_defaults(func=cmd_gold)

    seg = sub.add_parser("segment", help="split an archive into individual letters")
    seg.add_argument("path")
    seg.add_argument("--db", default=None, help="SQLite file to store letters in")
    seg.add_argument("--tier", default="docx", choices=["unicode", "docx", "pdf", "ocr"])
    seg.add_argument("--latin-digits", action="store_true")
    seg.add_argument("--vocab-min", type=int, default=3)
    seg.set_defaults(func=cmd_segment)

    aud = sub.add_parser("audit", help="are the anchors right for this office?")
    aud.add_argument("path")
    aud.add_argument("--latin-digits", action="store_true")
    aud.set_defaults(func=cmd_audit)

    st = sub.add_parser("stats", help="corpus health")
    st.add_argument("--db", required=True)
    st.set_defaults(func=cmd_stats)

    se = sub.add_parser("search", help="lexical search over the corpus")
    se.add_argument("query"); se.add_argument("--db", required=True)
    se.add_argument("--limit", type=int, default=10)
    se.add_argument("--min-trust", type=float, default=0.6)
    se.set_defaults(func=cmd_search)

    inv = sub.add_parser("inventory", help="Phase 1.1 archive triage")
    inv.add_argument("path"); inv.add_argument("--json", action="store_true")
    inv.set_defaults(func=cmd_inventory)

    ing = sub.add_parser("ingest", help="convert an archive and score it")
    ing.add_argument("path")
    ing.add_argument("-o", "--out", help="directory for converted .txt files")
    ing.add_argument("-t", "--table", default=None,
                     help="force one table instead of resolving per-run fonts")
    ing.add_argument("--latin-digits", action="store_true")
    ing.add_argument("--rescue-latin", action="store_true",
                     help="pass suspected mis-fonted Latin spans through unconverted")
    ing.add_argument("--repair", action="store_true")
    ing.add_argument("--vocab-min", type=int, default=3)
    ing.add_argument("--json", action="store_true")
    ing.set_defaults(func=cmd_ingest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
