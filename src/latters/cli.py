"""``latters`` command line -- Phase 0 and Phase 1 surface.

    latters fonts tables                 list mapping tables
    latters fonts convert  [-t TABLE]    convert text on stdin / --text
    latters fonts gold     [PATHS...]    run the gold-set regression
    latters inventory PATH               Phase 1.1 archive triage
    latters ingest PATH                  convert an archive to Unicode + scores

    latters gold extract ARCHIVE -o DIR  mine review sheet from the archive
    latters gold collect REVIEW.tsv      turn a filled sheet into a gold file
    latters gold coverage                which mapping slots are untested
    latters gold auto --docx A --pdf B   check the mapping against OCR
    latters gold run                     the regression (same as `fonts gold`)

    latters segment ARCHIVE --db X.db    convert, split into letters, score, store
    latters audit ARCHIVE                anchor firing and boundary diagnostics
    latters stats --db X.db              corpus health
    latters search QUERY --db X.db       lexical search over the corpus

    latters fields  ARCHIVE|--db X.db    field extraction coverage
    latters classify --db X.db           bootstrap labels, cross-validate, store
    latters templates --db X.db          mine per-cell skeletons

    latters retrieve REQUEST --db X.db   find the letters to draft from
    latters eval --db X.db               measure retrieval against baselines
    latters doctor                       check everything the app needs
    latters scorecard --db X.db          all three scorecards (Phase 7)
    latters backup --db X.db             consistent snapshot of the corpus

    latters draft REQUEST --db X.db      draft a letter from the archive
    latters serve --db X.db              the drafting and admin web pages
"""

from __future__ import annotations

import argparse
import json
import os
import re
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

#: Scans and photographs are here too, so `latters segment archive` sees
#: exactly what the admin upload box accepts. They diverged once and a
#: folder of scans silently produced an empty corpus from the CLI.
ARCHIVE_SUFFIXES = (".docx", ".doc", ".dot", ".rtf", ".pdf",
                    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _require_dir(path: str) -> Path | None:
    """An archive path that does not exist is a typo, not an empty archive.

    Reporting "0 files" for a mistyped path is the same output as a genuinely
    empty folder, so the mistake looks like a finding.
    """
    p = Path(path)
    if not p.exists():
        # Show where it actually looked. A bare "no such path: archive" makes
        # the reader work out that a relative path resolved against the
        # current directory; printing the resolved path makes it obvious.
        print(f"no such path: {p}\n"
              f"  looked in: {p.resolve()}\n"
              f"  current directory: {Path.cwd()}",
              file=sys.stderr)
        if not p.is_absolute():
            siblings = sorted(
                d.name for d in Path.cwd().iterdir()
                if d.is_dir() and not d.name.startswith((".", "__")))[:8]
            if siblings:
                print(f"  directories here: {', '.join(siblings)}",
                      file=sys.stderr)
        return None
    return p


def _require_db(path: str) -> str | None:
    """A corpus database must already exist for every command that reads one.

    sqlite3 creates a database on connect, and Store() creates the parent
    directory too, so a mistyped --db silently produced an EMPTY corpus that
    every later command then operated on quite happily. Only `segment`
    creates a database.
    """
    p = Path(path)
    if not p.exists():
        print(f"no corpus database at {p}\n"
              f"  looked in: {p.resolve()}\n"
              f"  Create one with:  latters segment <archive> --db {p}",
              file=sys.stderr)
        return None
    return path


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
    if _require_dir(args.archive) is None:
        return 2

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
def _convert_all(root: Path, args) -> list[tuple[Path, str, str, float]]:
    """Read everything readable. Returns (path, text, tier, confidence).

    The tier and confidence travel PER FILE, not per run. `--tier` used to
    label every letter in a batch identically, so a scanned PDF dropped into
    an archive of .docx was stored with docx trust -- the same bug the web
    upload path had, left behind in the CLI when OCR was added there first.
    """
    from .ocr import IMAGE_SUFFIXES

    out = []
    for path in _iter_files(root):
        suffix = path.suffix.lower()
        report = None
        try:
            if suffix == ".pdf" or suffix in IMAGE_SUFFIXES:
                from .ocr import read_image, read_pdf
                doc, report = (read_pdf(path) if suffix == ".pdf"
                               else read_image(path))
            else:
                doc = read_document(path)
        except Exception:
            continue
        text, _ = convert_document(doc, latin_digits=getattr(args, "latin_digits", False),
                                   rescue_latin=True)
        text, _ = repair_text(text)
        # An explicit --tier still wins for the formats that have no reader
        # of their own; OCR knows better than the flag about its own output.
        tier = report.tier if report is not None else getattr(args, "tier", "docx")
        out.append((path, text, tier, 1.0 if report is None else report.confidence))
    return out


def cmd_segment(args: argparse.Namespace) -> int:
    from .fields import extract as extract_fields
    from .segment import segment as split, trust, verdict
    from .store import LetterRow, Store

    if _require_dir(args.path) is None:
        return 2

    converted = _convert_all(Path(args.path), args)
    if not converted:
        print("nothing readable in that path", file=sys.stderr)
        return 1
    vocab = build_vocabulary([t for _, t, _, _ in converted],
                             min_count=args.vocab_min)

    rows: list[LetterRow] = []
    per_file: list[tuple[str, int, str]] = []
    for path, text, tier, penalty in converted:
        segments = split(text)
        per_file.append((path.name, len(segments), tier))
        for i, seg in enumerate(segments, 1):
            q = assess(seg.text, vocabulary=vocab or None)
            comp = seg.completeness()
            # Tesseract's own confidence multiplies the conversion score:
            # the validator only catches ILLEGAL Devanagari, and a
            # misrecognised word is usually legal Devanagari that is wrong.
            conv = q.score * penalty
            score = trust(conv, comp, tier)
            # The subject has to be stored here. It feeds the FTS subject
            # column and the retrieval evaluation's query set, and leaving it
            # NULL made both silently inert in a fresh pipeline.
            subject = extract_fields(seg.text).subject
            rows.append(LetterRow(
                source_file=path.name, seq=i, text=seg.text,
                start_line=seg.start_line, end_line=seg.end_line,
                source_tier=tier, conversion_confidence=conv,
                completeness=comp, trust=score, verdict=verdict(score),
                opened_by=seg.opened_by, form=seg.form.value,
                anchors=seg.anchors,
                violations=q.violations, missing=seg.missing(),
                subject=subject.value if subject else None))

    for name, n, tier in per_file:
        how = "" if tier == getattr(args, "tier", "docx") else f"  [{tier}]"
        print(f"  {name[:40]:42} {n:4d} letters{how}")
    buckets = Counter(r.verdict for r in rows)
    print(f"\n{len(rows)} letters from {len(converted)} file(s); "
          f"vocabulary {len(vocab)} words")
    for v in ("index", "review", "quarantine"):
        print(f"  {v:11} {buckets.get(v, 0)}")

    if args.db:
        with Store(args.db) as store:
            ins, dup = store.add(rows, refresh=args.refresh)
            verb = "re-scored" if args.refresh else "already present"
            print(f"\nstored {ins} new, {dup} {verb} -> {args.db}")
            print("  " + json.dumps(store.stats(), ensure_ascii=False))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    if _require_dir(args.path) is None:
        return 2

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
    forms: Counter[str] = Counter()
    lengths: list[int] = []
    completeness: list[float] = []
    longest = None
    for _, text, _, _ in converted:
        for line in tag_lines(text):
            fired.update(line.anchors)
        for seg in split(text):
            reasons[seg.opened_by] += 1
            forms[seg.form.value] += 1
            lengths.append(seg.body_chars)
            completeness.append(seg.completeness())
            if longest is None or seg.body_chars > longest.body_chars:
                longest = seg

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
        if lengths[-1] > 5 * pct(lengths, .95) and longest is not None:
            # A block of MERGED letters carries several complete anatomies --
            # two or more closings, two or more subjects. One genuinely long
            # order carries none of either. Saying "almost certainly several
            # letters" without checking was wrong on the first real archive
            # it met: the outlier was a single 9,000-character disciplinary
            # order.
            n_close = longest.anchors.get("closing", 0)
            n_subj = longest.anchors.get("subject", 0)
            merged = n_close >= 2 or n_subj >= 2
            print(f"\n!! The longest segment is {lengths[-1]} chars, far above "
                  f"p95 ({pct(lengths,.95)}).")
            if merged:
                print(f"!! It contains {n_close} closing(s) and {n_subj} "
                      "subject line(s), so it is several letters that never\n"
                      "!! got split. The closing anchors are missing a form "
                      "this office uses.")
            else:
                print(f"!! It contains {n_close} closing(s) and {n_subj} "
                      f"subject line(s) and reads as form '{longest.form.value}',\n"
                      "!! so it is most likely ONE long document rather than "
                      "a failed split.\n"
                      "!! Open it and confirm before changing any anchors.")

    print("\ndocument forms")
    for name, n in forms.most_common():
        print(f"  {name:12} {n:5d}  {n/total:5.1%}")
    print("  Orders are not letters: no addressee, no subject line, no "
          "valediction.\n  They are scored against their own anatomy, not a "
          "letter's. 'fragment'\n  means neither -- most likely a truncated "
          "segment.")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    if _require_db(args.db) is None:
        return 2

    from .store import Store
    with Store(args.db) as store:
        print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    if _require_db(args.db) is None:
        return 2

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
def _labelled_rows(db_path: str):
    from .classify import bootstrap
    from .fields import extract
    from .store import Store
    out = []
    with Store(db_path) as store:
        for row in store.db.execute("SELECT id, text FROM letters"):
            f = extract(row["text"])
            out.append((row["id"], row["text"], f, bootstrap(f, row["text"])))
    return out


def cmd_fields(args: argparse.Namespace) -> int:
    from .fields import extract

    if args.db and _require_db(args.db) is None:
        return 2
    if not args.db and (args.path is None or _require_dir(args.path) is None):
        return 2

    if args.db:
        texts = [r[1] for r in _labelled_rows(args.db)]
    else:
        texts = [t for _, t, _, _ in _convert_all(Path(args.path), args)]
    if not texts:
        print("nothing to read", file=sys.stderr)
        return 1

    tally = {k: Counter() for k in extract("").states()}
    for text in texts:
        for k, v in extract(text).states().items():
            tally[k][v] += 1
    n = len(texts)
    print(f"field extraction over {n} letters\n")
    print(f"  {'field':16} {'found':>7} {'blank':>7} {'absent':>7}   usable")
    for k, c in tally.items():
        print(f"  {k:16} {c['found']:7d} {c['blank']:7d} {c['absent']:7d}   "
              f"{(c['found'] + c['blank']) / n:>6.0%}")
    print("\n'blank' means the label is present with the value left to be filled")
    print("at dispatch -- 95% of a real archive looked like that. It is a usable")
    print("template, not a defect; 'absent' is the defect.")
    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    if _require_db(args.db) is None:
        return 2

    from .classify import (RARE_LABEL, UNLABELLED, coverage_gap,
                           cross_validate, department_features, fold_rare,
                           letter_type_features)
    from .store import Store

    rows = _labelled_rows(args.db)
    if not rows:
        print("no letters in that database", file=sys.stderr)
        return 1

    for name, attr, featurise in (
        ("DEPARTMENT", "department", department_features),
        ("LETTER TYPE", "letter_type", letter_type_features),
    ):
        X, y = [], []
        for _, text, f, b in rows:
            label = getattr(b, attr)
            if label != UNLABELLED:
                X.append(featurise(text, f))
                y.append(label)
        print(f"\n=== {name}: {len(X)}/{len(rows)} labelled ({len(X)/len(rows):.0%}) ===")
        if len(set(y)) < 2:
            print("   only one class present -- nothing to learn or evaluate")
            continue
        # Evaluate THE MODEL THAT SHIPS. TrainedClassifier.fit drops the
        # rare rows; cross-validating on the folded labels instead scored a
        # phantom 5-class model, reported an F1 for a class the shipped
        # classifier cannot emit, and understated the real macro-F1 by 0.147.
        folded, dropped = fold_rare(y, min_support=args.min_support)
        keep = [i for i, label in enumerate(folded) if label != RARE_LABEL]
        if len({folded[i] for i in keep}) < 2:
            print("   too few classes above the support floor to evaluate")
            continue
        print(cross_validate([X[i] for i in keep], [folded[i] for i in keep],
                             k=args.folds).render())

        # The dropped rows are not an evaluation detail -- they are whole
        # departments whose requests will be silently misfiled.
        gap = coverage_gap(y, min_support=args.min_support)
        if gap.folded:
            print()
            print(gap.render() if attr == "department" else
                  f"!! {gap.n_letters} letter(s) ({gap.share:.0%}) are in "
                  f"types the classifier cannot predict:\n"
                  f"!!   {', '.join(sorted(gap.folded))}")

    if args.write:
        with Store(args.db) as store:
            for rid, _, _, b in rows:
                store.db.execute(
                    "UPDATE letters SET department=?, letter_type=? WHERE id=?",
                    (None if b.department == UNLABELLED else b.department,
                     None if b.letter_type == UNLABELLED else b.letter_type, rid))
            store.db.commit()
        print(f"\nwrote bootstrapped labels to {args.db}")
        print("These are rule-derived, not human-verified. The cross-validation")
        print("above says how far they can be trusted.")
    return 0


def cmd_templates(args: argparse.Namespace) -> int:
    if _require_db(args.db) is None:
        return 2

    from .classify import UNLABELLED
    from .template import mine_all

    rows = [(text, b.department, b.letter_type)
            for _, text, _, b in _labelled_rows(args.db)
            if UNLABELLED not in (b.department, b.letter_type)]
    skeletons = mine_all(rows, min_cell=args.min_cell)
    if not skeletons:
        print(f"no (department, type) cell has {args.min_cell}+ letters. "
              "A skeleton mined from fewer is just those letters' quirks.",
              file=sys.stderr)
        return 1

    print(f"{len(rows)} labelled letters -> {len(skeletons)} cells "
          f"with {args.min_cell}+ letters\n")
    print(f"  {'department':14} {'type':20} {'n':>4} {'skeleton covers':>16}")
    for s in skeletons:
        print(f"  {s.department:14} {s.letter_type:20} {s.n_letters:4d} {s.coverage:>15.0%}")
    print("\n'skeleton covers' is the share of a typical letter the office's own")
    print("boilerplate already supplies. The model only has to write the rest,")
    print("which is what makes a 1B model viable at 9 Hindi words per second.")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for s in skeletons:
            safe = re.sub(r"[^\w\u0900-\u097F]+", "_", f"{s.department}_{s.letter_type}")
            (out / f"{safe}.md").write_text(s.render(), encoding="utf-8")
        print(f"\nwrote {len(skeletons)} skeleton(s) to {out}")
    if args.show:
        print("\n" + "=" * 72 + "\n")
        print(skeletons[0].render())
    return 0


# --------------------------------------------------------------------------
def _build_retriever(store, *, dense_model: str | None = None):
    from .retrieve import DenseIndex, Retriever, TfidfIndex, load_letters

    letters = load_letters(store.db)
    if not letters:
        return None, []
    tfidf = TfidfIndex().fit(letters)
    dense = None
    if dense_model:
        from .encoders import OnnxEncoder
        dense = DenseIndex(OnnxEncoder(dense_model)).fit(letters)
    return Retriever(store.db, letters, tfidf=tfidf, dense=dense), letters


def cmd_retrieve(args: argparse.Namespace) -> int:
    if _require_db(args.db) is None:
        return 2

    from .retrieve import Filters
    from .store import Store

    with Store(args.db) as store:
        retriever, letters = _build_retriever(store, dense_model=args.dense_model)
        if retriever is None:
            print("corpus is empty", file=sys.stderr)
            return 1
        use = tuple(args.use)
        filters = Filters(department=args.department, letter_type=args.letter_type,
                          min_trust=args.min_trust)
        hits = retriever.search(args.request, filters=filters,
                                limit=args.limit, use=use)

    if not hits:
        print("nothing matched. Try dropping --department, or lowering "
              "--min-trust.", file=sys.stderr)
        return 1
    for i, h in enumerate(hits, 1):
        l = h.letter
        headline = l.subject or l.text.splitlines()[0]
        print(f"\n[{i}] letter {l.id}  score {h.score:.4f}  trust {l.trust:.2f}")
        print(f"    {l.department or '?'} / {l.letter_type or '?'}   "
              f"{l.source_file}   ranks={h.ranks}")
        print(f"    {headline[:96]}")
        if args.full:
            for line in l.text.splitlines()[:12]:
                print(f"      {line[:96]}")
    print("\nTrust and source are shown on every hit deliberately: a draft is "
          "only\nas good as the letters it was built from, and the clerk has "
          "to be able\nto see which ones those were.")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    if _require_db(args.db) is None:
        return 2

    from .evaluate import build_queries, evaluate, random_baseline, render
    from .store import Store

    with Store(args.db) as store:
        retriever, letters = _build_retriever(store)
        if retriever is None:
            print("corpus is empty", file=sys.stderr)
            return 1
        queries = build_queries(letters, min_cell=args.min_cell)
        if not queries:
            n_subj = sum(1 for l in letters if l.subject)
            n_lab = sum(1 for l in letters if l.department and l.letter_type)
            print(
                f"nothing to evaluate: 0 of {len(letters)} letters qualify as "
                f"a query.\n"
                f"  with a subject line      {n_subj}\n"
                f"  with department and type {n_lab}\n"
                f"  needed: a subject AND {args.min_cell}+ letters sharing its "
                f"(department, type) cell\n"
                f"Run `latters classify --db {args.db} --write` first, and "
                f"lower --min-cell if the corpus is small.\n"
                "Printing an empty results table here would look like an "
                "evaluation that found nothing, rather than one that never "
                "ran.", file=sys.stderr)
            return 2
        if len(queries) < 30:
            print(f"only {len(queries)} evaluable queries -- too few to "
                  "distinguish configurations at two standard errors. Treat "
                  "what follows as indicative.", file=sys.stderr)
        print(f"{len(letters)} letters, {len(queries)} evaluable queries "
              f"(subject present, {args.min_cell}+ cell-mates)")

        configs = [
            ("bm25", ("bm25",), False, False),
            ("bm25 + dept", ("bm25",), True, False),
            ("tfidf", ("tfidf",), False, False),
            ("tfidf + dept", ("tfidf",), True, False),
            ("rrf(bm25,tfidf)", ("bm25", "tfidf"), False, False),
            ("rrf + dept", ("bm25", "tfidf"), True, False),
            ("rrf + dept + type", ("bm25", "tfidf"), True, True),
        ]
        for label, keep in (("FULL subject (verbatim -- favours lexical)", None),
                            (f"DEGRADED {args.degrade:.0%} (the realistic condition)",
                             args.degrade)):
            results = [random_baseline(letters, queries)]
            for name, use, fd, ft in configs:
                results.append(evaluate(retriever, queries, name=name, use=use,
                                        filter_department=fd, filter_type=ft,
                                        degrade_keep=keep))
            print(f"\n=== {label} ===")
            print(render(results))

    print("""
Both tasks are proxies, not the real one. Known-item recall is inflated
because the query is verbatim text from the target. Same-cell precision uses
the Phase 3 classifier's labels as ground truth, and those cross-validated at
0.81 macro-F1 for department and 0.64 for letter type -- so its ceiling is
well under 1.0 and some misses are mislabels.

The real evaluation is forty requests a clerk writes, with the letters they
would actually have wanted. Until that exists, read these as relative
comparisons between configurations, never as absolute quality.""")
    return 0


# --------------------------------------------------------------------------
def cmd_draft(args: argparse.Namespace) -> int:
    if _require_db(args.db) is None:
        return 2

    from .draft import Budget, build_service
    from .llm import Ollama, OllamaError, StubLLM
    from .store import Store

    llm = StubLLM(reply=args.stub_reply) if args.stub else Ollama(
        args.model, host=args.host,
        options={"num_ctx": args.num_ctx, "num_predict": args.num_predict,
                 "num_thread": args.threads, "temperature": args.temperature})
    if not args.stub and not llm.available():
        print(f"Ollama has no model '{args.model}' at {args.host}.\n"
              f"  ollama pull {args.model}\n"
              f"Or pass --stub to exercise everything except generation.",
              file=sys.stderr)
        return 2

    with Store(args.db) as store:
        if not store.count():
            print("corpus is empty; run `latters segment` first", file=sys.stderr)
            return 1
        try:
            service = build_service(
                store, llm,
            budget=Budget(context=args.num_ctx,
                          reserve_for_output=args.num_predict,
                          fertility=args.fertility),
            max_exemplars=args.exemplars, min_trust=args.min_trust,
                skeleton_overrides=args.skeletons)
        except (FileNotFoundError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        try:
            draft = service.draft(args.request, department=args.department,
                                  letter_type=args.letter_type, subject=args.subject)
        except OllamaError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    print("=" * 72)
    print(draft.text)
    print("=" * 72)
    print(f"\ncategory   {draft.department or '?'} / {draft.letter_type or '?'}")
    print(f"timing     {draft.seconds:.1f}s   "
          f"{draft.prompt_tokens} prompt + {draft.output_tokens} output tokens")
    print(f"quality    {draft.quality_score:.3f} ({draft.quality_verdict})")

    print(f"\ndrafted from {len(draft.sources)} past letter(s):")
    for lid, why in draft.sources:
        print(f"  letter {lid}  {why}")
    if not draft.sources:
        print("  (none -- this draft is not grounded in the archive)")

    if draft.removed_from_model_output:
        print(f"\nremoved from the model's output: "
              f"{', '.join(draft.removed_from_model_output)}")
    if draft.unsupported_numbers:
        print(f"\n!! NUMBERS THE MODEL INVENTED: "
              f"{', '.join(draft.unsupported_numbers)}")
        print("!! These appear nowhere in your request or in the source letters.")
        print("!! Check every one before this letter leaves the office.")
    for w in draft.warnings:
        print(f"\n!  {w}")

    if draft.needs_review:
        print("\n!! THIS DRAFT NEEDS REVIEW before it is used at all "
              "(see the warnings above).")
    print("\nThis is a draft. It must be read and corrected before dispatch.")
    return 1 if draft.needs_review else 0


# --------------------------------------------------------------------------
def cmd_gold_auto(args: argparse.Namespace) -> int:
    """Check the mapping tables against OCR, with no Hindi reader.

    Needs a PDF exported from the SAME .docx on a machine where the legacy
    fonts are installed -- Word renders the glyphs, Tesseract reads them,
    and the result never touches the mapping table. See goldauto.py.
    """
    from .goldauto import from_files

    for label, path in (("docx", args.docx), ("pdf", args.pdf)):
        if not Path(path).exists():
            print(f"no such {label}: {path}", file=sys.stderr)
            return 2

    result = from_files(Path(args.docx), Path(args.pdf),
                        latin_digits=args.latin_digits)
    print(result.render(show=args.show))
    for note in result.notes:
        print(f"\n  note: {note}")
    # Non-zero on a repeated disagreement: that is a mapping bug until a
    # Hindi reader says otherwise, and it should fail a check, not decorate
    # one.
    return 1 if result.systematic or result.agreement < 0.90 else 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Why does it not work yet? One command, in failure order.

    Deliberately does NOT require --db to exist: a missing corpus is the
    most likely thing to be wrong on a fresh install and refusing to run
    would withhold the diagnosis at the exact moment it is needed.
    """
    from .doctor import run

    report = run(db=args.db, skeletons=args.skeletons, model=args.model,
                 host=args.ollama, stub=args.stub)
    print(report.render())
    return 1 if report.blocking else 0


def cmd_scorecard(args: argparse.Namespace) -> int:
    """Phase 7: all three scorecards, loudly including the empty ones."""
    if _require_db(args.db) is None:
        return 2

    from .scorecard import build, render

    gold_dir = Path(args.gold) if args.gold else None
    if gold_dir is not None and _require_dir(args.gold) is None:
        return 2
    cards = build(args.db, gold_dir)
    print(render(cards))
    # Exit 1 when anything is unmeasured or failing. `latters scorecard`
    # belongs in whatever the office uses to decide the thing is ready, and
    # a command that returns 0 while two of three cards are empty would let
    # it pass.
    failing = [c for c in cards
               if not c.ok or (c.verdict or "").startswith(("NOT", "FAIL"))]
    return 1 if failing else 0


def cmd_backup(args: argparse.Namespace) -> int:
    """Phase 7: copy the corpus safely while the server may be writing.

    `copy corpus.db backup\` is what an office will do, and on a WAL
    database that is how you get a backup missing the last N transactions
    with no error anywhere. sqlite3's own backup API takes a consistent
    snapshot of a live database, so this is a one-liner that is correct
    rather than a one-liner that looks correct.
    """
    import sqlite3
    from datetime import datetime

    if _require_db(args.db) is None:
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    dest = out_dir / f"corpus-{stamp}.db"

    src = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        with sqlite3.connect(dest) as dst:
            src.backup(dst)
    finally:
        src.close()

    # Verify rather than assume. A backup nobody has opened is a hope.
    with sqlite3.connect(f"file:{dest}?mode=ro", uri=True) as check:
        integrity = check.execute("PRAGMA integrity_check").fetchone()[0]
        letters = check.execute("SELECT COUNT(*) FROM letters").fetchone()[0]
    if integrity != "ok":
        print(f"backup FAILED integrity check: {integrity}", file=sys.stderr)
        return 1

    size_mb = dest.stat().st_size / 1e6
    print(f"{dest}  {letters} letters  {size_mb:.1f} MB  integrity ok")

    if args.keep:
        old_backups = sorted(out_dir.glob("corpus-*.db"))[:-args.keep]
        for f in old_backups:
            f.unlink()
        if old_backups:
            print(f"removed {len(old_backups)} older backup(s), keeping "
                  f"{args.keep}")

    print("\nTo restore: stop the server, then copy this file over "
          "corpus.db.\nThe skeletons/ folder is the only other state; "
          "back that up too.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("the web pages need the optional extras:\n"
              "  pip install -e \".[web]\"", file=sys.stderr)
        return 2

    # Check the paths BEFORE binding the port. Without this a mistyped --db
    # served an empty corpus and a mistyped --skeletons served letters with no
    # letterhead -- both of which look like a working application until
    # somebody reads the output. Every other command already refuses; serve
    # is the one a non-technical user runs, so it is the one that matters.
    if _require_db(args.db) is None:
        return 2
    if args.skeletons and _require_dir(args.skeletons) is None:
        print("  (skeletons are optional: drop --skeletons to draft the body "
              "only)", file=sys.stderr)
        return 2

    from .web.app import create_app

    app = create_app(db=args.db, skeletons=args.skeletons, model=args.model,
                     host=args.ollama, stub=args.stub, exports=args.exports,
                     num_ctx=args.num_ctx)
    print(f"  drafting page   http://{args.bind}:{args.port}/")
    print(f"  admin page      http://{args.bind}:{args.port}/admin")
    print(f"  corpus          {args.db}")
    print(f"  model           {'stub (no LLM)' if args.stub else args.model}"
          + (f"  ({args.num_ctx}-token context)" if args.num_ctx else ""))
    if args.bind not in ("127.0.0.1", "localhost"):
        print("\n!! Bound to a non-loopback address. This application has no\n"
              "!! authentication and the corpus is official correspondence.\n"
              "!! Only do this on a trusted office LAN.", file=sys.stderr)
    # flush: stdout is block-buffered when it is not a terminal, so
    # `latters serve > log.txt` showed an empty log until the server
    # was killed -- the address you need is in that banner.
    print("\nCtrl-C to stop.", flush=True)
    uvicorn.run(app, host=args.bind, port=args.port, log_level=args.log_level)
    return 0


# --------------------------------------------------------------------------
def cmd_inventory(args: argparse.Namespace) -> int:
    if _require_dir(args.path) is None:
        return 2

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
    if _require_dir(args.path) is None:
        return 2

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
        for path, why in skipped[:10]:
            print(f"    {Path(path).name}: {why}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")
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

    ga = gsub.add_parser(
        "auto", help="check the mapping against OCR, no Hindi reader needed")
    ga.add_argument("--docx", required=True, help="one legacy-font letter")
    ga.add_argument("--pdf", required=True,
                    help="the SAME letter, exported to PDF from Word on a "
                         "machine with the legacy fonts installed")
    ga.add_argument("--latin-digits", action="store_true")
    ga.add_argument("--show", type=int, default=25)
    ga.set_defaults(func=cmd_gold_auto)

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
    seg.add_argument("--refresh", action="store_true",
                     help="re-score letters already in the corpus instead of "
                          "skipping them. Use after upgrading: dedupe is on "
                          "the TEXT, so an old corpus keeps NULL form and "
                          "subject for ever otherwise. Keeps hand corrections.")
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

    fl = sub.add_parser("fields", help="field extraction coverage")
    fl.add_argument("path", nargs="?", default=None)
    fl.add_argument("--db", default=None)
    fl.add_argument("--latin-digits", action="store_true")
    fl.set_defaults(func=cmd_fields)

    cl = sub.add_parser("classify", help="bootstrap labels and cross-validate")
    cl.add_argument("--db", required=True)
    cl.add_argument("--folds", type=int, default=5)
    cl.add_argument("--min-support", type=int, default=10)
    cl.add_argument("--write", action="store_true",
                    help="store the bootstrapped labels on each letter")
    cl.set_defaults(func=cmd_classify)

    tp = sub.add_parser("templates", help="mine per-cell skeletons")
    tp.add_argument("--db", required=True)
    tp.add_argument("--min-cell", type=int, default=8)
    tp.add_argument("-o", "--out", default=None, help="directory to write skeletons to")
    tp.add_argument("--show", action="store_true", help="print the largest skeleton")
    tp.set_defaults(func=cmd_templates)

    rt = sub.add_parser("retrieve", help="find the letters to draft from")
    rt.add_argument("request")
    rt.add_argument("--db", required=True)
    rt.add_argument("--limit", type=int, default=5)
    rt.add_argument("--department", default=None)
    rt.add_argument("--letter-type", default=None)
    rt.add_argument("--min-trust", type=float, default=0.6)
    rt.add_argument("--use", nargs="+", default=["bm25", "tfidf"],
                    choices=["bm25", "tfidf", "dense"])
    rt.add_argument("--dense-model", default=None,
                    help="directory holding an ONNX encoder (see encoders.py)")
    rt.add_argument("--full", action="store_true", help="print letter bodies")
    rt.set_defaults(func=cmd_retrieve)

    ev = sub.add_parser("eval", help="measure retrieval against baselines")
    ev.add_argument("--db", required=True)
    ev.add_argument("--min-cell", type=int, default=5)
    ev.add_argument("--degrade", type=float, default=0.5)
    ev.set_defaults(func=cmd_eval)

    dr = sub.add_parser("draft", help="draft a letter from the archive")
    dr.add_argument("request")
    dr.add_argument("--db", required=True)
    dr.add_argument("--model", default="gemma3:1b")
    dr.add_argument("--host", default="http://127.0.0.1:11434")
    dr.add_argument("--department", default=None)
    dr.add_argument("--letter-type", default=None)
    dr.add_argument("--subject", default=None)
    dr.add_argument("--exemplars", type=int, default=3)
    dr.add_argument("--min-trust", type=float, default=0.6)
    dr.add_argument("--num-ctx", type=int, default=4096)
    dr.add_argument("--num-predict", type=int, default=900)
    dr.add_argument("--threads", type=int, default=4)
    dr.add_argument("--temperature", type=float, default=0.3)
    dr.add_argument("--fertility", type=float, default=1.97,
                    help="tokens per Devanagari word for this model; "
                         "measured by scripts/phase0_bench.py")
    dr.add_argument("--skeletons", default=None,
                    help="directory of human-corrected skeletons, as written "
                         "by `latters templates -o`; these win over mined ones")
    dr.add_argument("--stub", action="store_true",
                    help="skip the LLM; exercises retrieval, skeleton, "
                         "assembly and the safety checks")
    dr.add_argument("--stub-reply", default=
                    "उपर्युक्त विषय के प्रसंग में कहना है कि आवश्यक कार्यवाही "
                    "सुनिश्चित करते हुए प्रतिवेदन इस कार्यालय को उपलब्ध कराएँ।")
    dr.set_defaults(func=cmd_draft)

    sv = sub.add_parser("serve", help="run the drafting and admin web pages")
    sv.add_argument("--db", required=True)
    sv.add_argument("--skeletons", default="skeletons")
    sv.add_argument("--exports", default="exports")
    sv.add_argument("--model", default="gemma3:1b")
    sv.add_argument("--ollama", default="http://127.0.0.1:11434")
    sv.add_argument("--bind", default="127.0.0.1",
                    help="loopback by default: there is no authentication")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--log-level", default="warning")
    sv.add_argument("--num-ctx", type=int, default=None,
                    help="context window; lower it (e.g. 2048) to fit a "
                         "bigger model in less RAM")
    sv.add_argument("--stub", action="store_true",
                    help="serve without an LLM; everything else works")
    sv.set_defaults(func=cmd_serve)

    dr = sub.add_parser("doctor", help="check everything the app needs")
    dr.add_argument("--db", default="corpus.db")
    dr.add_argument("--skeletons", default="skeletons")
    dr.add_argument("--model", default="gemma3:1b")
    dr.add_argument("--ollama", default="http://127.0.0.1:11434")
    dr.add_argument("--stub", action="store_true")
    dr.set_defaults(func=cmd_doctor)

    sc = sub.add_parser("scorecard", help="Phase 7: all three scorecards")
    sc.add_argument("--db", required=True)
    sc.add_argument("--gold", default=None,
                    help="directory of gold .tsv files (default: discovered)")
    sc.set_defaults(func=cmd_scorecard)

    bk = sub.add_parser("backup", help="consistent snapshot of the corpus")
    bk.add_argument("--db", required=True)
    bk.add_argument("-o", "--out", default="backups")
    bk.add_argument("--keep", type=int, default=0,
                    help="delete all but the newest N backups (0 = keep all)")
    bk.set_defaults(func=cmd_backup)

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


def _prepare_streams() -> None:
    """Make the console safe for Devanagari and for being piped.

    Two platform realities, both found by running the CLI rather than by
    reading it:

    * On Windows, Python encodes stdout with the console code page. In
      cmd.exe that is usually cp1252, and the first Devanagari character of a
      draft raises UnicodeEncodeError. The target machine for this project is
      a Windows laptop, so without this every command that prints a letter
      crashes there while working perfectly in development.
    * `latters templates | head` raises BrokenPipeError with a traceback when
      head exits. Office staff pipe to `head` and `more` constantly.

    `errors="replace"` rather than "strict": a console that genuinely cannot
    render Devanagari should print boxes, not abort the command. The files
    the tool writes are always UTF-8 regardless.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    _prepare_streams()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BrokenPipeError:
        # The reader went away (`| head`). Devnull stdout so the interpreter
        # does not report the same error again while flushing at exit.
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except OSError:
            pass
        return 0
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
