"""Measure draft quality on the target machine. Needs Ollama and a model.

WHY THIS SCRIPT EXISTS
----------------------
Everything else in Phase 5 is deterministic and tested: prompt budgeting,
stripping what the model was told not to write, assembling the skeleton,
flagging invented numbers. None of that says whether the *generated Hindi* is
any good, and that cannot be tested in a repository with no model.

THE METRIC THAT MATTERS IS EDITING EFFORT
-----------------------------------------
Not fluency, not a rubric score. The number that justifies this project to an
office is: how much of the draft does the clerk have to change before it goes
out? Character-level edit distance between the draft and the letter actually
dispatched, as a fraction of the dispatched letter's length.

    0.00-0.15   the draft is essentially usable
    0.15-0.40   faster than starting blank
    0.40-0.70   arguable
    > 0.70      slower than typing it, and the project has failed

To get that number you need pairs: a request, and the letter that was
actually sent. Until those exist this script reports what it can measure
without them -- latency, length, the safety checks, and how much of the draft
came from the skeleton rather than the model.

    python scripts/phase5_draft_eval.py --db corpus.db --model gemma3:1b
    python scripts/phase5_draft_eval.py --db corpus.db --pairs pairs.tsv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latters.draft import Budget, build_service  # noqa: E402
from latters.gold import levenshtein  # noqa: E402
from latters.llm import Ollama, OllamaError  # noqa: E402
from latters.store import Store  # noqa: E402

#: Requests spanning the categories a district office actually writes.
#: Replace these with real ones from your own office -- these are plausible,
#: not authentic, and a plausible request is not a test.
DEFAULT_REQUESTS = [
    "अंचल अधिकारी द्वारा दाखिल-खारिज में विलंब की जाँच हेतु प्रतिवेदन मांगना है",
    "कर्मचारी के विरुद्ध अनुशासनिक कार्यवाही हेतु कारण बताओ सूचना देनी है",
    "मासिक समीक्षा बैठक की सूचना सभी प्रखंड पदाधिकारियों को देनी है",
    "सरकारी भूमि पर अतिक्रमण हटाने हेतु अंचल अधिकारी को निर्देश देना है",
    "लोक शिकायत निवारण के परिवाद पर प्रतिवेदन मांगना है",
    "सेवानिवृत्त कर्मी के पेंशन भुगतान की स्वीकृति देनी है",
    "न्यायालय वाद में शपथ पत्र दाखिल करने हेतु प्रतिवेदन मांगना है",
    "बैंकों के किसान क्रेडिट कार्ड प्रदर्शन की समीक्षा हेतु बैठक बुलानी है",
]


def load_pairs(path: Path) -> list[tuple[str, str]]:
    """TSV: request<TAB>the letter that was actually dispatched."""
    out = []
    for row in csv.reader(path.read_text(encoding="utf-8-sig").splitlines(),
                          delimiter="\t"):
        if len(row) >= 2 and row[0].strip() and not row[0].startswith("#"):
            out.append((row[0].strip(), row[1].strip()))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--model", default="gemma3:1b")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--pairs", default=None,
                    help="TSV of request<TAB>dispatched-letter; without it, "
                         "editing effort cannot be measured")
    ap.add_argument("--skeletons", default=None)
    ap.add_argument("--fertility", type=float, default=1.97)
    ap.add_argument("--num-predict", type=int, default=900)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)

    llm = Ollama(args.model, host=args.host,
                 options={"num_predict": args.num_predict,
                          "num_thread": args.threads})
    if not llm.available():
        print(f"Ollama has no model '{args.model}'. Run: ollama pull {args.model}",
              file=sys.stderr)
        return 2

    pairs = load_pairs(Path(args.pairs)) if args.pairs else [
        (r, None) for r in DEFAULT_REQUESTS]
    if not args.pairs:
        print("!! No --pairs file, so EDITING EFFORT -- the only metric that\n"
              "!! justifies this project -- is not measured. What follows is\n"
              "!! latency and the safety checks only.\n")

    with Store(args.db) as store:
        service = build_service(store, llm,
                                budget=Budget(fertility=args.fertility,
                                              reserve_for_output=args.num_predict),
                                skeleton_overrides=args.skeletons)
        rows = []
        for i, (request, reference) in enumerate(pairs, 1):
            print(f"[{i}/{len(pairs)}] {request[:62]}…", flush=True)
            t0 = time.perf_counter()
            try:
                d = service.draft(request)
            except OllamaError as exc:
                print(f"    FAILED: {exc}", file=sys.stderr)
                continue
            wall = time.perf_counter() - t0
            effort = None
            if reference:
                effort = levenshtein(d.text, reference) / max(len(reference), 1)
            rows.append({
                "request": request, "draft": d, "wall": wall, "effort": effort,
                "skeleton_share": 1.0 - len(d.body) / max(len(d.text), 1),
            })
            print(f"    {wall:5.1f}s  {d.output_tokens:4d} tokens  "
                  f"quality {d.quality_score:.2f}  "
                  f"skeleton {rows[-1]['skeleton_share']:.0%}"
                  + (f"  EDIT EFFORT {effort:.2f}" if effort is not None else ""))
            if d.unsupported_numbers:
                print(f"    !! invented numbers: {d.unsupported_numbers}")

    if not rows:
        return 1
    print("\n" + "=" * 68)
    walls = sorted(r["wall"] for r in rows)
    print(f"drafts            {len(rows)}")
    print(f"latency           median {statistics.median(walls):.1f}s   "
          f"worst {walls[-1]:.1f}s")
    print(f"skeleton share    median "
          f"{statistics.median(r['skeleton_share'] for r in rows):.0%}  "
          "(the part the model did NOT have to write)")
    print(f"ungrounded drafts {sum(1 for r in rows if not r['draft'].sources)}")
    print(f"invented numbers  {sum(1 for r in rows if r['draft'].unsupported_numbers)}"
          f" of {len(rows)} drafts")
    print(f"malformed Hindi   "
          f"{sum(1 for r in rows if r['draft'].quality_verdict != 'clean')}"
          f" of {len(rows)}")
    print(f"truncated         {sum(1 for r in rows if r['draft'].truncated)}")

    efforts = [r["effort"] for r in rows if r["effort"] is not None]
    if efforts:
        med = statistics.median(efforts)
        print(f"\nEDITING EFFORT    median {med:.2f}   "
              f"best {min(efforts):.2f}   worst {max(efforts):.2f}")
        verdict = ("essentially usable" if med < 0.15 else
                   "faster than starting blank" if med < 0.40 else
                   "arguable" if med < 0.70 else
                   "SLOWER THAN TYPING IT -- the project has failed its purpose")
        print(f"                  {verdict}")
    else:
        print("\nEDITING EFFORT    not measured. Collect 30 pairs of "
              "(request, dispatched letter)\n"
              "                  into a TSV and pass --pairs. Nothing else "
              "here tells you\n                  whether the drafts are "
              "worth using.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
