"""Settle whether a dense encoder earns its place, on the target machine.

Phase 4 found that BM25, character n-gram TF-IDF and their fusion are
statistically indistinguishable on 547 real letters, and that only the
department metadata filter measurably helped. A neural encoder was therefore
NOT adopted -- but it could not be tested in the development environment,
which has no access to huggingface.co.

This script runs that test where the model can actually be downloaded:

    pip install onnxruntime tokenizers
    # fetch granite-embedding-97m-multilingual-r2 (ONNX) to a folder
    python scripts/phase4_encoder_eval.py --db corpus.db --model ./granite-97m

DECISION RULE, fixed in advance so the result cannot be rationalised after
the fact: adopt the encoder only if it beats `tfidf + dept filter` on
same-cell precision@5 by more than two standard errors (printed below, about
0.06 at 300 queries) in the DEGRADED condition. The degraded condition is the
one that matters -- it is the case where a clerk phrases the request
differently from the original letter, which is exactly what a semantic model
is supposed to fix and what a lexical one is supposed to miss.

Also reported: index build time and resident memory, because on an 8 GB
machine those are part of the decision.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latters.encoders import RandomProjectionEncoder  # noqa: E402
from latters.evaluate import build_queries, evaluate, random_baseline, render  # noqa: E402
from latters.retrieve import DenseIndex, Retriever, TfidfIndex, load_letters  # noqa: E402
from latters.store import Store  # noqa: E402


def rss_mb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--model", default=None,
                    help="directory holding an ONNX encoder; omitted means "
                         "only the no-download baseline runs")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--min-cell", type=int, default=5)
    ap.add_argument("--degrade", type=float, default=0.5)
    args = ap.parse_args(argv)

    with Store(args.db) as store:
        letters = load_letters(store.db)
        queries = build_queries(letters, min_cell=args.min_cell)
        print(f"{len(letters)} letters, {len(queries)} evaluable queries")
        if len(queries) < 100:
            print("\n!! Under 100 queries: two standard errors will be wide "
                  "enough\n!! that this test cannot settle anything. Ingest "
                  "more of the archive first.")

        before = rss_mb()
        t0 = time.perf_counter()
        tfidf = TfidfIndex().fit(letters)
        print(f"\ntfidf index: {time.perf_counter() - t0:.1f}s, "
              f"{tfidf.nbytes / 1e6:.1f} MB")

        dense = None
        if args.model:
            from latters.encoders import OnnxEncoder
            t0 = time.perf_counter()
            enc = OnnxEncoder(args.model, threads=args.threads)
            load = time.perf_counter() - t0
            t0 = time.perf_counter()
            dense = DenseIndex(enc).fit(letters)
            build = time.perf_counter() - t0
            print(f"onnx encoder: loaded in {load:.1f}s, encoded "
                  f"{len(letters)} letters in {build:.1f}s "
                  f"({len(letters) / max(build, 1e-9):.1f}/s)")
            print(f"  projected for 5,000 letters: "
                  f"{build * 5000 / max(len(letters), 1) / 60:.1f} min")
            print(f"  resident memory grew {rss_mb() - before:.0f} MB")
        else:
            print("no --model given; using the random-projection encoder as a "
                  "floor.\nA trained model that cannot beat this is "
                  "misconfigured.")
            dense = DenseIndex(RandomProjectionEncoder()).fit(letters)

        retriever = Retriever(store.db, letters, tfidf=tfidf, dense=dense)

        for label, keep in (("FULL subject", None),
                            (f"DEGRADED {args.degrade:.0%} (the deciding condition)",
                             args.degrade)):
            results = [random_baseline(letters, queries)]
            for name, use, filt in [
                ("tfidf + dept  [incumbent]", ("tfidf",), True),
                ("dense + dept", ("dense",), True),
                ("rrf(tfidf,dense) + dept", ("tfidf", "dense"), True),
                ("rrf(all three) + dept", ("bm25", "tfidf", "dense"), True),
            ]:
                results.append(evaluate(retriever, queries, name=name, use=use,
                                        filter_department=filt, degrade_keep=keep))
            print(f"\n=== {label} ===")
            print(render(results))

            if keep is not None:
                base = next(r for r in results if "incumbent" in r.name)
                se = base.stderr(base.same_cell_p5)
                print(f"\n  DECISION: adopt a dense encoder only if a row beats "
                      f"{base.same_cell_p5:.3f}")
                print(f"  by more than {2 * se:.3f} (two standard errors). "
                      "Otherwise keep")
                print("  tfidf + dept filter: it needs no model, no download and "
                      "no extra RAM.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
