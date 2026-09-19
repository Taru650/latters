"""Phase 0 -- measure the cost of running embeddings inside Ollama.

Settles one concrete question: if the embedding model lives in Ollama
alongside the drafting model, does serving an embedding request evict the
drafting model and force a reload from disk?

Why it matters on this machine
------------------------------
Ollama keeps a bounded number of models resident (``OLLAMA_MAX_LOADED_MODELS``,
and it will unload under memory pressure regardless). With 8 GB total and
roughly 3 GB free, a 1-2.6 GB drafting model plus a 0.6 B embedding model does
not comfortably fit. If the embedder evicts the drafter, the *next* draft pays
a full cold load -- measured at 30 s for a 1 B model and 100 s for a 1.7 B one
on this laptop's disk. A retrieval step that silently costs a 30-100 s reload
is not a slow feature, it is a broken one.

The alternative in the plan is a split runtime: the LLM stays in Ollama on the
CPU, and embeddings run in ONNX Runtime on the GPU via DirectML -- separate
process, separate memory, separate silicon, and the two can run at the same
time. This script produces the number that justifies (or refutes) that split.

Run:
    python scripts/phase0_embed.py --llm gemma3:1b --embed qwen3-embedding:0.6b
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

HOST = "http://127.0.0.1:11434"

HINDI_CHUNK = (
    "कार्यालय जिला शिक्षा अधिकारी, रायपुर। पत्र संख्या शिक्षा/समीक्षा/2024/118, "
    "दिनांक 15.03.2024। विषय: मासिक समीक्षा बैठक की सूचना। महोदय, उपरोक्त विषय के "
    "संदर्भ में सूचित किया जाता है कि दिनांक 25.03.2024 को प्रातः 11:00 बजे "
    "कार्यालय सभाकक्ष में मासिक समीक्षा बैठक आयोजित की जा रही है।"
)


def _post(path: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        HOST + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ns(v) -> float:
    return (v or 0) / 1e9


def ollama_ps() -> str:
    try:
        return subprocess.run(["ollama", "ps"], capture_output=True, text=True,
                              timeout=90).stdout.strip()
    except subprocess.TimeoutExpired:
        return ("(`ollama ps` did not answer within 90s -- the server is still "
                "paging models in or out. A timeout here is itself a result: "
                "the machine is in disk thrash, not merely busy.)")
    except Exception as exc:
        return f"(ollama ps failed: {exc})"


def draft(model: str, label: str) -> float:
    """One short generation. Returns load_duration in seconds.

    A non-zero load_duration on a model that was already resident means it was
    evicted and re-read from disk -- which is exactly the failure being tested.
    """
    r = _post("/api/generate", {
        "model": model, "prompt": "एक वाक्य में बैठक की सूचना लिखिए।",
        "stream": False, "keep_alive": "5m",
        "options": {"num_ctx": 2048, "num_predict": 24, "temperature": 0.3},
    })
    load_s = ns(r.get("load_duration"))
    print(f"  {label:34} load_duration = {load_s:6.2f}s   "
          f"total = {ns(r.get('total_duration')):6.2f}s")
    return load_s


def _utf8_console() -> None:
    """Windows encodes stdout with the console code page (usually cp1252),
    so the first Devanagari character aborts the script. See cli._prepare_streams."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    _utf8_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="gemma3:1b")
    ap.add_argument("--embed", default="qwen3-embedding:0.6b")
    ap.add_argument("--chunks", type=int, default=32,
                    help="embedding batch size, as at corpus ingest")
    args = ap.parse_args(argv)

    try:
        urllib.request.urlopen(HOST + "/api/tags", timeout=10).read()
    except Exception as exc:
        print(f"cannot reach Ollama at {HOST}: {exc}", file=sys.stderr)
        return 2

    print(f"LLM: {args.llm}   embedder: {args.embed}\n")

    print("1. warm the drafting model")
    draft(args.llm, "first call (cold)")
    draft(args.llm, "second call (should be warm)")
    print("\n   ollama ps:\n" + "\n".join("   " + l for l in ollama_ps().splitlines()))

    print(f"\n2. embed {args.chunks} chunks through Ollama")
    t0 = time.perf_counter()
    try:
        r = _post("/api/embed", {"model": args.embed,
                                 "input": [HINDI_CHUNK] * args.chunks,
                                 "keep_alive": "5m"})
    except urllib.error.HTTPError as exc:
        print(f"   embed failed: {exc} -- is {args.embed} pulled?", file=sys.stderr)
        return 1
    dt = time.perf_counter() - t0
    dims = len(r["embeddings"][0]) if r.get("embeddings") else 0
    print(f"   {args.chunks} chunks in {dt:.2f}s = {args.chunks/dt:.1f} chunks/s, "
          f"{dims} dimensions")
    print(f"   embed model load_duration = {ns(r.get('load_duration')):.2f}s")
    print("\n   ollama ps:\n" + "\n".join("   " + l for l in ollama_ps().splitlines()))

    print("\n3. draft again -- did the embedder evict the drafting model?")
    reload_s = draft(args.llm, "post-embed call")

    print("\n--- verdict ---")
    if reload_s > 1.0:
        print(f"EVICTED. The drafting model was re-read from disk, costing "
              f"{reload_s:.1f}s.\n"
              "Every retrieval step in production would add that to the user's\n"
              "wait. Do NOT run the embedder in Ollama on this machine. Use the\n"
              "split runtime: Ollama (CPU) for the LLM, ONNX Runtime + DirectML\n"
              "(GPU) for embeddings, as planned in Phase 4.")
    else:
        print("No eviction on this run -- both models stayed resident.\n"
              "That is not a clearance: it depends on what else is open. Re-run\n"
              "with a browser and Word open, which is the real desktop state.\n"
              "Check the RAM headroom in `ollama ps` above against the budget in\n"
              "docs/IMPLEMENTATION_PLAN.md before relying on it.")
    print("\nCompare against the split-runtime path: measure GPU embedding with\n"
          "scripts/phase0_directml.py, where the encoder sits in VRAM and costs\n"
          "the drafting model no system RAM and no CPU threads at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
