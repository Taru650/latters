"""Phase 0 -- benchmark candidate models on the target laptop and write BASELINE.md.

Run on the machine that will host the assistant, with Ollama running:

    ollama pull gemma3:1b
    ollama pull qwen3:1.7b
    python scripts/phase0_bench.py --models gemma3:1b qwen3:1.7b --out docs/BASELINE.md

Standard library only, so it runs on a locked-down office machine.

What it measures, and why each number decides something
-------------------------------------------------------
cold load      Seconds to read the model off disk. On a spinning disk this is
               the number that makes the app feel broken. It is also the
               justification for keep_alive=-1.
prefill t/s    Prompt processing. The only stage where GPU offload might pay,
               so this is the before/after number for the Vulkan experiment.
               Each repeat gets a unique marker prepended, because Ollama
               caches prompt KV and an unvaried prompt makes runs 2..n report
               physically impossible rates. Anything above PLAUSIBLE_PREFILL_TPS
               is flagged as suspected cache reuse rather than reported.
decode t/s     Token generation -- memory-bandwidth-bound. This is what the
               user actually waits through, and what a second SODIMM moves.
fertility      Tokens emitted per Devanagari word. A tokenizer that costs 1.5x
               more tokens per Hindi word costs 1.5x the wall-clock on every
               draft and eats the context budget that holds the exemplars.
               Compare this across models before believing any benchmark table.
thread sweep   num_thread 2/4/8. Hyperthread siblings contend for L3 and the
               memory controller on a bandwidth-bound decode, so 8 is often
               slower than 4 on a 4-core part. Measure, do not assume.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

HOST = "http://127.0.0.1:11434"

#: Above this, a CPU-only prefill measurement is not believable on a laptop
#: part and almost certainly reflects prompt-cache reuse. An i7-8550U does
#: roughly 190 GFLOP/s with AVX2; a 1B Q4 model needs ~2 GFLOP per prompt
#: token, which caps honest prefill near 100 t/s.
PLAUSIBLE_PREFILL_TPS = 400.0

#: A realistic drafting prompt: Hindi instruction, Hindi exemplar, Hindi output.
#: Benchmarking on English prompts is how people end up surprised by Indic
#: throughput in production.
HINDI_PROMPT = """आप एक सरकारी कार्यालय के लिए पत्र तैयार करने वाले सहायक हैं।

नमूना पत्र:
कार्यालय जिला शिक्षा अधिकारी, रायपुर
पत्र संख्या: शिक्षा/समीक्षा/2024/118
दिनांक: 15.03.2024
सेवा में, समस्त प्राचार्य, शासकीय उच्चतर माध्यमिक विद्यालय
विषय: मासिक समीक्षा बैठक की सूचना।
महोदय, उपरोक्त विषय के संदर्भ में सूचित किया जाता है कि दिनांक 25.03.2024 को
प्रातः 11:00 बजे कार्यालय सभाकक्ष में मासिक समीक्षा बैठक आयोजित की जा रही है।
आपसे अनुरोध है कि नियत समय पर उपस्थित होकर विभागीय प्रगति प्रतिवेदन प्रस्तुत करें।
भवदीय, जिला शिक्षा अधिकारी

उपरोक्त नमूने की शैली और प्रारूप में, त्रैमासिक निरीक्षण बैठक हेतु एक नया
सूचना पत्र तैयार कीजिए।"""

def _post(path: str, payload: dict, timeout: float = 900.0) -> dict:
    req = urllib.request.Request(
        HOST + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def devanagari_words(text: str) -> int:
    import re
    return len(re.findall(r"[ऀ-ॿ]+", text))


def generate(model: str, prompt: str, *, num_thread: int | None = None,
             num_ctx: int = 4096, num_predict: int = 200,
             keep_alive: str | int = "5m") -> dict:
    options: dict = {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": 0.3}
    if num_thread:
        options["num_thread"] = num_thread
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": options, "keep_alive": keep_alive}
    # Qwen3 base tags emit <think> blocks by default, which on a 15W CPU burns
    # 20-60s the user never sees. Suppress it so the comparison is fair.
    if "qwen3" in model.lower() and "instruct" not in model.lower():
        payload["think"] = False
    t0 = time.perf_counter()
    r = _post("/api/generate", payload)
    r["_wall_s"] = time.perf_counter() - t0
    return r


def unload(model: str) -> None:
    """Evict the model so the next call measures a genuine cold load."""
    try:
        _post("/api/generate", {"model": model, "prompt": "", "keep_alive": 0}, timeout=120)
    except urllib.error.URLError:
        pass
    time.sleep(2)


def ns_to_s(v) -> float:
    return (v or 0) / 1e9


def measure(model: str, prompt: str, repeats: int, threads: list[int]) -> dict:
    out: dict = {"model": model, "errors": []}

    unload(model)
    try:
        cold = generate(model, prompt, num_predict=32)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        out["errors"].append(f"cold run failed: {exc}")
        return out
    out["cold_load_s"] = round(ns_to_s(cold.get("load_duration")), 2)
    out["cold_wall_s"] = round(cold["_wall_s"], 2)

    prefill, decode, samples, prompt_counts = [], [], [], []
    for i in range(repeats):
        # Defeat the prompt KV cache: without a unique prefix, every repeat
        # after the first reuses cached keys and reports a prefill rate that
        # is an artefact of the cache, not of the hardware.
        varied = f"[अनुरोध सं. {i}-{time.time_ns()}]\n" + prompt
        r = generate(model, varied)
        pe_n, pe_d = r.get("prompt_eval_count", 0), ns_to_s(r.get("prompt_eval_duration"))
        ev_n, ev_d = r.get("eval_count", 0), ns_to_s(r.get("eval_duration"))
        prompt_counts.append(pe_n)
        if pe_d > 0 and pe_n > 0:
            prefill.append(pe_n / pe_d)
        if ev_d > 0:
            decode.append(ev_n / ev_d)
        samples.append(r)

    last = samples[-1]
    out["prompt_tokens"] = last.get("prompt_eval_count")
    out["prompt_token_counts"] = prompt_counts
    if prefill:
        median_prefill = statistics.median(prefill)
        if median_prefill > PLAUSIBLE_PREFILL_TPS:
            out["prefill_tps"] = None
            out["prefill_suspect"] = round(median_prefill, 1)
            out["errors"].append(
                f"prefill measured at {median_prefill:.0f} t/s, above the "
                f"{PLAUSIBLE_PREFILL_TPS:.0f} t/s plausibility cap -- prompt "
                "cache reuse, not a real rate. Reported as unmeasured."
            )
        else:
            out["prefill_tps"] = round(median_prefill, 1)
    else:
        out["prefill_tps"] = None
    out["decode_tps"] = round(statistics.median(decode), 2) if decode else None
    out["decode_stdev"] = round(statistics.stdev(decode), 2) if len(decode) > 1 else 0.0

    # Fertility needs a prompt the server has never seen, or a partial cache
    # hit understates the token count and flatters the tokenizer.
    n_words = devanagari_words(prompt)
    out["devanagari_words"] = n_words
    fertility_probe = f"[{time.time_ns()}]\n" + prompt
    try:
        fr = generate(model, fertility_probe, num_predict=1)
        probe_words = devanagari_words(fertility_probe)
        out["fertility"] = round(fr.get("prompt_eval_count", 0) / probe_words, 2) if probe_words else None
    except Exception as exc:
        out["fertility"] = None
        out["errors"].append(f"fertility probe failed: {exc}")

    gen = last.get("response", "")
    out["sample_output"] = gen[:400]
    out["output_is_devanagari"] = round(
        sum(1 for c in gen if "ऀ" <= c <= "ॿ") / max(len(gen), 1), 3)
    out["leaked_think_block"] = "<think>" in gen

    out["threads"] = {}
    for t in threads:
        try:
            r = generate(model, prompt, num_thread=t, num_predict=120)
            ev_d = ns_to_s(r.get("eval_duration"))
            out["threads"][t] = round(r.get("eval_count", 0) / ev_d, 2) if ev_d else None
        except Exception as exc:
            out["threads"][t] = None
            out["errors"].append(f"num_thread={t}: {exc}")

    # --- the derived metrics that actually decide the model ---------------
    #
    # Raw t/s is not comparable across tokenizers. A model emitting 2x the
    # tokens per Hindi word at 1.5x the token rate is slower in the only unit
    # that matters to the user, and it also fits half as much retrieved
    # context in the same window.
    if out["decode_tps"] and out["fertility"]:
        out["hindi_words_per_sec"] = round(out["decode_tps"] / out["fertility"], 2)
        out["seconds_for_400_word_letter"] = round(400 / out["hindi_words_per_sec"], 1)
        out["hindi_words_per_4k_context"] = int(4096 / out["fertility"])
    if out["decode_tps"]:
        out["seconds_for_600_token_draft"] = round(600 / out["decode_tps"], 1)
    return out


def render(results: list[dict], meta: dict) -> str:
    L = ["# Phase 0 baseline", "",
         f"Generated {meta['when']} on `{meta['host']}` ({meta['platform']}).", "",
         "Produced by `scripts/phase0_bench.py`. Re-run this after any hardware or",
         "Ollama change; it is the regression baseline for every later phase.", "",
         "## Results", "",
         "| model | cold load (s) | decode t/s | ± | fertility | **Hindi words/s** | 400-word letter (s) | words per 4k ctx | prefill t/s |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        if r.get("errors") and r.get("decode_tps") is None:
            L.append(f"| `{r['model']}` | FAILED | | | | | | |")
            continue
        prefill = r.get("prefill_tps")
        prefill_cell = str(prefill) if prefill else (
            f"_cache ({r['prefill_suspect']})_" if r.get("prefill_suspect") else "-")
        L.append(
            f"| `{r['model']}` | {r.get('cold_load_s','?')} | {r.get('decode_tps','?')} | "
            f"{r.get('decode_stdev',0)} | {r.get('fertility','?')} | "
            f"**{r.get('hindi_words_per_sec','?')}** | "
            f"{r.get('seconds_for_400_word_letter','?')} | "
            f"{r.get('hindi_words_per_4k_context','?')} | {prefill_cell} |")

    L += ["", "## Thread sweep (decode t/s by `num_thread`)", "",
          "| model | " + " | ".join(f"{t} threads" for t in meta["threads"]) + " |",
          "|---" * (len(meta["threads"]) + 1) + "|"]
    for r in results:
        cells = " | ".join(str(r.get("threads", {}).get(t, "-")) for t in meta["threads"])
        L.append(f"| `{r['model']}` | {cells} |")

    L += ["", "## How to read this", "",
          "- **Hindi words/s** is the decision metric: `decode t/s ÷ fertility`.",
          "  Raw t/s is not comparable across tokenizers, because a token is not",
          "  the same amount of Hindi in two different vocabularies. Rank models",
          "  on this column, not on decode t/s.",
          "- **words per 4k ctx** is the same effect on the input side: it caps how",
          "  many retrieved exemplars fit in the prompt. Below roughly 900 words",
          "  you cannot fit two full letters plus an instruction, which forces the",
          "  skeleton-plus-slots design from Phase 3 rather than raw exemplars.",
          "- **prefill t/s** shown as `_cache (N)_` means the measurement was",
          "  rejected as prompt-cache reuse. It is not a hardware result.",
          "- **fertility** is prompt tokens per Devanagari word. Lower is strictly",
          "  better: it is a direct multiplier on draft latency and on how many",
          "  exemplars fit in the context window. A model that wins on benchmark",
          "  tables and loses here will still feel slower in this application.",
          "- **decode t/s** is memory-bandwidth-bound. If the probe reported a",
          "  single memory module, a second matched SODIMM should roughly double it",
          "  — check that before optimising anything in software.",
          "- **cold load** is disk. If it is tens of seconds, set `keep_alive: -1`",
          "  so it is paid once per boot, and put an SSD on the shopping list.",
          "- The **thread sweep** frequently peaks at the physical core count, not",
          "  the logical one. Pin `num_thread` to whatever wins here.", ""]

    for r in results:
        L += [f"### `{r['model']}` sample output", "", "```", r.get("sample_output", "(none)"), "```", ""]
        if r.get("leaked_think_block"):
            L += ["> **A `<think>` block leaked into the output.** Those tokens are",
                  "> generated at full cost and never shown to the user. Disable",
                  "> thinking mode or switch to an instruct-only tag before comparing",
                  "> this model's latency against the others.", ""]
        if r.get("errors"):
            L += ["> errors: " + "; ".join(r["errors"]), ""]

    L += ["## Decisions this baseline feeds", "",
          "- [ ] Model choice for Phase 5 (win on decode t/s *and* fertility *and* output quality)",
          "- [ ] `num_thread` value for the production Ollama options block",
          "- [ ] Whether to buy a second SODIMM (single-channel?) and an SSD (cold load?)",
          "- [ ] Whether the Vulkan backend helps prefill (re-run with `OLLAMA_VULKAN=1`)",
          "- [ ] Peak RAM headroom, which caps the model tier (see the budget in",
          "      `docs/IMPLEMENTATION_PLAN.md`)", ""]
    return "\n".join(L)


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
    global HOST
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["gemma3:1b", "qwen3:1.7b"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--out", default="docs/BASELINE.md")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--host", default=HOST)
    args = ap.parse_args(argv)
    HOST = args.host

    try:
        with urllib.request.urlopen(HOST + "/api/tags", timeout=10) as r:
            tags = json.loads(r.read())
    except Exception as exc:
        print(f"cannot reach Ollama at {HOST}: {exc}\nStart it with `ollama serve`.",
              file=sys.stderr)
        return 2
    have = {m["name"] for m in tags.get("models", [])}
    print("models installed:", ", ".join(sorted(have)) or "(none)")

    results = []
    for model in args.models:
        if model not in have and f"{model}:latest" not in have:
            print(f"! {model} not installed -- run `ollama pull {model}`", file=sys.stderr)
        print(f"\n=== {model} ===", flush=True)
        r = measure(model, HINDI_PROMPT, args.repeats, args.threads)
        results.append(r)
        print(f"  cold load {r.get('cold_load_s')}s | prefill {r.get('prefill_tps')} t/s | "
              f"decode {r.get('decode_tps')} t/s | fertility {r.get('fertility')}")

    meta = {"when": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "host": platform.node(), "platform": platform.platform(),
            "threads": args.threads}
    report = render(results, meta)

    from pathlib import Path
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nwrote {out}")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"meta": meta, "results": results}, ensure_ascii=False,
                       indent=2), encoding="utf-8",
            )
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
