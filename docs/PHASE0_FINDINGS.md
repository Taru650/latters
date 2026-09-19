# Phase 0 findings — measured on the target laptop

All numbers below were measured on the machine, not estimated. Where a
prediction in `IMPLEMENTATION_PLAN.md` was wrong, it is marked.

---

## 1. RAM is single-channel — confirmed, and it is the headline

```
DeviceLocator GB Speed
------------- -- -----
DIMM B         8  2400
```

**One module, in the second slot. Slot A is empty.**

| | Measured / implied |
|---|---|
| Configuration | 1 × 8 GB DDR4-2400, single-channel |
| Memory bandwidth | ≈ **19.2 GB/s** (not the 38.4 GB/s a matched pair gives) |
| Free RAM at idle | ≈ 3 GB of 7.9 GB |

**Prediction that was wrong:** the plan inferred dual-channel from Gemma 3 1B
hitting 17.83 tok/s, which is 76% of a 19.2 GB/s ceiling and looked too
efficient to be real. The inference was bad: Gemma 3's embedding/output matrix
is a large share of the 815 MB file but only one row of it is read per token,
so the bytes actually moved per token are well below the file size. Efficiency
against *file size* is not efficiency against *bandwidth*.

### One part fixes both of this project's binding constraints

A single matched 8 GB DDR4-2400 SODIMM in slot A:

- **Doubles memory bandwidth** → [Likely] 1.5–1.8× on token generation, since
  decode is bandwidth-bound but not perfectly so.
- **Doubles capacity to 16 GB** → unblocks the 4 B model tier, which the RAM
  budget in the plan rules out at 8 GB.

Nothing in software comes close. Buy it before optimising anything.

---

## 2. Disk is spinning — confirmed by transfer rates

`MediaType` is unreliable behind the Intel RST RAID controller, but the load
timings are unambiguous:

| Model | Size | Cold load | Effective rate |
|---|---:|---:|---:|
| `gemma3:1b` | 815 MB | 25–30 s | ~27 MB/s |
| `qwen3:1.7b` | 1.4 GB | 100.4 s | ~14 MB/s |
| `qwen3-embedding:0.6b` | ~1.2 GB | 113.1 s | ~11 MB/s |

No SSD does 11–27 MB/s. Consequences, both now mandatory:

- `keep_alive: -1` on the drafting model.
- Exactly one model resident at a time (see §4).

---

## 3. Model benchmark — Gemma 3 wins by 4.8× on the metric that matters

| | cold load | decode t/s | fertility | **Hindi words/s** | 400-word letter | words per 4k ctx |
|---|---:|---:|---:|---:|---:|---:|
| `gemma3:1b` | 29.7 s | 17.83 | **1.97** | **9.05** | ~44 s | ~2,080 |
| `qwen3:1.7b` | 100.4 s | 11.27 | **6.03** | **1.87** | ~214 s | ~680 |

Raw token rate is not comparable across tokenizers. `Hindi words/s =
decode t/s ÷ fertility` is, and Gemma 3 1B beats Qwen3 1.7B by **4.8×** while
being the smaller model. The tokenizer gap alone is 3.06×, twice what the plan
predicted.

The context column matters as much: at ~680 Hindi words per 4k window, Qwen3
cannot hold two exemplar letters plus an instruction. Gemma 3 at ~2,080 can.

**Prefill figures from the first run (1575 and 6222 tok/s) were a harness bug** —
the identical prompt was sent on every repeat and runs 2–3 hit Ollama's prompt
KV cache. Fixed; prefill is now measured against a varied prompt and
implausible values are reported as `_cache (N)_` rather than as results.

---

## 4. Embedding inside Ollama is unusable here — measured, not argued

`scripts/phase0_embed.py --llm gemma3:1b --embed qwen3-embedding:0.6b`:

| Step | Result |
|---|---|
| Warm draft, before | **0.92 s** |
| Embed 32 chunks | 188.95 s total, of which **113.08 s** was loading the embedder |
| — net embedding rate | ≈ **0.42 chunks/s** |
| `ollama ps` during | **timed out** — the server was still paging models |
| Draft again, after | **98.34 s** (10.41 s of it re-loading the evicted model) |

**A draft that follows a retrieval went from 0.92 s to 98.34 s. That is a
107× regression**, and the `ollama ps` timeout says the machine was in disk
thrash, not merely busy.

Extrapolated to a real corpus: a 5,000-letter first ingest at 0.42 chunks/s is
**≈ 3.3 hours**, and that assumes no further eviction churn.

This settles the split-runtime design in the plan. Ollama cannot use DirectML,
so on this laptop it is CPU-only; putting the embedder there too forces both
models into the same ~3 GB of free RAM, where they evict each other across a
spinning disk.

| | Runtime | Silicon | Resident cost |
|---|---|---|---|
| Drafting LLM | Ollama | CPU, 4 threads | 0.8–2.6 GB system RAM |
| Embeddings | ONNX Runtime + DirectML | AMD R7 M4xx | ~200 MB VRAM, **0 system RAM** |

Caveat worth keeping: this result is amplified by the HDD and the
single-channel RAM. On a machine with an SSD and 16 GB it would be bad rather
than fatal. We are optimising for this machine.

---

## 5. Both GPUs lose to the CPU — the plan's GPU assignment is dead

`matmul 1024×1024, 20 iterations, onnxruntime 1.24.4, IR 9`:

| Device | Time | Throughput | vs CPU |
|---|---:|---:|---:|
| **CPU** (i7-8550U, AVX2) | 7.9 ms | **272.7 GFLOP/s** | — |
| DirectML `device_id=1` (AMD R7 M4xx) | 16.3 ms | 132.0 GFLOP/s | **0.48×** |
| DirectML `device_id=0` (Intel UHD 620) | 32.1 ms | 67.0 GFLOP/s | **0.25×** |

**Prediction that was wrong:** the plan asserted the discrete GPU would beat
the CPU by 2–3× on this workload, and assigned bulk embedding to it on that
basis. It loses by 2×. Two errors compounded:

1. **The CPU was underestimated by 2×.** The plan figured ~190 GFLOP/s from
   "4 cores × 16 FLOP/cycle × 3 GHz", which does not count AVX2 *FMA*. With
   FMA it is 32 FLOP/cycle/core: 4 × 32 × ~2.1 GHz ≈ 269 GFLOP/s, which is
   almost exactly what ONNX Runtime's MLAS kernels delivered.
2. **The GPU was overestimated by 5×.** 320 shaders × 2 × 1021 MHz gives
   ~650 GFLOP/s on paper; it returned 132, about 20% of peak. GCN mobile parts
   are register-starved with small caches, and DirectML's generic kernels are
   not tuned for a 2016 low-end part on a legacy driver.

### Methodological caveat, and what was done about it

The original benchmark shipped three 4 MB tensors over PCIe every iteration,
which a real encoder never does — its weights stay resident in VRAM and only
token ids go in and a few hundred floats come out. That penalises the GPU for
transfer it would not pay in production.

`phase0_directml.py` now also measures with **device-resident I/O binding**,
which keeps the operands on the device across iterations and isolates compute
from transfer. Re-run to see whether the 132 GFLOP/s was transfer-limited or
genuinely compute-limited. Unless the device-I/O row roughly doubles, the
conclusion below stands.

### Decision: embeddings run on the CPU

| | |
|---|---|
| Runtime | ONNX Runtime, `CPUExecutionProvider` |
| Model | `granite-embedding-97m-multilingual-r2` (ONNX, int8 dynamic) |
| Dependency dropped | `onnxruntime-directml` — plain `onnxruntime` is enough |

[Guessing] Projected ingest rate: a 97M encoder over ~512 tokens is roughly
99 GFLOP per letter; at a realistic 40–60% of the measured 272 GFLOP/s that is
~1.3–1.6 letters/s, so a 5,000-letter first ingest lands near **45–60 minutes**,
or ~30 with int8. Against the 0.42 chunks/s measured through Ollama, that is a
3–4× improvement *and* one less runtime.

**The one case that could still justify the GPU** is concurrency, not speed:
work placed there costs the LLM no CPU threads. That only matters if embedding
must run *while* someone is drafting. Queuing ingest to run when nobody is
drafting — which the plan's single-generation semaphore already implies — is
simpler and faster. Keep DirectML as a documented optional optimisation, not a
v1 dependency.

### Bug that blocked the first attempt

```
Unsupported model IR version: 14, max supported IR version: 13
```

Every provider failed, **including CPU** — which is how you tell a graph
problem from a GPU or driver problem. Recent `onnx` releases emit IR 14 by
default; the installed `onnxruntime` accepted at most 13. Fixed: the graph is
now built at the highest IR version the local runtime actually loads.

---

## Decisions this settles

- [x] **Model: `gemma3:1b`** for now. Re-test `gemma3:4b-it-qat` after the RAM upgrade.
- [x] **Drop `qwen3:1.7b`.** Worst measured option on Hindi throughput, context capacity and cold load.
- [x] **Do not run the embedder in Ollama.** Split runtime confirmed by a 107× measured regression.
- [x] **`keep_alive: -1`** is mandatory, not an optimisation.
- [ ] **Buy: 1 × 8 GB DDR4-2400 SODIMM (slot A) + a SATA/NVMe SSD.** Together they address bandwidth, capacity and load time — the three things every measurement above is limited by.
- [x] **DirectML: rejected.** Both GPUs lose to the CPU (0.48× and 0.25×). Embeddings run on the CPU via plain `onnxruntime`; the `onnxruntime-directml` dependency is dropped.
- [ ] Re-run `phase0_directml.py` for the device-I/O rows, to confirm the GPU deficit is compute and not PCIe transfer.
- [ ] `num_thread` sweep — not yet reported.
