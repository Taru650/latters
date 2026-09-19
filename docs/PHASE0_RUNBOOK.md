# Phase 0 runbook — what to run, and what each number decides

Every command here runs on the target laptop. Nothing in Phase 0 can be run
remotely; the whole point is to measure *this* machine.

Outputs land in `docs/` and are meant to be committed:

```
git add -f docs\probe.txt docs\directml.txt docs\BASELINE.md
git commit -m "Phase 0 baseline from <machine name>"
git push
```

---

## 1. Memory channels — the open question

**What decides:** whether a ~₹2,000 SODIMM roughly doubles drafting speed.

Token generation is memory-bandwidth-bound: every generated token reads the
whole active weight set from RAM once. Two matched modules let the controller
run dual-channel (~38 GB/s on DDR4-2400); one module halves that to ~19 GB/s,
and halves the token rate with it. No software change comes close.

```powershell
Get-CimInstance Win32_PhysicalMemory |
  Select-Object BankLabel, DeviceLocator,
    @{n='GB';e={[math]::Round($_.Capacity/1GB,1)}}, Speed, ConfiguredClockSpeed |
  Format-Table -AutoSize
```

**Reading it:** count the rows.

| Rows | Meaning | Action |
|---|---|---|
| 1 | Single-channel, ~19 GB/s | Fit a second matched SODIMM. Highest-ROI change available. |
| 2 | Dual-channel, ~38 GB/s | Nothing to do. RAM *capacity* is still the ceiling on model size. |

`ConfiguredClockSpeed` below `Speed` means the modules are running slower than
rated — usually a mismatched pair.

---

## 2. Disk type — already effectively answered

**What decides:** whether `keep_alive: -1` is an optimisation or a necessity,
and whether an SSD goes on the shopping list.

```powershell
Get-PhysicalDisk |
  Select-Object DeviceId, FriendlyName, MediaType, BusType,
    @{n='GB';e={[math]::Round($_.Size/1GB,0)}} | Format-Table -AutoSize
```

**Caveat:** behind an Intel RST RAID controller — which Task Manager reports as
`HDD (RAID)` on this machine — `MediaType` often reads `Unspecified`. That is
not an answer. The cold-load timings already are one: 1.4 GB in 100 s is
~14 MB/s, which no SSD does. Treat the disk as spinning until an SSD is fitted.

**Consequences, both mandatory:**
- `keep_alive: -1` on the drafting model, so the load is paid once per boot.
- Never let a second model evict the first (see step 4).

---

## 3. DirectML — does the AMD card earn its place?

**What decides:** whether bulk embedding at corpus ingest runs on the discrete
GPU or on the CPU.

```
pip install onnxruntime-directml numpy onnx
python scripts\phase0_directml.py
type docs\directml.txt
```

The script now writes `docs\directml.txt` as well as printing, because the
result table scrolls off above the explanatory footer.

**Reading it:** three lines matter.

```
  CPU                            xxx.x ms    xx.x GFLOP/s
  DirectML device_id=0           xxx.x ms    xx.x GFLOP/s     <- Intel UHD 620
  DirectML device_id=1           xxx.x ms    xx.x GFLOP/s     <- AMD R7 M4xx
```

| Outcome | Meaning |
|---|---|
| A GPU beats CPU by 2–3× | Assign batched ingest embedding to it. A 5,000-letter first ingest goes from ~30 min to ~10. |
| GPUs roughly match CPU | Keep embedding on the CPU. One less runtime, one less failure mode. |
| `UNAVAILABLE` | Driver too old, or plain `onnxruntime` is shadowing the DirectML build. `pip uninstall onnxruntime`, reinstall `onnxruntime-directml`, then update the GPU driver. |

**This says nothing about the LLM.** Matmul is compute-bound, where the AMD
card wins. Token generation is bandwidth-bound, where its DDR3 (~14 GB/s) is
~2.7× *slower* than system RAM. Do not offload LLM layers to it on the
strength of this benchmark.

---

## 4. Embedding placement — the split-runtime question

**What decides:** whether the embedding model may live in Ollama.

```
python scripts\phase0_embed.py --llm gemma3:1b --embed qwen3-embedding:0.6b
```

The script drafts, embeds 32 chunks, then drafts again, and reports
`load_duration` on that third call.

**Reading it:** a non-zero `load_duration` on the post-embed draft means the
embedder evicted the drafting model, and it was re-read from disk. On this
machine that is 30 s for a 1 B model and 100 s for a 1.7 B one — added to
every draft that follows a retrieval. That is not slow, it is broken.

### Why the plan splits the runtimes

Ollama cannot use DirectML, so on this laptop it runs the LLM on the CPU and
nothing else. ONNX Runtime *can* use any DirectX 12 GPU. They are separate
processes with separate memory:

| | Runtime | Silicon | Resident cost |
|---|---|---|---|
| Drafting LLM | Ollama | CPU, 4 threads | 0.8–2.6 GB system RAM |
| Embeddings | ONNX Runtime + DirectML | AMD R7 M4xx | ~200 MB **VRAM**, 0 system RAM |

Running the embedder inside Ollama instead puts both models in the same
bounded pool of ~3 GB free system RAM, competing for the same CPU threads, and
makes them serialise. Split across two runtimes they run *concurrently*: the
GPU can re-embed a freshly uploaded batch while the CPU decodes a draft.

`qwen3-embedding:0.6b` is ~1.2 GB resident and Ollama-only, so it is the worst
case on both counts. `granite-embedding-97m-multilingual-r2` is ~200 MB as
ONNX, scores higher on multilingual retrieval (60.3 vs 50.9 for the usual
`multilingual-e5-small` default), and runs on the card.

Quick check of what is resident right now:

```
ollama ps
```

---

## 5. Model benchmark

```
ollama pull gemma3:4b-it-qat
python scripts\phase0_bench.py --models gemma3:1b gemma3:4b-it-qat qwen3:1.7b --out docs\BASELINE.md
```

**Rank on `Hindi words/s`, not on `decode t/s`.** A token is not the same
amount of Hindi in two different vocabularies, so raw token rate is not
comparable across models. `Hindi words/s = decode t/s ÷ fertility`.

A `prefill t/s` cell reading `_cache (N)_` means the measurement was rejected
as prompt-cache reuse, not that the hardware did that.

---

## Checklist

- [ ] Memory modules: 1 or 2?
- [ ] Disk: SSD or spinning?
- [ ] DirectML: which device wins the matmul, and by how much?
- [ ] Embedding in Ollama: does it evict the drafting model?
- [ ] Model: which wins `Hindi words/s` at acceptable RAM?
- [ ] `num_thread`: which value wins the sweep?
