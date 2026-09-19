"""Phase 0.5 -- confirm ONNX Runtime DirectML can reach both GPUs.

Run on the target laptop:
    pip install onnxruntime-directml numpy
    python scripts\\phase0_directml.py

DirectML needs only a DirectX 12 device, which is why it is the right GPU path
on a Windows machine with a ROCm-unsupported AMD card. This script proves the
provider loads, then runs a matmul big enough to be compute-bound on each
device id and on the CPU, so you can see whether the discrete GPU is actually
faster for the batched-embedding workload it is meant to take.
"""

from __future__ import annotations

import sys
import time

try:
    import numpy as np
    import onnxruntime as ort
except ImportError as exc:
    sys.exit(f"missing dependency: {exc}\n  pip install onnxruntime-directml numpy")

N = 1024
ITERS = 20


#: ONNX Runtime refuses a graph whose IR version exceeds what it was built
#: against. Recent `onnx` releases default to IR 14, while the onnxruntime
#: wheels in circulation top out at 10-13, so a freshly built graph fails to
#: load on *every* provider -- including CPU, which is how you tell this apart
#: from a GPU or driver problem. Nothing about the model needs IR 14, so it is
#: pinned down and retried.
_IR_CANDIDATES = (9, 10, 8, 7)


def make_model(ir_version: int) -> bytes:
    """A single big matmul -- the shape of work an embedding encoder does."""
    from onnx import TensorProto, helper

    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [N, N])
    b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [N, N])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [N, N])
    node = helper.make_node("MatMul", ["A", "B"], ["Y"])
    graph = helper.make_graph([node], "bench", [a, b], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = ir_version
    return model.SerializeToString()


def build_loadable_model() -> tuple[bytes, int]:
    """Build the graph at the highest IR version this onnxruntime accepts."""
    last = None
    for ir in _IR_CANDIDATES:
        blob = make_model(ir)
        try:
            ort.InferenceSession(blob, providers=["CPUExecutionProvider"])
            return blob, ir
        except Exception as exc:
            last = exc
    raise RuntimeError(f"no IR version in {_IR_CANDIDATES} loaded: {last}")


def bench(model: bytes, providers, label: str) -> None:
    try:
        sess = ort.InferenceSession(model, providers=providers)
    except Exception as exc:
        print(f"  {label:28} UNAVAILABLE  ({type(exc).__name__}: {exc})")
        return
    rng = np.random.default_rng(0)
    a = rng.standard_normal((N, N), dtype=np.float32)
    b = rng.standard_normal((N, N), dtype=np.float32)
    sess.run(None, {"A": a, "B": b})  # warm up: first run includes compilation
    t0 = time.perf_counter()
    for _ in range(ITERS):
        sess.run(None, {"A": a, "B": b})
    dt = (time.perf_counter() - t0) / ITERS
    gflops = 2 * N**3 / dt / 1e9
    print(f"  {label:28} {dt*1000:8.1f} ms   {gflops:7.1f} GFLOP/s   "
          f"({sess.get_providers()[0]})")


class _Tee:
    """Print to the console and to a file at once.

    The benchmark table scrolls off the top of a small terminal behind the
    explanatory footer, so the result gets lost. Writing a file as well makes
    the output survive and makes it committable next to BASELINE.md.
    """

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.file.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()


def main() -> int:
    from pathlib import Path
    out_path = Path("docs/directml.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(out_path)

    print("available providers:", ort.get_available_providers())
    if "DmlExecutionProvider" not in ort.get_available_providers():
        print("\nDirectML provider is NOT available.")
        print("  pip uninstall onnxruntime && pip install onnxruntime-directml")
        print("  then update the GPU driver if it still does not appear.")
        return 1

    try:
        model, ir = build_loadable_model()
    except ImportError:
        return print("pip install onnx  (needed only to build the benchmark graph)") or 1
    except RuntimeError as exc:
        print(f"\ncould not build a loadable graph: {exc}")
        print("onnx and onnxruntime versions are incompatible. Either:")
        print("  pip install 'onnx<1.17'")
        print("  pip install --upgrade onnxruntime-directml")
        return 1

    print(f"onnx IR version accepted: {ir}   onnxruntime {ort.__version__}")
    print(f"\nmatmul {N}x{N}, {ITERS} iterations after warm-up:")
    bench(model, ["CPUExecutionProvider"], "CPU")
    for device_id in (0, 1):
        bench(model, [("DmlExecutionProvider", {"device_id": device_id})],
              f"DirectML device_id={device_id}")

    print("""
How to read this
  Device ids follow the order in Task Manager (GPU 0, GPU 1). Expect the
  discrete AMD card to beat the CPU here by roughly 2-3x: this workload is
  compute-bound, which is exactly why bulk embedding at ingest is assigned to
  it. Do NOT conclude anything about LLM token generation from this number --
  that is memory-bandwidth-bound, where the same card is ~2.7x SLOWER than
  system RAM.

  If a device is UNAVAILABLE, check its driver in the probe output. On GCN-era
  Radeons you may need the last legacy Adrenalin release for the family.""")
    print(f"\n[written to {out_path} -- commit it with: git add -f {out_path}]")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
