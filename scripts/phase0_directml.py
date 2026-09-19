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


def _run_plain(sess, a, b):
    sess.run(None, {"A": a, "B": b})


def _bind_on_device(sess, a, b, device: str):
    """Keep both operands resident on the device across iterations.

    Without this the benchmark ships 3 x 4 MB over PCIe on every iteration,
    which on a x4-linked mobile card is milliseconds of pure transfer against
    a ~16 ms kernel -- and it penalises the GPU for something a real encoder
    never does. An encoder's weights stay in VRAM; only token ids go in and a
    few hundred floats come out. Returns a callable, or None if this build
    cannot place tensors on the device.
    """
    try:
        oa = ort.OrtValue.ortvalue_from_numpy(a, device, 0)
        ob = ort.OrtValue.ortvalue_from_numpy(b, device, 0)
        oy = ort.OrtValue.ortvalue_from_shape_and_type((N, N), np.float32, device, 0)
        io = sess.io_binding()
        io.bind_ortvalue_input("A", oa)
        io.bind_ortvalue_input("B", ob)
        io.bind_ortvalue_output("Y", oy)
    except Exception:
        return None

    def _run():
        sess.run_with_iobinding(io)

    try:
        _run()
    except Exception:
        return None
    return _run


def bench(model: bytes, providers, label: str, *, device: str | None = None) -> None:
    try:
        sess = ort.InferenceSession(model, providers=providers)
    except Exception as exc:
        print(f"  {label:28} UNAVAILABLE  ({type(exc).__name__}: {exc})")
        return
    rng = np.random.default_rng(0)
    a = rng.standard_normal((N, N), dtype=np.float32)
    b = rng.standard_normal((N, N), dtype=np.float32)

    for mode, runner in (("host I/O", lambda: _run_plain(sess, a, b)),
                         ("device I/O", _bind_on_device(sess, a, b, device) if device else None)):
        if runner is None:
            continue
        runner()  # warm up: the first run includes kernel compilation
        t0 = time.perf_counter()
        for _ in range(ITERS):
            runner()
        dt = (time.perf_counter() - t0) / ITERS
        print(f"  {label:24} {mode:11} {dt*1000:8.1f} ms   "
              f"{2 * N**3 / dt / 1e9:7.1f} GFLOP/s")


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
              f"DirectML device_id={device_id}", device="dml")

    print("""
How to read this
  Device ids follow the order in Task Manager (GPU 0, GPU 1).

  Compare the "device I/O" rows, not "host I/O": the host rows include a
  PCIe round-trip of three 4 MB tensors per iteration, which a real encoder
  does not pay because its weights stay resident in VRAM.

  A GPU only earns the extra runtime if it clearly beats the CPU row. An
  i7-8550U doing AVX2 FMA is roughly 270 GFLOP/s, which is more than a
  low-end GCN mobile part delivers in practice -- so "CPU wins" is a normal
  and perfectly good outcome here, not a misconfiguration. It means one
  runtime instead of two.

  Even a GPU that loses on raw throughput can still be worth it for
  *concurrency*: work placed there costs the LLM no CPU threads. That only
  matters if embedding has to run while someone is drafting. If ingest can be
  queued to run when nobody is drafting, take the faster device and keep the
  stack simple.

  Do NOT conclude anything about LLM token generation from these numbers --
  that is memory-bandwidth-bound, where this card's DDR3 (~14 GB/s) loses to
  system RAM regardless.

  If a device is UNAVAILABLE, check its driver in the probe output. On GCN-era
  Radeons you may need the last legacy Adrenalin release for the family.""")
    print(f"\n[written to {out_path} -- commit it with: git add -f {out_path}]")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
