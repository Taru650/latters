"""Dense encoders for the optional semantic half of retrieval.

READ THE PHASE 4 RESULT BEFORE ADDING ONE
-----------------------------------------
Measured on 547 real letters, BM25, character n-gram TF-IDF and their RRF
fusion are statistically indistinguishable from each other, and all of them
degrade gently when the query is paraphrased. The only change that measurably
helped was the department metadata filter. On that evidence a neural encoder
is not justified: it costs ~200 MB of weights the target office cannot
download, an onnxruntime dependency, and resident RAM the drafting model
needs.

So nothing here is imported by default and the retriever works without it.
What this module provides is the plumbing, so the question can be *settled*
on the office's own machine with one command rather than argued about:

    python scripts/phase4_encoder_eval.py --db corpus.db --model <path-to-onnx>

Adopt a dense encoder only if it beats `tfidf + dept filter` on same-cell
precision by more than two standard errors (about 0.06 at 300 queries).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np

_WORD = re.compile(r"[ऀ-ॿA-Za-z0-9]+")


class RandomProjectionEncoder:
    """A real, testable encoder that needs no download.

    Hashes character n-grams into a low-dimensional space with a fixed random
    projection. This is not competitive with a trained model -- it captures
    surface form, not meaning -- but it is a genuine implementation of the
    `Encoder` protocol, so the dense index, the fusion and the evaluation
    harness can all be exercised end to end without network access. It also
    gives a floor: a trained encoder that cannot beat this is misconfigured.
    """

    def __init__(self, dim: int = 256, *, lo: int = 3, hi: int = 5,
                 buckets: int = 2 ** 14, seed: int = 0):
        self.dim, self.lo, self.hi, self.buckets = dim, lo, hi, buckets
        rng = np.random.default_rng(seed)
        # Achlioptas sparse projection: cheap, and preserves distances well.
        self._proj = rng.choice([-1.0, 0.0, 1.0], size=(buckets, dim),
                                p=[1 / 6, 2 / 3, 1 / 6]).astype(np.float32)

    def _bucket(self, gram: str) -> int:
        h = 2166136261
        for ch in gram:
            h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
        return h % self.buckets

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            t = re.sub(r"\s+", " ", text)[:1500]
            idx: dict[int, float] = {}
            for n in range(self.lo, self.hi + 1):
                for j in range(len(t) - n + 1):
                    b = self._bucket(t[j:j + n])
                    idx[b] = idx.get(b, 0.0) + 1.0
            if idx:
                keys = np.fromiter(idx.keys(), dtype=np.int32, count=len(idx))
                vals = np.fromiter(idx.values(), dtype=np.float32, count=len(idx))
                vals = 1.0 + np.log(vals)
                out[i] = vals @ self._proj[keys]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class OnnxEncoder:
    """A sentence-transformers-style ONNX encoder on the CPU.

    Intended for `granite-embedding-97m-multilingual-r2`, which ships ONNX
    weights, scores 60.3 on multilingual MTEB retrieval, and is ~200 MB.

    NOT EXERCISED IN THIS REPOSITORY'S TESTS. The development environment
    cannot reach huggingface.co, so this code path has never run against real
    weights. Treat it as a starting point to debug on the target machine, not
    as working code. `scripts/phase4_encoder_eval.py` is the harness for that.

    CPU, not DirectML: Phase 0 measured the CPU at 272 GFLOP/s against 132 on
    the discrete GPU and 67 on the iGPU.
    """

    def __init__(self, model_dir: str | Path, *, max_length: int = 512,
                 threads: int = 4, pooling: str = "mean"):
        import onnxruntime as ort  # imported lazily: not a hard dependency
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        onnx_files = sorted(model_dir.glob("*.onnx")) or sorted(
            (model_dir / "onnx").glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"no .onnx file under {model_dir}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(onnx_files[0]), opts, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding()
        self.pooling = pooling
        self._inputs = {i.name for i in self.session.get_inputs()}
        self.dim = int(self.session.get_outputs()[0].shape[-1] or 384)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        enc = self.tokenizer.encode_batch(list(texts))
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        feed = {k: v for k, v in feed.items() if k in self._inputs}

        out = self.session.run(None, feed)[0]
        if out.ndim == 3:
            if self.pooling == "cls":
                vecs = out[:, 0, :]
            else:
                m = mask[..., None].astype(np.float32)
                vecs = (out * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
        else:
            vecs = out
        vecs = vecs.astype(np.float32)
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
