"""Ollama client, configured from the Phase 0 measurements.

Standard library only: `urllib` speaks HTTP to Ollama perfectly well, and the
`ollama` package would be another wheel to install on a machine with no
internet.

Every default here traces to a measurement in docs/PHASE0_FINDINGS.md rather
than to a habit:

* ``keep_alive: -1`` -- the disk is spinning and a cold load measured 30s for
  a 1B model and 100s for a 1.7B one. Pinning the model means that is paid
  once per boot instead of once per draft.
* ``num_thread: 4`` -- four physical cores. Decode is memory-bandwidth-bound
  and the machine is single-channel, so hyperthread siblings contend for the
  same controller. Confirm with the thread sweep before changing it.
* ``num_ctx: 4096`` -- Gemma 3 fits ~2,080 Hindi words in that; Qwen3 fits
  ~680. Raising it costs KV cache RAM this machine does not have.
* ``think: False`` for Qwen3 base tags -- they emit <think> blocks by default,
  which at ~9 Hindi words/second is a minute the user never sees.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterator, Protocol

DEFAULT_HOST = "http://127.0.0.1:11434"

DEFAULT_OPTIONS: dict = {
    "num_ctx": 4096,
    "num_thread": 4,
    "num_batch": 256,
    "num_predict": 900,
    "temperature": 0.3,
    "top_p": 0.9,
    "repeat_penalty": 1.05,
}


class LLM(Protocol):
    """Minimal surface, so a stub can stand in for Ollama in tests."""

    def generate(self, prompt: str, *, system: str | None = None,
                 options: dict | None = None) -> "Completion": ...

    def stream(self, prompt: str, *, system: str | None = None,
               options: dict | None = None) -> Iterator[str]: ...


@dataclass
class Completion:
    text: str
    model: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    load_seconds: float = 0.0
    eval_seconds: float = 0.0
    wall_seconds: float = 0.0
    truncated: bool = False

    @property
    def tokens_per_second(self) -> float:
        return self.output_tokens / self.eval_seconds if self.eval_seconds else 0.0


class OllamaError(RuntimeError):
    pass


class Ollama:
    def __init__(self, model: str = "gemma3:1b", *, host: str = DEFAULT_HOST,
                 keep_alive: str | int = -1, options: dict | None = None,
                 timeout: float = 900.0):
        self.model = model
        self.host = host.rstrip("/")
        self.keep_alive = keep_alive
        self.options = {**DEFAULT_OPTIONS, **(options or {})}
        self.timeout = timeout

    # --- plumbing ---------------------------------------------------------
    def _payload(self, prompt: str, system: str | None, options: dict | None,
                 stream: bool) -> dict:
        p = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": {**self.options, **(options or {})},
        }
        if system:
            p["system"] = system
        # Qwen3 base tags think by default; the tokens are generated at full
        # cost and never shown.
        if "qwen3" in self.model.lower() and "instruct" not in self.model.lower():
            p["think"] = False
        return p

    def _request(self, payload: dict):
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"cannot reach Ollama at {self.host}: {exc}. Start it with "
                "`ollama serve`, and check the model is pulled.") from exc

    # --- api --------------------------------------------------------------
    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=10) as r:
                tags = json.loads(r.read())
        except Exception:
            return False
        names = {m.get("name", "") for m in tags.get("models", [])}
        return self.model in names or f"{self.model}:latest" in names

    def generate(self, prompt: str, *, system: str | None = None,
                 options: dict | None = None) -> Completion:
        t0 = time.perf_counter()
        with self._request(self._payload(prompt, system, options, False)) as resp:
            r = json.loads(resp.read())
        n_predict = (options or {}).get("num_predict", self.options["num_predict"])
        return Completion(
            text=r.get("response", ""), model=r.get("model", self.model),
            prompt_tokens=r.get("prompt_eval_count", 0),
            output_tokens=r.get("eval_count", 0),
            load_seconds=(r.get("load_duration") or 0) / 1e9,
            eval_seconds=(r.get("eval_duration") or 0) / 1e9,
            wall_seconds=time.perf_counter() - t0,
            # `length` means the model was cut off mid-sentence, which for a
            # letter means a missing closing block.
            truncated=r.get("done_reason") == "length"
            or r.get("eval_count", 0) >= n_predict)

    def stream(self, prompt: str, *, system: str | None = None,
               options: dict | None = None) -> Iterator[str]:
        """Token-by-token output.

        Not a nicety. At the measured ~9 Hindi words per second a 400-word
        letter takes about 45 seconds; streamed that reads as working, and
        unstreamed it reads as hung.
        """
        with self._request(self._payload(prompt, system, options, True)) as resp:
            for line in resp:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                piece = chunk.get("response", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    break


@dataclass
class StubLLM:
    """A deterministic stand-in, so the whole drafting pipeline is testable
    without a model. Also useful on the target machine for timing the
    non-LLM parts in isolation."""

    reply: str = "यह एक परीक्षण उत्तर है।"
    calls: list[dict] = field(default_factory=list)
    fail: bool = False

    def generate(self, prompt: str, *, system: str | None = None,
                 options: dict | None = None) -> Completion:
        if self.fail:
            raise OllamaError("stub configured to fail")
        self.calls.append({"prompt": prompt, "system": system, "options": options})
        return Completion(text=self.reply, model="stub",
                          prompt_tokens=len(prompt) // 4,
                          output_tokens=len(self.reply) // 4, eval_seconds=1.0)

    def stream(self, prompt: str, *, system: str | None = None,
               options: dict | None = None) -> Iterator[str]:
        yield from self.generate(prompt, system=system, options=options).text.split(" ")
