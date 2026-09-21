"""Build the offline install bundle -- run this on a machine WITH internet.

The target office has none. Everything the installer needs has to be on the
USB stick before it leaves: the wheel, its dependencies as wheels, the Ollama
installer, and the model as a GGUF file. `pip install latters` and
`ollama pull gemma3:1b` both need the network and both are what an installer
written without thinking about this would do.

    python packaging/make_bundle.py --out bundle --model gemma3:1b

Then copy `bundle/` to the USB stick and run `install.bat` from it on the
office machine.

What goes in, and why each piece is separate
--------------------------------------------
``wheels/``       the app and its dependencies, downloaded for the TARGET
                  platform, not this one. ``--platform win_amd64`` matters:
                  a Linux build machine otherwise silently collects Linux
                  wheels that fail on Windows with an error about the wrong
                  platform tag, at the office, with no internet to fix it.
``model/``        the GGUF plus a Modelfile, so ``ollama create`` works
                  offline. Exporting a model already pulled on this machine
                  is the only way to get the GGUF without the office
                  downloading 800 MB.
``OllamaSetup.exe`` fetched once. If it is missing the installer says so
                  rather than half-installing.
``*.bat``         copied verbatim from this directory.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

OLLAMA_URL = "https://ollama.com/download/OllamaSetup.exe"

MODELFILE = """FROM ./{gguf}

# Matches the options the application sends. Keeping them here too means a
# model run straight from `ollama run` behaves like the one the app sees,
# which is what someone debugging at the office will try first.
PARAMETER num_ctx 4096
PARAMETER num_thread 4
"""


def run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def collect_wheels(out: Path, python_version: str) -> None:
    wheels = out / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    # Build our own wheel first so the office installs a version, not a
    # source checkout it cannot rebuild without a compiler.
    run([sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(wheels),
         str(REPO)])
    # --only-binary=:all: is deliberate. A source distribution would need a
    # C compiler at the office, which there will not be.
    run([sys.executable, "-m", "pip", "download",
         "--only-binary=:all:",
         "--platform", "win_amd64",
         "--python-version", python_version,
         "-d", str(wheels),
         f"{REPO}[web]"])


def export_model(out: Path, model: str) -> bool:
    """Copy the GGUF for `model` out of the local Ollama store."""
    model_dir = out / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        blob = subprocess.run(
            ["ollama", "show", "--modelfile", model],
            capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"  !! could not read {model} from the local Ollama: {exc}")
        print(f"  !! run `ollama pull {model}` on this machine first")
        return False

    src = None
    for line in blob.splitlines():
        if line.upper().startswith("FROM ") and not line.strip().endswith(model):
            candidate = Path(line.split(None, 1)[1].strip())
            if candidate.exists():
                src = candidate
                break
    if src is None:
        print("  !! Ollama did not report a GGUF path for this model")
        return False

    gguf = model.replace(":", "-") + ".gguf"
    print(f"  copying {src} ({src.stat().st_size / 1e9:.2f} GB)")
    shutil.copy2(src, model_dir / gguf)
    (model_dir / "Modelfile").write_text(
        MODELFILE.format(gguf=gguf), encoding="utf-8")
    (model_dir / "MODEL_NAME").write_text(model, encoding="utf-8")
    return True


def fetch_ollama(out: Path) -> bool:
    dest = out / "OllamaSetup.exe"
    if dest.exists():
        print("  OllamaSetup.exe already present")
        return True
    try:
        print(f"  downloading {OLLAMA_URL}")
        urllib.request.urlopen(OLLAMA_URL, timeout=60)  # noqa: S310
    except Exception as exc:
        print(f"  !! could not download Ollama: {exc}")
        print("  !! download it by hand and put it in the bundle root")
        return False
    urllib.request.urlretrieve(OLLAMA_URL, dest)  # noqa: S310
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="bundle")
    ap.add_argument("--model", default="gemma3:1b")
    ap.add_argument("--python-version", default="311",
                    help="target CPython, e.g. 311 for the office's 3.11")
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--skip-ollama", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("wheels")
    collect_wheels(out, args.python_version)

    ok = True
    if not args.skip_model:
        print("model")
        ok &= export_model(out, args.model)
    if not args.skip_ollama:
        print("ollama installer")
        ok &= fetch_ollama(out)

    print("scripts")
    for name in ("install.bat", "start.bat", "backup.bat", "README.md"):
        src = HERE / name
        if src.exists():
            shutil.copy2(src, out / name)
            print(f"  {name}")

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\n{out}  {total / 1e9:.2f} GB")
    if not ok:
        print("\nINCOMPLETE -- see the !! lines above. Do not take this "
              "bundle to an office with no internet.")
        return 1
    print("\nCopy this folder to the USB stick, then run install.bat on the "
          "office machine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
