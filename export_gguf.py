"""
Export merged SFT model to GGUF format for llama.cpp / Ollama / LM Studio.

Handles:
  1. Cloning llama.cpp (if not already present)
  2. Installing its Python dependencies
  3. Converting HF safetensors → GGUF (f16 baseline)
  4. Quantising to requested types (default: Q4_K_M, Q5_K_M, Q8_0)

Usage:
    # Defaults — reads from checkpoints/merged_sft_qwen3_1.7b
    python export_gguf.py

    # Custom paths / quants
    RLHF_MERGED_MODEL=path/to/model \
    RLHF_GGUF_OUTPUT=checkpoints/gguf \
    RLHF_QUANT_TYPES=Q4_K_M,Q8_0 \
      python export_gguf.py

Environment variables:
    RLHF_MERGED_MODEL      Path to merged HF model dir          (default: checkpoints/merged_sft_qwen3_1.7b)
    RLHF_GGUF_OUTPUT       Directory for GGUF artefacts         (default: checkpoints/gguf)
    RLHF_QUANT_TYPES       Comma-separated quantisation types   (default: Q4_K_M,Q5_K_M,Q8_0)
    RLHF_LLAMA_CPP_DIR     Path to llama.cpp clone              (default: /tmp/llama.cpp)
    RLHF_GGUF_MODEL_NAME   Friendly name baked into filenames   (default: derived from model dir)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent

MERGED_MODEL = os.getenv(
    "RLHF_MERGED_MODEL",
    str(PROJECT_ROOT / "checkpoints" / "merged_sft_qwen3_1.7b"),
)
OUTPUT_DIR = os.getenv(
    "RLHF_GGUF_OUTPUT",
    str(PROJECT_ROOT / "checkpoints" / "gguf"),
)
QUANT_TYPES = os.getenv("RLHF_QUANT_TYPES", "Q4_K_M,Q5_K_M,Q8_0").split(",")
LLAMA_CPP = os.getenv("RLHF_LLAMA_CPP_DIR", "/tmp/llama.cpp")
MODEL_NAME = os.getenv(
    "RLHF_GGUF_MODEL_NAME",
    Path(MERGED_MODEL).name.replace(" ", "-"),
)

CONVERT_SCRIPT = os.path.join(LLAMA_CPP, "convert_hf_to_gguf.py")


# ── Helpers ──────────────────────────────────────────────────────────────────

def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, stream output, and abort on failure."""
    print(f"\n▶  {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")
    return result


def ensure_llama_cpp() -> None:
    """Clone llama.cpp and install its Python deps if needed."""
    if Path(CONVERT_SCRIPT).exists():
        print(f"llama.cpp already present at {LLAMA_CPP}")
        return

    print(f"Cloning llama.cpp → {LLAMA_CPP} ...")
    if Path(LLAMA_CPP).exists():
        shutil.rmtree(LLAMA_CPP)
    run([
        "git", "clone", "--depth=1",
        "https://github.com/ggerganov/llama.cpp.git",
        LLAMA_CPP,
    ])

    # Install conversion dependencies into current venv
    req_file = os.path.join(LLAMA_CPP, "requirements.txt")
    if Path(req_file).exists():
        run([sys.executable, "-m", "pip", "install", "-q", "-r", req_file])
    else:
        # Fallback: the converter needs gguf + numpy + sentencepiece at minimum
        run([sys.executable, "-m", "pip", "install", "-q", "gguf", "numpy", "sentencepiece"])


def find_llama_quantize() -> str | None:
    """Try to locate the llama-quantize binary (compiled C++)."""
    # Check common locations
    candidates = [
        os.path.join(LLAMA_CPP, "build", "bin", "llama-quantize"),
        os.path.join(LLAMA_CPP, "llama-quantize"),
        shutil.which("llama-quantize"),
    ]
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


def build_llama_quantize() -> str:
    """Compile llama-quantize from source."""
    build_dir = os.path.join(LLAMA_CPP, "build")
    os.makedirs(build_dir, exist_ok=True)
    run(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"], cwd=build_dir)
    run(["cmake", "--build", ".", "--target", "llama-quantize", "-j", str(os.cpu_count() or 4)], cwd=build_dir)
    binary = os.path.join(build_dir, "bin", "llama-quantize")
    if not Path(binary).is_file():
        raise FileNotFoundError(f"Build succeeded but binary not found at {binary}")
    return binary


def convert_to_f16_gguf(model_dir: str, output_path: str) -> None:
    """Convert HF safetensors to f16 GGUF using llama.cpp's converter."""
    run([
        sys.executable, CONVERT_SCRIPT,
        model_dir,
        "--outfile", output_path,
        "--outtype", "f16",
    ])


def quantize_gguf(f16_path: str, quant_path: str, quant_type: str, quantize_bin: str) -> None:
    """Quantise an f16 GGUF to a specific type."""
    run([quantize_bin, f16_path, quant_path, quant_type])


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    model_dir = Path(MERGED_MODEL)
    out_dir = Path(OUTPUT_DIR)

    # Validate
    if not model_dir.exists():
        sys.exit(f"ERROR: Merged model not found at {model_dir}\n"
                 f"Run merge_sft_lora.py first or set RLHF_MERGED_MODEL.")

    safetensors = list(model_dir.glob("*.safetensors"))
    if not safetensors:
        sys.exit(f"ERROR: No .safetensors files in {model_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GGUF EXPORT")
    print("=" * 60)
    print(f"  Model dir     : {model_dir}")
    print(f"  Output dir    : {out_dir}")
    print(f"  Model name    : {MODEL_NAME}")
    print(f"  Quant types   : {', '.join(QUANT_TYPES)}")
    print()

    # Step 1: Ensure llama.cpp is available
    print("[1/4] Ensuring llama.cpp is available ...")
    ensure_llama_cpp()

    # Step 2: Convert to f16 GGUF
    f16_path = str(out_dir / f"{MODEL_NAME}-f16.gguf")
    print(f"\n[2/4] Converting to f16 GGUF → {f16_path} ...")
    convert_to_f16_gguf(str(model_dir), f16_path)

    if not Path(f16_path).exists():
        sys.exit(f"ERROR: Conversion failed — {f16_path} not found")

    f16_size_gb = Path(f16_path).stat().st_size / (1024 ** 3)
    print(f"       f16 GGUF size: {f16_size_gb:.2f} GB")

    # Step 3: Build / locate llama-quantize
    print("\n[3/4] Locating llama-quantize ...")
    quantize_bin = find_llama_quantize()
    if quantize_bin:
        print(f"       Found: {quantize_bin}")
    else:
        print("       Not found — building from source ...")
        quantize_bin = build_llama_quantize()
        print(f"       Built: {quantize_bin}")

    # Step 4: Quantise
    print(f"\n[4/4] Quantising ({len(QUANT_TYPES)} variants) ...")
    results: list[tuple[str, str, float]] = []
    for qt in QUANT_TYPES:
        qt = qt.strip()
        quant_path = str(out_dir / f"{MODEL_NAME}-{qt}.gguf")
        print(f"\n  → {qt} ...")
        try:
            quantize_gguf(f16_path, quant_path, qt, quantize_bin)
            size_gb = Path(quant_path).stat().st_size / (1024 ** 3)
            results.append((qt, quant_path, size_gb))
        except RuntimeError as exc:
            print(f"  ✗ {qt} failed: {exc}")

    # Summary
    print("\n" + "=" * 60)
    print("GGUF EXPORT COMPLETE")
    print("=" * 60)
    print(f"  {'Type':<12} {'Size':>8}  Path")
    print(f"  {'─'*12} {'─'*8}  {'─'*40}")
    print(f"  {'f16':<12} {f16_size_gb:>7.2f}G  {f16_path}")
    for qt, path, size in results:
        print(f"  {qt:<12} {size:>7.2f}G  {path}")
    print()
    print("Use with llama.cpp:")
    if results:
        print(f"  llama-cli -m {results[0][1]} -p 'Hello'")
    print("Use with Ollama:")
    print(f"  ollama create {MODEL_NAME} -f Modelfile")
    print("=" * 60)


if __name__ == "__main__":
    main()
