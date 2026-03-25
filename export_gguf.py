"""
Export a merged Hugging Face checkpoint to GGUF for llama.cpp / Ollama / LM Studio.

Default behavior is intentionally conservative:
  1. Reuse or prepare a local llama.cpp checkout
  2. Convert HF safetensors -> GGUF
  3. Stop after writing a single GGUF file (`f16` by default)

Quantization is now an explicit later stage and only runs when requested.

Usage:
    # Defaults - reads from the merged DPO-baked fp32 model and writes one GGUF
    .venv/bin/python export_gguf.py

    # Custom source / output, still f16-only by default
    RLHF_MERGED_MODEL=path/to/model \
    RLHF_GGUF_OUTPUT=checkpoints/my_model_gguf \
      .venv/bin/python export_gguf.py

    # Emit a full-precision GGUF for A/B testing
    RLHF_GGUF_OUTTYPE=f32 \
      .venv/bin/python export_gguf.py

    # Opt in to quantization later
    RLHF_QUANTIZE_AFTER_EXPORT=1 \
    RLHF_QUANT_TYPES=Q4_K_M,Q8_0 \
      .venv/bin/python export_gguf.py

Environment variables:
    RLHF_MERGED_MODEL            Path to merged HF model dir
                                 (default: checkpoints/merged_qwen3_pinion_dpo_baked_fp32_20260312)
    RLHF_GGUF_OUTPUT             Directory for GGUF artefacts
                                 (default: checkpoints/gguf_qwen3_pinion_dpo_baked_20260312)
    RLHF_GGUF_OUTTYPE            GGUF precision for convert step (`f16`, `f32`, `bf16`)
                                 (default: f16)
    RLHF_GGUF_FILENAME           GGUF filename inside output dir
                                 (default: <outtype>.gguf)
    RLHF_QUANTIZE_AFTER_EXPORT   `1` to quantize after fp16 export, else skip
                                 (default: 0)
    RLHF_QUANT_TYPES             Comma-separated quantization types
                                 (default: Q4_K_M,Q5_K_M,Q8_0)
    RLHF_LLAMA_CPP_DIR           Path to llama.cpp clone
                                 (default: /tmp/llama.cpp)
    RLHF_LLAMA_CPP_PYTHON        Optional Python interpreter for llama.cpp
                                 conversion scripts
    RLHF_LLAMA_CPP_PYTHON_ENV    Dedicated helper venv for converter deps
                                 (default: ~/.cache/full_rlhf_pipeline/llama_cpp_py)
    RLHF_GGUF_MODEL_NAME         Friendly model name for summaries
                                 (default: derived from model dir)
"""

import os
import shutil
import subprocess
import sys
import tempfile
import json
import venv
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent

MERGED_MODEL = os.getenv(
    "RLHF_MERGED_MODEL",
    str(PROJECT_ROOT / "checkpoints" / "merged_qwen3_pinion_dpo_baked_fp32_20260312"),
)
OUTPUT_DIR = os.getenv(
    "RLHF_GGUF_OUTPUT",
    str(PROJECT_ROOT / "checkpoints" / "gguf_qwen3_pinion_dpo_baked_20260312"),
)
GGUF_OUTTYPE = os.getenv("RLHF_GGUF_OUTTYPE", "f16").lower()
VALID_OUTTYPES = {"f16", "f32", "bf16"}
GGUF_FILENAME = os.getenv("RLHF_GGUF_FILENAME", f"{GGUF_OUTTYPE}.gguf")
QUANT_TYPES = [
    qt.strip()
    for qt in os.getenv("RLHF_QUANT_TYPES", "Q4_K_M,Q5_K_M,Q8_0").split(",")
    if qt.strip()
]
QUANTIZE_AFTER_EXPORT = os.getenv("RLHF_QUANTIZE_AFTER_EXPORT", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LLAMA_CPP = os.getenv("RLHF_LLAMA_CPP_DIR", "/tmp/llama.cpp")
LLAMA_CPP_PYTHON = os.getenv("RLHF_LLAMA_CPP_PYTHON")
LLAMA_CPP_PYTHON_ENV = Path(
    os.getenv(
        "RLHF_LLAMA_CPP_PYTHON_ENV",
        str(Path.home() / ".cache" / "full_rlhf_pipeline" / "llama_cpp_py"),
    )
)
MODEL_NAME = os.getenv(
    "RLHF_GGUF_MODEL_NAME",
    Path(MERGED_MODEL).name.replace(" ", "-"),
)

CONVERT_SCRIPT = os.path.join(LLAMA_CPP, "convert_hf_to_gguf.py")
CONVERTER_PACKAGES = [
    "numpy",
    "torch",
    "transformers",
    "gguf",
    "sentencepiece",
    "safetensors",
    "protobuf",
    "jinja2",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, stream output, and abort on failure."""
    print(f"\n▶  {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")
    return result


def ensure_llama_cpp() -> None:
    """Clone llama.cpp if the converter script is not already available."""
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


def resolve_venv_python(venv_dir: Path) -> Path:
    """Return the platform-specific python path for a virtual environment."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def ensure_converter_python() -> str:
    """
    Return a Python interpreter suitable for llama.cpp conversion scripts.

    Preference order:
    1. `RLHF_LLAMA_CPP_PYTHON` if explicitly provided
    2. A dedicated helper venv outside the repo training environment
    """
    if LLAMA_CPP_PYTHON:
        override = Path(LLAMA_CPP_PYTHON).expanduser()
        if not override.exists():
            raise FileNotFoundError(
                f"RLHF_LLAMA_CPP_PYTHON points to a missing interpreter: {override}"
            )
        print(f"Using explicit llama.cpp converter Python: {override}")
        return str(override)

    helper_python = resolve_venv_python(LLAMA_CPP_PYTHON_ENV)
    if not helper_python.exists():
        print(f"Creating dedicated llama.cpp converter venv -> {LLAMA_CPP_PYTHON_ENV}")
        LLAMA_CPP_PYTHON_ENV.parent.mkdir(parents=True, exist_ok=True)
        venv.EnvBuilder(with_pip=True, clear=False, symlinks=(os.name != "nt")).create(
            LLAMA_CPP_PYTHON_ENV
        )

    import_probe = (
        "import importlib\n"
        "mods = ['numpy', 'torch', 'transformers', 'gguf', 'sentencepiece', "
        "'safetensors', 'google.protobuf', 'jinja2']\n"
        "missing = []\n"
        "for name in mods:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception:\n"
        "        missing.append(name)\n"
        "raise SystemExit(0 if not missing else 1)\n"
    )
    probe_result = subprocess.run(
        [str(helper_python), "-c", import_probe],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe_result.returncode != 0:
        print("Installing minimal unpinned converter dependencies into helper env ...")
        run([
            str(helper_python),
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "-q",
            *CONVERTER_PACKAGES,
        ])

    return str(helper_python)


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


def convert_to_gguf(model_dir: str, output_path: str, outtype: str, python_exe: str) -> None:
    """Convert HF safetensors to a requested GGUF precision."""
    run([
        python_exe, CONVERT_SCRIPT,
        model_dir,
        "--outfile", output_path,
        "--outtype", outtype,
    ])


def prepare_model_dir_for_conversion(model_dir: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """
    Return a conversion-safe model directory.

    Some recent transformers builds expect `extra_special_tokens` in
    tokenizer_config.json to be a dict (not a list). Qwen3 checkpoints in this
    repo use the list form, which breaks converter fallback tokenizer loading.
    We avoid mutating source checkpoints by creating a temporary symlinked view
    with a normalized tokenizer config only when needed.
    """
    cfg_path = model_dir / "tokenizer_config.json"
    if not cfg_path.exists():
        return model_dir, None

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return model_dir, None

    extra_tokens = cfg.get("extra_special_tokens")
    if not isinstance(extra_tokens, list):
        return model_dir, None

    tempdir = tempfile.TemporaryDirectory(prefix="gguf_tokfix_")
    temp_root = Path(tempdir.name)

    for entry in model_dir.iterdir():
        dst = temp_root / entry.name
        try:
            dst.symlink_to(entry.resolve())
        except Exception:
            if entry.is_dir():
                shutil.copytree(entry, dst, dirs_exist_ok=True)
            elif entry.is_file():
                shutil.copy2(entry, dst)

    # Replace symlink with a concrete normalized config in the temp view.
    tmp_cfg_path = temp_root / "tokenizer_config.json"
    if tmp_cfg_path.exists() and tmp_cfg_path.is_symlink():
        tmp_cfg_path.unlink()
    cfg["extra_special_tokens"] = {
        f"extra_token_{idx}": token for idx, token in enumerate(extra_tokens)
    }
    tmp_cfg_path.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Applied temporary tokenizer_config normalization for GGUF conversion.")
    return temp_root, tempdir


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
    if GGUF_OUTTYPE not in VALID_OUTTYPES:
        sys.exit(
            f"ERROR: Unsupported RLHF_GGUF_OUTTYPE='{GGUF_OUTTYPE}'. "
            f"Use one of: {', '.join(sorted(VALID_OUTTYPES))}."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GGUF EXPORT")
    print("=" * 60)
    print(f"  Model dir     : {model_dir}")
    print(f"  Output dir    : {out_dir}")
    print(f"  Model name    : {MODEL_NAME}")
    print(f"  GGUF outtype  : {GGUF_OUTTYPE}")
    print(f"  GGUF file     : {GGUF_FILENAME}")
    print(f"  Quantize      : {'yes' if QUANTIZE_AFTER_EXPORT else 'no'}")
    print(f"  Quant types   : {', '.join(QUANT_TYPES) if QUANT_TYPES else '(none)'}")
    print()

    # Step 1: Ensure llama.cpp is available
    print("[1/4] Ensuring llama.cpp is available ...")
    ensure_llama_cpp()
    converter_python = ensure_converter_python()
    print(f"       Converter Python: {converter_python}")

    # Step 2: Convert to GGUF
    gguf_path = str(out_dir / GGUF_FILENAME)
    print(f"\n[2/4] Converting to {GGUF_OUTTYPE} GGUF → {gguf_path} ...")
    conversion_dir, conversion_tempdir = prepare_model_dir_for_conversion(model_dir)
    try:
        convert_to_gguf(str(conversion_dir), gguf_path, GGUF_OUTTYPE, converter_python)
    finally:
        if conversion_tempdir is not None:
            conversion_tempdir.cleanup()

    if not Path(gguf_path).exists():
        sys.exit(f"ERROR: Conversion failed - {gguf_path} not found")

    gguf_size_gb = Path(gguf_path).stat().st_size / (1024 ** 3)
    print(f"       {GGUF_OUTTYPE} GGUF size: {gguf_size_gb:.2f} GB")

    results: list[tuple[str, str, float]] = []
    if QUANTIZE_AFTER_EXPORT and QUANT_TYPES and GGUF_OUTTYPE == "f16":
        # Step 3: Build / locate llama-quantize
        print("\n[3/4] Locating llama-quantize ...")
        quantize_bin = find_llama_quantize()
        if quantize_bin:
            print(f"       Found: {quantize_bin}")
        else:
            print("       Not found — building from source ...")
            quantize_bin = build_llama_quantize()
            print(f"       Built: {quantize_bin}")

        # Step 4: Quantize
        print(f"\n[4/4] Quantising ({len(QUANT_TYPES)} variants) ...")
        for qt in QUANT_TYPES:
            quant_path = str(out_dir / f"{qt}.gguf")
            print(f"\n  → {qt} ...")
            try:
                quantize_gguf(gguf_path, quant_path, qt, quantize_bin)
                size_gb = Path(quant_path).stat().st_size / (1024 ** 3)
                results.append((qt, quant_path, size_gb))
            except RuntimeError as exc:
                print(f"  ✗ {qt} failed: {exc}")
    elif QUANTIZE_AFTER_EXPORT and QUANT_TYPES and GGUF_OUTTYPE != "f16":
        print("\n[3/4] Quantization requested but skipped.")
        print(f"       Current outtype is '{GGUF_OUTTYPE}'. Quantization expects f16 input.")
        print("[4/4] Done after base GGUF export.")
    else:
        print("\n[3/4] Quantization skipped by policy.")
        print("[4/4] Done after base GGUF export.")

    # Summary
    print("\n" + "=" * 60)
    print("GGUF EXPORT COMPLETE")
    print("=" * 60)
    print(f"  {'Type':<12} {'Size':>8}  Path")
    print(f"  {'─'*12} {'─'*8}  {'─'*40}")
    print(f"  {GGUF_OUTTYPE:<12} {gguf_size_gb:>7.2f}G  {gguf_path}")
    for qt, path, size in results:
        print(f"  {qt:<12} {size:>7.2f}G  {path}")
    print()
    print("Use with llama.cpp:")
    print(f"  llama-cli -m {gguf_path} -p 'Hello'")
    print("Use with Ollama:")
    print(f"  ollama create {MODEL_NAME} -f Modelfile")
    print("=" * 60)


if __name__ == "__main__":
    main()
