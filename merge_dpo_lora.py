"""
Bake the DPO LoRA adapter into an already-merged SFT base model.

Input:
- RLHF_SFT_MERGED_BASE: merged SFT base checkpoint directory
- RLHF_DPO_ADAPTER: DPO adapter directory (expects adapter_config + weights)

Output:
- RLHF_DPO_MERGED_OUTPUT_DIR: standalone merged model directory
"""

import gc
import os
import warnings
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

BASE_MODEL = os.getenv("RLHF_SFT_MERGED_BASE", "checkpoints/merged_sft_qwen3_1.7b")
DPO_ADAPTER = os.getenv("RLHF_DPO_ADAPTER", "checkpoints/dpo_qwen3_1.7b/final")
OUTPUT_PATH = os.getenv("RLHF_DPO_MERGED_OUTPUT_DIR", "checkpoints/merged_dpo_qwen3_1.7b")
OUTPUT_DTYPE = os.getenv("RLHF_MERGED_DTYPE", "float32")
MAX_SHARD_SIZE = os.getenv("RLHF_MERGE_MAX_SHARD_SIZE", "1GB")


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(
            f"Unsupported RLHF_MERGED_DTYPE='{name}'. "
            f"Use one of: {', '.join(mapping.keys())}"
        )
    return mapping[name]


def _validate_dpo_adapter(adapter_path: str) -> Path:
    p = Path(adapter_path)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"DPO adapter directory not found: {p}")

    config = p / "adapter_config.json"
    safetensors = p / "adapter_model.safetensors"
    bin_file = p / "adapter_model.bin"

    if not config.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {p}")
    if not safetensors.exists() and not bin_file.exists():
        raise FileNotFoundError(
            f"Missing adapter weights in {p} (expected adapter_model.safetensors or adapter_model.bin)"
        )

    return p


def _load_tokenizer(adapter_path: Path, base_model: str):
    try:
        return AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    except Exception:
        return AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)


def main():
    adapter_path = _validate_dpo_adapter(DPO_ADAPTER)

    print("=" * 60)
    print("MERGE DPO ADAPTER ON TOP OF MERGED SFT BASE")
    print("=" * 60)
    print(f"Merged SFT base : {BASE_MODEL}")
    print(f"DPO adapter     : {adapter_path}")
    print(f"Output path     : {OUTPUT_PATH}")
    print(f"Output dtype    : {OUTPUT_DTYPE}")
    print(f"Max shard size  : {MAX_SHARD_SIZE}")

    print("\n[1/4] Loading tokenizer...")
    tokenizer = _load_tokenizer(adapter_path, BASE_MODEL)

    print(f"[2/4] Loading merged SFT base ({BASE_MODEL}) via mmap...")
    load_dtype = torch.float32
    load_kwargs = dict(
        low_cpu_mem_usage=True,
        use_safetensors=True,
        trust_remote_code=True,
        dtype=load_dtype,
    )
    try:
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_kwargs)
    except TypeError:
        load_kwargs.pop("dtype", None)
        load_kwargs["torch_dtype"] = load_dtype
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_kwargs)

    print(f"[3/4] Applying DPO adapter from {adapter_path} and merging...")
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model = model.merge_and_unload()
    print(f"       Merged model params: {sum(p.numel() for p in model.parameters()):,}")

    del base
    gc.collect()

    output_dtype = _resolve_dtype(OUTPUT_DTYPE)
    if output_dtype != torch.float32:
        print(f"       Casting merged model to {OUTPUT_DTYPE} before save...")
        model = model.to(dtype=output_dtype)
        gc.collect()

    print(f"[4/4] Saving merged model to {OUTPUT_PATH} (max_shard_size={MAX_SHARD_SIZE})...")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    model.save_pretrained(
        OUTPUT_PATH,
        safe_serialization=True,
        max_shard_size=MAX_SHARD_SIZE,
    )
    tokenizer.save_pretrained(OUTPUT_PATH)

    written = list(Path(OUTPUT_PATH).iterdir())
    print(f"\nWrote {len(written)} files:")
    for f in sorted(written):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:40s}  {size_mb:>8.2f} MB")

    print("\n" + "=" * 60)
    print("DONE â€” merged DPO-on-SFT model ready:")
    print(f"  {os.path.abspath(OUTPUT_PATH)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
