"""
Bake the SFT LoRA adapter into the base Qwen3-1.7B weights.

Produces a single standalone model — no more base + adapter spaghetti.
Uses mmap loading so RAM stays low.
"""

import gc
import os
import warnings

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

BASE_MODEL = os.getenv("RLHF_BASE_MODEL", "Qwen/Qwen3-1.7B")
ADAPTER_PATH = os.getenv("RLHF_SFT_ADAPTER", "checkpoints/checkpoints/full_pipeline/sft")
OUTPUT_PATH = os.getenv("RLHF_MERGED_OUTPUT_DIR", "checkpoints/merged_sft_qwen3_1.7b")
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


def main():
    print("=" * 60)
    print("MERGE LoRA INTO BASE — single model output")
    print("=" * 60)
    print(f"Base model      : {BASE_MODEL}")
    print(f"Adapter path    : {ADAPTER_PATH}")
    print(f"Output path     : {OUTPUT_PATH}")
    print(f"Output dtype    : {OUTPUT_DTYPE}")
    print(f"Max shard size  : {MAX_SHARD_SIZE}")

    # 1. Tokenizer (from adapter dir — has the chat template)
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)

    # 2. Base model via mmap (streams from disk, low RAM)
    print(f"[2/4] Loading base model ({BASE_MODEL}) via mmap...")
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
        # Backward compatibility with older transformers that still require torch_dtype
        load_kwargs.pop("dtype", None)
        load_kwargs["torch_dtype"] = load_dtype
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_kwargs)

    # 3. Apply LoRA then merge_and_unload → bakes weights in-place
    print(f"[3/4] Applying LoRA from {ADAPTER_PATH} and merging...")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model = model.merge_and_unload()  # ← this is the money line
    print(f"       Merged model params: {sum(p.numel() for p in model.parameters()):,}")
    del base
    gc.collect()

    # Optional post-merge dtype cast to reduce disk footprint and save time.
    output_dtype = _resolve_dtype(OUTPUT_DTYPE)
    if output_dtype != torch.float32:
        print(f"       Casting merged model to {OUTPUT_DTYPE} before save...")
        model = model.to(dtype=output_dtype)
        gc.collect()

    # 4. Save as standalone model
    print(f"[4/4] Saving merged model to {OUTPUT_PATH} (max_shard_size={MAX_SHARD_SIZE})...")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    model.save_pretrained(
        OUTPUT_PATH,
        safe_serialization=True,
        max_shard_size=MAX_SHARD_SIZE,
    )
    tokenizer.save_pretrained(OUTPUT_PATH)

    # Quick sanity: list what got written
    written = list(Path(OUTPUT_PATH).iterdir())
    print(f"\nWrote {len(written)} files:")
    for f in sorted(written):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:40s}  {size_mb:>8.2f} MB")

    print("\n" + "=" * 60)
    print("DONE — use this path for all downstream stages:")
    print(f"  {os.path.abspath(OUTPUT_PATH)}")
    print("No more base + adapter loading needed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
