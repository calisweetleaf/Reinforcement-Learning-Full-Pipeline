"""
Bake the SFT LoRA adapter into the base Qwen3-1.7B weights.

Produces a single standalone model — no more base + adapter spaghetti.
Uses mmap loading so RAM stays low.
"""

import torch
import os
import warnings
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

BASE_MODEL = "Qwen/Qwen3-1.7B"
ADAPTER_PATH = "checkpoints/checkpoints/full_pipeline/sft"
OUTPUT_PATH = "checkpoints/merged_sft_qwen3_1.7b"


def main():
    print("=" * 60)
    print("MERGE LoRA INTO BASE — single model output")
    print("=" * 60)

    # 1. Tokenizer (from adapter dir — has the chat template)
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)

    # 2. Base model via mmap (streams from disk, low RAM)
    print(f"[2/4] Loading base model ({BASE_MODEL}) via mmap...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        trust_remote_code=True,
    )

    # 3. Apply LoRA then merge_and_unload → bakes weights in-place
    print(f"[3/4] Applying LoRA from {ADAPTER_PATH} and merging...")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model = model.merge_and_unload()  # ← this is the money line
    print(f"       Merged model params: {sum(p.numel() for p in model.parameters()):,}")

    # 4. Save as standalone model
    print(f"[4/4] Saving merged model to {OUTPUT_PATH}...")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
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
