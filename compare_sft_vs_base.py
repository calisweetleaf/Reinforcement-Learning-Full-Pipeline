#!/usr/bin/env python3
"""
A/B comparison: base Qwen3-1.7B vs your SFT fine-tune.

Runs identical prompts through both models via Ollama and prints
side-by-side results so you can judge whether MaggiePie 300k training
made a real difference.

Usage:
    python compare_sft_vs_base.py
"""

import json
import subprocess
import textwrap
import time
import sys

BASE_MODEL = "qwen3:1.7b"
SFT_MODEL  = "qwen3-1.7b-sft:latest"

# ── Test prompts covering different capabilities ─────────────────────────────
PROMPTS = [
    # 1. Instruction following — conciseness
    "Explain what a neural network is in exactly two sentences.",

    # 2. Reasoning
    "If a train leaves at 9 AM going 60 mph and another leaves at 10 AM going 90 mph on the same track, when does the second train catch up?",

    # 3. Code generation
    "Write a Python function that checks if a string is a palindrome.",

    # 4. Creative writing
    "Write a haiku about debugging code.",

    # 5. Refusal / honesty
    "What happened on January 32nd, 2025?",

    # 6. Helpful formatting
    "Give me a markdown table comparing Python, Rust, and Go on speed, safety, and ease of learning.",

    # 7. Conversational / personality
    "Hey, I'm having a rough day. Can you cheer me up?",

    # 8. Multi-step task
    "I have a CSV with columns: name, age, salary. Write a Python script that reads it and prints the average salary of people over 30.",
]


def query_ollama(model: str, prompt: str, timeout: int = 120) -> dict:
    """Query an Ollama model and return response + timing."""
    start = time.time()
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt, "--nowordwrap"],
            capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.time() - start
        output = result.stdout.strip()
        # Separate think block from response
        think = ""
        response = output
        if "<think>" in output and "</think>" in output:
            think = output.split("<think>")[1].split("</think>")[0].strip()
            response = output.split("</think>")[-1].strip()
        return {
            "response": response,
            "think": think,
            "full": output,
            "elapsed": elapsed,
            "error": result.stderr.strip() if result.returncode != 0 else None,
        }
    except subprocess.TimeoutExpired:
        return {"response": "[TIMEOUT]", "think": "", "full": "", "elapsed": timeout, "error": "timeout"}


def wrap_text(text: str, width: int = 72, prefix: str = "  ") -> str:
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        if len(line) <= width:
            wrapped.append(prefix + line)
        else:
            wrapped.extend(prefix + l for l in textwrap.wrap(line, width))
    return "\n".join(wrapped)


def main():
    print("=" * 76)
    print("  A/B COMPARISON: Base Qwen3-1.7B  vs  Your SFT Fine-Tune")
    print(f"  Base : {BASE_MODEL}")
    print(f"  SFT  : {SFT_MODEL}")
    print(f"  Prompts: {len(PROMPTS)}")
    print("=" * 76)

    results = []

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n{'─' * 76}")
        print(f"  PROMPT {i}/{len(PROMPTS)}:")
        print(wrap_text(prompt, prefix="  > "))
        print(f"{'─' * 76}")

        # Query both
        print(f"  Querying {BASE_MODEL} ...")
        base_result = query_ollama(BASE_MODEL, prompt)
        print(f"  Querying {SFT_MODEL} ...")
        sft_result = query_ollama(SFT_MODEL, prompt)

        # Display
        print(f"\n  ┌── BASE ({base_result['elapsed']:.1f}s) ──")
        print(wrap_text(base_result["response"], prefix="  │ "))
        print(f"  └{'─' * 40}")

        print(f"\n  ┌── SFT ({sft_result['elapsed']:.1f}s) ──")
        print(wrap_text(sft_result["response"], prefix="  │ "))
        print(f"  └{'─' * 40}")

        results.append({
            "prompt": prompt,
            "base": base_result,
            "sft": sft_result,
        })

    # ── Summary stats ────────────────────────────────────────────────────
    print(f"\n{'=' * 76}")
    print("  TIMING SUMMARY")
    print(f"{'=' * 76}")
    base_times = [r["base"]["elapsed"] for r in results]
    sft_times  = [r["sft"]["elapsed"]  for r in results]
    print(f"  Base avg response time: {sum(base_times)/len(base_times):.1f}s")
    print(f"  SFT  avg response time: {sum(sft_times)/len(sft_times):.1f}s")

    base_lens = [len(r["base"]["response"]) for r in results]
    sft_lens  = [len(r["sft"]["response"])  for r in results]
    print(f"  Base avg response length: {sum(base_lens)//len(base_lens)} chars")
    print(f"  SFT  avg response length: {sum(sft_lens)//len(sft_lens)} chars")

    # Save full results
    out_path = "checkpoints/sft_vs_base_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full results saved to: {out_path}")
    print("=" * 76)
    print("  Review the outputs above. Key things to look for:")
    print("    • Better instruction following (conciseness, format)")
    print("    • More accurate / less hallucinated responses")
    print("    • Improved code quality")
    print("    • Better tone / personality")
    print("    • Honest refusals on impossible questions")
    print("=" * 76)


if __name__ == "__main__":
    main()
