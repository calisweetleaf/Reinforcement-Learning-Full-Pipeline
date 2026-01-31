#!/usr/bin/env python3
"""
Qwen3 1.7B RLHF Training Script
The Ultimate Tune-Up - Memory Efficient CPU Training

This script runs the full RLHF pipeline on Qwen3 1.7B with:
- QLoRA for memory efficiency
- SimPO (reference-free) to avoid loading 2 models
- Small but high-quality datasets
- CPU-optimized settings

Usage:
    python scripts/train_qwen3_1.7b.py --method simpo --epochs 2
    python scripts/train_qwen3_1.7b.py --method dpo --epochs 1
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import from our RLHF pipeline
from rlhf import (
    RLHFOrchestrator,
    PolicyModel,
    SFTTrainer,
    DPOTrainer,
    SimPOTrainer,
    SFTConfig,
    DPOConfig,
    SimPOConfig,
    DeviceManager,
    SFTDataset,
    PreferenceDataset,
    setup_logging
)

# Optional PEFT imports
try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

logger = setup_logging(level=logging.INFO)

# ==============================================================================
# SAMPLE DATASETS - High quality, small size
# ==============================================================================

SFT_DATA = [
    {"prompt": "Explain the concept of recursion in programming.", "response": "Recursion is a programming technique where a function calls itself to solve a problem. It works by breaking down complex problems into smaller, identical subproblems. Every recursive function needs a base case (stopping condition) and a recursive case (where it calls itself). For example, calculating factorial: factorial(n) = n * factorial(n-1), with base case factorial(0) = 1. Recursion is powerful for tree traversals, divide-and-conquer algorithms, and problems with naturally recursive structures."},
    {"prompt": "What are the main differences between Python and JavaScript?", "response": "Python and JavaScript differ in several key ways: 1) Python uses indentation for blocks while JavaScript uses curly braces. 2) Python is dynamically typed with optional type hints, JavaScript is dynamically typed with TypeScript as an alternative. 3) Python excels in data science, ML, and scripting; JavaScript dominates web development. 4) Python has simpler syntax and is more readable; JavaScript has more complex scoping rules. 5) Python uses pip for packages; JavaScript uses npm/yarn."},
    {"prompt": "How do neural networks learn?", "response": "Neural networks learn through backpropagation with gradient descent. Data passes forward through layers of neurons, each applying weights and activation functions. The output is compared to the target using a loss function. Gradients of this loss are computed backwards through the network. Weights are updated in the direction that reduces the loss. This cycle repeats over many examples until predictions improve. Learning rate controls update magnitude, and techniques like batch normalization and dropout prevent overfitting."},
    {"prompt": "Write a Python function to check if a string is a palindrome.", "response": "```python\ndef is_palindrome(s: str) -> bool:\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n\n# Examples:\n# is_palindrome(\"A man, a plan, a canal: Panama\") -> True\n# is_palindrome(\"race a car\") -> False\n```"},
    {"prompt": "Explain quantum entanglement in simple terms.", "response": "Quantum entanglement is a phenomenon where two particles become connected such that measuring one instantly affects the other, regardless of distance. Imagine two magic coins: when you flip one and it lands heads, the other instantly becomes tails, even across the universe. Einstein called this 'spooky action at a distance.' Entanglement is real, experimentally verified, and is the basis for quantum computing and cryptography."},
    {"prompt": "What is the difference between machine learning and deep learning?", "response": "Machine learning (ML) is a broad field where algorithms learn patterns from data. Deep learning (DL) is a subset of ML using neural networks with many layers. Key differences: 1) ML includes various algorithms (decision trees, SVMs); DL specifically uses neural networks. 2) ML often requires manual feature engineering; DL learns features automatically. 3) DL excels with unstructured data (images, audio, text); classical ML works well with structured data. 4) DL needs more data and compute power."},
    {"prompt": "How would you design a rate limiter for an API?", "response": "A rate limiter controls request frequency. Common approaches: 1) Token Bucket: Users have tokens that refill at a fixed rate. Each request consumes one token. 2) Sliding Window Log: Track timestamps of recent requests, reject if count exceeds limit. 3) Sliding Window Counter: Combine fixed window counts with weighted overlap. Implementation: use Redis for distributed rate limiting, include user identification (API key, IP), return appropriate headers (X-RateLimit-Remaining), and implement backoff strategies."},
    {"prompt": "Explain the CAP theorem in distributed systems.", "response": "The CAP theorem states that a distributed system can only guarantee two of three properties: Consistency (all nodes see same data), Availability (every request gets a response), and Partition tolerance (system continues despite network failures). Since network partitions are inevitable, you choose between CP or AP systems. CP systems sacrifice availability during partitions to maintain consistency. AP systems remain available but may serve stale data. Most modern systems make nuanced tradeoffs based on use cases."},
]

PREFERENCE_DATA = [
    {"prompt": "Write a function to find the sum of an array.", "chosen": "```python\ndef array_sum(arr: list) -> int | float:\n    \"\"\"Calculate the sum of all elements in an array.\n    \n    Args:\n        arr: A list of numbers (int or float)\n        \n    Returns:\n        The sum of all elements\n    \"\"\"\n    return sum(arr)\n```", "rejected": "def sum(a): return sum(a)"},
    {"prompt": "Explain what an API is.", "chosen": "An API (Application Programming Interface) is a set of rules and protocols that allows different software applications to communicate with each other. Think of it as a waiter in a restaurant: you (the client) give your order to the waiter (the API), who takes it to the kitchen (the server), and brings back your food (the response). APIs define what requests you can make, how to make them, and what format the response will be in.", "rejected": "API means application programming interface. It lets programs talk to each other."},
    {"prompt": "How do I handle errors in Python?", "chosen": "Python uses try-except blocks for error handling:\n\n```python\ntry:\n    result = risky_operation()\nexcept ValueError as e:\n    print(f\"Invalid value: {e}\")\nexcept Exception as e:\n    logger.error(f\"Unexpected error: {e}\")\n    raise\nelse:\n    print(\"Success!\")\nfinally:\n    cleanup()\n```\n\nBest practices: catch specific exceptions, don't silence errors, log useful information.", "rejected": "Use try except. Put code in try block and handle errors in except."},
    {"prompt": "What's the best programming language to learn first?", "chosen": "Python is widely recommended for beginners: 1) Clean, readable syntax. 2) Versatile - used in web development, data science, automation, and AI. 3) Huge community with abundant learning resources. 4) Immediate feedback through interactive interpreter. 5) In-demand skill with good job prospects. However, the 'best' language depends on your goals: JavaScript for web, Swift for iOS, or C++ for games. The key is to pick one and build projects.", "rejected": "learn python its easy"},
    {"prompt": "Explain inheritance in object-oriented programming.", "chosen": "Inheritance is an OOP principle where a class (child) can inherit properties and methods from another class (parent). This promotes code reuse and establishes an 'is-a' relationship.\n\n```python\nclass Animal:\n    def __init__(self, name):\n        self.name = name\n\nclass Dog(Animal):\n    def speak(self):\n        return f\"{self.name} says Woof!\"\n```\n\nDog inherits `__init__` from Animal. Python also supports multiple inheritance.", "rejected": "Inheritance means a class can get stuff from another class. Like Dog extends Animal."},
    {"prompt": "How do databases index data?", "chosen": "Database indexes are data structures that speed up data retrieval. The most common is the B-tree index, which organizes data in a balanced tree allowing O(log n) lookups. Hash indexes provide O(1) lookups for exact matches. For text search, inverted indexes map words to documents. Creating an index: `CREATE INDEX idx_email ON users(email);`. Indexes speed up reads but slow down writes. Choose columns that are frequently queried and have high cardinality.", "rejected": "Indexes make databases faster. They're like a table of contents."},
]


def create_data_files(output_dir: str = "data"):
    """Create sample data files for training."""
    os.makedirs(output_dir, exist_ok=True)
    
    sft_path = os.path.join(output_dir, "sft_train.json")
    with open(sft_path, 'w', encoding='utf-8') as f:
        json.dump(SFT_DATA, f, indent=2, ensure_ascii=False)
    logger.info(f"Created SFT dataset: {sft_path} ({len(SFT_DATA)} examples)")
    
    pref_path = os.path.join(output_dir, "preference_train.json")
    with open(pref_path, 'w', encoding='utf-8') as f:
        json.dump(PREFERENCE_DATA, f, indent=2, ensure_ascii=False)
    logger.info(f"Created preference dataset: {pref_path} ({len(PREFERENCE_DATA)} examples)")
    
    return sft_path, pref_path


def load_model_for_training(model_name: str, use_qlora: bool = True):
    """Load model with memory-efficient settings."""
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Quantization config for 4-bit loading (only works with CUDA)
    quantization_config = None
    if use_qlora and BNB_AVAILABLE and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization (QLoRA)")
        except Exception as e:
            logger.warning(f"QLoRA not available: {e}")
            quantization_config = None
    
    # Load base model directly, then wrap in PolicyModel
    if quantization_config is not None:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map="auto",
        )
    else:
        # FP32 for CPU
        logger.info("Loading model in FP32 (CPU mode)")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
    
    # Enable gradient checkpointing for memory efficiency
    base_model.gradient_checkpointing_enable()
    
    # Wrap in PolicyModel by passing pre-loaded model
    policy_model = PolicyModel(
        model_name,
        use_gradient_checkpointing=True,
        model=base_model
    )
    
    # Apply LoRA if available and CUDA present
    if PEFT_AVAILABLE and use_qlora and torch.cuda.is_available():
        try:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            if quantization_config is not None:
                policy_model.model = prepare_model_for_kbit_training(policy_model.model)
            policy_model.model = get_peft_model(policy_model.model, lora_config)
            policy_model.model.print_trainable_parameters()
        except Exception as e:
            logger.warning(f"Failed to apply LoRA: {e}")
    elif PEFT_AVAILABLE and use_qlora:
        # LoRA on CPU (without quantization)
        try:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            policy_model.model = get_peft_model(policy_model.model, lora_config)
            policy_model.model.print_trainable_parameters()
            logger.info("Applied LoRA (CPU mode, no quantization)")
        except Exception as e:
            logger.warning(f"Failed to apply LoRA on CPU: {e}")
    
    return policy_model, tokenizer


def run_sft(policy_model, tokenizer, sft_data, config: SFTConfig, device_manager):
    """Run supervised fine-tuning."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Supervised Fine-Tuning (SFT)")
    logger.info("=" * 60)
    
    from torch.utils.data import DataLoader
    
    dataset = SFTDataset(sft_data, tokenizer, max_length=config.max_seq_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    trainer = SFTTrainer(policy_model, tokenizer, config, device_manager)
    trainer.train(dataloader)
    
    logger.info("SFT completed!")
    return policy_model


def run_simpo(policy_model, tokenizer, preference_data, config: SimPOConfig, device_manager):
    """Run SimPO (reference-free, memory efficient)."""
    logger.info("=" * 60)
    logger.info("STAGE 2: SimPO (Reference-Free Preference Optimization)")
    logger.info("=" * 60)
    
    from torch.utils.data import DataLoader
    
    dataset = PreferenceDataset(preference_data, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    trainer = SimPOTrainer(policy_model, config, device_manager)
    trainer.train(dataloader)
    
    logger.info("SimPO completed!")
    return policy_model


def run_dpo(policy_model, tokenizer, preference_data, config: DPOConfig, device_manager):
    """Run DPO (requires reference model)."""
    logger.info("=" * 60)
    logger.info("STAGE 2: DPO (Direct Preference Optimization)")
    logger.info("=" * 60)
    
    from torch.utils.data import DataLoader
    
    # Create reference model (frozen copy)
    logger.info("Creating reference model (frozen copy)...")
    reference_model = PolicyModel(policy_model.base_model_name)
    reference_model.model.eval()
    for param in reference_model.model.parameters():
        param.requires_grad = False
    
    dataset = PreferenceDataset(preference_data, tokenizer, max_length=config.max_seq_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    trainer = DPOTrainer(policy_model, reference_model, config, device_manager)
    trainer.train(dataloader)
    
    logger.info("DPO completed!")
    return policy_model


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3 1.7B with RLHF")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="Model name or path")
    parser.add_argument("--method", type=str, default="simpo", choices=["sft", "simpo", "dpo"],
                        help="Training method (simpo recommended for CPU)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--output-dir", type=str, default="checkpoints/qwen3_1.7b",
                        help="Output directory for checkpoints")
    parser.add_argument("--no-qlora", action="store_true", help="Disable QLoRA/LoRA")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Qwen3 1.7B RLHF Training - The Ultimate Tune-Up")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Device: {args.device}")
    logger.info(f"LoRA: {not args.no_qlora}")
    
    # Create sample data
    sft_path, pref_path = create_data_files()
    
    with open(sft_path, 'r') as f:
        sft_data = json.load(f)
    with open(pref_path, 'r') as f:
        preference_data = json.load(f)
    
    # Load model
    policy_model, tokenizer = load_model_for_training(
        args.model, 
        use_qlora=not args.no_qlora
    )
    
    # Device manager
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_manager = DeviceManager(device=device, use_amp=(device != "cpu"))
    logger.info(f"Using device: {device_manager.device}")
    
    # Stage 1: SFT (always run first if doing full pipeline)
    if args.method in ["sft", "simpo", "dpo"]:
        sft_config = SFTConfig(
            learning_rate=args.lr,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            max_seq_length=args.max_length,
            output_dir=os.path.join(args.output_dir, "sft"),
            gradient_accumulation_steps=16,
            use_lora=not args.no_qlora,
        )
        policy_model = run_sft(policy_model, tokenizer, sft_data, sft_config, device_manager)
    
    # Stage 2: Preference optimization
    if args.method == "simpo":
        simpo_config = SimPOConfig(
            learning_rate=args.lr / 10,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            beta=2.0,
            gamma=0.5,
            output_dir=os.path.join(args.output_dir, "simpo"),
            gradient_accumulation_steps=16,
        )
        policy_model = run_simpo(policy_model, tokenizer, preference_data, simpo_config, device_manager)
        
    elif args.method == "dpo":
        dpo_config = DPOConfig(
            learning_rate=args.lr / 10,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            beta=0.1,
            output_dir=os.path.join(args.output_dir, "dpo"),
            gradient_accumulation_steps=16,
        )
        policy_model = run_dpo(policy_model, tokenizer, preference_data, dpo_config, device_manager)
    
    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    
    logger.info(f"Saving final model to {final_dir}")
    policy_model.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Model saved to: {final_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
