#!/usr/bin/env python3
"""
SimPO-Only Script - Resume from SFT Checkpoint
Runs ONLY SimPO preference optimization on an already-SFT'd model.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer

from rlhf import (
    PolicyModel,
    SimPOTrainer,
    SimPOConfig,
    DeviceManager,
    PreferenceDataset,
    setup_logging
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logger = setup_logging(level=logging.INFO)

# Same preference data as main script
PREFERENCE_DATA = [
    {"prompt": "Write a function to find the sum of an array.", "chosen": "```python\ndef array_sum(arr: list) -> int | float:\n    \"\"\"Calculate the sum of all elements in an array.\n    \n    Args:\n        arr: A list of numbers (int or float)\n        \n    Returns:\n        The sum of all elements\n    \"\"\"\n    return sum(arr)\n```", "rejected": "def sum(a): return sum(a)"},
    {"prompt": "Explain what an API is.", "chosen": "An API (Application Programming Interface) is a set of rules and protocols that allows different software applications to communicate with each other. Think of it as a waiter in a restaurant: you (the client) give your order to the waiter (the API), who takes it to the kitchen (the server), and brings back your food (the response). APIs define what requests you can make, how to make them, and what format the response will be in.", "rejected": "API means application programming interface. It lets programs talk to each other."},
    {"prompt": "How do I handle errors in Python?", "chosen": "Python uses try-except blocks for error handling:\n\n```python\ntry:\n    result = risky_operation()\nexcept ValueError as e:\n    print(f\"Invalid value: {e}\")\nexcept Exception as e:\n    logger.error(f\"Unexpected error: {e}\")\n    raise\nelse:\n    print(\"Success!\")\nfinally:\n    cleanup()\n```\n\nBest practices: catch specific exceptions, don't silence errors, log useful information.", "rejected": "Use try except. Put code in try block and handle errors in except."},
    {"prompt": "What's the best programming language to learn first?", "chosen": "Python is widely recommended for beginners: 1) Clean, readable syntax. 2) Versatile - used in web development, data science, automation, and AI. 3) Huge community with abundant learning resources. 4) Immediate feedback through interactive interpreter. 5) In-demand skill with good job prospects. However, the 'best' language depends on your goals: JavaScript for web, Swift for iOS, or C++ for games. The key is to pick one and build projects.", "rejected": "learn python its easy"},
    {"prompt": "Explain inheritance in object-oriented programming.", "chosen": "Inheritance is an OOP principle where a class (child) can inherit properties and methods from another class (parent). This promotes code reuse and establishes an 'is-a' relationship.\n\n```python\nclass Animal:\n    def __init__(self, name):\n        self.name = name\n\nclass Dog(Animal):\n    def speak(self):\n        return f\"{self.name} says Woof!\"\n```\n\nDog inherits `__init__` from Animal. Python also supports multiple inheritance.", "rejected": "Inheritance means a class can get stuff from another class. Like Dog extends Animal."},
    {"prompt": "How do databases index data?", "chosen": "Database indexes are data structures that speed up data retrieval. The most common is the B-tree index, which organizes data in a balanced tree allowing O(log n) lookups. Hash indexes provide O(1) lookups for exact matches. For text search, inverted indexes map words to documents. Creating an index: `CREATE INDEX idx_email ON users(email);`. Indexes speed up reads but slow down writes. Choose columns that are frequently queried and have high cardinality.", "rejected": "Indexes make databases faster. They're like a table of contents."},
]


def load_sft_checkpoint(checkpoint_path: str, model_name: str):
    """Load model from SFT checkpoint."""
    logger.info(f"Loading SFT checkpoint from: {checkpoint_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Create policy model
    policy_model = PolicyModel(model_name, use_gradient_checkpointing=True)
    
    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        policy_model.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        policy_model.model.load_state_dict(checkpoint)
    
    logger.info("Checkpoint loaded successfully!")
    
    # Apply LoRA if available
    if PEFT_AVAILABLE:
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
            logger.info("Applied LoRA")
        except Exception as e:
            logger.warning(f"Failed to apply LoRA: {e}")
    
    return policy_model, tokenizer


def run_simpo_only(policy_model, tokenizer, preference_data, config: SimPOConfig, device_manager):
    """Run SimPO only."""
    logger.info("=" * 60)
    logger.info("SimPO (Reference-Free Preference Optimization)")
    logger.info("=" * 60)
    
    from torch.utils.data import DataLoader
    
    dataset = PreferenceDataset(preference_data, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    trainer = SimPOTrainer(policy_model, config, device_manager)
    trainer.train(dataloader)
    
    logger.info("SimPO completed!")
    return policy_model


def main():
    parser = argparse.ArgumentParser(description="Run SimPO only from SFT checkpoint")
    parser.add_argument("--checkpoint", type=str, 
                        default="checkpoints/qwen3_1.7b/sft/SFT_step_0/model.pt",
                        help="Path to SFT checkpoint")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="Base model name (for tokenizer)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of SimPO epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate for SimPO")
    parser.add_argument("--output-dir", type=str, default="checkpoints/qwen3_1.7b/simpo",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("SimPO-Only Training (from SFT checkpoint)")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {args.device}")
    
    # Load from checkpoint
    policy_model, tokenizer = load_sft_checkpoint(args.checkpoint, args.model)
    
    # Device manager
    device_manager = DeviceManager(device=args.device, use_amp=(args.device != "cpu"))
    
    # SimPO config
    simpo_config = SimPOConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        beta=2.0,
        gamma=0.5,
        output_dir=args.output_dir,
        gradient_accumulation_steps=16,
    )
    
    # Run SimPO
    policy_model = run_simpo_only(policy_model, tokenizer, PREFERENCE_DATA, simpo_config, device_manager)
    
    # Save final
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    
    logger.info(f"Saving SimPO model to {final_dir}")
    policy_model.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    logger.info("=" * 60)
    logger.info("SimPO Training Complete!")
    logger.info(f"Model saved to: {final_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
