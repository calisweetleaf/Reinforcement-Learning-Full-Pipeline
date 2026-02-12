#!/usr/bin/env python3.12
"""
Full RLHF Pipeline with Curated Datasets
=========================================

Runs the complete pipeline:
1. SFT: Magpie-Align/Magpie-Pro-300K-Filtered
2. Reward Model: nvidia/HelpSteer2
3. DPO: argilla/distilabel-intel-orca-dpo-pairs
4. SimPO: HuggingFaceH4/ultrafeedback_binarized
5. (Optional) GRPO: AI-MO/NuminaMath-CoT
6. (Optional) KTO: trl-lib/kto-mix-14k

Optimized for CPU/low-memory VPS (8GB RAM).
"""

import os
import sys
import json
import argparse
import logging
import gc
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

# Import from RLHF pipeline
from rlhf import (
    PolicyModel,
    RewardModel,
    SFTTrainer,
    RewardModelTrainer,
    DPOTrainer,
    SimPOTrainer,
    GRPOTrainer,
    KTOTrainer,
    SFTConfig,
    RewardModelConfig,
    DPOConfig,
    SimPOConfig,
    GRPOConfig,
    KTOConfig,
    DeviceManager,
    SFTDataset,
    PreferenceDataset,
    KTODataset,
    GRPODataset,
    setup_logging,
)

# Import dataset loader
from scripts.dataset_loader import load_dataset_for_stage, DATASET_CONFIGS

# Optional PEFT
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logger = setup_logging(level=logging.INFO)

# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

PIPELINE_STAGES = {
    "sft": {
        "dataset": "sft",
        "description": "Supervised Fine-Tuning on Magpie-Pro-300K",
        "required": True,
    },
    "reward_model": {
        "dataset": "reward_model",
        "description": "Reward Model on HelpSteer2",
        "required": False,  # Only needed for PPO
    },
    "dpo": {
        "dataset": "dpo",
        "description": "Direct Preference Optimization on Orca-DPO",
        "required": False,
    },
    "simpo": {
        "dataset": "simpo",
        "description": "Simple Preference Optimization on UltraFeedback",
        "required": False,
    },
    "grpo": {
        "dataset": "grpo",
        "description": "Group Relative Policy Optimization on NuminaMath",
        "required": False,
    },
    "kto": {
        "dataset": "kto",
        "description": "Kahneman-Tversky Optimization on KTO-mix",
        "required": False,
    },
}


def clear_memory():
    """Aggressively clear memory between stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_base_model(model_name: str, checkpoint_path: Optional[str] = None):
    """Load model with memory-mapped weights (no RAM limits)."""
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model with mmap - streams weights from disk, not RAM
    logger.info("Loading model with mmap (streaming from disk)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        # Use mmap to stream weights from disk instead of loading to RAM
        use_safetensors=True,
    )
    
    # Enable gradient checkpointing
    base_model.gradient_checkpointing_enable()
    
    # Wrap in PolicyModel
    policy_model = PolicyModel(
        model_name,
        use_gradient_checkpointing=True,
        model=base_model
    )
    
    # Apply LoRA
    if PEFT_AVAILABLE:
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
    
    # Load from checkpoint if specified
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading weights from {checkpoint_path}")
        from peft import PeftModel
        policy_model.model = PeftModel.from_pretrained(
            base_model, checkpoint_path
        )
    
    return policy_model, tokenizer


def run_sft_stage(
    policy_model,
    tokenizer,
    device_manager,
    max_samples: int,
    epochs: int,
    lr: float,
    output_dir: str
):
    """Stage 1: Supervised Fine-Tuning."""
    logger.info("=" * 70)
    logger.info("STAGE 1: Supervised Fine-Tuning (SFT)")
    logger.info("Dataset: Magpie-Align/Magpie-Pro-300K-Filtered")
    logger.info("=" * 70)
    
    # Load data
    sft_data = load_dataset_for_stage("sft", max_samples=max_samples, streaming=True)
    logger.info(f"Loaded {len(sft_data)} SFT examples")
    
    # Create dataset and dataloader
    dataset = SFTDataset(sft_data, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Config
    config = SFTConfig(
        learning_rate=lr,
        batch_size=1,
        num_epochs=epochs,
        max_seq_length=512,
        output_dir=os.path.join(output_dir, "sft"),
        gradient_accumulation_steps=16,
        logging_steps=1,
        save_steps=50,
    )
    
    # Train
    trainer = SFTTrainer(policy_model, tokenizer, config, device_manager)
    trainer.train(dataloader)
    
    # Save
    save_path = os.path.join(output_dir, "sft_final")
    os.makedirs(save_path, exist_ok=True)
    policy_model.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"SFT model saved to {save_path}")
    
    clear_memory()
    return policy_model


def run_dpo_stage(
    policy_model,
    tokenizer,
    device_manager,
    max_samples: int,
    epochs: int,
    lr: float,
    output_dir: str
):
    """Stage 2a: Direct Preference Optimization."""
    logger.info("=" * 70)
    logger.info("STAGE 2: Direct Preference Optimization (DPO)")
    logger.info("Dataset: argilla/distilabel-intel-orca-dpo-pairs")
    logger.info("=" * 70)
    
    # Load data
    dpo_data = load_dataset_for_stage("dpo", max_samples=max_samples, streaming=True)
    logger.info(f"Loaded {len(dpo_data)} DPO examples")
    
    # Create dataset and dataloader
    dataset = PreferenceDataset(dpo_data, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Create reference model (frozen copy) - but use same model for memory
    # DPO will handle this internally
    
    # Config
    config = DPOConfig(
        learning_rate=lr / 10,  # Lower LR for preference training
        batch_size=1,
        num_epochs=epochs,
        beta=0.1,
        output_dir=os.path.join(output_dir, "dpo"),
        gradient_accumulation_steps=16,
        logging_steps=1,
        save_steps=50,
    )
    
    # For DPO we need a reference model - create a frozen copy
    logger.info("Creating reference model (this uses extra memory)...")
    ref_model = PolicyModel(policy_model.base_model_name)
    ref_model.model.eval()
    for param in ref_model.model.parameters():
        param.requires_grad = False
    
    # Train
    trainer = DPOTrainer(policy_model, ref_model, config, device_manager)
    trainer.train(dataloader)
    
    # Clean up reference model
    del ref_model
    clear_memory()
    
    # Save
    save_path = os.path.join(output_dir, "dpo_final")
    os.makedirs(save_path, exist_ok=True)
    policy_model.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"DPO model saved to {save_path}")
    
    return policy_model


def run_simpo_stage(
    policy_model,
    tokenizer,
    device_manager,
    max_samples: int,
    epochs: int,
    lr: float,
    output_dir: str
):
    """Stage 2b: Simple Preference Optimization (reference-free)."""
    logger.info("=" * 70)
    logger.info("STAGE 2: Simple Preference Optimization (SimPO)")
    logger.info("Dataset: HuggingFaceH4/ultrafeedback_binarized")
    logger.info("=" * 70)
    
    # Load data
    simpo_data = load_dataset_for_stage("simpo", max_samples=max_samples, streaming=True)
    logger.info(f"Loaded {len(simpo_data)} SimPO examples")
    
    # Create dataset and dataloader
    dataset = PreferenceDataset(simpo_data, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Config - SimPO is reference-free (memory efficient!)
    config = SimPOConfig(
        learning_rate=lr / 10,
        batch_size=1,
        num_epochs=epochs,
        beta=2.0,
        gamma=0.5,
        output_dir=os.path.join(output_dir, "simpo"),
        gradient_accumulation_steps=16,
        logging_steps=1,
        save_steps=50,
    )
    
    # Train
    trainer = SimPOTrainer(policy_model, config, device_manager)
    trainer.train(dataloader)
    
    # Save
    save_path = os.path.join(output_dir, "simpo_final")
    os.makedirs(save_path, exist_ok=True)
    policy_model.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"SimPO model saved to {save_path}")
    
    clear_memory()
    return policy_model


def run_kto_stage(
    policy_model,
    tokenizer,
    device_manager,
    max_samples: int,
    epochs: int,
    lr: float,
    output_dir: str
):
    """Stage 3: Kahneman-Tversky Optimization (binary feedback)."""
    logger.info("=" * 70)
    logger.info("STAGE 3: Kahneman-Tversky Optimization (KTO)")
    logger.info("Dataset: trl-lib/kto-mix-14k")
    logger.info("=" * 70)
    
    # Load data
    kto_data = load_dataset_for_stage("kto", max_samples=max_samples, streaming=True)
    logger.info(f"Loaded {len(kto_data)} KTO examples")
    
    # Create dataset and dataloader
    dataset = KTODataset(kto_data, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Config
    config = KTOConfig(
        learning_rate=lr / 10,
        batch_size=1,
        num_epochs=epochs,
        beta=0.1,
        output_dir=os.path.join(output_dir, "kto"),
        gradient_accumulation_steps=16,
        logging_steps=1,
        save_steps=50,
    )
    
    # Reference model for KTO
    ref_model = PolicyModel(policy_model.base_model_name)
    ref_model.model.eval()
    for param in ref_model.model.parameters():
        param.requires_grad = False
    
    # Train
    trainer = KTOTrainer(policy_model, ref_model, config, device_manager)
    trainer.train(dataloader)
    
    del ref_model
    clear_memory()
    
    # Save
    save_path = os.path.join(output_dir, "kto_final")
    os.makedirs(save_path, exist_ok=True)
    policy_model.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"KTO model saved to {save_path}")
    
    return policy_model


def main():
    parser = argparse.ArgumentParser(description="Full RLHF Pipeline")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B",
                        help="Base model name")
    parser.add_argument("--stages", nargs="+", 
                        default=["sft", "simpo"],
                        choices=["sft", "dpo", "simpo", "kto", "grpo"],
                        help="Pipeline stages to run")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max samples per stage (streaming, no RAM limit)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Epochs per stage")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Base learning rate")
    parser.add_argument("--output-dir", type=str, default="checkpoints/full_pipeline",
                        help="Output directory")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from checkpoint path")
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("FULL RLHF PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Stages: {args.stages}")
    logger.info(f"Max samples/stage: {args.max_samples}")
    logger.info(f"Epochs/stage: {args.epochs}")
    logger.info(f"Output: {args.output_dir}")
    
    # Device manager - MUST use float32 on CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else None
    device_manager = DeviceManager(device=device, dtype=dtype, use_amp=(device != "cpu"))
    logger.info(f"Device: {device_manager.device}, dtype: {device_manager.dtype}")
    
    # Load model
    policy_model, tokenizer = load_base_model(args.model, args.resume_from)
    
    # Run pipeline stages
    for stage in args.stages:
        if stage == "sft":
            policy_model = run_sft_stage(
                policy_model, tokenizer, device_manager,
                args.max_samples, args.epochs, args.lr, args.output_dir
            )
        elif stage == "dpo":
            policy_model = run_dpo_stage(
                policy_model, tokenizer, device_manager,
                args.max_samples, args.epochs, args.lr, args.output_dir
            )
        elif stage == "simpo":
            policy_model = run_simpo_stage(
                policy_model, tokenizer, device_manager,
                args.max_samples, args.epochs, args.lr, args.output_dir
            )
        elif stage == "kto":
            policy_model = run_kto_stage(
                policy_model, tokenizer, device_manager,
                args.max_samples, args.epochs, args.lr, args.output_dir
            )
        elif stage == "grpo":
            logger.warning("GRPO stage not yet implemented in full pipeline")
            continue
        
        logger.info(f"Stage {stage} completed!")
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    policy_model.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE!")
    logger.info(f"Final model saved to: {final_path}")
    logger.info("=" * 70)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Stages completed: {args.stages}")
    print(f"Samples per stage: {args.max_samples}")
    print(f"Final model: {final_path}")
    print("\nTo test the model:")
    print(f"  python3 -c \"")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  from peft import PeftModel")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{args.model}', trust_remote_code=True)")
    print(f"  model = PeftModel.from_pretrained(model, '{final_path}')")
    print(f"  ...\"")
    print("=" * 70)


if __name__ == "__main__":
    main()
