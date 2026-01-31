"""
Basic DPO Training Example

This example demonstrates a minimal DPO training pipeline:
1. Load a base model
2. Run SFT warmup
3. Train with DPO on preference data
4. Evaluate results
"""

import json
from rlhf import RLHFOrchestrator, SFTConfig, DPOConfig

DEFAULT_BASE_MODEL = "gpt2"
DEFAULT_OUTPUT_DIR = "./rlhf_output"

def load_sample_data():
    """Create sample training data for demonstration."""
    
    # SFT data: instruction-response pairs
    sft_data = [
        {
            "prompt": "Explain the concept of machine learning.",
            "response": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make decisions with minimal human intervention."
        },
        {
            "prompt": "What is the capital of France?",
            "response": "The capital of France is Paris, which is also the largest city in the country and serves as its cultural, economic, and political center."
        },
        {
            "prompt": "Write a haiku about nature.",
            "response": "Gentle morning breeze\nWhispers through the ancient trees\nNature's symphony"
        },
    ]
    
    # Preference data: prompt with chosen (better) and rejected (worse) responses
    preference_data = [
        {
            "prompt": "Explain quantum computing.",
            "chosen": "Quantum computing harnesses quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical bits that are either 0 or 1, quantum bits (qubits) can exist in multiple states simultaneously, enabling certain computations to be performed exponentially faster.",
            "rejected": "Quantum computers are computers that use quantum mechanics. They are faster than normal computers."
        },
        {
            "prompt": "How do you make a good cup of coffee?",
            "chosen": "To make excellent coffee: 1) Use freshly roasted, high-quality beans, 2) Grind just before brewing to preserve aroma, 3) Use filtered water at 195-205°F (90-96°C), 4) Maintain a 1:16 coffee-to-water ratio, 5) Brew for 4-5 minutes for pour-over, and 6) Serve immediately for optimal flavor.",
            "rejected": "Just put coffee in water and heat it up."
        },
    ]
    
    return sft_data, preference_data


def main():
    print("=" * 60)
    print("Basic DPO Training Example")
    print("=" * 60)
    
    # Load data
    sft_data, preference_data = load_sample_data()
    print(f"Loaded {len(sft_data)} SFT examples")
    print(f"Loaded {len(preference_data)} preference pairs")
    
    # Initialize orchestrator
    print("\nInitializing RLHF Orchestrator...")
    orchestrator = RLHFOrchestrator(
        base_model=DEFAULT_BASE_MODEL,
        output_dir=DEFAULT_OUTPUT_DIR
    )
    
    # Stage 1: SFT Warmup
    print("\n" + "=" * 60)
    print("Stage 1: Supervised Fine-Tuning (Warmup)")
    print("=" * 60)
    
    sft_config = SFTConfig(
        learning_rate=5e-6,
        batch_size=4,
        num_epochs=1,  # Small for demo
        max_seq_length=512,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        output_dir="./checkpoints/sft_demo"
    )
    
    print(f"SFT Config: {sft_config}")
    # Note: In practice, you would call:
    # orchestrator.run_sft(sft_data, sft_config)
    print("[SFT training would run here with real model]")
    
    # Stage 2: DPO Training
    print("\n" + "=" * 60)
    print("Stage 2: Direct Preference Optimization")
    print("=" * 60)
    
    dpo_config = DPOConfig(
        learning_rate=5e-7,
        batch_size=2,
        num_epochs=1,  # Small for demo
        beta=0.1,  # KL penalty coefficient
        loss_type="sigmoid",
        use_lora=True,  # Continue with LoRA adapters
        output_dir="./checkpoints/dpo_demo"
    )
    
    print(f"DPO Config: {dpo_config}")
    # Note: In practice, you would call:
    # orchestrator.run_policy_optimization(
    #     method="dpo",
    #     data=preference_data,
    #     config=dpo_config
    # )
    print("[DPO training would run here with real model]")
    
    # Evaluation
    print("\n" + "=" * 60)
    print("Stage 3: Evaluation")
    print("=" * 60)
    
    test_prompts = [
        "Explain neural networks.",
        "What makes a good leader?",
    ]
    
    print("Test prompts:")
    for prompt in test_prompts:
        print(f"  - {prompt}")
    
    # Note: In practice, you would call:
    # metrics = orchestrator.evaluate(test_prompts)
    # print(f"KL Divergence: {metrics['kl_div']:.4f}")
    # print(f"Reward Accuracy: {metrics['reward_accuracy']:.4f}")
    print("[Evaluation would run here with real model]")
    
    # Save final model
    print("\n" + "=" * 60)
    print("Stage 4: Saving Model")
    print("=" * 60)
    
    # orchestrator.save_models("./final_dpo_model")
    print("[Model would be saved here]")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
