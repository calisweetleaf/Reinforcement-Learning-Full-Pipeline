"""
Full RLHF Pipeline Example

This example demonstrates the complete end-to-end RLHF pipeline:
1. Supervised Fine-Tuning (SFT)
2. Reward Model Training
3. Policy Optimization (with method selection)
4. Self-Play Enhancement
5. Evaluation
"""

import json
from typing import List, Dict
from rlhf import (
    RLHFOrchestrator,
    SFTConfig,
    RewardModelConfig,
    DPOConfig,
    GRPOConfig,
    PPOConfig,
    SimPOConfig,
    KTOConfig
)

DEFAULT_BASE_MODEL = "gpt2"
DEFAULT_OUTPUT_DIR = "./rlhf_output"


def create_sample_datasets():
    """Create sample datasets for the full pipeline."""
    
    # SFT dataset
    sft_data = [
        {
            "prompt": "What are the benefits of exercise?",
            "response": "Regular exercise offers numerous benefits including improved cardiovascular health, stronger muscles and bones, better mental health through endorphin release, weight management, reduced risk of chronic diseases like diabetes and heart disease, improved sleep quality, and enhanced cognitive function."
        },
        {
            "prompt": "Explain photosynthesis.",
            "response": "Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy. Using sunlight, water, and carbon dioxide, they produce glucose (food) and oxygen. This process occurs primarily in the chloroplasts and is essential for life on Earth as it provides oxygen and forms the base of most food chains."
        },
        {
            "prompt": "How do you make pasta?",
            "response": "To make pasta: 1) Bring a large pot of salted water to boil, 2) Add pasta and cook according to package directions until al dente, 3) Reserve some pasta water before draining, 4) Toss with your chosen sauce, adding pasta water as needed for consistency, 5) Serve immediately with grated cheese if desired."
        },
    ]
    
    # Preference dataset (paired comparisons)
    preference_data = [
        {
            "prompt": "What is climate change?",
            "chosen": "Climate change refers to significant, long-term changes in the average weather patterns that have come to define Earth's local, regional and global climates. It is primarily driven by human activities, especially the burning of fossil fuels which releases greenhouse gases like carbon dioxide and methane into the atmosphere, trapping heat and causing global temperatures to rise.",
            "rejected": "Climate change is when the weather gets warmer because of pollution."
        },
        {
            "prompt": "Describe the water cycle.",
            "chosen": "The water cycle, also known as the hydrologic cycle, describes the continuous movement of water on, above, and below Earth's surface. It includes evaporation (water turning to vapor), condensation (vapor forming clouds), precipitation (rain, snow, etc.), and collection (water gathering in oceans, rivers, groundwater). This cycle is powered by solar energy and gravity and is essential for distributing fresh water across the planet.",
            "rejected": "Water goes up and down in a cycle. It evaporates and rains."
        },
    ]
    
    # Unpaired data for KTO
    unpaired_data = [
        {
            "prompt": "Tell me about dogs.",
            "completion": "Dogs are domesticated mammals and popular pets known for their loyalty, intelligence, and companionship. They come in various breeds with different sizes, temperaments, and characteristics.",
            "label": "desirable"
        },
        {
            "prompt": "Write a poem.",
            "completion": "roses are red violets are blue",
            "label": "undesirable"
        },
    ]
    
    # Prompts for PPO/GRPO
    prompts = [
        "Explain the importance of education.",
        "What are the effects of sleep deprivation?",
        "How does the internet work?",
    ]
    
    return sft_data, preference_data, unpaired_data, prompts


def run_full_pipeline(method: str = "dpo"):
    """
    Run the complete RLHF pipeline with specified method.
    
    Args:
        method: One of "dpo", "grpo", "ppo", "simpo", "kto"
    """
    print("=" * 70)
    print(f"Full RLHF Pipeline - Method: {method.upper()}")
    print("=" * 70)
    
    # Load data
    sft_data, preference_data, unpaired_data, prompts = create_sample_datasets()
    
    # Initialize orchestrator
    orchestrator = RLHFOrchestrator(
        base_model=DEFAULT_BASE_MODEL,
        output_dir=DEFAULT_OUTPUT_DIR
    )
    
    # Stage 1: SFT
    print("\n" + "-" * 70)
    print("STAGE 1: Supervised Fine-Tuning (SFT)")
    print("-" * 70)
    
    sft_config = SFTConfig(
        learning_rate=5e-6,
        batch_size=4,
        num_epochs=2,
        max_seq_length=512,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        output_dir="./checkpoints/sft"
    )
    
    print(f"Training on {len(sft_data)} examples...")
    print(f"Config: LR={sft_config.learning_rate}, Epochs={sft_config.num_epochs}")
    # orchestrator.run_sft(sft_data, sft_config)
    print("[SFT complete]")
    
    # Stage 2: Reward Model (if needed)
    if method in ["ppo"]:
        print("\n" + "-" * 70)
        print("STAGE 2: Reward Model Training")
        print("-" * 70)
        
        rm_config = RewardModelConfig(
            learning_rate=1e-5,
            batch_size=4,
            num_epochs=2,
            ensemble_size=1,
            output_dir="./checkpoints/reward_model"
        )
        
        print(f"Training reward model on {len(preference_data)} pairs...")
        # orchestrator.run_reward_model_training(preference_data, rm_config)
        print("[Reward model training complete]")
    
    # Stage 3: Policy Optimization
    print("\n" + "-" * 70)
    print(f"STAGE 3: Policy Optimization ({method.upper()})")
    print("-" * 70)
    
    if method == "dpo":
        config = DPOConfig(
            learning_rate=5e-7,
            batch_size=2,
            num_epochs=2,
            beta=0.1,
            loss_type="sigmoid",
            output_dir="./checkpoints/dpo"
        )
        data = preference_data
        
    elif method == "grpo":
        config = GRPOConfig(
            learning_rate=1e-6,
            batch_size=2,
            group_size=4,
            num_epochs=2,
            kl_coeff=0.1,
            output_dir="./checkpoints/grpo"
        )
        data = prompts
        
    elif method == "ppo":
        config = PPOConfig(
            learning_rate=1e-6,
            batch_size=2,
            num_epochs=2,
            clip_ratio=0.2,
            kl_coeff=0.02,
            output_dir="./checkpoints/ppo"
        )
        data = prompts
        
    elif method == "simpo":
        config = SimPOConfig(
            learning_rate=5e-7,
            batch_size=2,
            num_epochs=2,
            beta=2.0,
            gamma=0.5,
            output_dir="./checkpoints/simpo"
        )
        data = preference_data
        
    elif method == "kto":
        config = KTOConfig(
            learning_rate=5e-7,
            batch_size=2,
            num_epochs=2,
            beta=0.1,
            lambda_d=1.0,
            lambda_u=0.5,
            output_dir="./checkpoints/kto"
        )
        data = unpaired_data
    
    print(f"Training with {method.upper()}...")
    print(f"Config: {config}")
    # orchestrator.run_policy_optimization(method=method, data=data, config=config)
    print(f"[{method.upper()} training complete]")
    
    # Stage 4: Self-Play Enhancement (optional)
    print("\n" + "-" * 70)
    print("STAGE 4: Self-Play Enhancement")
    print("-" * 70)
    
    print("Running self-play for capability improvement...")
    print("  - Generating synthetic training data")
    print("  - Iterative refinement")
    print("  - Capability testing")
    # synthetic_data = orchestrator.run_self_play(n_games=100)
    print("[Self-play complete]")
    
    # Stage 5: Evaluation
    print("\n" + "-" * 70)
    print("STAGE 5: Evaluation")
    print("-" * 70)
    
    eval_prompts = [
        "What is artificial intelligence?",
        "How do computers work?",
    ]
    
    print(f"Evaluating on {len(eval_prompts)} test prompts...")
    # metrics = orchestrator.evaluate(eval_prompts)
    print("Evaluation metrics:")
    print("  - KL Divergence: [would be computed]")
    print("  - Reward Accuracy: [would be computed]")
    print("  - Response Diversity: [would be computed]")
    print("  - Win Rate: [would be computed]")
    
    # Save final model
    print("\n" + "-" * 70)
    print("Saving Final Model")
    print("-" * 70)
    
    # orchestrator.save_models(f"./final_model_{method}")
    print(f"Model saved to ./final_model_{method}")
    
    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)


def compare_methods():
    """Compare different RLHF methods."""
    print("\n" + "=" * 70)
    print("Method Comparison")
    print("=" * 70)
    
    comparison = {
        "DPO": {
            "memory": "Medium (2 models)",
            "speed": "Fast",
            "stability": "High",
            "best_for": "General preference learning"
        },
        "GRPO": {
            "memory": "Medium (2 models)",
            "speed": "Medium",
            "stability": "High",
            "best_for": "Reasoning with verifiable rewards"
        },
        "PPO": {
            "memory": "High (4 models)",
            "speed": "Slow",
            "stability": "Medium",
            "best_for": "Maximum control, online learning"
        },
        "SimPO": {
            "memory": "Low (1 model)",
            "speed": "Fastest",
            "stability": "Medium",
            "best_for": "Resource-constrained training"
        },
        "KTO": {
            "memory": "Medium (2 models)",
            "speed": "Fast",
            "stability": "High",
            "best_for": "Unpaired preference data"
        }
    }
    
    for method, attrs in comparison.items():
        print(f"\n{method}:")
        for key, value in attrs.items():
            print(f"  {key}: {value}")


def main():
    """Main entry point with multiple examples."""
    
    print("=" * 70)
    print("Full RLHF Pipeline Examples")
    print("=" * 70)
    
    # Show method comparison
    compare_methods()
    
    # Run example pipelines
    methods = ["dpo", "grpo", "simpo"]
    
    for method in methods:
        print("\n\n")
        run_full_pipeline(method)
        input("\nPress Enter to continue to next method...")
    
    print("\n\n")
    print("=" * 70)
    print("All Examples Complete!")
    print("=" * 70)
    
    print("\nNext steps:")
    print("1. Replace sample data with your own datasets")
    print("2. Adjust hyperparameters for your use case")
    print("3. Enable actual training by uncommenting orchestrator calls")
    print("4. Monitor training with WandB/TensorBoard")
    print("5. Evaluate thoroughly before deployment")


if __name__ == "__main__":
    main()
