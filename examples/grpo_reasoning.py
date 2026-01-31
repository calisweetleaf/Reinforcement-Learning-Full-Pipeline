"""
GRPO Training Example for Mathematical Reasoning

This example demonstrates training a model to solve math problems
using Group Relative Policy Optimization (GRPO) with verifiable rewards.
"""

import re
import random
from typing import List, Dict
from rlhf import RLHFOrchestrator, GRPOConfig


def generate_math_problems(n: int = 100) -> List[Dict]:
    """Generate simple math problems with verifiable answers."""
    problems = []
    
    for _ in range(n):
        # Generate linear equations: ax + b = c
        a = random.randint(2, 10)
        b = random.randint(1, 20)
        x = random.randint(1, 20)
        c = a * x + b
        
        problem = f"Solve for x: {a}x + {b} = {c}"
        
        problems.append({
            "prompt": problem,
            "answer": str(x),
            "type": "linear_equation"
        })
    
    return problems


def extract_final_answer(text: str) -> str:
    """Extract the final numerical answer from model completion."""
    # Look for patterns like "x = 5", "answer is 5", "= 5", etc.
    patterns = [
        r'x\s*=\s*(-?\d+)',
        r'answer\s*(?:is|:)?\s*(-?\d+)',
        r'result\s*(?:is|:)?\s*(-?\d+)',
        r'=\s*(-?\d+)(?:\s*$|\s*[^\d])',
        r'(?:^|\s)(-?\d+)(?:\s*$)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            return matches[-1]  # Return last match (likely final answer)
    
    return ""


def math_reward_fn(completion: str, ground_truth: str) -> float:
    """
    Reward function for math problems.
    Returns 1.0 for correct answers, 0.0 for incorrect.
    """
    predicted = extract_final_answer(completion)
    
    if not predicted:
        return 0.0  # No answer found
    
    try:
        if int(predicted) == int(ground_truth):
            return 1.0
    except ValueError:
        pass
    
    return 0.0


def format_reward_fn(completion: str) -> float:
    """
    Reward for proper formatting (step-by-step reasoning).
    Encourages models to show their work.
    """
    score = 0.0
    
    # Has step indicators
    if any(marker in completion.lower() for marker in ['step', 'first', 'then', 'next']):
        score += 0.2
    
    # Has mathematical working
    if '=' in completion and len(completion.split('=')) >= 2:
        score += 0.2
    
    # Reasonable length (not too short, not too long)
    word_count = len(completion.split())
    if 20 <= word_count <= 200:
        score += 0.1
    
    return score


def combined_reward_fn(completion: str, problem: Dict) -> float:
    """
    Combined reward: correctness + format quality.
    """
    correctness = math_reward_fn(completion, problem["answer"])
    formatting = format_reward_fn(completion)
    
    # Weight correctness heavily
    return 0.8 * correctness + 0.2 * formatting


def main():
    print("=" * 70)
    print("GRPO Training Example: Mathematical Reasoning")
    print("=" * 70)
    
    # Generate training problems
    print("\nGenerating math problems...")
    train_problems = generate_math_problems(n=100)
    test_problems = generate_math_problems(n=20)
    
    print(f"Training problems: {len(train_problems)}")
    print(f"Test problems: {len(test_problems)}")
    
    # Show example
    example = train_problems[0]
    print(f"\nExample problem: {example['prompt']}")
    print(f"Answer: {example['answer']}")
    
    # Initialize orchestrator
    print("\nInitializing RLHF Orchestrator...")
    orchestrator = RLHFOrchestrator()
    
    # GRPO Configuration
    print("\n" + "=" * 70)
    print("GRPO Configuration")
    print("=" * 70)
    
    grpo_config = GRPOConfig(
        learning_rate=1e-6,
        batch_size=8,          # Number of prompts per batch
        group_size=8,          # Completions per prompt
        num_epochs=3,
        kl_coeff=0.1,          # KL penalty
        clip_ratio=0.2,        # PPO clipping
        max_completion_length=256,
        temperature=1.0,       # Sampling temperature
        use_verifiable_rewards=True,
        output_dir="./checkpoints/grpo_math"
    )
    
    print(f"Learning rate: {grpo_config.learning_rate}")
    print(f"Batch size: {grpo_config.batch_size}")
    print(f"Group size: {grpo_config.group_size}")
    print(f"KL coefficient: {grpo_config.kl_coeff}")
    print(f"Clip ratio: {grpo_config.clip_ratio}")
    
    # Training
    print("\n" + "=" * 70)
    print("Training with GRPO")
    print("=" * 70)
    
    print("Starting training loop...")
    print("For each batch:")
    print("  1. Generate group_size completions per prompt")
    print("  2. Score each with verifiable reward function")
    print("  3. Compute group-relative advantages")
    print("  4. Update policy with clipped objective")
    
    # Note: In practice, you would call:
    # orchestrator.run_policy_optimization(
    #     method="grpo",
    #     data=train_problems,
    #     config=grpo_config,
    #     reward_fn=lambda comp, item: combined_reward_fn(comp, item)
    # )
    print("\n[GRPO training would run here with real model]")
    
    # Demonstrate reward function
    print("\n" + "=" * 70)
    print("Reward Function Demonstration")
    print("=" * 70)
    
    example_problem = "Solve for x: 3x + 5 = 20"
    example_answer = "5"
    
    good_completion = """
To solve for x, I need to isolate x on one side of the equation.

Starting with: 3x + 5 = 20

Step 1: Subtract 5 from both sides
3x = 20 - 5
3x = 15

Step 2: Divide both sides by 3
x = 15 / 3
x = 5

So the answer is 5.
"""
    
    bad_completion = "The answer is 7."
    
    print(f"\nProblem: {example_problem}")
    print(f"Correct answer: {example_answer}")
    
    print("\nGood completion (step-by-step):")
    print("-" * 40)
    print(good_completion)
    reward_good = combined_reward_fn(good_completion, {"answer": example_answer})
    print(f"Reward: {reward_good:.2f}")
    
    print("\nBad completion (wrong answer):")
    print("-" * 40)
    print(bad_completion)
    reward_bad = combined_reward_fn(bad_completion, {"answer": example_answer})
    print(f"Reward: {reward_bad:.2f}")
    
    # Evaluation
    print("\n" + "=" * 70)
    print("Evaluation")
    print("=" * 70)
    
    print("Testing on held-out problems...")
    
    correct = 0
    for problem in test_problems[:5]:  # Just show 5
        # Simulate model generation
        # In practice: completion = orchestrator.policy.generate(problem["prompt"])
        print(f"\nProblem: {problem['prompt']}")
        print(f"Expected: {problem['answer']}")
        print("[Model would generate answer here]")
    
    # Save model
    print("\n" + "=" * 70)
    print("Saving Model")
    print("=" * 70)
    
    # orchestrator.save_models("./grpo_math_model")
    print("[Model would be saved here]")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    print("\nKey advantages of GRPO demonstrated:")
    print("  ✓ No value model needed (group-relative baseline)")
    print("  ✓ Verifiable rewards (exact answer checking)")
    print("  ✓ Natural variance reduction through grouping")
    print("  ✓ Works well for reasoning tasks")


if __name__ == "__main__":
    main()
