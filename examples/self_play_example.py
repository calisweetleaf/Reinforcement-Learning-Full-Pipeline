"""
Self-Play Training Example

This example demonstrates self-play training where the model:
1. Competes against previous versions of itself
2. Uses iterative refinement (generate → critique → improve)
3. Implements adversarial validation
"""

import copy
from typing import List, Dict
from rlhf import RLHFOrchestrator, SFTConfig


class SelfPlayTrainer:
    """
    Self-play training implementation.
    """
    
    def __init__(self, orchestrator: RLHFOrchestrator):
        self.orchestrator = orchestrator
        self.current_policy = None
        self.previous_policy = None
        self.iteration = 0
    
    def initialize(self, base_model_path: str):
        """Initialize policies from base model."""
        print(f"Initializing policies from {base_model_path}")
        
        # Current policy (being trained)
        self.current_policy = self.orchestrator.load_policy(base_model_path)
        
        # Previous policy (frozen for competition)
        self.previous_policy = copy.deepcopy(self.current_policy)
        self.previous_policy.freeze()
        
        print("Policies initialized")
    
    def competitive_round(
        self, 
        prompts: List[str],
        reward_fn
    ) -> List[Dict]:
        """
        Run a competitive round: current vs previous policy.
        
        Returns preference data for training.
        """
        print(f"Running competitive round {self.iteration}")
        
        preferences = []
        
        for prompt in prompts:
            # Both policies generate
            current_output = self.current_policy.generate(prompt)
            previous_output = self.previous_policy.generate(prompt)
            
            # Score both
            current_reward = reward_fn(current_output)
            previous_reward = reward_fn(previous_output)
            
            # Create preference based on winner
            if current_reward > previous_reward:
                preferences.append({
                    "prompt": prompt,
                    "chosen": current_output,
                    "rejected": previous_output,
                    "margin": current_reward - previous_reward
                })
            elif previous_reward > current_reward:
                preferences.append({
                    "prompt": prompt,
                    "chosen": previous_output,
                    "rejected": current_output,
                    "margin": previous_reward - current_reward
                })
        
        print(f"Generated {len(preferences)} preference pairs")
        return preferences
    
    def iterative_refinement(
        self,
        prompt: str,
        max_iterations: int = 3
    ) -> List[Dict]:
        """
        Run iterative refinement: generate → critique → improve.
        
        Returns training data from the refinement trajectory.
        """
        trajectory = []
        
        # Initial generation
        response = self.current_policy.generate(prompt)
        trajectory.append({
            "stage": "initial",
            "content": response
        })
        
        for i in range(max_iterations):
            # Self-critique
            critique_prompt = f"""
You are evaluating a response to the following task:

Task: {prompt}

Response: {response}

Analyze this response critically. Identify:
1. Factual errors or inaccuracies
2. Missing information
3. Logical flaws
4. Areas for improvement
5. How you would improve it

Your critique:"""
            
            critique = self.current_policy.generate(critique_prompt)
            
            # Refinement
            refine_prompt = f"""
Original task: {prompt}

Current response: {response}

Critique of current response: {critique}

Based on this critique, provide an improved response that addresses all issues.

Improved response:"""
            
            improved = self.current_policy.generate(refine_prompt)
            
            # Store for training
            trajectory.append({
                "stage": f"refinement_{i+1}",
                "original": response,
                "critique": critique,
                "improved": improved
            })
            
            response = improved
        
        return trajectory
    
    def adversarial_validation(
        self,
        test_prompts: List[str],
        capability_threshold: float = 0.6
    ) -> bool:
        """
        Validate current policy against test suite.
        
        Returns True if policy passes validation.
        """
        print("Running adversarial validation...")
        
        passed = 0
        failed = 0
        
        for prompt in test_prompts:
            # Generate multiple samples
            samples = [
                self.current_policy.generate(prompt, temperature=1.0)
                for _ in range(5)
            ]
            
            # Check diversity (prevent mode collapse)
            unique_samples = len(set(samples))
            diversity_score = unique_samples / len(samples)
            
            # Check quality (would use reward model in practice)
            quality_score = self.evaluate_quality(samples[0])
            
            if diversity_score > 0.6 and quality_score > capability_threshold:
                passed += 1
            else:
                failed += 1
                print(f"  Failed on: {prompt[:50]}...")
                print(f"    Diversity: {diversity_score:.2f}, Quality: {quality_score:.2f}")
        
        pass_rate = passed / (passed + failed)
        print(f"Validation: {passed}/{passed + failed} passed ({pass_rate:.1%})")
        
        return pass_rate >= capability_threshold
    
    def evaluate_quality(self, response: str) -> float:
        """
        Simple quality evaluation (placeholder).
        In practice, use trained reward model or human evaluation.
        """
        score = 0.0
        
        # Length check (not too short, not too long)
        words = len(response.split())
        if 20 <= words <= 500:
            score += 0.3
        
        # Structure check
        if '.' in response and response[0].isupper():
            score += 0.3
        
        # Coherence check (simple heuristic)
        sentences = response.split('.')
        if len(sentences) >= 2:
            score += 0.4
        
        return score
    
    def update_reference(self):
        """Update the reference (previous) policy."""
        print("Updating reference policy...")
        self.previous_policy = copy.deepcopy(self.current_policy)
        self.previous_policy.freeze()
        self.iteration += 1
    
    def train_iteration(
        self,
        prompts: List[str],
        reward_fn,
        config
    ):
        """Run one iteration of self-play training."""
        print(f"\n{'='*60}")
        print(f"Self-Play Iteration {self.iteration}")
        print('='*60)
        
        # 1. Competitive self-play
        print("\n1. Competitive Self-Play")
        preferences = self.competitive_round(prompts, reward_fn)
        
        # 2. Iterative refinement (on subset)
        print("\n2. Iterative Refinement")
        refinement_data = []
        for prompt in prompts[:5]:  # Subset for demo
            trajectory = self.iterative_refinement(prompt)
            refinement_data.extend(trajectory)
            print(f"  Refined: {prompt[:50]}...")
        
        # 3. Train on combined data
        print("\n3. Training")
        # In practice:
        # self.orchestrator.run_policy_optimization(
        #     method="dpo",
        #     data=preferences,
        #     config=config
        # )
        print(f"  Would train on {len(preferences)} preferences")
        print(f"  Plus {len(refinement_data)} refinement examples")
        
        # 4. Validation
        print("\n4. Validation")
        test_prompts = prompts[-10:]  # Use last 10 as test
        passed = self.adversarial_validation(test_prompts)
        
        if passed:
            print("Validation PASSED - updating reference")
            self.update_reference()
        else:
            print("Validation FAILED - rolling back")
            # Rollback logic here
        
        return passed


def example_reward_fn(response: str) -> float:
    """
    Example reward function.
    In practice, use trained reward model or rule-based rewards.
    """
    score = 0.0
    
    # Prefer longer, detailed responses
    words = len(response.split())
    score += min(words / 100, 1.0) * 0.3
    
    # Prefer responses with structure
    if any(marker in response for marker in ['1.', '2.', '3.', '- ', '* ']):
        score += 0.3
    
    # Prefer responses with explanations
    if any(word in response.lower() for word in ['because', 'therefore', 'thus', 'as a result']):
        score += 0.4
    
    return score


def main():
    print("=" * 70)
    print("Self-Play Training Example")
    print("=" * 70)
    
    # Initialize
    orchestrator = RLHFOrchestrator()
    trainer = SelfPlayTrainer(orchestrator)
    
    # In practice: trainer.initialize("path/to/base/model")
    print("\n[Would initialize from base model here]")
    
    # Training prompts
    prompts = [
        "Explain the importance of critical thinking.",
        "What are the benefits of regular exercise?",
        "How does photosynthesis work?",
        "Describe the water cycle.",
        "What causes climate change?",
        "Explain the concept of supply and demand.",
        "What is the scientific method?",
        "How do vaccines work?",
    ]
    
    print(f"\nTraining with {len(prompts)} prompts")
    
    # Training configuration
    config = {
        "learning_rate": 5e-7,
        "batch_size": 4,
        "num_epochs": 1,
        "beta": 0.1
    }
    
    # Run self-play iterations
    n_iterations = 3
    for i in range(n_iterations):
        passed = trainer.train_iteration(
            prompts=prompts,
            reward_fn=example_reward_fn,
            config=config
        )
        
        if not passed:
            print(f"\nStopping early due to validation failure")
            break
    
    print("\n" + "=" * 70)
    print("Self-Play Training Complete")
    print("=" * 70)
    
    print("\nKey takeaways:")
    print("  ✓ Model improves by competing against itself")
    print("  ✓ Iterative refinement creates training data")
    print("  ✓ Adversarial validation prevents regression")
    print("  ✓ Reference policy provides stable baseline")


if __name__ == "__main__":
    main()
