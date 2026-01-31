# Self-Play in RLHF

## Overview

Self-play is a training paradigm where a model improves by competing against itself or previous versions of itself. In RLHF, self-play enables autonomous capability improvement without requiring constant human feedback.

## Why Self-Play Matters

Traditional RLHF has limitations:
1. **Human bottleneck**: Quality feedback is expensive and slow
2. **Distribution shift**: Models can overfit to specific reward models
3. **Capability ceiling**: Limited by the quality of human annotators

Self-play breaks these limits by:
- Generating its own training data through competition
- Discovering strategies humans might miss
- Continuously improving through iterative refinement

## Self-Play Architecture

### 1. Competitive Self-Play

```
Current Policy ──▶ Generates completion A
    │
    ├───▶ Previous Policy ──▶ Generates completion B
    │
    └───▶ Judge (reward model or rules) ──▶ Winner
                │
                ▼
        Update policy to favor winning strategy
```

**Implementation**:
```python
class CompetitiveSelfPlay:
    def __init__(self, policy, reference_policy, reward_fn):
        self.policy = policy  # Current
        self.ref_policy = reference_policy  # Previous snapshot
        self.reward_fn = reward_fn
    
    def train_step(self, prompts):
        # Generate from both policies
        current_outputs = self.policy.generate(prompts)
        ref_outputs = self.ref_policy.generate(prompts)
        
        # Score both
        current_rewards = self.reward_fn(current_outputs)
        ref_rewards = self.reward_fn(ref_outputs)
        
        # Create preference pairs from winners
        preferences = []
        for i, prompt in enumerate(prompts):
            if current_rewards[i] > ref_rewards[i]:
                preferences.append({
                    "prompt": prompt,
                    "chosen": current_outputs[i],
                    "rejected": ref_outputs[i]
                })
        
        # Train on preferences
        return preferences
```

### 2. Iterative Refinement

The model generates, critiques, and refines its own outputs:

```
Prompt ──▶ Generate(v0) ──▶ Critique ──▶ Refine(v1)
                              │            │
                              └────────────┘
                                   (loop)
```

**Process**:
1. **Generate**: Initial response to prompt
2. **Critique**: Model analyzes its own output for errors
3. **Refine**: Model produces improved version
4. **Train**: Use (v0, critique, v1) as training data

**Implementation**:
```python
class IterativeRefiner:
    def __init__(self, policy):
        self.policy = policy
        self.critique_template = """
        Original response: {response}
        
        Analyze the above response. Identify:
        1. Factual errors
        2. Logical flaws
        3. Areas for improvement
        4. Missing information
        
        Critique:"""
        
        self.refine_template = """
        Original: {response}
        Critique: {critique}
        
        Provide an improved response addressing all issues:
        
        Improved:"""
    
    def refine(self, prompt, max_iterations=3):
        response = self.policy.generate(prompt)
        
        for i in range(max_iterations):
            # Generate critique
            critique_prompt = self.critique_template.format(response=response)
            critique = self.policy.generate(critique_prompt)
            
            # Generate refinement
            refine_prompt = self.refine_template.format(
                response=response, 
                critique=critique
            )
            improved = self.policy.generate(refine_prompt)
            
            # Store for training
            yield {
                "iteration": i,
                "original": response,
                "critique": critique,
                "improved": improved
            }
            
            response = improved
```

### 3. Constitutional Self-Improvement

Model improves by following explicit principles (constitution):

```
Constitutional Principles:
1. Be helpful and honest
2. Admit uncertainty
3. Provide evidence for claims
4. Consider multiple perspectives
5. Acknowledge limitations

Prompt ──▶ Response ──▶ Evaluate against constitution ──▶ Refine
```

**Implementation**:
```python
class ConstitutionalRewardWrapper:
    def __init__(self, base_reward_fn, constitution):
        self.base_reward = base_reward_fn
        self.constitution = constitution  # List of principles
    
    def score(self, prompt, response):
        # Base reward
        score = self.base_reward(prompt, response)
        
        # Constitutional evaluation
        for principle in self.constitution:
            adherence = self.evaluate_principle(response, principle)
            score += self.principle_weights[principle] * adherence
        
        return score
    
    def evaluate_principle(self, response, principle):
        # Use auxiliary model or rules to evaluate
        # Returns score in [0, 1]
        pass
```

## Advanced Self-Play Mechanisms

### 1. League Training

Maintain a population of policies at different skill levels:

```python
class LeagueTraining:
    def __init__(self, n_leagues=5):
        self.leagues = [
            PolicySnapshot(f"league_{i}") 
            for i in range(n_leagues)
        ]
    
    def sample_opponent(self, current_performance):
        # Sample from leagues near current skill
        # Encourage learning from slightly better opponents
        probabilities = self.compute_matchmaking_probs(current_performance)
        return np.random.choice(self.leagues, p=probabilities)
```

### 2. Skill-Conditioned Training

Explicitly condition on skill level:

```python
# Add skill token to prompt
skill_tokens = ["<novice>", "<intermediate>", "<expert>"]
prompt = f"<expert>{original_prompt}"

# Train across skill levels
for skill in skill_tokens:
    # Train policy to produce skill-appropriate outputs
    pass
```

### 3. Adversarial Filtering

Use the model to find its own failure modes:

```python
class AdversarialFilter:
    def find_adversarial_examples(self, dataset, n_examples=100):
        """Find prompts where model performs poorly."""
        failures = []
        
        for prompt in dataset:
            outputs = self.policy.generate(prompt, n_samples=10)
            rewards = [self.reward_fn(o) for o in outputs]
            
            if max(rewards) - min(rewards) > threshold:
                # High variance = model is uncertain
                failures.append({
                    "prompt": prompt,
                    "best": outputs[np.argmax(rewards)],
                    "worst": outputs[np.argmin(rewards)]
                })
        
        return failures[:n_examples]
```

## Self-Play Training Loop

```python
class SelfPlayTrainer:
    def __init__(self, config):
        self.policy = PolicyModel(config.model_name)
        self.ref_policy = copy.deepcopy(self.policy)
        self.reward_fn = self.setup_reward()
        self.refiner = IterativeRefiner(self.policy)
        
    def train(self, prompts, n_iterations=100):
        for iteration in range(n_iterations):
            # 1. Generate and refine
            refined_data = []
            for prompt in prompts:
                for step in self.refiner.refine(prompt):
                    refined_data.append(step)
            
            # 2. Self-play comparison
            preferences = self.compete(prompts)
            
            # 3. Train on combined data
            self.train_step(refined_data + preferences)
            
            # 4. Update reference periodically
            if iteration % 10 == 0:
                self.ref_policy = copy.deepcopy(self.policy)
            
            # 5. Evaluate and checkpoint
            metrics = self.evaluate()
            if metrics["win_rate"] > 0.6:  # Beating previous version
                self.save_checkpoint(iteration)
```

## Integration with RLHF Pipeline

```python
from rlhf import RLHFOrchestrator

orchestrator = RLHFOrchestrator()

# Standard RLHF stages
orchestrator.run_sft(sft_data)
orchestrator.run_reward_model_training(pref_data)

# Self-play enhancement
for round in range(5):  # Self-play rounds
    # Generate synthetic training data through self-play
    synthetic_data = orchestrator.run_self_play(
        n_games=1000,
        refinement_iterations=3
    )
    
    # Mix with real data
    mixed_data = combine(pref_data, synthetic_data)
    
    # Continue training
    orchestrstrator.run_policy_optimization(
        method="dpo",
        data=mixed_data
    )
```

## Best Practices

1. **Start with good initialization**: Self-play amplifies initial capabilities
2. **Regular reference updates**: Prevents catastrophic forgetting
3. **Diverse prompts**: Need variety to learn general strategies
4. **Quality over quantity**: Better to have fewer high-quality self-play games
5. **Monitor for mode collapse**: Check diversity metrics regularly

## Evaluation

Track these metrics during self-play:

- **Win rate vs previous version**: Should increase over time
- **Elo rating**: Track skill progression
- **Response diversity**: Prevent mode collapse
- **Human evaluation**: Ultimate judge of quality
- **Capability retention**: Don't lose previous abilities

## Applications

### Reasoning Tasks
- Math problem solving
- Code generation
- Logical puzzles

### Creative Tasks
- Story generation
- Dialogue systems
- Content creation

### Strategic Tasks
- Game playing
- Debate
- Negotiation
