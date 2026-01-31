# Usage Guide

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Full-RLHF-Pipeline.git
cd Full-RLHF-Pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Example

```python
from rlhf import RLHFOrchestrator, SFTConfig, DPOConfig

# Initialize orchestrator
orchestrator = RLHFOrchestrator()

# Stage 1: SFT
sft_config = SFTConfig(
    learning_rate=5e-6,
    batch_size=32,
    num_epochs=3
)

orchestrator.run_sft(
    data=sft_data,  # List of {"prompt": ..., "response": ...}
    config=sft_config
)

# Stage 2: DPO
dpo_config = DPOConfig(
    beta=0.1,
    learning_rate=5e-7,
    num_epochs=2
)

orchestrator.run_policy_optimization(
    method="dpo",
    data=preference_data,  # List of {"prompt": ..., "chosen": ..., "rejected": ...}
    config=dpo_config
)

# Save final model
orchestrator.save_models("./final_model")
```

## Complete Pipeline Examples

### Example 1: Full DPO Pipeline

```python
from rlhf import (
    RLHFOrchestrator,
    SFTConfig, RewardModelConfig, DPOConfig,
    PolicyModel
)

# Initialize
orchestrator = RLHFOrchestrator(device="cuda")

# SFT Stage
print("Stage 1: Supervised Fine-Tuning")
sft_config = SFTConfig(
    learning_rate=5e-6,
    batch_size=64,
    num_epochs=3,
    max_seq_length=2048,
    use_lora=True,
    lora_r=16,
    lora_alpha=32
)

orchestrator.run_sft(
    data=sft_dataset,
    config=sft_config
)

# Reward Model Training
print("Stage 2: Reward Model Training")
rm_config = RewardModelConfig(
    learning_rate=1e-5,
    batch_size=32,
    num_epochs=5,
    ensemble_size=3  # Train ensemble for uncertainty
)

orchestrator.run_reward_model_training(
    data=preference_dataset,
    config=rm_config
)

# DPO Training
print("Stage 3: Direct Preference Optimization")
dpo_config = DPOConfig(
    beta=0.1,
    learning_rate=5e-7,
    num_epochs=2,
    loss_type="sigmoid",  # or "hinge", "ipo"
    label_smoothing=0.0
)

orchestrator.run_policy_optimization(
    method="dpo",
    data=preference_dataset,
    config=dpo_config
)

# Evaluation
print("Stage 4: Evaluation")
metrics = orchestrator.evaluate(
    eval_prompts=test_prompts,
    reference_model=orchestrator.reference_model
)

print(f"KL Divergence: {metrics['kl_div']:.4f}")
print(f"Reward Accuracy: {metrics['reward_accuracy']:.4f}")
print(f"Win Rate: {metrics['win_rate']:.2%}")
```

### Example 2: GRPO for Math Reasoning

```python
from rlhf import RLHFOrchestrator, GRPOConfig
import re

# Initialize
orchestrator = RLHFOrchestrator()

# Math problem dataset
math_problems = [
    {
        "prompt": "Solve: 2x + 5 = 13",
        "answer": "4"
    },
    # ... more problems
]

# Verifiable reward function
def math_reward_fn(completion: str, answer: str) -> float:
    """Extract and check mathematical answer."""
    # Extract final answer (assuming format "... = 4" or "Answer: 4")
    patterns = [
        r'[=:]\s*(-?\d+)',
        r'answer is\s*(-?\d+)',
        r'final answer[,:]?\s*(-?\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, completion.lower())
        if match:
            predicted = match.group(1)
            return 1.0 if predicted == answer else 0.0
    
    return 0.0  # No answer found

# GRPO Configuration
grpo_config = GRPOConfig(
    learning_rate=1e-6,
    batch_size=16,  # Number of prompts
    group_size=8,   # Completions per prompt
    num_epochs=3,
    kl_coeff=0.1,
    clip_ratio=0.2,
    max_completion_length=512,
    use_verifiable_rewards=True
)

# Run GRPO
orchestrator.run_policy_optimization(
    method="grpo",
    data=math_problems,
    config=grpo_config,
    reward_fn=lambda comp, item: math_reward_fn(comp, item["answer"])
)

# Test
response = orchestrator.policy.generate(
    "Solve: 3x - 7 = 20",
    max_new_tokens=256
)
print(response)
```

### Example 3: KTO with Unpaired Data

```python
from rlhf import RLHFOrchestrator, KTOConfig

# Unpaired data examples
data = [
    # Desirable examples
    {"prompt": "Write a poem about nature", 
     "completion": "The trees sway gently...", 
     "label": "desirable"},
    
    # Undesirable examples
    {"prompt": "Explain quantum physics", 
     "completion": "Quantum physics is hard...", 
     "label": "undesirable"},
    # ...
]

kto_config = KTOConfig(
    learning_rate=5e-7,
    batch_size=32,
    num_epochs=2,
    beta=0.1,
    lambda_d=1.0,   # Weight for desirable
    lambda_u=0.5,   # Weight for undesirable (loss aversion)
    kl_ema_decay=0.99
)

orchestrator.run_policy_optimization(
    method="kto",
    data=data,
    config=kto_config
)
```

### Example 4: Full PPO Training

```python
from rlhf import RLHFOrchestrator, PPOConfig

# PPO requires a reward function/model
def custom_reward_fn(completions: List[str]) -> List[float]:
    rewards = []
    for completion in completions:
        score = 0.0
        # Length penalty (prefer concise)
        score += max(0, 1.0 - len(completion) / 1000)
        # Fluency score (example)
        score += fluency_model.score(completion)
        # Task-specific score
        score += task_reward(completion)
        rewards.append(score)
    return rewards

ppo_config = PPOConfig(
    learning_rate=1e-6,
    batch_size=256,
    num_epochs=4,
    clip_ratio=0.2,
    kl_coeff=0.02,
    kl_target=0.01,  # Adaptive KL
    value_loss_coef=0.5,
    entropy_coef=0.01,
    lam=0.95,  # GAE lambda
    gamma=0.99,
    ppo_epochs=4,
    mini_batch_size=64,
    rollout_multiplier=4,  # Collect 4× batch size rollouts
    max_completion_length=512
)

orchestrator.run_policy_optimization(
    method="ppo",
    data=prompts,
    config=ppo_config,
    reward_fn=custom_reward_fn
)
```

### Example 5: Self-Play with Iterative Refinement

```python
from rlhf import RLHFOrchestrator

orchestrator = RLHFOrchestrator()

# Initial SFT
orchestrator.run_sft(sft_data, sft_config)

# Self-play rounds
for round_num in range(5):
    print(f"Self-Play Round {round_num + 1}")
    
    # Generate synthetic training data
    synthetic_data = orchestrator.run_self_play(
        n_games=1000,
        refinement_iterations=3,
        capability_test_prompts=test_prompts
    )
    
    # Mix with real data
    mixed_data = combine_datasets(real_preference_data, synthetic_data)
    
    # Continue training with DPO
    orchestrator.run_policy_optimization(
        method="dpo",
        data=mixed_data,
        config=dpo_config
    )
    
    # Save checkpoint
    orchestrator.save_models(f"./checkpoints/round_{round_num}")
```

## Configuration Presets

### 7B Model Config

```python
from rlhf import get_7b_model_config

config = get_7b_model_config()
# Sets appropriate batch sizes, gradient accumulation, etc.
# for training 7B models on single/multi-GPU setups
```

### 70B Model Config

```python
from rlhf import get_70b_model_config

config = get_70b_model_config()
# Sets up for large model training with:
# - Aggressive gradient accumulation
# - LoRA by default
# - Offloading strategies
# - Multi-node settings
```

## Advanced Features

### Context Compression

```python
orchestrator = RLHFOrchestrator(
    use_context_compression=True,
    compression_config={
        "method": "attention",
        "compression_ratio": 16,
        "chunk_size": 512
    }
)

# Now handles long contexts efficiently
response = orchestrator.generate(long_document + question)
```

### Streaming Datasets

```python
from rlhf import StreamingPreferenceDataset

# For datasets too large to fit in memory
dataset = StreamingPreferenceDataset(
    data_path="path/to/sharded/data",
    tokenizer=tokenizer,
    shard_size=10000
)

orchestrator.run_policy_optimization(
    method="dpo",
    data=dataset,  # Streaming dataset
    config=config
)
```

### Custom Reward Functions

```python
# Rule-based rewards
def rule_based_reward(completion: str) -> float:
    score = 0.0
    if has_valid_json(completion):
        score += 0.3
    if passes_syntax_check(completion):
        score += 0.4
    if meets_length_requirements(completion):
        score += 0.3
    return score

# Model-based rewards
def model_based_reward(completion: str) -> float:
    return reward_model.score(completion)

# Hybrid rewards
def hybrid_reward(completion: str) -> float:
    hard_reward = rule_based_reward(completion)
    if hard_reward == 1.0:  # Perfect on rules
        soft_reward = model_based_reward(completion)
        return 0.7 + 0.3 * soft_reward
    return hard_reward
```

### Checkpoint Management

```python
# Automatic checkpointing
config.save_steps = 500
config.save_total_limit = 3  # Keep only 3 most recent

# Resume from checkpoint
config.resume_from_checkpoint = "./checkpoints/checkpoint-1000"
orchestrator.run_policy_optimization(...)

# Manual save/load
orchestrator.save_models("./my_checkpoint")
orchestrator.load_models("./my_checkpoint")
```

## Monitoring & Logging

### Weights & Biases

```python
config.use_wandb = True
config.wandb_project = "my-rlhf-project"
config.wandb_run_name = "experiment-1"
```

### TensorBoard

```python
config.use_tensorboard = True
config.tensorboard_dir = "./logs"
```

### Custom Metrics

```python
def custom_evaluation(policy, dataset):
    metrics = {}
    # Your custom metrics
    return metrics

metrics = orchestrator.evaluate(
    eval_prompts=test_prompts,
    custom_metrics_fn=custom_evaluation
)
```

## Troubleshooting

### Out of Memory

```python
# Solutions:
config.batch_size = 8  # Reduce batch size
config.gradient_accumulation_steps = 4  # Effective batch = 32
config.use_lora = True  # Use parameter-efficient fine-tuning
config.fp16 = True  # Use mixed precision
```

### Training Instability

```python
# Solutions:
config.learning_rate = 1e-7  # Lower learning rate
config.beta = 0.2  # Increase KL penalty (DPO/GRPO)
config.clip_ratio = 0.1  # Tighter clipping (PPO)
config.loss_type = "ipo"  # More stable than sigmoid
```

### Mode Collapse

```python
# Solutions:
config.entropy_coef = 0.02  # Increase exploration (PPO)
config.temperature = 1.2  # Increase sampling diversity
# Use response length penalties in reward function
```

## Best Practices

1. **Start with SFT**: Always begin with supervised fine-tuning
2. **Validate frequently**: Evaluate on held-out set every N steps
3. **Monitor KL**: Keep KL divergence in reasonable range (0.1-0.5)
4. **Use checkpoints**: Save frequently, rollback on regression
5. **Mix methods**: Combine multiple methods for best results
6. **Test thoroughly**: Run capability tests before deploying

## Performance Tips

- Use `torch.compile()` for 10-20% speedup
- Enable gradient checkpointing for large models
- Use DeepSpeed for multi-GPU training
- Pre-tokenize and cache datasets
- Use streaming for large datasets
