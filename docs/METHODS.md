# RLHF Methods Reference

Complete technical reference for all preference learning methods implemented in this pipeline.

## Table of Contents

1. [Direct Preference Optimization (DPO)](#direct-preference-optimization-dpo)
2. [Group Relative Policy Optimization (GRPO)](#group-relative-policy-optimization-grpo)
3. [Simple Preference Optimization (SimPO)](#simple-preference-optimization-simpo)
4. [Kahneman-Tversky Optimization (KTO)](#kahneman-tversky-optimization-kto)
5. [Identity Preference Optimization (IPO)](#identity-preference-optimization-ipo)
6. [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)

---

## Direct Preference Optimization (DPO)

### Overview

DPO eliminates the need for explicit reward modeling by directly optimizing the policy from preference data. It derives a closed-form expression for the optimal policy given a reward function, then optimizes it directly.

**Paper**: *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (Rafailov et al., 2023)

### Mathematical Foundation

**Reward Model**: Under the Bradley-Terry model, the probability of preferring completion y_w over y_l given prompt x is:

```
p*(y_w ≻ y_l | x) = σ(r*(x, y_w) - r*(x, y_l))
```

**Optimal Policy**: The optimal policy under KL constraint is:

```
π_r(y|x) = (1/Z(x)) * π_ref(y|x) * exp(r(x,y)/β)
```

**DPO Loss**: Substituting the reward expression:

```
L_DPO(π_θ; π_ref) = -E[(x, y_w, y_l)~D] [
    log σ(β * log(π_θ(y_w|x)/π_ref(y_w|x)) 
         - β * log(π_θ(y_l|x)/π_ref(y_l|x)))
]
```

### Variants

#### Sigmoid DPO (Standard)

```python
loss = -F.logsigmoid(beta * (chosen_logratios - rejected_logratios))
```

#### Hinge DPO

Maximum margin approach:
```python
loss = torch.relu(1 - beta * (chosen_logratios - rejected_logratios))
```

#### IPO (Identity Preference Optimization)

Removes β from the log-ratio scaling:
```
L_IPO = (log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x)) - 1/(2β))^2
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta` | 0.1 | KL penalty coefficient. Higher = stay closer to reference |
| `loss_type` | "sigmoid" | "sigmoid", "hinge", or "ipo" |
| `label_smoothing` | 0.0 | Smoothing for regularization |

### When to Use

✅ **Use DPO when**:
- You have high-quality paired preference data
- You want simple, stable training
- You need to stay close to the reference model
- Training budget is limited

❌ **Avoid when**:
- You need online learning from environment rewards
- You have unpaired preference data (use KTO instead)
- You need group-based sampling (use GRPO)

### Implementation Notes

```python
from rlhf import RLHFOrchestrator, DPOConfig

config = DPOConfig(
    beta=0.1,
    learning_rate=5e-7,
    loss_type="sigmoid"
)

orchestrator = RLHFOrchestrator()
orchestrator.run_policy_optimization(
    method="dpo",
    data=preference_data,
    config=config
)
```

---

## Group Relative Policy Optimization (GRPO)

### Overview

GRPO is the training method used in DeepSeek-R1. It eliminates the need for a value model by using group-relative advantages, making it particularly effective for reasoning tasks with verifiable rewards.

**Reference**: DeepSeek-R1 training methodology (DeepSeek-AI, 2025)

### Key Innovation

Instead of training a critic network to estimate values, GRPO:
1. Samples G completions for each prompt
2. Computes rewards for each completion
3. Uses the mean reward as baseline (advantage = R_i - mean(R))
4. Applies PPO-style clipped updates

### Algorithm

```python
# For each batch of prompts:
completions = [generate(prompt, G) for prompt in prompts]  # G per prompt
rewards = [reward_fn(c) for c in completions]  # Score each

# Group-relative advantages
mean_rewards = [mean(r_group) for r_group in rewards]
advantages = [r - mean_r for r, mean_r in zip(rewards, mean_rewards)]

# Clipped policy update
ratio = π_new(completions) / π_old(completions)
clipped_ratio = clip(ratio, 1-ε, 1+ε)
objective = min(ratio * A, clipped_ratio * A) - β * KL(π_new || π_ref)
```

### Mathematical Formulation

**Objective**:
```
J_GRPO(θ) = E_{x~D, {y_i}_{i=1}^G ~ π_θ} [
    (1/G) Σ_i min(
        (π_θ(y_i|x) / π_θ_old(y_i|x)) * A_i,
        clip(...) * A_i
    ) - β * KL(π_θ || π_ref)
]

where A_i = R(y_i, x) - (1/G) Σ_j R(y_j, x)
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `group_size` | 8 | Completions per prompt |
| `kl_coeff` | 0.1 | KL penalty coefficient |
| `clip_ratio` | 0.2 | PPO clipping epsilon |
| `temperature` | 1.0 | Sampling temperature |
| `use_verifiable_rewards` | True | Use rule-based rewards when available |

### Reward Functions

**Verifiable Rewards** (best for GRPO):
- Math problems: Check if answer matches ground truth
- Code: Execute and check test cases
- Logic puzzles: Verify solution correctness

**Model-Based Rewards**:
- Use trained reward model
- Process reward model for step-by-step

### When to Use

✅ **Use GRPO when**:
- You have verifiable rewards (math, code, logic)
- You want to avoid training a value model
- You're doing reasoning-focused training
- You can afford multiple samples per prompt

❌ **Avoid when**:
- Rewards are expensive to compute (sampling G times)
- You have purely subjective preferences
- Memory is severely constrained

### Implementation Notes

```python
from rlhf import RLHFOrchestrator, GRPOConfig

config = GRPOConfig(
    group_size=16,  # More samples = better baseline
    kl_coeff=0.1,
    use_verifiable_rewards=True
)

def math_reward_fn(completion: str, answer: str) -> float:
    # Extract final answer and compare
    predicted = extract_answer(completion)
    return 1.0 if check_equivalent(predicted, answer) else 0.0

orchestrator.run_policy_optimization(
    method="grpo",
    data=math_problems,
    config=config,
    reward_fn=math_reward_fn
)
```

---

## Simple Preference Optimization (SimPO)

### Overview

SimPO removes the need for a reference model entirely, making it the most resource-efficient preference learning method. It directly optimizes for a margin between chosen and rejected responses.

**Paper**: *SimPO: Simple Preference Optimization with a Reference-Free Reward* (Meng et al., 2024)

### Key Idea

Instead of comparing to a reference policy, SimPO defines reward as the average log-probability per token:

```
r_simpo(x, y) = (1/|y|) * log π(y|x) = (1/|y|) * Σ_t log π(y_t|x, y_{<t})
```

Then optimizes:
```
L_SimPO = -log σ(β * (r(x, y_w) - r(x, y_l)) - γ)
```

where γ is a target margin.

### Mathematical Formulation

**Length-Normalized Reward**:
```
r(x, y) = (1/|y|) * log π(y|x)
```

**SimPO Loss**:
```
L = -E[(x, y_w, y_l)~D] [log σ(β * (r(x, y_w) - r(x, y_l)) - γ)]
```

The length normalization is crucial - without it, the model learns to generate short responses.

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta` | 2.0 | Reward scaling coefficient (typically higher than DPO) |
| `gamma` | 0.5 | Target reward margin |
| `label_smoothing` | 0.0 | Regularization |

### When to Use

✅ **Use SimPO when**:
- You have strict memory constraints
- You want the fastest training
- You don't have a good reference model
- You're doing continued training

❌ **Avoid when**:
- You need strong regularization to reference
- Your data has high variance in response lengths
- You need precise control over KL divergence

### Comparison with DPO

| Aspect | DPO | SimPO |
|--------|-----|-------|
| Reference Model | Required | Not needed |
| Memory | 2× model | 1× model |
| Training Speed | Moderate | Fast |
| Stability | High | Moderate |
| KL Control | Explicit | Implicit via γ |

### Implementation Notes

```python
from rlhf import RLHFOrchestrator, SimPOConfig

config = SimPOConfig(
    beta=2.0,  # Higher than DPO
    gamma=0.5  # Target margin
)

orchestrator.run_policy_optimization(
    method="simpo",
    data=preference_data,
    config=config
)
```

---

## Kahneman-Tversky Optimization (KTO)

### Overview

KTO enables training from non-paired preference data. You only need examples labeled as "desirable" or "undesirable" - no need for matched pairs. Based on prospect theory's observation that humans are loss-averse.

**Paper**: *Human-Centered Loss Functions (HALO) Part 2: KTO* (Ethayarajh et al., 2024)

### Key Innovation

KTO models the human preference process as:
1. Humans imagine a reference outcome
2. They compare actual outcome to reference
3. Losses hurt more than equivalent gains feel good (loss aversion)

### Mathematical Formulation

**Ideal Generation Probability**:
```
π(y|x) = π(y|x)^λ_d if y is desirable
π(y|x) = π(y|x)^λ_u if y is undesirable
```

**KTO Loss**:
```
L_KTO = E_{x,y~D_desirable} [λ_d * (1 - σ(β * KL(π(y|x) || π_ref(y|x))))]
      + E_{x,y~D_undesirable} [λ_u * σ(β * KL(π(y|x) || π_ref(y|x)))]
```

**KL Estimation via EMA**:
```
KL_EMA = α * KL_current + (1-α) * KL_EMA
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta` | 0.1 | KL coefficient |
| `lambda_d` | 1.0 | Weight for desirable examples |
| `lambda_u` | 1.0 | Weight for undesirable examples |
| `kl_ema_decay` | 0.99 | EMA decay for KL estimate |

**Note**: Typically λ_d > λ_u (loss aversion), e.g., λ_d=1.0, λ_u=0.5

### When to Use

✅ **Use KTO when**:
- You have natural data without paired comparisons
- Collecting paired data is expensive
- You want to use all available data (including unpaired)
- You have implicit feedback (clicks, time-spent)

❌ **Avoid when**:
- You have high-quality paired data (DPO may work better)
- You need fine-grained preference distinctions

### Data Format

```python
# Desirable examples
desirable_data = [
    {"prompt": "...", "completion": "...", "label": "desirable"},
    ...
]

# Undesirable examples
undesirable_data = [
    {"prompt": "...", "completion": "...", "label": "undesirable"},
    ...
]
```

### Implementation Notes

```python
from rlhf import RLHFOrchestrator, KTOConfig

config = KTOConfig(
    beta=0.1,
    lambda_d=1.0,
    lambda_u=0.5,  # Loss aversion
    kl_ema_decay=0.99
)

orchestrator.run_policy_optimization(
    method="kto",
    data=mixed_preference_data,
    config=config
)
```

---

## Identity Preference Optimization (IPO)

### Overview

IPO is a variant of DPO that removes the β parameter from inside the log-ratio, placing it outside as a regularization term. This can lead to more stable training.

**Paper**: *A General Theoretical Paradigm to Understand Learning from Human Preferences* (Azar et al., 2023)

### Mathematical Formulation

**IPO Loss**:
```
L_IPO = E[(x, y_w, y_l)~D] [
    (log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x)) - 1/(2β))^2
]
```

This is a squared loss instead of log-loss, which:
- Prevents the model from over-optimizing preferences
- Provides more stable gradients
- Has a natural interpretation as matching a target gap

### When to Use

✅ **Use IPO when**:
- DPO training is unstable
- You're seeing over-optimization (reward hacking)
- You want more conservative updates

### Comparison

```
DPO:  L = -log σ(β * (logratio_w - logratio_l))
IPO:  L = (logratio_w - logratio_l - 1/(2β))^2
```

IPO tends to be more stable but may converge slower.

---

## Proximal Policy Optimization (PPO)

### Overview

PPO is the full reinforcement learning approach used in the original RLHF work (InstructGPT, etc.). It trains a value model to estimate advantages and uses clipped importance sampling for stable updates.

**Paper**: *Proximal Policy Optimization Algorithms* (Schulman et al., 2017)
**RLHF**: *Training Language Models to Follow Instructions* (Ouyang et al., 2022)

### Components

1. **Policy Model (Actor)**: Generates completions
2. **Value Model (Critic)**: Estimates state values
3. **Reward Model**: Provides final rewards
4. **Reference Model**: For KL penalty

### Algorithm

```
1. Collect rollouts:
   For each prompt:
     Generate completion with current policy
     Compute reward from reward model
     Store (prompt, completion, reward, log_prob, value)

2. Compute advantages with GAE:
   δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
   A_t = Σ_{l=0}^∞ (γλ)^l * δ_{t+l}

3. Update policy (multiple epochs):
   ratio = π_new / π_old
   objective = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
   loss = -objective + β * KL(π_new || π_ref) - c * entropy

4. Update value model:
   V_loss = MSE(V_pred, returns)
```

### Generalized Advantage Estimation (GAE)

```
A_t^GAE(γ, λ) = Σ_{l=0}^∞ (γλ)^l * δ_{t+l}

where δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

- λ = 0: High bias, low variance (TD(0))
- λ = 1: Low bias, high variance (Monte Carlo)
- λ = 0.95: Typical compromise

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip_ratio` | 0.2 | PPO clipping parameter |
| `kl_coeff` | 0.02 | KL penalty coefficient |
| `kl_target` | None | Adaptive KL if set |
| `value_loss_coef` | 0.5 | Value loss weight |
| `entropy_coef` | 0.01 | Entropy bonus |
| `lam` | 0.95 | GAE lambda |
| `gamma` | 0.99 | Discount factor |
| `ppo_epochs` | 4 | Update epochs per batch |
| `mini_batch_size` | 64 | Minibatch size for updates |

### Reward Hacking Prevention

**KL Penalty**: Prevent policy from drifting too far from reference
**Reward Whiting**: Normalize rewards to reduce variance
**Entropy Bonus**: Encourage exploration

### When to Use

✅ **Use PPO when**:
- You have a trained reward model
- You want maximum control over training
- You need online learning capability
- You have sufficient compute for value model training

❌ **Avoid when**:
- You want simpler implementation (use DPO/GRPO)
- Memory is constrained (needs 3+ models)
- You don't have a good reward model

### Implementation Notes

```python
from rlhf import RLHFOrchestrator, PPOConfig

config = PPOConfig(
    clip_ratio=0.2,
    kl_coeff=0.02,
    lam=0.95,
    gamma=0.99,
    ppo_epochs=4
)

# Reward function can be model-based or rule-based
def reward_fn(completions):
    return reward_model.score(completions)

orchestrator.run_policy_optimization(
    method="ppo",
    data=prompts,
    config=config,
    reward_fn=reward_fn
)
```

---

## Method Selection Guide

```
Do you have paired preference data?
├── YES → Do you need reference-free training?
│         ├── YES → SimPO
│         └── NO  → Do you want maximum stability?
│                   ├── YES → IPO
│                   └── NO  → DPO
└── NO  → Do you have verifiable rewards?
          ├── YES → GRPO
          └── NO  → KTO (unpaired data)

Do you need online learning / have a good reward model?
└── YES → PPO
```

## Performance Characteristics

| Method | Memory | Speed | Stability | Flexibility |
|--------|--------|-------|-----------|-------------|
| DPO | Medium | Fast | High | Medium |
| GRPO | Medium | Medium | High | High |
| SimPO | Low | Fastest | Medium | Low |
| KTO | Medium | Fast | Medium | Medium |
| IPO | Medium | Fast | Very High | Medium |
| PPO | High | Slow | Medium | Very High |
