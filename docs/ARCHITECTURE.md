# Architecture Deep Dive

## System Overview

The Full-RLHF-Pipeline is architected as a modular, composable system where each stage can operate independently or as part of an end-to-end pipeline. The design follows these core principles:

1. **Separation of Concerns**: Data, models, trainers, and orchestration are cleanly separated
2. **Configurability**: Every component is configurable through dataclass configs with validation
3. **Observability**: Comprehensive logging at every layer
4. **Extensibility**: New methods can be added without modifying existing code

## Core Components

### 1. Configuration System

All configurations inherit from `BaseConfig` and provide:

- Type-safe configuration with dataclasses
- JSON serialization/deserialization via `dataclasses-json`
- Runtime validation with meaningful error messages

```python
@dataclass
class DPOConfig(BaseConfig):
    beta: float = 0.1  # KL coefficient
    loss_type: str = "sigmoid"  # "sigmoid", "hinge", "ipo"
    
    def validate(self) -> None:
        super().validate()
        assert self.beta > 0, "beta must be positive"
```

### 2. Data Pipeline

#### Dataset Classes

**PreferenceDataset**

- Handles paired preference data (prompt, chosen, rejected)
- Tokenizes with proper truncation and padding
- Returns tensor dictionaries ready for training

**SFTDataset**

- Supervised fine-tuning data
- Optional prompt masking (only train on responses)
- Packing support for efficiency

**Streaming Variants**

- File-backed datasets for TB-scale training
- Sharded data loading for distributed training
- On-the-fly tokenization

#### Data Flow

```
Raw Data → Tokenizer → Dataset → DataLoader → Trainer
    ↓           ↓          ↓          ↓
 JSON/LMDB  Transform  Batching   Collate
```

### 3. Model Architecture

#### PolicyModel

Wraps `AutoModelForCausalLM` with RLHF-specific functionality:

```python
class PolicyModel:
    def forward(self, input_ids, attention_mask) -> CausalLMOutput
    def generate(self, prompts, **kwargs) -> List[str]
    def get_log_probs(self, sequences, attention_mask) -> torch.Tensor
    def save_pretrained(path)
    def from_pretrained(path)
```

**Key Features**:

- Efficient log-probability computation for KL divergence
- Generation with custom stopping criteria
- Automatic device placement

#### RewardModel

Standalone reward model with ranking head:

```
Base LM → [Transformer Layers] → [Linear] → [ReLU] → [Linear] → Scalar Reward
```

**Ensemble Support**: Multiple reward models can be combined for uncertainty estimation.

#### ProcessRewardModel

Step-level reward model for chain-of-thought reasoning:

- Predicts reward at each reasoning step
- Aggregates to final outcome reward
- Enables process supervision (like OpenAI's ORM)

#### ValueModel

Critic network for PPO:

- Predicts state values for advantage estimation
- Can return per-token or final values
- Separate from policy to allow different architectures

#### ContextCompressor

Long-context handling module:

- Compresses extended contexts into fixed-size representations
- Uses attention-based compression or learned compression
- Enables processing beyond base model context limits

### 4. Training Infrastructure

#### DeviceManager

Centralized device and precision management:

```python
class DeviceManager:
    def __init__(self, device, dtype, use_amp)
    def autocast_context()  # Mixed precision context
    def backward(loss, optimizer)  # With gradient scaling
    def step(optimizer, max_grad_norm)  # With clipping
```

**Features**:

- Automatic BF16/FP16 selection based on hardware
- Gradient scaling for AMP
- Gradient clipping

#### CheckpointManager

Robust checkpointing:

- Rolling checkpoints (keep only N most recent)
- Atomic saves (write to temp, then move)
- Automatic recovery from latest checkpoint
- Metadata tracking (step, epoch, metrics)

#### TrainingLogger

Multi-backend logging:

- Console (with Rich formatting)
- Weights & Biases
- TensorBoard
- Local JSONL files

### 5. Training Algorithms

#### Trainer Hierarchy

```
BaseTrainer (abstract)
    ├── SFTTrainer
    ├── RewardModelTrainer
    ├── DPOTrainer
    ├── GRPOTrainer
    ├── SimPOTrainer
    ├── KTOTrainer
    └── PPOTrainer
```

Each trainer implements:

- `train_epoch()` - One epoch of training
- `train_step()` - Single batch update
- `evaluate()` - Validation metrics
- `save_checkpoint()` - Persist state

#### DPO (Direct Preference Optimization)

**Loss Function**:

```
L_DPO = -log σ(β * log(π(y_w|x)/π_ref(y_w|x)) - β * log(π(y_l|x)/π_ref(y_l|x)))
```

**Variants**:

- Sigmoid (standard DPO)
- Hinge (max-margin)
- IPO (no β parameter)

#### GRPO (Group Relative Policy Optimization)

**Key Innovation**: No value model needed - uses group-relative advantages.

**Process**:

1. Sample G completions per prompt
2. Score each with reward function
3. Compute advantages: A_i = R_i - mean(R)
4. Update with PPO-style clipped objective

**Advantages**:

- No critic training
- Natural variance reduction through grouping
- Works well for verifiable rewards (math, code)

#### PPO (Proximal Policy Optimization)

**Full RL implementation**:

1. Collect rollouts with current policy
2. Compute advantages with GAE
3. Multiple epochs of clipped updates
4. KL penalty to prevent drift

**Components**:

- Value model for advantage estimation
- Reward function (can be model or rule-based)
- Old policy for importance sampling

### 6. Self-Improvement System

#### AdversarialValidator

Automated quality assessment:

- Perplexity-based fluency scoring
- Repetition detection
- Coherence checking
- Factual consistency (via NLI)

#### CapabilityTester

Regression testing suite:

- Task-specific evaluations
- Win-rate against baseline
- Performance thresholds
- Automatic rollback on degradation

#### IterativeRefiner

Generate-critique-refine loop:

```
Prompt → Generate → Critique → Refine → Final Output
           ↑___________________________|
```

**Strategies**:

- Self-critique (model critiques own output)
- Constitutional critique (principle-guided)
- Iterative refinement (multiple rounds)

### 7. Orchestration

#### RLHFOrchestrator

Central coordinator for end-to-end pipelines:

```python
class RLHFOrchestrator:
    def run_sft(self, data, config)
    def run_reward_model_training(self, data, config)
    def run_policy_optimization(self, method, data, config)
    def run_full_pipeline(self, stages)
    def save_models(self, path)
    def load_models(self, path)
```

**Responsibilities**:

- Reference model creation (frozen copy)
- Stage sequencing
- Checkpoint management between stages
- Rollback on failure/regression
- History aggregation

## Data Flow Examples

### Example 1: DPO Pipeline

```
1. Load SFT model checkpoint
2. Create frozen reference copy
3. Load preference dataset
4. For each batch:
   a. Compute log-probs for chosen/rejected under π and π_ref
   b. Calculate DPO loss
   c. Backprop and update
5. Evaluate on held-out set
6. Save best checkpoint
```

### Example 2: GRPO with Self-Play

```
1. Initialize policy
2. For each training step:
   a. Sample prompts from task distribution
   b. Generate G completions per prompt
   c. Score with reward function (verifiable tasks)
   d. Compute group-relative advantages
   e. Update policy with clipped objective
3. Run adversarial validation
4. If regression detected, rollback
5. Otherwise, continue training
```

## Extension Points

### Adding a New Method

1. Create config class inheriting from `BaseConfig`
2. Implement trainer class with required methods
3. Register in orchestrator

Example:

```python
@dataclass
class MyMethodConfig(BaseConfig):
    my_param: float = 0.1

class MyMethodTrainer:
    def __init__(self, policy, ref_model, config):
        ...
    def train_step(self, batch):
        ...
```

### Custom Reward Functions

```python
def my_reward_fn(completions: List[str]) -> List[float]:
    # Your logic here
    return rewards

# Use in GRPO/PPO
orchestrator.run_policy_optimization(
    method="grpo",
    reward_fn=my_reward_fn
)
```

### Custom Datasets

```python
class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        ...
    def __getitem__(self, idx):
        return {
            'input_ids': ...,
            'attention_mask': ...,
            'custom_field': ...
        }
```

## Performance Considerations

### Memory Optimization

1. **Gradient Checkpointing**: Trade compute for memory
2. **LoRA**: Train only adapter layers
3. **8-bit Optimizers**: Reduce optimizer state
4. **Gradient Accumulation**: Effective larger batches

### Throughput Optimization

1. **DataLoader Workers**: Parallel data loading
2. **Mixed Precision**: BF16/FP16 training
3. **Compiled Models**: torch.compile() support
4. **Distributed Training**: DDP/FSDP

### Scaling Strategies

| Setup | Strategy | Expected Speedup |
|-------|----------|------------------|
| Single GPU | LoRA + Gradient Accumulation | 1x (baseline) |
| Multi-GPU | DDP | ~N× (N GPUs) |
| Large Models | FSDP + Offloading | Enables training |
| TB Data | Streaming + Sharding | Unbounded data |

## Monitoring & Debugging

### Key Metrics

**All Methods**:

- Loss curves
- KL divergence from reference
- Gradient norms
- Learning rate

**Method-Specific**:

- DPO: Reward margin (chosen - rejected)
- GRPO: Group reward variance, advantage magnitude
- PPO: Value loss, entropy, explained variance
- KTO: KL estimate from EMA

### Debugging Tools

1. **Gradient Flow Visualization**: Check for vanishing/exploding gradients
2. **Activation Histograms**: Monitor distribution shifts
3. **Sample Generation**: Periodic qualitative evaluation
4. **Reward Debugging**: Log reward components separately

## Future Architecture Directions

1. **Online RL**: Continuous learning from live feedback
2. **Multi-Agent**: Multiple policies competing/cooperating
3. **Hierarchical**: High-level + low-level policy decomposition
4. **Neural Architecture Search**: Auto-discover optimal model architectures
