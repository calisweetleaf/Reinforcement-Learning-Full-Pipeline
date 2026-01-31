# Full-RLHF-Pipeline: End-to-End Specification

## Abstract

This document specifies the end-to-end RLHF pipeline implemented in `rlhf.py`, with runtime extensions in `inference_optimizations.py` and `model_merging.py`. The focus is operational: data contracts, model roles, training stages, and the concrete behaviors encoded in the codebase. The intent is to describe what exists and how it executes, not to propose new features.

The pipeline is designed as a modular training system that allows supervised fine-tuning, reward modeling, and multiple preference-optimization methods to be composed into a single production-grade training flow. Each method occupies a precise position in the pipeline and has a distinct contract for inputs, outputs, and model state transitions.

## Scope and Sources of Truth

The specification is grounded in the implementation. The primary source is `rlhf.py` (all stages, trainers, and orchestration). Inference-time compute optimizations are sourced from `inference_optimizations.py`. Model combination and merging procedures are sourced from `model_merging.py`. Existing documentation in `docs/` is used to strengthen explanatory framing where it matches the code.

## System Overview

The pipeline runs as a staged workflow: (1) supervised fine-tuning to establish a strong policy baseline, (2) reward model training to learn preference signals, (3) a policy-optimization stage (DPO, GRPO, SimPO, KTO, or PPO), followed by evaluation and optional self-improvement loops. Supporting infrastructure (device management, logging, checkpointing) is shared across all stages to ensure consistent training behavior, reproducibility, and operational stability.

## Architecture and Component Map

The architecture is organized into configurations, datasets, models, training infrastructure, trainers, and orchestration. The control plane is intentionally thin: most behavior is encoded in the trainer classes, with shared utilities enforcing consistent device, precision, logging, and checkpoint semantics.

**Configuration system.** All stage configs inherit from `BaseConfig`, which defines optimizer parameters, scheduling, gradient accumulation, mixed precision flags, logging and checkpointing cadence, and optional experiment tracking. Each method-specific config adds only its core hyperparameters and validation rules. This ensures consistent behavior across stages and simplifies orchestration.

**Device and precision management.** `DeviceManager` resolves device placement, selects BF16 or FP16 when available, and wraps forward passes in an AMP autocast context. It centralizes gradient scaling and clipping, guaranteeing that every trainer applies the same numerical stability rules.

**Optimization stack.** `create_optimizer` constructs AdamW with epsilon and weight decay, and optionally attaches a cosine warmup schedule when training steps are known. The same optimizer factory is used by all trainers.

**Logging and checkpoints.** `TrainingLogger` multiplexes metrics to console, WandB, and TensorBoard based on availability and config flags. It maintains a local history and standardizes metric naming by stage. `CheckpointManager` implements rolling checkpoints by stage, saving model and optimizer state with metadata and pruning older checkpoints beyond a configurable limit.

**Parameter-efficient fine-tuning.** `apply_lora` integrates PEFT adapters when enabled and logs the proportion of trainable parameters. This supports large-model training within constrained hardware while keeping the core training loops unchanged.

## Data Contracts and Datasets

The pipeline standardizes data at the dataset layer so every trainer receives consistent tensors. Four primary data contracts are implemented, each with an in-memory and a streaming variant.

**SFT data.** Supervised fine-tuning expects `{prompt, response}` pairs. `SFTDataset` concatenates prompt and response, tokenizes to a fixed length, and produces `input_ids`, `attention_mask`, and `labels`. Prompt tokens are optionally masked in `labels` (set to `-100`) to focus loss on the response while preserving the full context in the forward pass.

**Preference pairs.** Preference learning (DPO, SimPO, reward modeling) uses `{prompt, chosen, rejected}`. `PreferenceDataset` tokenizes prompt+chosen and prompt+rejected independently, and provides both tokenized variants along with `prompt_length` computed from a prompt-only pass. The prompt length is used to restrict losses to response tokens.

**Unpaired labels.** KTO consumes `{prompt, response, label}` where `label` is binary (1 desirable, 0 undesirable). `KTODataset` returns tokenized prompt+response with `prompt_length` and a float label tensor for asymmetric loss computation.

**Prompt-only rollouts.** GRPO and PPO operate on prompts only and generate completions online. `GRPODataset` tokenizes prompts with a configurable maximum prompt length and returns both text and tensors so the trainer can decode completions for reward evaluation.

**Streaming variants.** `StreamingPreferenceDataset`, `StreamingSFTDataset`, `StreamingKTODataset`, and `StreamingGRPODataset` accept file-backed datasets for large-scale training. They support JSONL or raw-text line formats, maintain a shuffle buffer for on-the-fly randomization, and yield the same tensor structures as their in-memory counterparts. This provides a drop-in path to scale without changing trainer logic.

This specification treats dataset selection as an operational decision layered on top of these contracts. The user-provided dataset stack is documented separately as an example configuration, not as a requirement of the code.

### Operational Dataset Stack (User-Specified)

The following dataset stack reflects the current target configuration supplied by the user. It is presented as an operational mapping to the pipeline stages and trainer classes.

| Pipeline Stage | Trainer Class | Dataset (HF ID) | Rationale (condensed) |
| --- | --- | --- | --- |
| Stage 1: SFT | `SFTTrainer` | `Magpie-Align/Magpie-Pro-300K-Filtered` | Synthetic instruction data distilled from aligned LLMs; curated and filtered. |
| Stage 2: RM | `RewardModelTrainer` | `nvidia/HelpSteer2` | Multi-attribute scores suitable for reward modeling. |
| Stage 3: DPO | `DPOTrainer` | `argilla/distilabel-intel-orca-dpo-pairs` | High-margin preference pairs for stable DPO. |
| Stage 3: GRPO | `GRPOTrainer` | `AI-MO/NuminaMath-CoT` (prompts only) | Reasoning-heavy prompts with verifiable answers. |
| Stage 3: SimPO | `SimPOTrainer` | `princeton-nlp/SimPO-UltraFeedback` | Length-normalized preferences for SimPO. |
| Stage 3: KTO | `KTOTrainer` | `trl-lib/kto-mix-14k` | Binary-labeled data formatted for KTO. |
| Stage 3: PPO | `PPOTrainer` | `openbmb/UltraFeedback` (prompts only) | Diverse prompts for rollout generation. |

## Method Specifications (Section by Method)

Each method section includes: pipeline placement, objective and training signal, inputs/outputs, and practical behavior as implemented in the trainer and configuration classes.

1. Supervised Fine-Tuning (SFT)
2. Reward Model Training (RM)
3. Direct Preference Optimization (DPO)
4. Group Relative Policy Optimization (GRPO)
5. Simple Preference Optimization (SimPO)
6. Kahneman-Tversky Optimization (KTO)
7. Proximal Policy Optimization (PPO)

### Supervised Fine-Tuning (SFT)

**Pipeline placement.** SFT is the entry point for training. It establishes a strong instruction-following policy before any preference learning is applied. In the pipeline this occurs immediately after initialization and before reward modeling or preference optimization, because downstream methods assume a competent base policy and rely on its logits and generations as stable inputs.

**Inputs and data contract.** The SFT stage consumes records with `prompt` and `response`. The `SFTDataset` concatenates prompt and response, tokenizes the full sequence to a fixed maximum length, and constructs labels with a masked prompt region. Prompt masking is implemented by setting label positions corresponding to prompt tokens to `-100`, which disables loss contribution at those positions while keeping the full input context for the model’s forward pass. Padding positions are also masked to `-100`, preserving stable loss scaling under variable-length inputs.

**Training signal and objective.** The trainer delegates the loss to the underlying causal language model by passing `labels` into the `PolicyModel` forward call. The effective objective is standard cross-entropy over the response tokens (or over the full sequence if prompt masking is disabled), with gradient accumulation applied to reach the configured effective batch size.

**Implementation behavior.** The `SFTTrainer` uses `DeviceManager` for device placement and mixed-precision autocasting, `create_optimizer` for AdamW and optional cosine scheduling with warmup, and `CheckpointManager` for rolling checkpoints. Logging occurs at configured step intervals, evaluation can run periodically on a held-out dataloader, and training can resume from the latest checkpoint. Gradient clipping and AMP scaling are applied consistently across steps, and the loop is structured to ensure deterministic update boundaries when gradient accumulation is enabled.

**Outputs and downstream use.** The result is an updated `PolicyModel` and training loss history. This model becomes the base policy for subsequent stages, and in DPO-like workflows it is also the source for the frozen reference model copy used to anchor preference learning.

### Reward Model Training (RM)

**Pipeline placement.** Reward modeling follows SFT because it requires a competent policy distribution and preference data over that distribution. The reward model is trained before any policy-optimization method that consumes reward signals or preference margins. This ordering ensures that preference learning is grounded in a stable scoring function and that later methods can reuse a consistent reward interface.

**Model architecture.** `RewardModel` wraps a base transformer encoder (`AutoModel`) and adds a two-layer MLP reward head with GELU activation and dropout. The head maps pooled hidden states to a scalar reward. Pooling defaults to the last non-padding token, with optional mean pooling over masked tokens. The reward head weights are explicitly initialized for stability, and the class provides `save_pretrained` and `from_pretrained` utilities with explicit metadata tracking.

**Inputs and data contract.** The stage consumes paired preference data (`prompt`, `chosen`, `rejected`). The `PreferenceDataset` tokenizes prompt+chosen and prompt+rejected, producing paired input IDs and attention masks, along with the prompt length for diagnostics.

**Training signal and objective.** `RewardModelTrainer` implements Bradley-Terry style pairwise ranking. For each pair it computes `logits = reward(chosen) - reward(rejected)`, applies an optional margin, and optimizes binary cross-entropy against a positive target with label smoothing. Optional L2 regularization is applied to the reward model parameters. An ensemble of reward models is supported: each model is trained independently with its own optimizer and schedule, and evaluation aggregates across the ensemble.

**Evaluation and outputs.** The trainer’s evaluation computes preference accuracy by comparing mean ensemble rewards for chosen and rejected completions. The `predict` API returns mean and standard deviation across the ensemble, enabling uncertainty-aware scoring. The primary output is one or more trained reward models used by downstream policy-optimization stages and by inference-time reranking when enabled.

### Direct Preference Optimization (DPO)

**Pipeline placement.** DPO is a policy-optimization stage that follows SFT and optional reward modeling. It is positioned after a reference policy exists, because its objective explicitly compares the current policy against a frozen reference model. DPO does not require a reward model during training; it consumes paired preference data directly.

**Inputs and data contract.** The trainer expects paired preference batches containing `prompt`, `chosen`, and `rejected` sequences with tokenized `input_ids` and `attention_mask`. The dataset provides `prompt_length` so the trainer can isolate response tokens.

**Training signal and objective.** `DPOTrainer` computes per-sample log-probabilities for response tokens under both the current policy and a frozen reference. The loss is based on the difference between policy and reference log-prob ratios for chosen versus rejected responses. Three loss types are implemented: standard sigmoid DPO, hinge-loss DPO, and IPO-style squared loss. Optional label smoothing blends the positive and negative log-sigmoid terms.

**Implementation behavior.** The trainer uses the same optimizer creation and logging stack as other stages, and can checkpoint the policy at configured intervals. The reference model is moved to the same device and is optionally frozen at initialization. The response-only log-prob computation masks out prompt tokens to ensure the loss is driven exclusively by response quality.

**Outputs and downstream use.** The result is an updated policy that encodes preference ordering while remaining anchored to the reference distribution. The output policy can be evaluated directly or used as a new reference for subsequent iterations.

### Group Relative Policy Optimization (GRPO)

**Pipeline placement.** GRPO is a policy-optimization stage used when rewards are verifiable or reliably scored and the system wants to avoid training a value model. It operates after SFT and uses a frozen reference model for KL anchoring, but it does not require a reward model if a rule-based reward is available.

**Inputs and data contract.** The trainer consumes prompts only. `GRPODataset` tokenizes prompts and yields prompt text along with input IDs and attention masks. For each prompt, the trainer generates a group of completions (`group_size`) and scores each completion with the provided `reward_fn(prompt, completion)`.

**Training signal and objective.** GRPO computes group-relative advantages by normalizing each completion reward against the mean and standard deviation of its group. The policy update uses a PPO-style clipped objective on per-token log-prob ratios, plus an explicit KL penalty computed via the low-variance estimator `exp(ref - policy) - (ref - policy) - 1`. The objective is applied only to the completion portion of each sequence.

**Implementation behavior.** The trainer expands prompts to generate grouped completions, decodes completions for scoring, and supports batched reward computation when the reward function is backed by a model. It stores policy log-probabilities as the “old” log-probs for ratio computation, performs optional multiple policy updates per batch, and logs both loss and mean reward. The reference model is always frozen and kept in evaluation mode.

**Outputs and downstream use.** GRPO returns a policy updated with group-relative advantages, along with per-epoch loss and reward metrics. The output policy can be evaluated directly or fed into later stages of the pipeline.

### Simple Preference Optimization (SimPO)

**Pipeline placement.** SimPO is a reference-free alternative to DPO that fits in the same policy-optimization slot. It is used after SFT when avoiding a reference model is desirable due to memory or simplicity, while still relying on paired preference data.

**Inputs and data contract.** SimPO consumes paired preference batches with the same structure as DPO. The trainer uses `prompt_length` to isolate response tokens and computes implicit rewards from policy log-probabilities.

**Training signal and objective.** The SimPO reward is the length-normalized log-probability of the response under the current policy. The loss optimizes a margin between chosen and rejected rewards, scaled by `beta` and offset by a target margin `gamma`. Optional label smoothing blends positive and negative log-sigmoid terms.

**Implementation behavior.** The trainer computes per-token log-probabilities, masks prompt tokens, normalizes by response length, and applies the SimPO objective without a reference model. It follows the standard optimization and logging flow shared across methods.

**Outputs and downstream use.** The result is an updated policy optimized for preference separation without a reference model. It can be evaluated directly or used as the base for subsequent stages.

### Kahneman-Tversky Optimization (KTO)

**Pipeline placement.** KTO is used when preference data is not paired. It occupies the same policy-optimization slot as DPO and SimPO but relies on binary labels (`desirable` vs `undesirable`) rather than chosen/rejected pairs. A frozen reference model is used to anchor the policy via KL terms.

**Inputs and data contract.** `KTODataset` consumes records with `prompt`, `response`, and `label` (1 for desirable, 0 for undesirable), plus tokenized inputs and `prompt_length`. The trainer uses this label to apply asymmetric loss terms.

**Training signal and objective.** The trainer computes policy and reference log-probabilities over response tokens, forming a KL-like term. A running EMA of the reference KL is maintained with a warmup buffer to stabilize early training. The loss applies different coefficients for desirable and undesirable examples (`lambda_u` and `lambda_d`) consistent with loss-aversion framing, producing an asymmetric preference signal.

**Implementation behavior.** The reference model is frozen and kept in evaluation mode. The EMA warmup collects a fixed number of batches before activating decay-based updates. The loss is computed per batch and optimized with the standard mixed-precision, optimizer, and logging infrastructure shared across trainers.

**Outputs and downstream use.** The result is a policy that separates desirable from undesirable responses without needing paired comparisons, anchored to the reference distribution through the KL term.

### Proximal Policy Optimization (PPO)

**Pipeline placement.** PPO is the full reinforcement learning stage used when explicit rewards and value estimation are required. It follows SFT and typically follows reward modeling, because it needs a reward function (model-based or rule-based) and a value model to estimate advantages.

**Inputs and data contract.** The trainer consumes prompt-only batches (prompts, tokenized IDs, and attention masks). A reward function `reward_fn(prompt, response)` provides scalar rewards that are applied to the final response token. The trainer uses a separate `ValueModel` to predict per-token values for GAE.

**Training signal and objective.** PPO collects rollouts by sampling responses from the policy, computes per-token rewards (with an explicit KL penalty against a frozen reference model), and estimates advantages using vectorized GAE. Policy updates use the clipped surrogate objective with entropy regularization, while value updates minimize clipped value loss. Advantages are normalized over response tokens for stability.

**Implementation behavior.** The trainer maintains an experience buffer, computes log-probabilities and values for each rollout, and applies reward whitening when enabled. It runs multiple PPO epochs per rollout, uses separate optimizers for policy and value, and adapts the KL penalty coefficient to track a target KL when configured. Checkpoints and logging follow the shared training infrastructure.

**Outputs and downstream use.** PPO produces a policy optimized under explicit reward signals with tracked KL behavior and value-function alignment. The resulting policy can be evaluated with the RLHF evaluator or used as a base for subsequent iterations.

## Cross-Cutting Capabilities

Several capabilities are shared across methods and can be enabled regardless of which policy-optimization technique is selected.

**Model components.** `PolicyModel` wraps `AutoModelForCausalLM` and provides standardized `generate` and `get_log_probs` utilities for loss computation and KL penalties. `ValueModel` provides per-token or final value predictions for PPO, with optional shared backbone support. `ProcessRewardModel` augments the reward layer with step-wise scoring and optional boundary detection for process supervision. `ContextCompressor` provides a learned attention-based compression module for long-context workflows and includes save/load utilities compatible with the rest of the pipeline.

**Evaluation.** `RLHFEvaluator` computes KL divergence against a reference policy, reward accuracy, diversity statistics, and win-rate estimates between policies. The evaluator is designed for periodic gating and regression detection in long runs.

**Self-improvement and validation.** The pipeline includes an adversarial validator and a capability tester used for automated quality checks. `AdversarialValidator` scores outputs with a flaw detector, coherence scorer, and multi-head quality metrics. `CapabilityTester` runs task-specific regression checks and reports deltas relative to a baseline, enabling rollback in the orchestrator when degradations exceed a configured threshold. `IterativeRefiner` implements a generate-critique-refine loop to synthesize improvement data directly from the policy.

## Orchestration and Pipeline Flow

`RLHFOrchestrator` is the control surface for end-to-end runs. It initializes the tokenizer, manages model lifecycles, and exposes `run_sft`, `run_reward_model_training`, `run_process_reward_model_training`, and `run_policy_optimization` as stage-level entry points. Each stage initializes the appropriate trainer, builds the matching dataset, and records results into a training history dictionary keyed by stage.

The orchestrator also maintains a frozen reference model when required, manages optional context compression utilities, and provides `compress_prompts` and `compress_context_from_ids` as shared helpers. When self-improvement is enabled, the orchestrator wires in validators and capability testers to gate progression and register rollbacks on regressions. This orchestration layer keeps the pipeline cohesive without duplicating method-specific logic.

## Inference Optimizations (Runtime)

`inference_optimizations.py` provides runtime components that improve quality or latency without retraining the policy. These components are designed to wrap existing policy and reward/value models.

**Optimized attention and KV cache.** `OptimizedAttention` switches between Flash Attention 2 (when available on CUDA) and PyTorch SDPA fallback, preserving correctness while reducing memory in supported environments. `PagedKVCache` implements page-based KV storage with explicit allocation and sequence tracking to reduce fragmentation and enable efficient batching during long generation.

**Speculative decoding.** `SpeculativeDecoder` accelerates generation by using a small draft model to propose `gamma` tokens and a large target model to verify. Accepted draft tokens are appended directly; rejected tokens are resampled from the target distribution. The interface is designed for drop-in use with existing causal LM outputs.

**Best-of-N reranking.** `BestOfNSampler` generates multiple candidates from the policy, scores them with a reward model, and selects the highest scoring candidate. An optional diversity bonus is computed via token-level edit distance to discourage near-duplicates. `BestOfNConfig` exposes sampling temperature, top-p, and aggregation controls.

**MCTS for reasoning.** `MCTSGenerator` performs Monte Carlo Tree Search over textual actions. It expands nodes using policy priors, evaluates nodes via a value model or terminal reward function, and backpropagates values to drive UCB-based selection. `MCTSConfig` controls simulations, exploration constants, temperature, depth, and branching width.

## Model Merging and Ensembling

`model_merging.py` implements weight-level merging and ensembling utilities for combining multiple fine-tuned models into a single policy or for ensemble generation at inference time.

**MergeConfig and ModelMerger.** `MergeConfig` selects a merge method (`task_arithmetic`, `ties`, `slerp`, `dare`) and controls weight normalization and sparsity density. `ModelMerger` computes parameter deltas between a base model and fine-tuned variants, merges deltas according to the selected method, and applies the merged delta to the base state dict. TIES merging performs trim-elect-sign-merge over parameter deltas, SLERP interpolates between two models on a spherical path, and DARE drops and rescales a sparse subset of deltas.

**Model soups.** `ModelSoup` supports uniform or weighted averaging across models, and a greedy soup mode that adds models only when they improve an evaluation function. This provides a practical path to merge multiple specializations without explicit task routing.

**EnsemblePolicy.** `EnsemblePolicy` supports average-logit decoding or simple voting across multiple models. The interface mirrors standard generation so it can be wrapped around existing policies with minimal changes.

**Layer-wise interpolation.** `layer_wise_interpolation` enables per-layer weighting between two models, which is useful when preserving lower-layer behavior while tuning higher layers for task specialization.

## Appendices

Appendix A enumerates the configuration classes (`SFTConfig`, `RewardModelConfig`, `DPOConfig`, `GRPOConfig`, `SimPOConfig`, `KTOConfig`, `PPOConfig`) and their stage-specific hyperparameters. Appendix B lists the dataset schemas for SFT, preference pairs, unpaired labels, and prompt-only rollouts. Appendix C inventories the primary classes in `rlhf.py` and their roles in the pipeline.
