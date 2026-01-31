# Full-RLHF-Pipeline: End-to-End Specification (Draft Outline)

## Abstract

This document specifies the end-to-end RLHF pipeline implemented in `rlhf.py`, with runtime extensions in `inference_optimizations.py` and `model_merging.py`. The focus is operational: data contracts, model roles, training stages, and the concrete behaviors encoded in the codebase. The intent is to describe what exists and how it executes, not to propose new features.

The pipeline is designed as a modular training system that allows supervised fine-tuning, reward modeling, and multiple preference-optimization methods to be composed into a single production-grade training flow. Each method occupies a precise position in the pipeline and has a distinct contract for inputs, outputs, and model state transitions.

## Scope and Sources of Truth

The specification is grounded in the implementation. The primary source is `rlhf.py` (all stages, trainers, and orchestration). Inference-time compute optimizations are sourced from `inference_optimizations.py`. Model combination and merging procedures are sourced from `model_merging.py`. Existing documentation in `docs/` is used to strengthen explanatory framing where it matches the code.

## System Overview

The pipeline runs as a staged workflow: (1) supervised fine-tuning to establish a strong policy baseline, (2) reward model training to learn preference signals, (3) a policy-optimization stage (DPO, GRPO, SimPO, KTO, or PPO), followed by evaluation and optional self-improvement loops. Supporting infrastructure (device management, logging, checkpointing) is shared across all stages to ensure consistent training behavior, reproducibility, and operational stability.

## Architecture and Component Map

This section will describe how configurations, datasets, models, and trainers are composed, including the orchestration surface exposed by `RLHFOrchestrator`, and the supporting services (checkpointing, logging, early stopping, optimizer creation, and LoRA integration).

## Data Contracts and Datasets

This section will enumerate the canonical data formats used by each stage (SFT, preference pairs, unpaired labels, prompt-only rollouts) and the streaming dataset variants that support large-scale data. It will also include the recommended dataset stack provided by the user as an operational example, separate from the core code contracts.

## Method Specifications (Section by Method)

Each method section will include: pipeline placement, objective and training signal, inputs/outputs, and practical behavior as implemented in the trainer and configuration classes.

1. Supervised Fine-Tuning (SFT)
2. Reward Model Training (RM)
3. Direct Preference Optimization (DPO)
4. Group Relative Policy Optimization (GRPO)
5. Simple Preference Optimization (SimPO)
6. Kahneman-Tversky Optimization (KTO)
7. Proximal Policy Optimization (PPO)

## Cross-Cutting Capabilities

This section will cover process reward modeling, context compression, streaming datasets, self-improvement and validation utilities, and evaluation metrics, strictly as implemented in code.

## Inference Optimizations (Runtime)

This section will document test-time compute scaling and inference speedups implemented in `inference_optimizations.py`, including reranking, tree search, speculative decoding, and attention/KV-cache optimizations.

## Model Merging and Ensembling

This section will document the model-merging methods and ensembling utilities implemented in `model_merging.py`, including TIES-merging and model soups, and how they integrate with the policy model state.

## Appendices

This section will include configuration parameter tables, data schemas, and key class inventories.
