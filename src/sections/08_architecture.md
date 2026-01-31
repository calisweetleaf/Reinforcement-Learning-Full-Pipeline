## Architecture and Component Map

The architecture is organized into configurations, datasets, models, training infrastructure, trainers, and orchestration. The control plane is intentionally thin: most behavior is encoded in the trainer classes, with shared utilities enforcing consistent device, precision, logging, and checkpoint semantics.

All stage configs inherit from `BaseConfig`, which defines optimizer parameters, scheduling, gradient accumulation, mixed precision flags, logging and checkpointing cadence, and optional experiment tracking. Each method-specific config adds only its core hyperparameters and validation rules. This keeps training behavior consistent across stages and simplifies orchestration.

`DeviceManager` resolves device placement, selects BF16 or FP16 when available, and wraps forward passes in an AMP autocast context. It centralizes gradient scaling and clipping, guaranteeing that every trainer applies the same numerical stability rules.

`create_optimizer` constructs AdamW with epsilon and weight decay, and optionally attaches a cosine warmup schedule when training steps are known. The same optimizer factory is used by all trainers.

`TrainingLogger` multiplexes metrics to console, WandB, and TensorBoard based on availability and config flags. It maintains a local history and standardizes metric naming by stage. `CheckpointManager` implements rolling checkpoints by stage, saving model and optimizer state with metadata and pruning older checkpoints beyond a configurable limit.

`apply_lora` integrates PEFT adapters when enabled and logs the proportion of trainable parameters. This supports large-model training within constrained hardware while keeping the core training loops unchanged.
