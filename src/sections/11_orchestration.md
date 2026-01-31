## Orchestration and Pipeline Flow

`RLHFOrchestrator` is the control surface for end-to-end runs. It initializes the tokenizer, manages model lifecycles, and exposes `run_sft`, `run_reward_model_training`, `run_process_reward_model_training`, and `run_policy_optimization` as stage-level entry points. Each stage initializes the appropriate trainer, builds the matching dataset, and records results into a training history dictionary keyed by stage.

The orchestrator also maintains a frozen reference model when required, manages optional context compression utilities, and provides `compress_prompts` and `compress_context_from_ids` as shared helpers. When self-improvement is enabled, the orchestrator wires in validators and capability testers to gate progression and register rollbacks on regressions. This orchestration layer keeps the pipeline cohesive without duplicating method-specific logic.
