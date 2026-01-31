## Cross-Cutting Capabilities

Several capabilities are shared across methods and can be enabled regardless of which policy-optimization technique is selected.

`PolicyModel` wraps `AutoModelForCausalLM` and provides standardized `generate` and `get_log_probs` utilities for loss computation and KL penalties. `ValueModel` provides per-token or final value predictions for PPO, with optional shared backbone support. `ProcessRewardModel` augments the reward layer with step-wise scoring and optional boundary detection for process supervision. `ContextCompressor` provides a learned attention-based compression module for long-context workflows and includes save/load utilities compatible with the rest of the pipeline.

`RLHFEvaluator` computes KL divergence against a reference policy, reward accuracy, diversity statistics, and win-rate estimates between policies. The evaluator is designed for periodic gating and regression detection in long runs.

The pipeline includes an adversarial validator and a capability tester used for automated quality checks. `AdversarialValidator` scores outputs with a flaw detector, coherence scorer, and multi-head quality metrics. `CapabilityTester` runs task-specific regression checks and reports deltas relative to a baseline, enabling rollback in the orchestrator when degradations exceed a configured threshold. `IterativeRefiner` implements a generate-critique-refine loop to synthesize improvement data directly from the policy.
