## Reward Model Training (RM)

Reward modeling follows SFT because it requires a competent policy distribution and preference data over that distribution. The reward model is trained before any policy-optimization method that consumes reward signals or preference margins. This ordering ensures that preference learning is grounded in a stable scoring function and that later methods can reuse a consistent reward interface.

`RewardModel` wraps a base transformer encoder (`AutoModel`) and adds a two-layer MLP reward head with GELU activation and dropout. The head maps pooled hidden states to a scalar reward. Pooling defaults to the last non-padding token, with optional mean pooling over masked tokens. The reward head weights are explicitly initialized for stability, and the class provides `save_pretrained` and `from_pretrained` utilities with explicit metadata tracking.

The stage consumes paired preference data (`prompt`, `chosen`, `rejected`). The `PreferenceDataset` tokenizes prompt+chosen and prompt+rejected, producing paired input IDs and attention masks, along with the prompt length for diagnostics.

`RewardModelTrainer` implements Bradley-Terry style pairwise ranking. For each pair it computes `logits = reward(chosen) - reward(rejected)`, applies an optional margin, and optimizes binary cross-entropy against a positive target with label smoothing. Optional L2 regularization is applied to the reward model parameters. An ensemble of reward models is supported: each model is trained independently with its own optimizer and schedule, and evaluation aggregates across the ensemble.

Evaluation computes preference accuracy by comparing mean ensemble rewards for chosen and rejected completions. The `predict` API returns mean and standard deviation across the ensemble, enabling uncertainty-aware scoring. The primary output is one or more trained reward models used by downstream policy-optimization stages and by inference-time reranking when enabled.
