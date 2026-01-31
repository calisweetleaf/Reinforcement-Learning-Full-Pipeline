## Proximal Policy Optimization (PPO)

PPO is the full reinforcement learning stage used when explicit rewards and value estimation are required. It follows SFT and typically follows reward modeling, because it needs a reward function (model-based or rule-based) and a value model to estimate advantages.

The trainer consumes prompt-only batches (prompts, tokenized IDs, and attention masks). A reward function `reward_fn(prompt, response)` provides scalar rewards that are applied to the final response token. The trainer uses a separate `ValueModel` to predict per-token values for GAE.

PPO collects rollouts by sampling responses from the policy, computes per-token rewards (with an explicit KL penalty against a frozen reference model), and estimates advantages using vectorized GAE. Policy updates use the clipped surrogate objective with entropy regularization, while value updates minimize clipped value loss. Advantages are normalized over response tokens for stability.

The trainer maintains an experience buffer, computes log-probabilities and values for each rollout, and applies reward whitening when enabled. It runs multiple PPO epochs per rollout, uses separate optimizers for policy and value, and adapts the KL penalty coefficient to track a target KL when configured. Checkpoints and logging follow the shared training infrastructure.

The output is a policy optimized under explicit reward signals with tracked KL behavior and value-function alignment. The resulting policy can be evaluated with the RLHF evaluator or used as a base for subsequent iterations.
