## Group Relative Policy Optimization (GRPO)

GRPO is a policy-optimization stage used when rewards are verifiable or reliably scored and the system wants to avoid training a value model. It operates after SFT and uses a frozen reference model for KL anchoring, but it does not require a reward model if a rule-based reward is available.

The trainer consumes prompts only. `GRPODataset` tokenizes prompts and yields prompt text along with input IDs and attention masks. For each prompt, the trainer generates a group of completions (`group_size`) and scores each completion with the provided `reward_fn(prompt, completion)`.

GRPO computes group-relative advantages by normalizing each completion reward against the mean and standard deviation of its group. The policy update uses a PPO-style clipped objective on per-token log-prob ratios, plus an explicit KL penalty computed via the low-variance estimator `exp(ref - policy) - (ref - policy) - 1`. The objective is applied only to the completion portion of each sequence.

The trainer expands prompts to generate grouped completions, decodes completions for scoring, and supports batched reward computation when the reward function is backed by a model. It stores policy log-probabilities as the "old" log-probs for ratio computation, performs optional multiple policy updates per batch, and logs both loss and mean reward. The reference model is always frozen and kept in evaluation mode.

The output is a policy updated with group-relative advantages, along with per-epoch loss and reward metrics.
