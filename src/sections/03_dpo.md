## Direct Preference Optimization (DPO)

DPO is a policy-optimization stage that follows SFT and optional reward modeling. It is positioned after a reference policy exists, because its objective explicitly compares the current policy against a frozen reference model. DPO does not require a reward model during training; it consumes paired preference data directly.

The trainer expects paired preference batches containing `prompt`, `chosen`, and `rejected` sequences with tokenized `input_ids` and `attention_mask`. The dataset provides `prompt_length` so the trainer can isolate response tokens.

`DPOTrainer` computes per-sample log-probabilities for response tokens under both the current policy and a frozen reference. The loss is based on the difference between policy and reference log-prob ratios for chosen versus rejected responses. Three loss types are implemented: standard sigmoid DPO, hinge-loss DPO, and IPO-style squared loss. Optional label smoothing blends the positive and negative log-sigmoid terms.

The trainer uses the standard optimizer creation and logging stack, and can checkpoint the policy at configured intervals. The reference model is moved to the same device and is optionally frozen at initialization. The response-only log-prob computation masks out prompt tokens to ensure the loss is driven exclusively by response quality.

The output is an updated policy that encodes preference ordering while remaining anchored to the reference distribution. The output policy can be evaluated directly or used as a new reference for subsequent iterations.
