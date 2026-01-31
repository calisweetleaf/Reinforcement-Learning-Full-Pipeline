## Simple Preference Optimization (SimPO)

SimPO is a reference-free alternative to DPO that fits in the same policy-optimization slot. It is used after SFT when avoiding a reference model is desirable due to memory or simplicity, while still relying on paired preference data.

SimPO consumes paired preference batches with the same structure as DPO. The trainer uses `prompt_length` to isolate response tokens and computes implicit rewards from policy log-probabilities.

The SimPO reward is the length-normalized log-probability of the response under the current policy. The loss optimizes a margin between chosen and rejected rewards, scaled by `beta` and offset by a target margin `gamma`. Optional label smoothing blends positive and negative log-sigmoid terms.

The trainer computes per-token log-probabilities, masks prompt tokens, normalizes by response length, and applies the SimPO objective without a reference model. It follows the standard optimization and logging flow shared across methods.

The result is an updated policy optimized for preference separation without a reference model. It can be evaluated directly or used as the base for subsequent stages.
