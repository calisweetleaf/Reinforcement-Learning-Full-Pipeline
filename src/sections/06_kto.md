## Kahneman-Tversky Optimization (KTO)

KTO is used when preference data is not paired. It occupies the same policy-optimization slot as DPO and SimPO but relies on binary labels (desirable vs undesirable) rather than chosen/rejected pairs. A frozen reference model is used to anchor the policy via KL terms.

`KTODataset` consumes records with `prompt`, `response`, and `label` (1 for desirable, 0 for undesirable), plus tokenized inputs and `prompt_length`. The trainer uses this label to apply asymmetric loss terms.

The trainer computes policy and reference log-probabilities over response tokens, forming a KL-like term. A running EMA of the reference KL is maintained with a warmup buffer to stabilize early training. The loss applies different coefficients for desirable and undesirable examples (`lambda_u` and `lambda_d`) consistent with loss-aversion framing, producing an asymmetric preference signal.

The reference model is frozen and kept in evaluation mode. The EMA warmup collects a fixed number of batches before activating decay-based updates. The loss is computed per batch and optimized with the standard mixed-precision, optimizer, and logging infrastructure shared across trainers.

The result is a policy that separates desirable from undesirable responses without needing paired comparisons, anchored to the reference distribution through the KL term.
