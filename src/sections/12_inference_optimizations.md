## Inference Optimizations (Runtime)

`inference_optimizations.py` provides runtime components that improve quality or latency without retraining the policy. These components are designed to wrap existing policy and reward/value models.

`OptimizedAttention` switches between Flash Attention 2 (when available on CUDA) and PyTorch SDPA fallback, preserving correctness while reducing memory in supported environments. `PagedKVCache` implements page-based KV storage with explicit allocation and sequence tracking to reduce fragmentation and enable efficient batching during long generation.

`SpeculativeDecoder` accelerates generation by using a small draft model to propose `gamma` tokens and a large target model to verify. Accepted draft tokens are appended directly; rejected tokens are resampled from the target distribution. The interface is designed for drop-in use with existing causal LM outputs.

`BestOfNSampler` generates multiple candidates from the policy, scores them with a reward model, and selects the highest scoring candidate. An optional diversity bonus is computed via token-level edit distance to discourage near-duplicates. `BestOfNConfig` exposes sampling temperature, top-p, and aggregation controls.

`MCTSGenerator` performs Monte Carlo Tree Search over textual actions. It expands nodes using policy priors, evaluates nodes via a value model or terminal reward function, and backpropagates values to drive UCB-based selection. `MCTSConfig` controls simulations, exploration constants, temperature, depth, and branching width.
