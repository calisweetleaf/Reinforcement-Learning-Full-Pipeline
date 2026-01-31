## Supervised Fine-Tuning (SFT)

SFT is the entry point for training. It establishes a strong instruction-following policy before any preference learning is applied. In the pipeline this occurs immediately after initialization and before reward modeling or preference optimization, because downstream methods assume a competent base policy and rely on its logits and generations as stable inputs.

The SFT stage consumes records with `prompt` and `response`. The `SFTDataset` concatenates prompt and response, tokenizes the full sequence to a fixed maximum length, and constructs labels with a masked prompt region. Prompt masking is implemented by setting label positions corresponding to prompt tokens to `-100`, which disables loss contribution at those positions while keeping the full input context for the modelâ€™s forward pass. Padding positions are also masked to `-100`, preserving stable loss scaling under variable-length inputs.

The trainer delegates the loss to the underlying causal language model by passing `labels` into the `PolicyModel` forward call. The effective objective is standard cross-entropy over the response tokens (or over the full sequence if prompt masking is disabled), with gradient accumulation applied to reach the configured effective batch size.

`SFTTrainer` uses `DeviceManager` for device placement and mixed-precision autocasting, `create_optimizer` for AdamW and optional cosine scheduling with warmup, and `CheckpointManager` for rolling checkpoints. Logging occurs at configured step intervals, evaluation can run periodically on a held-out dataloader, and training can resume from the latest checkpoint. Gradient clipping and AMP scaling are applied consistently across steps, and the loop is structured to ensure deterministic update boundaries when gradient accumulation is enabled.

The output is an updated `PolicyModel` and training loss history. This model becomes the base policy for subsequent stages, and in DPO-like workflows it is also the source for the frozen reference model copy used to anchor preference learning.
