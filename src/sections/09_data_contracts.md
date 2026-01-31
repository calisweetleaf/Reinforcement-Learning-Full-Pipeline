## Data Contracts and Datasets

The pipeline standardizes data at the dataset layer so every trainer receives consistent tensors. Four primary data contracts are implemented, each with an in-memory and a streaming variant.

Supervised fine-tuning expects `{prompt, response}` pairs. `SFTDataset` concatenates prompt and response, tokenizes to a fixed length, and produces `input_ids`, `attention_mask`, and `labels`. Prompt tokens are optionally masked in `labels` (set to `-100`) to focus loss on the response while preserving the full context in the forward pass.

Preference learning (DPO, SimPO, reward modeling) uses `{prompt, chosen, rejected}`. `PreferenceDataset` tokenizes prompt+chosen and prompt+rejected independently, and provides both tokenized variants along with `prompt_length` computed from a prompt-only pass. The prompt length is used to restrict losses to response tokens.

KTO consumes `{prompt, response, label}` where `label` is binary (1 desirable, 0 undesirable). `KTODataset` returns tokenized prompt+response with `prompt_length` and a float label tensor for asymmetric loss computation.

GRPO and PPO operate on prompts only and generate completions online. `GRPODataset` tokenizes prompts with a configurable maximum prompt length and returns both text and tensors so the trainer can decode completions for reward evaluation.

`StreamingPreferenceDataset`, `StreamingSFTDataset`, `StreamingKTODataset`, and `StreamingGRPODataset` accept file-backed datasets for large-scale training. They support JSONL or raw-text line formats, maintain a shuffle buffer for on-the-fly randomization, and yield the same tensor structures as their in-memory counterparts. This provides a drop-in path to scale without changing trainer logic.

### Operational Dataset Stack (User-Specified)

The following dataset stack reflects the current target configuration supplied by the user. It is presented as an operational mapping to the pipeline stages and trainer classes.

| Pipeline Stage | Trainer Class | Dataset (HF ID) | Rationale (condensed) |
| --- | --- | --- | --- |
| Stage 1: SFT | `SFTTrainer` | `Magpie-Align/Magpie-Pro-300K-Filtered` | Synthetic instruction data distilled from aligned LLMs; curated and filtered. |
| Stage 2: RM | `RewardModelTrainer` | `nvidia/HelpSteer2` | Multi-attribute scores suitable for reward modeling. |
| Stage 3: DPO | `DPOTrainer` | `argilla/distilabel-intel-orca-dpo-pairs` | High-margin preference pairs for stable DPO. |
| Stage 3: GRPO | `GRPOTrainer` | `AI-MO/NuminaMath-CoT` (prompts only) | Reasoning-heavy prompts with verifiable answers. |
| Stage 3: SimPO | `SimPOTrainer` | `princeton-nlp/SimPO-UltraFeedback` | Length-normalized preferences for SimPO. |
| Stage 3: KTO | `KTOTrainer` | `trl-lib/kto-mix-14k` | Binary-labeled data formatted for KTO. |
| Stage 3: PPO | `PPOTrainer` | `openbmb/UltraFeedback` (prompts only) | Diverse prompts for rollout generation. |
