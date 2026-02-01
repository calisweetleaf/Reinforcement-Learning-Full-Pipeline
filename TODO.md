# Release TODO / Status (2026-01-31)

## Scope and Status

- This release ships the base, fully-implemented RLHF pipeline as described in `rlhf.py` plus runtime modules `inference_optimizations.py` and `model_merging.py`.
- The pipeline has not been run end-to-end in this release window. Individual components compile and are implemented, but full E2E validation was not executed.

## Confirmed Present (Implementation + Docs)

- Context compression: `ContextCompressor` in `rlhf.py` and `docs/CONTEXT_COMPRESSION.md`.
- Self-play: `IterativeRefiner` in `rlhf.py` and `docs/SELF_PLAY.md`.
- Missing pieces added: `docs/MISSING_PIECES_ADDED.md` describes the SOTA++ execution-layer additions.
- Inference optimizations: `inference_optimizations.py` (Best-of-N, MCTS, speculative decoding, FA2/KV cache).
- Model merging: `model_merging.py` (TIES/SLERP/DARE, soups, ensemble policy).

## Known Non-Validated Items

- Full end-to-end run of SFT -> RM -> PO (DPO/GRPO/SimPO/KTO/PPO) not executed for this release.
- No automated test suite executed (no `tests/` in this repo).

## Recent Fixes (2026-02-01)

- PPO rollout reward computation now batches reward model inference when available and keeps sequential fallback for custom reward functions.
- Reward specificity regex patterns are compiled once per ConstitutionalRewardWrapper instance.
- Inference optimizations: draft generation uses cache-aware sampling, diversity scoring uses faster pairwise computation with optional RapidFuzz, MCTS rollout uses incremental decoding with KV cache fallback.

## Dataset Stack (Recommended Default)

- Stage 1: SFT -> `Magpie-Align/Magpie-Pro-300K-Filtered`
- Stage 2: RM -> `nvidia/HelpSteer2` (requires pairing or RM adaptation)
- Stage 3: DPO -> `argilla/distilabel-intel-orca-dpo-pairs`
- Stage 3: GRPO -> `AI-MO/NuminaMath-CoT` (prompts only)
- Stage 3: SimPO -> `princeton-nlp/SimPO-UltraFeedback` (confirm HF ID)
- Stage 3: KTO -> `trl-lib/kto-mix-14k`
- Stage 3: PPO -> `openbmb/UltraFeedback` (prompts only)

## Release Note

- This is the baseline release. Advanced ideas and next-wave optimizations are documented in `docs/ADVANCED_OPTIMIZATIONS.md` and will be expanded in a future iteration.
