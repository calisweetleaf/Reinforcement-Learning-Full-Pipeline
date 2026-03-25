# STATE

Date: 2026-03-12
Scope: Full-RLHF-Pipeline repo state synthesis from the actual file tree, state docs, contract docs, live artifact manifests, and MCP memory/journal context.
Intent: provide one current-state document that separates architectural intent from on-disk reality.

## 1. Source Corpus Used

Primary file-tree source:

- `current-dpov2-filetree.md`

Normative or near-normative docs:

- `AGENTS.md`
- `OPSEC.md`
- `DESIRED_TRAIN_DPO_CONTRACT.md`
- `workflows.md`
- `/home/daeron/.codex/workflows.md`

Dataset and corpus-lineage docs:

- `datasets/claude-gpt-dpo-and-analysis/DATASET_INDEX.md`
- `datasets/claude-gpt-dpo-and-analysis/Distill-The-Flow-README.md`

Historical and forensic docs:

- `CURRENT_DPO_STATE.md`
- `DPO_STATE_AUDIT.md`
- `DPO_PIPELINE_FORENSIC_REPORT.md`

Repo-local memory/state docs:

- `memory/MEMORY.md`
- `memory/CONTEXT.md`
- `checkpoints/output-tree.md`

Live artifact manifests read directly:

- `checkpoints/dpo_qwen3_1.7b/training_manifest.json`
- `checkpoints/dpo_contract_smoke/training_manifest.json`
- `checkpoints/dpo_contract_smoke/telemetry.json`

MCP context used:

- active session `72c01e35-682e-4c47-bebc-213f7035a217`
- memory key `full_rlhf_pipeline_dpo_v2_overnight_run_pathology_2026_03_11`
- relevant journal observation: current `train_dpo_v2.py` does not persist resumable DPO adapter state per step

## 1A. Authority Stack Update (2026-03-12)

For onboarding and external review, the repo now needs to be read as a layered authority stack rather than a flat set of docs.

Current recommended read order:

1. `README.md` for orientation
2. `AGENTS.md` for repo-local operating rules
3. `OPSEC.md` for operator mindset and integration doctrine
4. `workflows.md` for MCP/exoskeleton execution posture
5. `STATE.md` for current repo-wide truth
6. `CURRENT_DPO_STATE.md` for historical `train_dpo.py` forensics only
7. `datasets/claude-gpt-dpo-and-analysis/DATASET_INDEX.md` for in-repo DPO file lineage
8. `datasets/claude-gpt-dpo-and-analysis/Distill-The-Flow-README.md` for Moonshine / corpus-forensics lineage

These docs do not carry equal authority:

- `STATE.md`, current checkpoint manifests, and current artifact roots describe current reality
- `CURRENT_DPO_STATE.md` describes a historical March 9 `train_dpo.py` world
- `DATASET_INDEX.md` explains how in-repo DPO files were derived
- `Distill-The-Flow-README.md` explains the upstream Moonshine authority surface and streamer/DB contract

## 2. Global Repo Model

This repository is a modular RLHF research pipeline centered on an immutable merged SFT base, followed by adapter-only DPO and additional capability modules.

The stable architectural intent across the repo is:

- keep `checkpoints/merged_sft_qwen3_1.7b` as the immutable base
- train DPO as adapter-only work on top of that base
- keep merging as a separate later process
- treat context compression, self-play, inference artifacts, and telemetry as part of the desired pipeline output

The repo currently contains two overlapping DPO realities:

- an older `train_dpo.py` reality documented in `CURRENT_DPO_STATE.md` and the forensic report
- a newer `train_dpo_v2.py` reality documented in `AGENTS.md`, `memory/MEMORY.md`, `memory/CONTEXT.md`, and the current checkpoint manifests

## 3. Canonical Current Interpretation

As of March 11, 2026, the repo-local current implementation target is:

- `train_dpo_v2.py`

This is stated explicitly in `AGENTS.md`.

The current intended DPO posture is:

- base model: `checkpoints/merged_sft_qwen3_1.7b`
- output root: `checkpoints/dpo_qwen3_1.7b`
- adapter-only DPO
- modular artifact saving
- merge deferred

This is consistent with:

- `AGENTS.md`
- `DESIRED_TRAIN_DPO_CONTRACT.md`
- `memory/MEMORY.md`
- `memory/CONTEXT.md`

## 3A. Moonshine / Dataset Lineage Reality

The repo's DPO data should be understood as downstream of Project Moonshine rather than as an isolated local dataset.

Upstream authority:

- `datasets/claude-gpt-dpo-and-analysis/Distill-The-Flow-README.md`
- Moonshine Mash active DB and corpus-forensics lane described there
- `moonshine_streamer.py` as the extraction and packaging utility

In-repo training lineage authority:

- `datasets/claude-gpt-dpo-and-analysis/DATASET_INDEX.md`

Practical file lineage:

- Moonshine Mash DB
- provider candidate files
- `balanced_100.raw.*`
- `dpo_100.unified.*`
- `dpo_100.curated.*`
- optional subsets such as `dpo_50.curated.*`

Critical interpretation:

- `dpo_100.curated.json` is the provider-balanced in-repo DPO set
- `dpo_50.curated.json` is the deterministic head subset and therefore should not be treated as balanced

## 4. What The File Tree Proves

The full repo snapshot in `current-dpov2-filetree.md` proves several important things at once.

### 4.1 The immutable merged SFT base exists and is intact

`checkpoints/merged_sft_qwen3_1.7b/` contains:

- all 8 safetensor shards
- config/tokenizer assets
- README/model card assets

This remains the load substrate, not the training output.

### 4.2 There is a contract-shaped smoke run artifact root

`checkpoints/dpo_contract_smoke/` contains:

- `epoch_1/`
- `final/`
- `context_compressor/epoch_1/`
- `context_compressor/final/`
- `telemetry.json`
- `training_manifest.json`
- `online_inference_artifacts/epoch_1.json`
- `self_play_artifacts/epoch_1.json`

This directory demonstrates a successful small run that produced resumable adapter output and supporting artifacts.

Important limitation:

- this is a smoke-style proof root, not the main `dpo_qwen3_1.7b` production directory
- its self-play outputs are JSON artifacts, not the final "all-on" production artifact contract for the larger run

### 4.3 The main `dpo_qwen3_1.7b` directory is partial, not complete

`checkpoints/dpo_qwen3_1.7b/` contains:

- `reward_model/`
- `context_compressor/compressor_state.pt`
- `adversarial_validator/validator_state.pt`
- `ref_cache_eval5.pt`
- `ref_cache_train45.pt`
- multiple logs and PID files
- `training_manifest.json`
- empty `ref_cache/`
- empty `self_play/`

It does not contain:

- `epoch_1/`
- `final/`
- `mini_dpo_final/`
- `draft_model/`
- `iterative_refiner/`
- `telemetry.json`
- self-play JSONL outputs
- speculative decoder stats

This is the central artifact truth of the repo right now.

## 5. Current `train_dpo_v2.py` Code Reality

Current code inspection shows that `train_dpo_v2.py` now contains the following phases/features:

- per-epoch adapter checkpoints under `epoch_N/`
- final adapter save under `final/`
- optional reward-model phase
- inline `ContextCompressor`
- `AdversarialValidator` state save/resume
- self-play phase writing to `self_play/`
- mini-DPO phase writing `mini_dpo_final/`
- draft-model training phase writing `draft_model/`
- final artifact phase saving `iterative_refiner` and speculative decoder stats
- telemetry emission to `telemetry.json`
- training manifest updates throughout the run

This matters because some older docs are now partially stale.

Most important staleness:

- `DPO_STATE_AUDIT.md` described a March 10 version of `train_dpo_v2.py` that explicitly excluded RewardModel and EWC
- current `train_dpo_v2.py` now includes optional RewardModel again
- current `checkpoints/dpo_qwen3_1.7b/training_manifest.json` confirms RewardModel was actually run in the live v2 path

So the correct read is:

- `DPO_STATE_AUDIT.md` is historically valuable
- it is not the final March 11 truth for v2 behavior

## 6. Main Artifact Truth: `dpo_qwen3_1.7b`

Direct manifest and file inventory show:

- script: `train_dpo_v2.py`
- dataset: `datasets/claude-gpt-dpo-and-analysis/dpo_100.curated.json`
- base: `checkpoints/merged_sft_qwen3_1.7b`
- output: `checkpoints/dpo_qwen3_1.7b`
- reward model enabled: `True`
- draft model configured: `Qwen/Qwen3-0.6B`
- self-play not disabled
- context compressor not disabled

The manifest records only one completed phase artifact:

- `reward_model`

The directory also contains saved module states:

- `context_compressor/compressor_state.pt`
- `adversarial_validator/validator_state.pt`

The run therefore produced real reusable module-side artifacts, but not the core DPO adapter output.

## 7. Reusable Versus Missing

### 7.1 Reusable now

These are real artifacts worth keeping:

- `checkpoints/merged_sft_qwen3_1.7b/`
- `checkpoints/dpo_qwen3_1.7b/reward_model/model.pt`
- `checkpoints/dpo_qwen3_1.7b/reward_model/reward_model_meta.json`
- `checkpoints/dpo_qwen3_1.7b/context_compressor/compressor_state.pt`
- `checkpoints/dpo_qwen3_1.7b/adversarial_validator/validator_state.pt`
- `checkpoints/dpo_qwen3_1.7b/ref_cache_eval5.pt`
- `checkpoints/dpo_qwen3_1.7b/ref_cache_train45.pt`
- `checkpoints/dpo_qwen3_1.7b/training_manifest.json`
- `checkpoints/dpo_qwen3_1.7b/dpo_resume.log`
- `checkpoints/dpo_qwen3_1.7b/dpo_rolling.log`
- `checkpoints/dpo_qwen3_1.7b/dpo_v2_detached.log`
- `checkpoints/dpo_contract_smoke/` as a known-good small proof root

### 7.2 Missing from the main production root

These are the major absent artifacts in `checkpoints/dpo_qwen3_1.7b/`:

- DPO adapter checkpoint directory `epoch_1/`
- final adapter directory `final/`
- mini-DPO output `mini_dpo_final/`
- self-play pair files
- draft-model adapter output
- iterative refiner output
- speculative decoder stats
- final telemetry JSON

### 7.3 Practical interpretation

This repo is not in a "nothing exists" state.

It is in a:

- "advanced components and logs exist"
- "main DPO adapter output for the large run does not exist yet"

state.

## 8. Checkpointing Truth

The single most important operational truth is this:

`train_dpo_v2.py` streams step logs, but durable DPO adapter resume points are created only at epoch boundaries.

What that means in practice:

- the logs can make a run look alive and productive
- if the run stalls or dies before `epoch_1/` is written, the adapter-side DPO progress is not resumable
- module artifacts like reward model, compressor, and validator may still exist and be reusable

This partially satisfies the contract, but not as strongly as the repo discussion often implied.

Contract expectation:

- save checkpoints at intervals during training
- make crashes survivable

Current code reality:

- epoch-based checkpoints exist in code
- there is no per-step adapter durability
- long pathological first epochs remain a real failure mode

## 9. The Major Document Drift Map

### 9.1 `CURRENT_DPO_STATE.md`

Status:

- historically valuable
- not the current canonical DPO v2 truth

Why:

- it is centered on `train_dpo.py`
- it documents the older v1 integrated-memory crisis
- it references a much larger swap-tuned environment than the repo-local `AGENTS.md` currently describes

### 9.2 `DPO_STATE_AUDIT.md`

Status:

- valuable as a March 10 audit snapshot
- partially stale for March 11 v2 reality

Why:

- it describes a version of `train_dpo_v2.py` that excluded RewardModel
- current code and current manifest show RewardModel reintroduced and used in v2

### 9.3 `DPO_PIPELINE_FORENSIC_REPORT.md`

Status:

- still load-bearing for root-cause doctrine

Why:

- it remains the clearest explanation of the original v1 crash path
- its loader/memory and contract-drift analysis still matters conceptually
- it should not be confused with a literal description of the current v2 file

### 9.4 `AGENTS.md`

Status:

- closest repo-local current working truth

Why:

- it names `train_dpo_v2.py` as primary
- it includes the March 11 addendum about the live v2 run pathology
- it captures the MCP caveat that project-context tools may drift to Somnus-MCP unless paths are pinned

## 10. MCP/Context Tooling Caveat

One stable operational caveat is now part of repo truth:

Some MCP project-context and exoskeleton tooling can default to `/home/daeron/Somnus-MCP` rather than this repo.

Therefore, for Full-RLHF-Pipeline truth:

- use absolute-path reads
- use repo-local shell commands with explicit working directory
- treat path-pinned file inspection as authoritative over generic project-context summaries

This caveat was re-confirmed during this state pass.

## 11. Best Single-Sentence Read Of The Repo Right Now

The repository contains a valid architectural contract, a successful small smoke proof of that contract, and a partially successful large `train_dpo_v2.py` output root whose advanced components were saved but whose actual DPO adapter training did not yet reach the first durable checkpoint.

## 12. Operational Meaning

The repo should not be read as a total loss.

It should be read as:

- architecture preserved
- immutable SFT base preserved
- advanced module artifacts partially salvaged
- main DPO adapter still missing in the production output root
- historical docs need interpretation by date, not blind trust

## 13. Current Best State Table

| Area | Current Best Read |
|---|---|
| Canonical DPO target | `train_dpo_v2.py` |
| Immutable base | `checkpoints/merged_sft_qwen3_1.7b/` |
| Contract authority | `DESIRED_TRAIN_DPO_CONTRACT.md` plus repo-local `AGENTS.md` |
| Historical crash doctrine | `DPO_PIPELINE_FORENSIC_REPORT.md` |
| Main production output root | `checkpoints/dpo_qwen3_1.7b/` |
| Small proof root | `checkpoints/dpo_contract_smoke/` |
| Main output root status | partial advanced artifacts, no durable DPO adapter checkpoint yet |
| Reward model in v2 | present and used in current manifest/code |
| Per-step resumable DPO saves | not present |
| First real DPO resume boundary | `epoch_1/` |
| MCP project-context trust level | use with caution unless path-pinned |

## 14. Bottom Line

If someone reads this repo cold, the safest interpretation is:

1. The design intent is modular adapter-only DPO on top of an immutable merged SFT base.
2. `train_dpo_v2.py` is the current implementation target, not `train_dpo.py`.
3. The repo already contains one small contract-shaped success root (`dpo_contract_smoke`).
4. The main `dpo_qwen3_1.7b` root is not empty or useless, but it is incomplete.
5. The missing load-bearing artifact in the main root is the DPO adapter checkpoint/final adapter, not the entire pipeline state.


## 15. Addendum: 2026-03-11 Document-Omniscient Reconciliation

A second architecture pass was completed against:

- `docs/ARCHITECTURE.md`
- `docs/CONTEXT_COMPRESSION.md`
- `docs/METHODS.md`
- `docs/USAGE.md`
- `docs/SELF_PLAY.md`
- `docs/ADVANCED_OPTIMIZATIONS.md`
- `SPEC.md`
- `telemetry.py`
- `train_dpo_v2.py`
- `train_dpo_full_pipeline.py`
- `rlhf.py`
- `Operation-sota.md`
- the current checkpoint tree docs

### 15.1 Telemetry is real and active

`telemetry.py` is not just a companion utility sitting unused in the repo.

Current file-truth:

- `train_dpo_v2.py` imports `TelemetryRecorder`
- initializes it in Phase 0
- records DPO step latencies during training
- increments counters / steps during the run
- emits `telemetry.json` at each epoch boundary and again at final artifact save
- `train_dpo_full_pipeline.py` inherits that same telemetry behavior because it is a thin wrapper over `train_dpo_v2.run_with_args(...)`
- `train_dpo.py` also uses `TelemetryRecorder`, and its older path is actually richer in snapshot/event usage than the current v2 path

So the correct repo interpretation is:

- telemetry is operational in both DPO entrypoints
- v2 uses a narrower subset of the recorder than v1
- wrapper-based full-pipeline DPO retains telemetry automatically

### 15.2 Current DPO-plus runtime truth

The present `train_dpo_v2.py` path is not "just DPO" in the narrow sense.

Current defaults and wiring:

- Reward model phase exists and runs when `--reward-model` is enabled
- `train_dpo_full_pipeline.py` sets `reward_model=True` by default
- learned `ContextCompressor` training is active unless `--no-context-compressor` is passed
- self-play runs unless `--no-self-play` is passed
- mini-DPO on self-play pairs is present
- draft-model adapter training is present when a draft model is configured and not disabled
- final artifact save persists compressor, validator, refiner, speculative-decoder stats, telemetry, and manifest state when those stages succeed

### 15.3 The new wrapper is now part of repo reality

`train_dpo_full_pipeline.py` now exists as a thin orchestration layer over `train_dpo_v2.py`.

Its current purpose is not to replace the advanced DPO engine, but to pin the intended qwen3-pinion DPO posture:

- base: `checkpoints/merged_sft_qwen3_1.7b`
- output root: `checkpoints/dpo_qwen3_pinion_full_pipeline`
- HF fallback: `Somnus-Sovereign-Systems/qwen3-pinion`
- default dataset preset: balanced `dpo_100.curated.json`
- reward-model phase: enabled by default

This should now be considered part of the active DPO entrypoint surface for repo-local work.

### 15.4 Doc drift map after re-reading the broader corpus

The broader docs remain valuable, but they are not all equally implementation-grounded.

Current best reading order for truth is:

1. `train_dpo_v2.py` / `train_dpo_full_pipeline.py`
2. `telemetry.py`
3. `SPEC.md`
4. `rlhf.py`
5. checkpoint tree + manifests
6. the broader docs set for intent and design direction

Most important drift points:

- `docs/USAGE.md` and `Operation-sota.md` describe a broader full-stack pipeline posture than the exact current runnable DPO v2 path
- `docs/SELF_PLAY.md` describes self-play as a more unified subsystem than what `rlhf.py` currently exposes directly
- `docs/CONTEXT_COMPRESSION.md` describes compression as a more fully threaded long-context mode than the current `rlhf.py` trainer integration actually provides

### 15.5 `rlhf.py` truth that matters for DPO-plus interpretation

The underlying `rlhf.py` capability surface is still broad and load-bearing:

- full RewardModel and ProcessRewardModel implementations exist
- `ContextCompressor` is real, saveable, loadable, and exposed through orchestrator utilities
- self-improvement primitives are real: validator, capability testing, iterative refiner, rollback logic, reward wrappers
- `RLHFOrchestrator.run_full_pipeline()` does sequence SFT -> reward model -> optional process reward model -> policy optimization

But two boundaries matter:

- `rlhf.py` does not currently provide a single unified `run_self_play()` stage in the orchestrator
- context compression in `rlhf.py` is an initialized utility/persistence surface, not a universally threaded training-forward path across the trainer stack

That means the repo's advanced components are real, but some of the surrounding docs describe the intended topology more strongly than the exact currently wired runtime does.


## 16) 2026-03-12: qwen3-pinion DPO Bake (Safe Output-Only Merge)

Execution completed with strict non-overwrite contract.

### Inputs
- HF base snapshot: `checkpoints/hf_snapshots/qwen3-pinion`
- DPO adapter: `checkpoints/dpo_qwen3_pinion_full_pipeline/final`

### Output
- Merged model dir: `checkpoints/merged_qwen3_pinion_dpo_baked_20260312`
- Files emitted:
  - `model-00001-of-00005.safetensors`
  - `model-00002-of-00005.safetensors`
  - `model-00003-of-00005.safetensors`
  - `model-00004-of-00005.safetensors`
  - `model-00005-of-00005.safetensors`
  - `model.safetensors.index.json`
  - `config.json`, `generation_config.json`, tokenizer artifacts

### Boundary Conditions
- No GGUF/export performed.
- No in-place mutation of `checkpoints/merged_sft_qwen3_1.7b`.
- Sidecar .pt artifacts remain sidecars; only DPO LoRA weights were baked into base model weights in this merge step.

## 17) 2026-03-12: Adapter Composition Rule

The repo now has enough artifact truth to state a clean rule about adapter experimentation.

### 17.1 Canonical SFT adapter identity

The canonical SFT adapter that produced `checkpoints/merged_sft_qwen3_1.7b/` is:

- `checkpoints/checkpoints/full_pipeline/sft/`

This is the path documented in:

- `checkpoints/merged_sft_qwen3_1.7b/README.md`
- `merge_sft_lora.py`

There is another adapter-like artifact at:

- `checkpoints/checkpoints/qwen3_1.7b/final/`

but it should not be silently substituted for the canonical SFT adapter without a separate provenance check.

### 17.2 `model_merging.py` boundary

`model_merging.py` should be interpreted as a **full-model merge utility**, not a raw PEFT-adapter merge utility.

Direct code truth:

- it takes a concrete `base_model`
- it takes a list of concrete fine-tuned `nn.Module` models
- it computes deltas from `state_dict()` alignment against that base

Therefore:

- valid use: merge multiple full fine-tuned models against a common base
- invalid/unsafe use: point it directly at raw adapter dirs and treat them as standalone models

### 17.3 Experimental adapter-composition lane

If the operator wants a high-output test lane with many variants, the correct interpretation is:

- use PEFT-native LoRA adapter composition first
- then bake the resulting composed adapter into a **new** output root
- keep the canonical shipped SFT->DPO sequential path intact

This experimental lane is compatible with the current artifact reality because:

- the canonical SFT adapter targets the Qwen3 attention projections
- the qwen3-pinion DPO adapter targets the same modules
- both are LoRA adapters over the same model family

### 17.4 Mixed-rank constraint

Current adapter ranks:

- canonical SFT adapter: `r=8`
- qwen3-pinion DPO adapter: `r=16`

Operational meaning:

- equal-rank-only composition methods should not be treated as valid for this pair
- mixed-rank-safe composition methods are the appropriate experiment class

### 17.5 No-double-apply rule

This is the most important constraint for future experimentation:

- if SFT and DPO are composed into one adapter, that composed adapter must be baked onto raw `Qwen/Qwen3-1.7B`
- do not bake a composed SFT+DPO adapter onto `checkpoints/merged_sft_qwen3_1.7b/`

Why:

- `checkpoints/merged_sft_qwen3_1.7b/` already contains the SFT effect in full weights
- reapplying a composed SFT+DPO adapter there would double-apply the SFT delta

### 17.6 Best current interpretation

The repo now has two legitimate paths:

- **canonical path**: merged SFT full weights -> DPO LoRA on top -> output-only bake/export
- **experimental path**: canonical SFT adapter + current DPO adapter -> PEFT-native composition -> bake onto raw Qwen3-1.7B -> A/B test in isolated output roots

The canonical path remains the default truth for shipping. The adapter-composition path is an experiment lane for testing, not a replacement for the established sequential lineage.

<!-- OP_SOTA_STATE_SYNC_START -->
## OP-SOTA Auto Sync (2026-03-13T08:29:00+00:00)

- Run id: `opsota-20260313T082900Z`
- Posture: AGENTS-first lock reconciliation
- Canonical spine assertion: `rlhf.py` remains canonical across systems
- Adapter policy: adapter files are hook/extension lanes for new runs and extra models

### Spec Authority Snapshot
- Core Implementation: `rlhf.py` (7334 lines, main RLHF methods)
- Inference Optimizations: `inference_optimizations.py` (Flash Attn, MCTS, Best-of-N)
- Model Merging: `model_merging.py` (TIES-Merging, Model Soups)
- Primary orchestrator: `RLHFOrchestrator`

### Checkpoint Snapshot
- SFT_BASE: path=`checkpoints/merged_sft_qwen3_1.7b`, exists=yes, files=16, bytes=8138468125
- HF_SNAPSHOT: path=`checkpoints/hf_snapshots/qwen3-pinion`, exists=no, files=0, bytes=0
- DPO_RUN: path=`checkpoints/dpo_qwen3_pinion_full_pipeline`, exists=yes, files=38, bytes=15352084422
- DPO_ADAPTER_FINAL: path=`checkpoints/dpo_qwen3_pinion_full_pipeline/final`, exists=yes, files=6, bytes=37154653
- MERGED_DPO_FP32: path=`checkpoints/merged_qwen3_pinion_dpo_baked_fp32_20260312`, exists=yes, files=14, bytes=8138450132
- MERGED_DPO_FP16_RESHARD: path=`checkpoints/merged_qwen3_pinion_dpo_baked_20260312`, exists=yes, files=11, bytes=4074970236
<!-- OP_SOTA_STATE_SYNC_END -->
