"""
RLHF Pipeline Orchestrator — YAML-Driven Entrypoint

This is the production entrypoint for the Full-RLHF-Pipeline.
It reads config/qwen3_1.7b_model.yaml (or any --config path), builds the
appropriate rlhf.py config objects, and calls RLHFOrchestrator methods.

Zero reimplementation. Every trainer, dataset, model, and evaluation
operation delegates to existing rlhf.py classes.

This entrypoint now wires the active orchestration/config surface across
`rlhf.py`, `inference_optimizations.py`, and `inference_protocols.py`.

Usage:
    # Dry run (parse config, print plan, no model loading):
    .venv/bin/python run_pipeline.py --config config/qwen3_1.7b_model.yaml --dry-run

    # Full execution with DPO:
    .venv/bin/python run_pipeline.py --config config/qwen3_1.7b_model.yaml

    # Override method from CLI:
    .venv/bin/python run_pipeline.py --config config/qwen3_1.7b_model.yaml --method grpo

    # Force SFT re-training:
    .venv/bin/python run_pipeline.py --config config/qwen3_1.7b_model.yaml --force-sft
"""

import argparse
import atexit
import gc
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import yaml
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# rlhf.py imports — the ONLY source of training/model/dataset logic
# ---------------------------------------------------------------------------
from rlhf import (
    # Orchestrator
    RLHFOrchestrator,
    # Infrastructure
    DeviceManager,
    CheckpointManager,
    TrainingLogger,
    # Config dataclasses
    BaseConfig,
    SFTConfig,
    RewardModelConfig,
    DPOConfig,
    GRPOConfig,
    TreeGRPOConfig,
    SimPOConfig,
    KTOConfig,
    PPOConfig,
    # Core model classes for artifact reuse
    RewardModel,
    ProcessRewardModel,
    ContextCompressor,
    # In-memory datasets
    PreferenceDataset,
    SFTDataset,
    KTODataset,
    GRPODataset,
    # Streaming datasets (RAM-efficient, for JSONL)
    StreamingPreferenceDataset,
    StreamingSFTDataset,
    StreamingKTODataset,
    StreamingGRPODataset,
    # Self-improvement components
    IterativeRefiner,
    AdversarialValidator,
    CapabilityTester,
    # Evaluation
    RLHFEvaluator,
    # Reward wiring
    RewardFunctionFactory,
    ConstitutionalRewardWrapper,
)

# ---------------------------------------------------------------------------
# Advanced Capability Formation & Optimizations
# ---------------------------------------------------------------------------
from inference_protocols import PolicyAdapter, ProcessRewardModelAdapter
from inference_optimizations import (
    # Already in use
    BestOfNSampler,
    MCTSGenerator,
    MCTSConfig,
    # Newly wired
    BestOfNConfig,
    MCTSNode,
    OptimizedAttention,
    PagedKVCache,
    SpeculativeDecoderConfig,
    SpeculativeDecoder,
    MDPConfig,
    LexicalMDP,
    VerifiableRewardFactory,
    ChainOfThoughtConfig,
    ChainOfThoughtGenerator,
    AStarConfig,
    AStarNode,
    AStarDecoder,
    RolloutSample,
    AStarGenerator,
    TreeRolloutCollector,
    compile_model,
)

from benchmark_harness import BenchmarkHarness, BudgetProfileSelector, RegressionGate
from model_merging import ModelMerger, MergeConfig, save_merge_artifact
import telemetry as _telemetry_module
from telemetry import TelemetryRecorder

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("run_pipeline")

# Valid method identifiers that map 1:1 to rlhf.py trainer dispatch
VALID_METHODS = frozenset({"dpo", "grpo", "tree_grpo", "hidden_cot_sft", "simpo", "kto", "ppo"})

# Methods that require a reward signal
METHODS_REQUIRING_REWARD = frozenset({"ppo", "grpo", "tree_grpo"})

# Methods that use preference pair data (chosen/rejected)
METHODS_PREFERENCE_DATA = frozenset({"dpo", "simpo"})

# Methods that use unpaired data
METHODS_UNPAIRED_DATA = frozenset({"kto"})

# Methods that use prompt-only data
METHODS_PROMPT_DATA = frozenset({"grpo", "tree_grpo", "ppo"})

# Methods that must use the canonical PRM path. No legacy RM fallback.
STRICT_PRM_METHODS = frozenset({"grpo", "tree_grpo"})

# Valid session selectors when YAML session splitting is enabled.
SESSION_CHOICES = frozenset({"1", "2", "all"})


def _is_truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _config_dir(config: Dict[str, Any]) -> Path:
    raw = config.get("__config_dir__")
    return Path(raw).resolve() if raw else Path.cwd().resolve()


def _resolve_path_or_id_from_config(config: Dict[str, Any], value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    candidate = Path(value).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return str(candidate.resolve())

    if not candidate.is_absolute():
        local_candidate = (_config_dir(config) / candidate).resolve()
        if local_candidate.exists():
            return str(local_candidate)

    return value


def _resolve_required_path_from_config(
    config: Dict[str, Any],
    value: str,
    *,
    label: str,
) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (_config_dir(config) / candidate).resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _resolve_output_path_from_config(config: Dict[str, Any], value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (_config_dir(config) / candidate).resolve()


def _active_method_section(config: Dict[str, Any], method: str) -> Dict[str, Any]:
    section = config.get(method, {})
    if not section and method == "tree_grpo":
        section = config.get("grpo", {})
    return section if isinstance(section, dict) else {}


def get_sessions_config(config: Dict[str, Any]) -> Dict[str, Any]:
    top_level = config.get("sessions")
    if isinstance(top_level, dict):
        return top_level

    pipeline = config.get("pipeline", {})
    nested = pipeline.get("sessions")
    if isinstance(nested, dict):
        return nested

    return {}


def get_benchmark_config(config: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("benchmark_harness", "benchmarking"):
        cfg = config.get(key)
        if isinstance(cfg, dict):
            return cfg
    return {}


_EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "\u2600-\u26FF"
    "\uFE0F\u200D"
    "]+",
    flags=re.UNICODE,
)


def _strip_emoji_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    cleaned = _EMOJI_RE.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _strip_emoji_payload(value: Any) -> Any:
    if isinstance(value, str):
        return _strip_emoji_text(value)
    if isinstance(value, list):
        return [_strip_emoji_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: _strip_emoji_payload(item) for key, item in value.items()}
    return value


def _sync_default_telemetry_recorder(recorder: TelemetryRecorder) -> None:
    _telemetry_module.default_recorder = recorder
    try:
        import rlhf as _rlhf_module
        _rlhf_module.default_recorder = recorder
    except Exception as exc:  # pragma: no cover - defensive sync only
        logger.warning("Could not sync telemetry recorder into rlhf.py: %s", exc)


def _init_runtime_telemetry(
    config: Dict[str, Any],
    output_base: Path,
    *,
    method: str,
    session: str,
) -> Dict[str, Any]:
    log_cfg = config.get("logging", {})
    tb_writer = None
    tb_dir: Optional[Path] = None

    if _is_truthy(log_cfg.get("use_tensorboard", False)):
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir_raw = log_cfg.get("tensorboard_dir")
            if tb_dir_raw:
                tb_dir = _resolve_output_path_from_config(config, tb_dir_raw)
            else:
                tb_dir = (output_base / "tensorboard").resolve()

            sessions_cfg = get_sessions_config(config)
            if (
                session in {"1", "2"}
                and _is_truthy(sessions_cfg.get("enabled", False))
                and str(sessions_cfg.get("split_mode", "single")).lower() != "single"
            ):
                tb_dir = tb_dir / f"session_{session}"

            tb_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(str(tb_dir))
        except Exception as exc:
            logger.warning("TensorBoard requested but SummaryWriter could not be created: %s", exc)

    recorder = TelemetryRecorder(tb_writer=tb_writer)
    _sync_default_telemetry_recorder(recorder)
    json_path = str((output_base / "telemetry_run.json").resolve())
    atexit.register(_finalize_runtime_telemetry, recorder, json_path)
    recorder.record_event(
        "runtime_activation",
        {
            "component": "telemetry",
            "method": method,
            "session": session,
            "output_dir": str(output_base),
            "tensorboard_dir": str(tb_dir) if tb_dir is not None else None,
            "json_path": json_path,
        },
    )
    recorder.record_memory_snapshot("telemetry_online")
    return {
        "recorder": recorder,
        "tensorboard_dir": str(tb_dir) if tb_dir is not None else None,
        "json_path": json_path,
    }


def _finalize_runtime_telemetry(recorder: Optional[TelemetryRecorder], json_path: str) -> None:
    if recorder is None or getattr(recorder, "_runtime_finalized", False):
        return
    try:
        recorder.emit_json(json_path)
        writer = getattr(recorder, "_tb_writer", None)
        if writer is not None:
            try:
                writer.flush()
                writer.close()
            except Exception:
                pass
    finally:
        setattr(recorder, "_runtime_finalized", True)


def _record_runtime_activation(
    recorder: Optional[TelemetryRecorder],
    component: str,
    **payload: Any,
) -> None:
    if recorder is None:
        return
    recorder.record_event("runtime_activation", {"component": component, **payload})


def _has_prm_source(config: Dict[str, Any]) -> bool:
    pipeline = config.get("pipeline", {})
    prm_cfg = pipeline.get("process_reward_model")
    if isinstance(prm_cfg, dict):
        if _is_truthy(prm_cfg.get("enabled", False)):
            return True
        if prm_cfg.get("load_from_checkpoint"):
            return True

    artifact_reuse = get_artifact_reuse_config(config)
    if _is_truthy(artifact_reuse.get("enabled", False)) and _is_truthy(
        artifact_reuse.get("load_process_reward_model", False)
    ):
        return True

    return False


def validate_runtime_contracts(
    config: Dict[str, Any],
    method: str,
    *,
    session: str = "all",
) -> None:
    if session not in SESSION_CHOICES:
        raise ValueError(
            f"Invalid session selector '{session}'. Valid: {sorted(SESSION_CHOICES)}"
        )

    pipeline = config["pipeline"]
    method_cfg = _active_method_section(config, method)

    if method in STRICT_PRM_METHODS:
        if "process_reward_model" not in pipeline:
            legacy_rm = pipeline.get("reward_model")
            if isinstance(legacy_rm, dict):
                raise ValueError(
                    f"{method.upper()} requires pipeline.process_reward_model. "
                    "pipeline.reward_model is legacy compatibility only and is refused "
                    "for this Chiron reward path."
                )
            raise ValueError(
                f"{method.upper()} requires pipeline.process_reward_model in the YAML."
            )

        if not _has_prm_source(config):
            raise ValueError(
                f"{method.upper()} requires a canonical PRM source. Enable pipeline.process_reward_model, "
                "set load_from_checkpoint, or enable artifact_reuse.load_process_reward_model."
            )

    if method in {"tree_grpo", "hidden_cot_sft"}:
        cot_cfg = config.get("inference_optimizations", {}).get("chain_of_thought", {})
        cot_required = method == "hidden_cot_sft" or _is_truthy(method_cfg.get("cot_mode", False))
        if cot_required and not _is_truthy(cot_cfg.get("enabled", False)):
            raise ValueError(
                f"{method.upper()} requires inference_optimizations.chain_of_thought.enabled=true for the active hidden-CoT lane. "
                "Refusing silent flat-generation fallback."
            )
        if cot_required and int(cot_cfg.get("max_thinking_tokens", 0) or 0) <= 0:
            raise ValueError(
                f"{method.upper()} requires chain_of_thought.max_thinking_tokens > 0."
            )
        if cot_required and int(cot_cfg.get("max_answer_tokens", 0) or 0) <= 0:
            raise ValueError(
                f"{method.upper()} requires chain_of_thought.max_answer_tokens > 0."
            )

    sessions_cfg = get_sessions_config(config)
    if session in {"1", "2"} and sessions_cfg and not _is_truthy(
        sessions_cfg.get("enabled", False)
    ):
        raise ValueError(
            f"--session {session} requested, but sessions.enabled is not true in the YAML."
        )


def get_prm_pipeline_config(pipeline: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve Stage-2 Process Reward Model controls from pipeline config.

    Priority:
    1) pipeline.process_reward_model (current canon)
    2) pipeline.reward_model (legacy compatibility fallback)
    """
    prm_cfg = pipeline.get("process_reward_model")
    if isinstance(prm_cfg, dict):
        return prm_cfg

    legacy_rm = pipeline.get("reward_model")
    if isinstance(legacy_rm, dict):
        return {
            "enabled": legacy_rm.get("enabled", False),
            "load_from_checkpoint": legacy_rm.get("load_from_checkpoint"),
            "step_detection": "newline",
            "process_reward_weight": 0.0,
        }

    return {
        "enabled": False,
        "load_from_checkpoint": None,
        "step_detection": "newline",
        "process_reward_weight": 0.0,
    }


def get_artifact_reuse_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve optional artifact-reuse controls.

    Supports both:
    - top-level: artifact_reuse: ...
    - nested:    pipeline.artifact_reuse: ...
    """
    top_level = config.get("artifact_reuse")
    if isinstance(top_level, dict):
        return top_level

    pipeline = config.get("pipeline", {})
    nested = pipeline.get("artifact_reuse")
    if isinstance(nested, dict):
        return nested

    return {}


def _load_state_dict_flexible(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Load a checkpoint that may be either a raw state_dict or a wrapped dict.
    """
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(
            f"Checkpoint payload at {checkpoint_path} is not a dict "
            f"(got {type(payload).__name__})"
        )

    for key in ("state_dict", "model_state_dict", "module_state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate

    return payload


def maybe_load_artifact_bundle(
    orchestrator: RLHFOrchestrator,
    config: Dict[str, Any],
    device_manager: DeviceManager,
) -> Dict[str, Any]:
    """
    Optionally reuse previously trained artifacts (RM/PRM/context/self-improvement)
    from an existing checkpoint directory.
    """
    reuse_cfg = get_artifact_reuse_config(config)
    summary: Dict[str, Any] = {
        "enabled": _is_truthy(reuse_cfg.get("enabled", False)),
        "checkpoint_dir": None,
        "loaded": [],
        "ref_cache": {},
        "skip_prm_training": False,
    }
    if not summary["enabled"]:
        return summary

    checkpoint_root_raw = reuse_cfg.get("checkpoint_dir") or reuse_cfg.get("from_checkpoint")
    if not checkpoint_root_raw:
        raise ValueError(
            "artifact_reuse.enabled=true but no artifact_reuse.checkpoint_dir provided."
        )

    checkpoint_root = _resolve_required_path_from_config(
        config,
        checkpoint_root_raw,
        label="artifact_reuse checkpoint_dir",
    )
    summary["checkpoint_dir"] = str(checkpoint_root)

    def _resolve_path(key: str, default_relative: str) -> Path:
        configured = reuse_cfg.get(key)
        if configured:
            return _resolve_output_path_from_config(config, configured)
        return (checkpoint_root / default_relative).resolve()

    # ------------------------------------------------------------------
    # Reward model (outcome RM)
    # ------------------------------------------------------------------
    if reuse_cfg.get("load_reward_model", True):
        reward_model_dir = _resolve_path("reward_model_dir", "reward_model")
        if reward_model_dir.exists():
            try:
                reward_model = RewardModel.from_pretrained(str(reward_model_dir))
                reward_model = device_manager.to_device(reward_model)
                reward_model.eval()
                orchestrator.reward_models = [reward_model]
                summary["loaded"].append("reward_model")
                logger.info(f"Artifact reuse: loaded reward model from {reward_model_dir}")
            except Exception as exc:
                logger.warning(
                    f"Artifact reuse: failed loading reward model from {reward_model_dir}: {exc}"
                )
        else:
            logger.info(
                f"Artifact reuse: reward_model dir not found at {reward_model_dir}; skipping."
            )

    # ------------------------------------------------------------------
    # Process reward model (PRM)
    # ------------------------------------------------------------------
    if reuse_cfg.get("load_process_reward_model", False):
        process_reward_model_dir = _resolve_path(
            "process_reward_model_dir", "process_reward_model"
        )
        if process_reward_model_dir.exists():
            try:
                prm = ProcessRewardModel.from_pretrained(str(process_reward_model_dir))
                prm = device_manager.to_device(prm)
                prm.eval()
                orchestrator.process_reward_model = prm
                summary["loaded"].append("process_reward_model")
                logger.info(
                    "Artifact reuse: loaded process reward model from "
                    f"{process_reward_model_dir}"
                )
            except Exception as exc:
                logger.warning(
                    "Artifact reuse: failed loading process reward model from "
                    f"{process_reward_model_dir}: {exc}"
                )
        else:
            logger.info(
                "Artifact reuse: process_reward_model dir not found at "
                f"{process_reward_model_dir}; skipping."
            )

    # ------------------------------------------------------------------
    # Context compressor
    # ------------------------------------------------------------------
    if reuse_cfg.get("load_context_compressor", True):
        cc_dir = _resolve_path("context_compressor_dir", "context_compressor")
        cc_model = cc_dir / "model.pt"
        cc_meta = cc_dir / "context_compressor_meta.json"
        cc_legacy_state = cc_dir / "compressor_state.pt"

        loaded_cc = None
        try:
            if cc_model.exists() and cc_meta.exists():
                loaded_cc = ContextCompressor.from_pretrained(str(cc_dir))
                logger.info(
                    f"Artifact reuse: loaded context compressor (pretrained format) from {cc_dir}"
                )
            elif cc_legacy_state.exists():
                if orchestrator.policy_model is None:
                    logger.warning(
                        "Artifact reuse: cannot load legacy compressor_state.pt before "
                        "policy model initialization."
                    )
                else:
                    cc_cfg = config.get("context_compression", {})
                    loaded_cc = ContextCompressor(
                        hidden_size=orchestrator.policy_model.config.hidden_size,
                        compression_ratio=int(
                            cc_cfg.get("compression_ratio", orchestrator.context_compression_ratio)
                        ),
                        num_compression_heads=int(
                            cc_cfg.get("num_heads", orchestrator.context_compression_heads)
                        ),
                        dropout=float(
                            cc_cfg.get("dropout", orchestrator.context_compression_dropout)
                        ),
                    )
                    cc_state = _load_state_dict_flexible(cc_legacy_state)
                    missing, unexpected = loaded_cc.load_state_dict(cc_state, strict=False)
                    if missing:
                        logger.warning(
                            "Artifact reuse: context compressor missing keys: "
                            f"{missing[:8]}{'...' if len(missing) > 8 else ''}"
                        )
                    if unexpected:
                        logger.warning(
                            "Artifact reuse: context compressor unexpected keys: "
                            f"{unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
                        )
                    logger.info(
                        "Artifact reuse: loaded context compressor (legacy compressor_state.pt) "
                        f"from {cc_legacy_state}"
                    )
            else:
                logger.info(
                    "Artifact reuse: no context compressor weights found in "
                    f"{cc_dir}; skipping."
                )
        except Exception as exc:
            logger.warning(f"Artifact reuse: context compressor load failed: {exc}")

        if loaded_cc is not None:
            orchestrator.context_compressor = device_manager.to_device(loaded_cc)
            orchestrator.use_context_compressor = True
            summary["loaded"].append("context_compressor")

    # ------------------------------------------------------------------
    # Adversarial validator
    # ------------------------------------------------------------------
    if reuse_cfg.get("load_validator", True):
        validator_path = _resolve_path(
            "validator_state_path", "adversarial_validator/validator_state.pt"
        )
        if validator_path.exists():
            if orchestrator.validator is None:
                logger.info(
                    "Artifact reuse: validator_state exists but self-improvement is disabled; "
                    "skipping validator load."
                )
            else:
                try:
                    validator_state = _load_state_dict_flexible(validator_path)
                    missing, unexpected = orchestrator.validator.load_state_dict(
                        validator_state,
                        strict=False,
                    )
                    if missing:
                        logger.warning(
                            "Artifact reuse: validator missing keys: "
                            f"{missing[:8]}{'...' if len(missing) > 8 else ''}"
                        )
                    if unexpected:
                        logger.warning(
                            "Artifact reuse: validator unexpected keys: "
                            f"{unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
                        )
                    orchestrator.validator = device_manager.to_device(orchestrator.validator)
                    orchestrator.validator.eval()
                    summary["loaded"].append("validator")
                    logger.info(f"Artifact reuse: loaded validator state from {validator_path}")
                except Exception as exc:
                    logger.warning(f"Artifact reuse: validator load failed: {exc}")

    # ------------------------------------------------------------------
    # Iterative refiner
    # ------------------------------------------------------------------
    if reuse_cfg.get("load_iterative_refiner", True):
        refiner_path = _resolve_path(
            "iterative_refiner_state_path", "iterative_refiner/refiner_state.pt"
        )
        if refiner_path.exists():
            if orchestrator.policy_model is None or orchestrator.validator is None:
                logger.info(
                    "Artifact reuse: refiner state exists but policy/validator are not ready; "
                    "skipping iterative refiner load."
                )
            else:
                try:
                    sp_cfg = config.get("self_play", {})
                    refiner = IterativeRefiner(
                        policy_model=orchestrator.policy_model,
                        validator=orchestrator.validator,
                        tokenizer=orchestrator.tokenizer,
                        max_iterations=int(sp_cfg.get("refinement_iterations", 3)),
                        quality_threshold=float(sp_cfg.get("quality_threshold", 0.85)),
                        device=device_manager.device,
                    )
                    refiner_state = _load_state_dict_flexible(refiner_path)
                    missing, unexpected = refiner.load_state_dict(refiner_state, strict=False)
                    if missing:
                        logger.warning(
                            "Artifact reuse: iterative refiner missing keys: "
                            f"{missing[:8]}{'...' if len(missing) > 8 else ''}"
                        )
                    if unexpected:
                        logger.warning(
                            "Artifact reuse: iterative refiner unexpected keys: "
                            f"{unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
                        )
                    refiner = device_manager.to_device(refiner)
                    refiner.eval()
                    # Dynamic attachment for downstream users that opt-in.
                    setattr(orchestrator, "iterative_refiner", refiner)
                    summary["loaded"].append("iterative_refiner")
                    logger.info(
                        f"Artifact reuse: loaded iterative refiner state from {refiner_path}"
                    )
                except Exception as exc:
                    logger.warning(f"Artifact reuse: iterative refiner load failed: {exc}")

    # ------------------------------------------------------------------
    # Reference cache paths (optional metadata pass-through)
    # ------------------------------------------------------------------
    if reuse_cfg.get("use_ref_cache", True):
        ref_cache_dir = _resolve_path("ref_cache_dir", "ref_cache")
        for split in ("train", "eval"):
            split_path = ref_cache_dir / f"ref_cache_{split}.pt"
            if split_path.exists():
                summary["ref_cache"][split] = str(split_path)
        if summary["ref_cache"]:
            logger.info(f"Artifact reuse: available ref_cache files: {summary['ref_cache']}")

    summary["skip_prm_training"] = (
        bool(reuse_cfg.get("skip_prm_training_if_loaded", True))
        and orchestrator.process_reward_model is not None
    )
    return summary


# ============================================================================
# CONFIG LOADING + VALIDATION
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate the YAML config file.

    Performs structural validation — checks that required sections exist
    and that method-specific sections are present for the active method.

    Args:
        config_path: Absolute or relative path to the YAML config file.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If config is structurally invalid.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Loading config from: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(config).__name__}")

    config["__config_path__"] = str(config_path)
    config["__config_dir__"] = str(config_path.parent.resolve())

    # --- Required top-level sections ---
    required_sections = ["model", "training", "pipeline"]
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")

    # --- Pipeline section validation ---
    pipeline = config["pipeline"]
    if "policy_optimization" not in pipeline:
        raise ValueError("Config pipeline section must include 'policy_optimization'")

    po = pipeline["policy_optimization"]
    if po.get("enabled", True):
        method = po.get("method", "").lower()
        if method not in VALID_METHODS:
            raise ValueError(
                f"Invalid policy_optimization.method: '{method}'. "
                f"Valid options: {sorted(VALID_METHODS)}"
            )
        # Check that the method-specific config section exists
        if method not in config and not (method == "tree_grpo" and "grpo" in config):
            raise ValueError(
                f"Method '{method}' is enabled but config has no '{method}' section. "
                f"Add a '{method}:' block to the YAML."
            )

    # --- Model section validation ---
    model_cfg = config["model"]
    if "name" not in model_cfg:
        raise ValueError("Config model section must include 'name' (model path or HF identifier)")

    logger.info(f"Config loaded: model={model_cfg['name']}, method={po.get('method', 'N/A')}")
    return config


# ============================================================================
# CONFIG OBJECT BUILDERS
# ============================================================================
# These functions map YAML dict sections → rlhf.py dataclass config objects.
# Every field is explicit. No silent defaults that diverge from rlhf.py.

def _extract_base_fields(section: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Extract BaseConfig fields from a YAML section.

    Handles the common fields shared by all config dataclasses:
    learning_rate, batch_size, num_epochs, weight_decay, max_grad_norm,
    warmup_ratio, gradient_accumulation_steps, output_dir, etc.
    """
    training_cfg = {}  # Will be populated from multiple YAML sections

    # Direct mappings from the method-specific section
    field_map = {
        "learning_rate": "learning_rate",
        "batch_size": "batch_size",
        "num_epochs": "num_epochs",
    }
    for yaml_key, config_key in field_map.items():
        if yaml_key in section:
            training_cfg[config_key] = section[yaml_key]

    training_cfg["output_dir"] = section.get("output_dir", output_dir)
    return training_cfg


def build_sft_config(config: Dict[str, Any]) -> SFTConfig:
    """Build SFTConfig from the 'sft' YAML section."""
    sft = config.get("sft", {})
    training = config.get("training", {})
    lora = config.get("lora", {})

    kwargs = _extract_base_fields(sft, output_dir="./checkpoints/sft")
    if "max_seq_length" in sft:
        kwargs["max_seq_length"] = sft["max_seq_length"]

    # LoRA settings
    if lora.get("use_lora", False):
        kwargs["use_lora"] = True
        kwargs["lora_r"] = lora.get("r", 16)
        kwargs["lora_alpha"] = lora.get("alpha", 32)
        kwargs["lora_dropout"] = lora.get("dropout", 0.05)
        if "target_modules" in lora:
            kwargs["lora_target_modules"] = lora["target_modules"]

    # Training infrastructure
    kwargs["gradient_accumulation_steps"] = training.get("gradient_accumulation_steps", 1)
    kwargs["max_grad_norm"] = training.get("max_grad_norm", 1.0)
    kwargs["weight_decay"] = training.get("weight_decay", 0.01)

    # Logging
    log_cfg = config.get("logging", {})
    kwargs["use_wandb"] = log_cfg.get("use_wandb", False)
    kwargs["use_tensorboard"] = log_cfg.get("use_tensorboard", False)
    if log_cfg.get("use_tensorboard"):
        kwargs["tensorboard_dir"] = log_cfg.get("tensorboard_dir")
    if log_cfg.get("use_wandb"):
        kwargs["wandb_project"] = log_cfg.get("wandb_project", "rlhf-qwen3-1.7b")

    return SFTConfig(**kwargs)


def build_process_reward_model_config(config: Dict[str, Any]) -> RewardModelConfig:
    """Build RewardModelConfig for Process Reward Model training."""
    prm = config.get("process_reward_model", config.get("reward_model", {}))
    training = config.get("training", {})

    kwargs = _extract_base_fields(
        prm, output_dir="./checkpoints/process_reward_model"
    )

    # PRM uses RewardModelConfig as the base config object
    prm_fields = {
        "dropout": "dropout",
        "label_smoothing": "label_smoothing",
        "l2_reg": "l2_reg",
        "margin": "margin",
        "ensemble_size": "ensemble_size",
    }
    for yaml_key, config_key in prm_fields.items():
        if yaml_key in prm:
            kwargs[config_key] = prm[yaml_key]

    # Training infrastructure
    kwargs["gradient_accumulation_steps"] = training.get("gradient_accumulation_steps", 1)
    kwargs["max_grad_norm"] = training.get("max_grad_norm", 1.0)

    return RewardModelConfig(**kwargs)


def build_method_config(
    method: str,
    config: Dict[str, Any],
) -> Union[SFTConfig, DPOConfig, GRPOConfig, TreeGRPOConfig, SimPOConfig, KTOConfig, PPOConfig]:
    """
    Build the method-specific config dataclass from its YAML section.

    Args:
        method: One of 'hidden_cot_sft', 'dpo', 'grpo', 'tree_grpo', 'simpo', 'kto', 'ppo'.
        config: Full parsed YAML config dict.

    Returns:
        The appropriate config dataclass instance.

    Raises:
        ValueError: If method is unknown or section is missing.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"Unknown method: '{method}'. Valid: {sorted(VALID_METHODS)}")

    section = config.get(method, {})
    if not section and method == "tree_grpo":
        # Allow tree_grpo to inherit from grpo section when explicit tree block
        # has not yet been added to YAML.
        section = config.get("grpo", {})
    if not section:
        raise ValueError(f"No '{method}' section found in config")

    training = config.get("training", {})
    lora = config.get("lora", {})
    log_cfg = config.get("logging", {})

    base = _extract_base_fields(section, output_dir=f"./checkpoints/{method}")

    # Training infrastructure
    base["gradient_accumulation_steps"] = training.get("gradient_accumulation_steps", 1)
    base["max_grad_norm"] = training.get("max_grad_norm", 1.0)
    base["weight_decay"] = training.get("weight_decay", 0.01)

    # LoRA
    if lora.get("use_lora", False):
        base["use_lora"] = True
        base["lora_r"] = lora.get("r", 16)
        base["lora_alpha"] = lora.get("alpha", 32)
        base["lora_dropout"] = lora.get("dropout", 0.05)
        if "target_modules" in lora:
            base["lora_target_modules"] = lora["target_modules"]

    # Logging
    base["use_wandb"] = log_cfg.get("use_wandb", False)
    base["use_tensorboard"] = log_cfg.get("use_tensorboard", False)
    if log_cfg.get("use_tensorboard"):
        base["tensorboard_dir"] = log_cfg.get("tensorboard_dir")

    # ----- Method-specific fields -----
    if method == "hidden_cot_sft":
        hidden_cot_fields = {
            "max_seq_length": "max_seq_length",
            "dropout": "dropout",
        }
        for yaml_key, config_key in hidden_cot_fields.items():
            if yaml_key in section:
                base[config_key] = section[yaml_key]
        return SFTConfig(**base)

    if method == "dpo":
        dpo_fields = {
            "beta": "beta",
            "label_smoothing": "label_smoothing",
            "loss_type": "loss_type",
        }
        for yaml_key, config_key in dpo_fields.items():
            if yaml_key in section:
                base[config_key] = section[yaml_key]
        return DPOConfig(**base)

    elif method == "grpo":
        grpo_fields = {
            "group_size": "group_size",
            "kl_coeff": "kl_coeff",
            "clip_ratio": "clip_ratio",
            "num_policy_updates": "num_policy_updates",
            "max_completion_length": "max_completion_length",
            "temperature": "temperature",
            "use_verifiable_rewards": "use_verifiable_rewards",
        }
        for yaml_key, config_key in grpo_fields.items():
            if yaml_key in section:
                base[config_key] = section[yaml_key]
        return GRPOConfig(**base)

    elif method == "tree_grpo":
        # Start from GRPO-compatible fields
        tree_grpo_fields = {
            "group_size": "group_size",
            "kl_coeff": "kl_coeff",
            "clip_ratio": "clip_ratio",
            "num_policy_updates": "num_policy_updates",
            "max_completion_length": "max_completion_length",
            "temperature": "temperature",
            "use_verifiable_rewards": "use_verifiable_rewards",
            # TreeGRPO-specific extensions
            "depth_discount": "depth_discount",
            "visit_count_weighting": "visit_count_weighting",
            "branch_comparison_mode": "branch_comparison_mode",
            "use_prm_preference": "use_prm_preference",
            "prm_min_aggregation": "prm_min_aggregation",
            "rollout_source": "rollout_source",
            "min_group_size": "min_group_size",
            "replay_buffer_size": "replay_buffer_size",
            "replay_priority": "replay_priority",
            "cot_mode": "cot_mode",
            "reward_on_answer_only": "reward_on_answer_only",
        }
        for yaml_key, config_key in tree_grpo_fields.items():
            if yaml_key in section:
                base[config_key] = section[yaml_key]
        return TreeGRPOConfig(**base)

    elif method == "simpo":
        simpo_fields = {
            "beta": "beta",
            "gamma": "gamma",
            "label_smoothing": "label_smoothing",
        }
        for yaml_key, config_key in simpo_fields.items():
            if yaml_key in section:
                base[config_key] = section[yaml_key]
        return SimPOConfig(**base)

    elif method == "kto":
        kto_fields = {
            "beta": "beta",
            "lambda_d": "lambda_d",
            "lambda_u": "lambda_u",
            "kl_ema_decay": "kl_ema_decay",
        }
        for yaml_key, config_key in kto_fields.items():
            if yaml_key in section:
                base[config_key] = section[yaml_key]
        return KTOConfig(**base)

    elif method == "ppo":
        ppo_fields = {
            "clip_ratio": "clip_ratio",
            "kl_coeff": "kl_coeff",
            "kl_target": "kl_target",
            "value_loss_coef": "value_loss_coef",
            "entropy_coef": "entropy_coef",
            "lam": "lam",
            "gamma": "gamma",
            "normalize_advantage": "normalize_advantage",
            "whiten_rewards": "whiten_rewards",
            "ppo_epochs": "ppo_epochs",
            "mini_batch_size": "mini_batch_size",
            "rollout_multiplier": "rollout_multiplier",
            "temperature": "temperature",
            "max_completion_length": "max_completion_length",
        }
        for yaml_key, config_key in ppo_fields.items():
            if yaml_key in section:
                base[config_key] = section[yaml_key]
        return PPOConfig(**base)

    # Should never reach here — guarded by VALID_METHODS check above
    raise ValueError(f"Unhandled method: {method}")


# ============================================================================
# DEVICE MANAGER BUILDER
# ============================================================================

def build_device_manager(config: Dict[str, Any]) -> DeviceManager:
    """
    Build a DeviceManager from training + hardware YAML sections.

    Resolves device (CPU/CUDA), AMP settings, and dtype from config.
    """
    training = config.get("training", {})
    hardware = config.get("hardware", {})

    device_str = training.get("device", "cpu")
    use_amp = training.get("use_amp", False)

    # Determine dtype
    model_cfg = config.get("model", {})
    torch_dtype_str = model_cfg.get("torch_dtype", "float32")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(torch_dtype_str, torch.float32)

    # Set torch thread count from hardware config
    num_threads = hardware.get("torch_num_threads")
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))
        logger.info(f"Set torch.num_threads={num_threads}")

    dm = DeviceManager(device=device_str, use_amp=use_amp, dtype=dtype)
    logger.info(
        f"DeviceManager: device={dm.device}, use_amp={use_amp}, dtype={dtype}"
    )
    return dm


# ============================================================================
# DATASET LOADING
# ============================================================================

def _resolve_data_path(config: Dict[str, Any], path_str: str) -> Path:
    """
    Resolve a dataset path from config.

    Relative paths are resolved against the directory containing the YAML file,
    not the current working directory.
    """
    return _resolve_required_path_from_config(config, path_str, label="Dataset file")


def _is_jsonl(path: Path) -> bool:
    """Check if a file is JSONL format by extension."""
    return path.suffix.lower() in (".jsonl", ".ndjson")


def _is_json(path: Path) -> bool:
    """Check if a file is JSON format by extension."""
    return path.suffix.lower() == ".json"


def _load_json_data(path: Path) -> List[Dict[str, Any]]:
    """Load a JSON file containing a list of dicts."""
    logger.info(f"Loading JSON dataset: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}, got {type(data).__name__}")
    logger.info(f"Loaded {len(data)} examples from {path}")
    return data


def _load_jsonl_data(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL/NDJSON file into a list of dict records."""
    logger.info(f"Loading JSONL dataset into memory: {path}")
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(
                    f"Expected JSON object on line {line_no} of {path}, got {type(item).__name__}"
                )
            data.append(item)
    logger.info(f"Loaded {len(data)} examples from {path}")
    return data


def load_training_data(
    method: str,
    config: Dict[str, Any],
    tokenizer,
    max_length: int = 2048,
) -> Union[List[Dict], "torch.utils.data.Dataset"]:
    """
    Load training data appropriate for the active method.

    For JSONL files, returns a streaming IterableDataset.
    For JSON files, returns either the raw list (for orchestrator dispatch)
    or a map-style Dataset.

    The orchestrator's internal method runners (_run_dpo, _run_grpo, etc.)
    can accept either raw lists or DataLoaders, so we can return either.

    Args:
        method: Active training method.
        config: Full parsed YAML config.
        tokenizer: Initialized tokenizer from the orchestrator.
        max_length: Maximum sequence length for tokenization.

    Returns:
        Dataset or list suitable for the method.
    """
    method_section = config.get(method, {})
    if not method_section and method == "tree_grpo":
        method_section = config.get("grpo", {})
    train_file = method_section.get("train_file")

    if train_file is None:
        raise ValueError(
            f"No 'train_file' specified in '{method}' config section. "
            f"Set {method}.train_file in the YAML."
        )

    data_path = _resolve_data_path(config, train_file)

    # --- Check for Moonshine data_source before filesystem path resolution ---
    data_source = method_section.get("data_source", "").strip().lower()
    if data_source == "moonshine":
        moonshine_cfg = method_section.get("moonshine", {})
        return load_moonshine_data(config, moonshine_cfg, mode="grpo")

    # --- hidden CoT warmstart and tree_grpo expect prompt records in memory ---
    if method in {"tree_grpo", "hidden_cot_sft"}:
        logger.info(
            "Tree-GRPO data path: materializing records in memory so "
            "RLHFOrchestrator._run_tree_grpo can collect grouped rollouts."
        )
        if _is_jsonl(data_path):
            return _load_jsonl_data(data_path)
        if _is_json(data_path):
            return _load_json_data(data_path)
        raise ValueError(
            f"Unsupported data file format: {data_path.suffix}. "
            f"Use .json or .jsonl/.ndjson"
        )

    # --- JSONL → Streaming dataset ---
    if _is_jsonl(data_path):
        logger.info(f"Using streaming dataset for JSONL: {data_path}")

        batch_size = int(method_section.get("batch_size", 1))
        hardware = config.get("hardware", {})
        num_workers = int(hardware.get("num_workers", 0))
        pin_memory = bool(hardware.get("pin_memory", False))

        if method in METHODS_PREFERENCE_DATA:
            dataset = StreamingPreferenceDataset(
                filepath=str(data_path),
                tokenizer=tokenizer,
                max_length=max_length,
            )
        elif method in METHODS_UNPAIRED_DATA:
            dataset = StreamingKTODataset(
                filepath=str(data_path),
                tokenizer=tokenizer,
                max_length=max_length,
            )
        elif method in METHODS_PROMPT_DATA:
            dataset = StreamingGRPODataset(
                filepath=str(data_path),
                tokenizer=tokenizer,
                max_prompt_length=max_length,
            )
        else:
            raise ValueError(f"No streaming dataset class for method '{method}'")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    # --- JSON → Load into memory ---
    elif _is_json(data_path):
        data = _load_json_data(data_path)
        # Return raw list — the orchestrator's _run_* methods accept lists
        # and internally build the appropriate Dataset + DataLoader.
        return data

    else:
        raise ValueError(
            f"Unsupported data file format: {data_path.suffix}. "
            f"Use .json or .jsonl/.ndjson"
        )


def load_sft_data(
    config: Dict[str, Any],
    tokenizer,
    max_length: int = 2048,
) -> Union[List[Dict], "torch.utils.data.Dataset"]:
    """Load SFT training data from config."""
    sft = config.get("sft", {})
    train_file = sft.get("train_file")

    if train_file is None:
        raise ValueError(
            "SFT is enabled but no 'train_file' specified in 'sft' config section."
        )

    data_path = _resolve_data_path(config, train_file)

    if _is_jsonl(data_path):
        logger.info(
            "Materializing JSONL SFT data into raw records because "
            "RLHFOrchestrator.run_sft() expects list input."
        )
        return _load_jsonl_data(data_path)
    elif _is_json(data_path):
        return _load_json_data(data_path)
    else:
        raise ValueError(f"Unsupported SFT data format: {data_path.suffix}")


def load_prm_data(
    config: Dict[str, Any],
    tokenizer,
    max_length: int = 2048,
) -> Union[List[Dict], "torch.utils.data.Dataset"]:
    """
    Load preference data for Process Reward Model training.

    Falls back to the active method's train_file if no separate
    process_reward_model.train_file is specified.
    """
    prm = config.get("process_reward_model", config.get("reward_model", {}))
    train_file = prm.get("train_file")

    # Fallback: use the DPO/SimPO training data (it's preference pairs)
    if train_file is None:
        pipeline = config.get("pipeline", {})
        method = pipeline.get("policy_optimization", {}).get("method", "dpo")
        method_section = config.get(method, {})
        train_file = method_section.get("train_file")

    if train_file is None:
        raise ValueError(
            "Process reward model training enabled but no training data found. "
            "Set process_reward_model.train_file or provide preference data through "
            "the active method's train_file."
        )

    data_path = _resolve_data_path(config, train_file)

    if _is_jsonl(data_path):
        logger.info(
            "Materializing JSONL PRM data into raw preference records because "
            "RLHFOrchestrator.run_process_reward_model_training() expects list input."
        )
        return _load_jsonl_data(data_path)
    elif _is_json(data_path):
        return _load_json_data(data_path)
    else:
        raise ValueError(f"Unsupported PRM data format: {data_path.suffix}")


def load_moonshine_data(
    config: Dict[str, Any],
    moonshine_cfg: Dict[str, Any],
    mode: str = "grpo",
) -> List[Dict[str, Any]]:
    """
    Load training data from the Moonshine corpus via distill-the-flow/moonshine_streamer.

    Delegates to moonshine_streamer.stream() with mode='grpo' (prompt-only records)
    or mode='dpo'/'sft' depending on the caller's lane.

    Args:
        config: Full parsed YAML config dict.
        moonshine_cfg: Moonshine-specific sub-config from the method section.
        mode: Stream mode — 'grpo', 'sft', or 'dpo'.

    Returns:
        Materialized list of prompt records.

    Raises:
        FileNotFoundError: If the Moonshine DB/input path does not exist.
        ImportError: If moonshine_streamer cannot be imported.
        ValueError: If no records are extracted.
    """
    input_path = moonshine_cfg.get("input_path") or moonshine_cfg.get("db_path")
    if not input_path:
        raise ValueError(
            "Moonshine data_source is active but no moonshine.input_path or moonshine.db_path "
            "is configured in the method section. Set tree_grpo.moonshine.input_path in the YAML."
        )

    resolved = _resolve_required_path_from_config(
        config, input_path, label="Moonshine input path"
    )

    try:
        # moonshine_streamer lives inside distill-the-flow/
        streamer_dir = Path(__file__).resolve().parent / "distill-the-flow"
        if str(streamer_dir) not in sys.path:
            sys.path.insert(0, str(streamer_dir))
        import moonshine_streamer
    except ImportError as exc:
        raise ImportError(
            f"Cannot import moonshine_streamer from {streamer_dir}. "
            "Ensure distill-the-flow/moonshine_streamer.py exists and its dependencies "
            "(ijson) are installed in the active .venv."
        ) from exc

    input_kind = moonshine_cfg.get("input_kind", "auto")
    provider = moonshine_cfg.get("provider")
    topic = moonshine_cfg.get("topic")
    min_gain = float(moonshine_cfg.get("min_gain", 0.50))
    max_sycophancy = float(moonshine_cfg.get("max_sycophancy", 0.25))
    limit = moonshine_cfg.get("limit")
    if limit is not None:
        limit = int(limit)
    strip_meta = bool(moonshine_cfg.get("strip_meta", True))
    strip_emoji = _is_truthy(moonshine_cfg.get("strip_emoji", False))
    provider_policy_mode = str(moonshine_cfg.get("provider_policy_mode", "legacy"))
    provider_policy = moonshine_cfg.get("provider_policy")

    logger.info(
        "Loading Moonshine data: path=%s mode=%s kind=%s provider=%s limit=%s policy_mode=%s strip_emoji=%s",
        resolved, mode, input_kind, provider or "all", limit, provider_policy_mode, strip_emoji,
    )

    records: List[Dict[str, Any]] = list(
        moonshine_streamer.stream(
            input_path=str(resolved),
            mode=mode,
            input_kind=input_kind,
            provider=provider,
            topic=topic,
            min_gain=min_gain,
            max_sycophancy=max_sycophancy,
            limit=limit,
            strip_meta=strip_meta,
            provider_policy_mode=provider_policy_mode,
            provider_policy=provider_policy,
        )
    )

    if strip_emoji:
        records = [_strip_emoji_payload(record) for record in records]

    if not records:
        raise ValueError(
            f"Moonshine extraction yielded zero records from {resolved} with mode={mode}. "
            "Check input_path, provider filter, and min_gain threshold."
        )

    logger.info(
        "Moonshine data loaded: %d %s records from %s",
        len(records), mode, resolved,
    )
    return records


# ============================================================================
# EXECUTION PLAN (for --dry-run)
# ============================================================================

def _parse_ram_gb(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().lower().replace("gib", "").replace("gb", "")
        try:
            return float(stripped)
        except ValueError:
            digits = "".join(ch if (ch.isdigit() or ch == ".") else " " for ch in stripped).split()
            if digits:
                try:
                    return float(digits[0])
                except ValueError:
                    return None
    return None


def _load_prompt_strings_from_file(path: Path, max_prompts: int) -> List[str]:
    if _is_jsonl(path):
        records = _load_jsonl_data(path)
        return _extract_prompt_strings(records, max_prompts)
    if _is_json(path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("prompts", payload.get("data", []))
        return _extract_prompt_strings(payload, max_prompts)
    return [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:max_prompts]


def _extract_prompt_strings(records: Any, max_prompts: int) -> List[str]:
    prompts: List[str] = []
    if isinstance(records, list):
        for item in records:
            prompt: Optional[str] = None
            if isinstance(item, str):
                prompt = item
            elif isinstance(item, dict):
                for key in ("prompt", "text", "input", "question"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        prompt = value.strip()
                        break
            if prompt:
                prompts.append(prompt)
            if len(prompts) >= max_prompts:
                break
    return prompts[:max_prompts]


def _select_session_records(
    records: List[Any],
    config: Dict[str, Any],
    session: str,
) -> Tuple[List[Any], Dict[str, Any]]:
    sessions_cfg = get_sessions_config(config)
    if session == "all" or not _is_truthy(sessions_cfg.get("enabled", False)):
        return records, {
            "enabled": _is_truthy(sessions_cfg.get("enabled", False)),
            "session": session,
            "mode": "single",
            "selected": len(records),
            "total": len(records),
        }

    if not isinstance(records, list):
        raise ValueError(
            "Session splitting currently requires in-memory list data. "
            "Use JSON/JSONL materialized list input for this lane."
        )

    mode = str(sessions_cfg.get("split_mode", "half")).lower()
    total = len(records)
    if total == 0:
        return [], {"enabled": True, "session": session, "mode": mode, "selected": 0, "total": 0}

    if mode == "single":
        start, end = 0, total
    elif mode == "half":
        midpoint = max(1, total // 2)
        if session == "1":
            start, end = 0, midpoint
        else:
            start, end = midpoint, total
    elif mode == "custom":
        if session == "1":
            start = int(sessions_cfg.get("session1_start", 0))
            end = int(sessions_cfg.get("session1_end", max(1, total // 2)))
        else:
            start = int(sessions_cfg.get("session2_start", int(sessions_cfg.get("session1_end", max(1, total // 2)))))
            end = int(sessions_cfg.get("session2_end", total))
    else:
        raise ValueError(
            f"Unsupported sessions.split_mode '{mode}'. Use 'single', 'half', or 'custom'."
        )

    start = max(0, min(start, total))
    end = max(start, min(end, total))
    selected = records[start:end]
    return selected, {
        "enabled": True,
        "session": session,
        "mode": mode,
        "start_index": start,
        "end_index": end,
        "selected": len(selected),
        "total": total,
    }


def _session_output_dir(base_output: Path, config: Dict[str, Any], session: str) -> Path:
    sessions_cfg = get_sessions_config(config)
    if _is_truthy(sessions_cfg.get("enabled", False)) and session in {"1", "2"}:
        return (base_output / f"session_{session}").resolve()
    return base_output.resolve()


def _session_resume_dir(config: Dict[str, Any], base_output: Path) -> Path:
    sessions_cfg = get_sessions_config(config)
    configured = (
        sessions_cfg.get("resume_from")
        or sessions_cfg.get("session_1_output_dir")
        or sessions_cfg.get("session1_output_dir")
    )
    if configured:
        return _resolve_required_path_from_config(config, configured, label="Session resume directory")
    return (base_output / "session_1").resolve()


def _benchmark_constraints(
    config: Dict[str, Any],
    device_manager: DeviceManager,
    benchmark_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    hardware = config.get("hardware", {})
    return {
        "max_ram_gb": _parse_ram_gb(hardware.get("max_memory_gb", hardware.get("max_memory"))) or 16.0,
        "has_gpu": device_manager.device.type == "cuda",
        "latency_budget_ms": int(benchmark_cfg.get("latency_budget_ms", 1000)),
        "quality_priority": _is_truthy(benchmark_cfg.get("quality_priority", True)),
    }


def _regression_record_from_benchmarks(
    benchmark_results: Dict[str, Any],
    target: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    target = (target or "search_best").strip()
    search = benchmark_results.get("search_quality") or {}
    if target == "search_best" and isinstance(search, dict) and search:
        valid_items = [
            (name, payload) for name, payload in search.items()
            if isinstance(payload, dict) and "error" not in payload
        ]
        if valid_items:
            name, payload = max(
                valid_items,
                key=lambda item: float(item[1].get("mean_reward", item[1].get("reward_score_mean", float("-inf")))),
            )
            record = dict(payload)
            record["eval_score"] = float(record.get("mean_reward", record.get("reward_score_mean", float("nan"))))
            return record, f"search_quality:{name}"

    if target.startswith("search_quality:"):
        name = target.split(":", 1)[1]
        payload = search.get(name)
        if isinstance(payload, dict):
            record = dict(payload)
            record["eval_score"] = float(record.get("mean_reward", record.get("reward_score_mean", float("nan"))))
            return record, target

    if target == "hidden_cot_fidelity":
        payload = benchmark_results.get("hidden_cot_fidelity")
        if isinstance(payload, dict):
            record = dict(payload)
            record["eval_score"] = float(record.get("validity_rate", float("nan")))
            return record, target

    if target == "tree_grpo":
        payload = benchmark_results.get("tree_grpo")
        if isinstance(payload, dict):
            record = dict(payload)
            record["eval_score"] = float(record.get("mean_reward", float("nan")))
            return record, target

    return None, target


def _load_benchmark_baseline(
    config: Dict[str, Any],
    gate_cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    baseline_path_raw = gate_cfg.get("baseline_results_file") or gate_cfg.get("baseline_report")
    if not baseline_path_raw:
        return None
    baseline_path = _resolve_required_path_from_config(
        config,
        baseline_path_raw,
        label="Benchmark baseline results file",
    )
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    results = payload.get("results", payload)
    record, _ = _regression_record_from_benchmarks(results, gate_cfg.get("target", "search_best"))
    return record


def run_benchmark_suite(
    *,
    config: Dict[str, Any],
    output_base: Path,
    device_manager: DeviceManager,
    orchestrator: RLHFOrchestrator,
    method: str,
    method_config: Any,
    training_data: Any,
    policy_adapter: Optional[PolicyAdapter],
    prm_adapter: Optional[ProcessRewardModelAdapter],
    cot_generator: Optional[Any],
    generator_op: Optional[Any],
    telemetry_recorder: Optional[TelemetryRecorder] = None,
) -> Dict[str, Any]:
    benchmark_cfg = get_benchmark_config(config)
    if not _is_truthy(benchmark_cfg.get("enabled", False)):
        return {}

    harness = BenchmarkHarness(telemetry=telemetry_recorder)
    inf_cfg = config.get("inference_optimizations", {})
    prompts_limit = int(benchmark_cfg.get("max_prompts", 8))

    if isinstance(benchmark_cfg.get("prompts"), list):
        prompts = _extract_prompt_strings(benchmark_cfg.get("prompts"), prompts_limit)
    else:
        prompts_file = benchmark_cfg.get("prompts_file") or benchmark_cfg.get("prompt_file")
        if prompts_file:
            prompts_path = _resolve_required_path_from_config(
                config,
                prompts_file,
                label="Benchmark prompts file",
            )
            prompts = _load_prompt_strings_from_file(prompts_path, prompts_limit)
        else:
            prompts = _extract_prompt_strings(training_data, prompts_limit)

    if not prompts:
        if _is_truthy(benchmark_cfg.get("required", False)):
            raise ValueError(
                "benchmark_harness.enabled=true but no benchmark prompts were available. "
                "Provide benchmark_harness.prompts or benchmark_harness.prompts_file."
            )
        logger.warning("Benchmark harness enabled but no prompts were available; skipping benchmark stage.")
        return {"status": "skipped", "reason": "no_prompts"}

    selector = BudgetProfileSelector(
        history_path=str(_resolve_output_path_from_config(config, benchmark_cfg.get("history_path")))
        if benchmark_cfg.get("history_path") else None
    )
    profile_name = benchmark_cfg.get("profile", "auto")
    bench_bon_cfg = None
    bench_mcts_cfg = None
    if profile_name is not False:
        profile_override = None
        if isinstance(profile_name, str) and profile_name.strip() and profile_name != "auto":
            profile_override = profile_name
        _, bench_bon_cfg, bench_mcts_cfg, _ = selector.select(
            _benchmark_constraints(config, device_manager, benchmark_cfg),
            profile_override=profile_override,
        )

    results: Dict[str, Any] = {
        "status": "ok",
        "profile": selector.last_selected_profile,
        "n_prompts": len(prompts),
        "prompts_preview": prompts[: min(3, len(prompts))],
    }

    max_new_tokens = int(
        benchmark_cfg.get(
            "max_new_tokens",
            getattr(method_config, "max_completion_length", 128),
        )
    )

    search_cfg = benchmark_cfg.get("search_quality", {})
    if _is_truthy(search_cfg.get("enabled", True)):
        if policy_adapter is None or prm_adapter is None:
            if _is_truthy(search_cfg.get("required", False)):
                raise ValueError(
                    "Search-quality benchmark requires both policy_adapter and prm_adapter."
                )
            results["search_quality"] = {
                "status": "skipped",
                "reason": "missing_policy_or_prm",
            }
        else:
            bon_cfg_dict = dict(inf_cfg.get("best_of_n", {}))
            if bench_bon_cfg is not None:
                bon_cfg_dict = {**vars(bench_bon_cfg), **bon_cfg_dict}
            bon_cfg = BestOfNConfig(
                n_samples=int(bon_cfg_dict.get("n_samples", 4)),
                temperature=float(bon_cfg_dict.get("temperature", 1.0)),
                top_p=float(bon_cfg_dict.get("top_p", 0.95)),
                reward_aggregation=bon_cfg_dict.get("reward_aggregation", "mean"),
                use_diversity_bonus=_is_truthy(bon_cfg_dict.get("use_diversity_bonus", True)),
                diversity_weight=float(bon_cfg_dict.get("diversity_weight", 0.1)),
                step_rerank=_is_truthy(bon_cfg_dict.get("step_rerank", False)),
                step_prm=prm_adapter if _is_truthy(bon_cfg_dict.get("step_rerank", False)) else None,
                prm_process_weight=float(bon_cfg_dict.get("prm_process_weight", 0.0)),
            )
            astar_cfg_dict = inf_cfg.get("astar", {})
            astar_cfg = None
            if _is_truthy(astar_cfg_dict.get("enabled", False)):
                astar_cfg = AStarConfig(
                    max_nodes=int(astar_cfg_dict.get("max_nodes", 256)),
                    max_depth=int(astar_cfg_dict.get("max_depth", 64)),
                    n_actions=int(astar_cfg_dict.get("n_actions", 8)),
                    heuristic_weight=float(astar_cfg_dict.get("heuristic_weight", 1.0)),
                    temperature=float(astar_cfg_dict.get("temperature", 0.8)),
                    use_value_heuristic=_is_truthy(astar_cfg_dict.get("use_value_heuristic", True)),
                )
            if bench_mcts_cfg is None:
                mcts_cfg_dict = inf_cfg.get("mcts", {})
                bench_mcts_cfg = MCTSConfig(
                    n_simulations=int(mcts_cfg_dict.get("n_simulations", 50)),
                    max_depth=int(mcts_cfg_dict.get("max_depth", 50)),
                    max_rollout_depth=int(mcts_cfg_dict.get("max_rollout_depth", 30)),
                    n_actions=int(mcts_cfg_dict.get("n_actions", 8)),
                    temperature=float(mcts_cfg_dict.get("temperature", 1.0)),
                    c_puct=float(mcts_cfg_dict.get("c_puct", 1.414)),
                    depth_discount=float(mcts_cfg_dict.get("depth_discount", 0.95)),
                )
            results["search_quality"] = harness.run_search_quality_benchmark(
                policy=policy_adapter,
                reward_fn=lambda prompt, completion="": prm_adapter.score(prompt, completion),
                tokenizer=orchestrator.tokenizer,
                prompts=prompts,
                mcts_config=bench_mcts_cfg,
                astar_config=astar_cfg,
                bon_config=bon_cfg,
                max_new_tokens=max_new_tokens,
            )

    cot_bench_cfg = benchmark_cfg.get("hidden_cot_fidelity", {})
    if _is_truthy(cot_bench_cfg.get("enabled", cot_generator is not None)):
        if cot_generator is None:
            if _is_truthy(cot_bench_cfg.get("required", False)):
                raise ValueError("Hidden-CoT fidelity benchmark requested but cot_generator is unavailable.")
            results["hidden_cot_fidelity"] = {
                "status": "skipped",
                "reason": "missing_cot_generator",
            }
        else:
            cot_cfg = inf_cfg.get("chain_of_thought", {})
            results["hidden_cot_fidelity"] = harness.run_hidden_cot_fidelity_benchmark(
                cot_generator=cot_generator,
                prompts=prompts,
                think_start_tag=cot_cfg.get("think_start_tag", "<think>"),
                think_end_tag=cot_cfg.get("think_end_tag", "</think>"),
            )

    tree_bench_cfg = benchmark_cfg.get("tree_grpo", {})
    if _is_truthy(tree_bench_cfg.get("enabled", False)):
        if method != "tree_grpo":
            results["tree_grpo"] = {
                "status": "skipped",
                "reason": f"method={method}",
            }
        elif not isinstance(generator_op, MCTSGenerator):
            if _is_truthy(tree_bench_cfg.get("required", False)):
                raise ValueError(
                    "Tree-GRPO benchmark currently requires an MCTSGenerator search source."
                )
            results["tree_grpo"] = {
                "status": "skipped",
                "reason": "missing_mcts_generator",
            }
        elif prm_adapter is None:
            if _is_truthy(tree_bench_cfg.get("required", False)):
                raise ValueError("Tree-GRPO benchmark requires a PRM adapter reward surface.")
            results["tree_grpo"] = {
                "status": "skipped",
                "reason": "missing_prm_adapter",
            }
        else:
            results["tree_grpo"] = harness.run_tree_grpo_benchmark(
                policy=orchestrator.policy_model,
                reference_model=orchestrator.reference_model,
                tokenizer=orchestrator.tokenizer,
                mcts_generator=generator_op,
                reward_fn=lambda prompt, completion="": prm_adapter.score(prompt, completion),
                prompts=prompts,
                tree_grpo_config=method_config,
                n_steps=int(tree_bench_cfg.get("n_steps", 5)),
            )

    gate_cfg = benchmark_cfg.get("regression_gate", {})
    if _is_truthy(gate_cfg.get("enabled", False)):
        current_record, selected_target = _regression_record_from_benchmarks(
            results,
            gate_cfg.get("target", "search_best"),
        )
        if current_record is None:
            raise ValueError(
                f"Regression gate target '{gate_cfg.get('target', 'search_best')}' could not be resolved."
            )
        baseline_record = _load_benchmark_baseline(config, gate_cfg)
        gate = RegressionGate(thresholds=gate_cfg.get("thresholds", {}))
        passed = gate.check(current_record, baseline_record)
        results["regression_gate"] = {
            "passed": passed,
            "target": selected_target,
            "thresholds": gate_cfg.get("thresholds", {}),
        }
        if not passed and _is_truthy(gate_cfg.get("fail_on_violation", True)):
            raise RuntimeError(
                f"Benchmark regression gate failed for target '{selected_target}'."
            )

    report_path_raw = benchmark_cfg.get("report_path")
    report_path = (
        _resolve_output_path_from_config(config, report_path_raw)
        if report_path_raw else (output_base / "benchmark_harness_report.json").resolve()
    )
    harness.emit_report(results, str(report_path))
    results["report_path"] = str(report_path)
    return results


def print_execution_plan(
    config: Dict[str, Any],
    method_override: Optional[str] = None,
    session: str = "all",
) -> None:
    """
    Print what the pipeline would do without loading any models.

    Used by --dry-run to verify config before committing resources.
    """
    pipeline = config["pipeline"]
    model_name = config["model"]["name"]
    method = method_override or pipeline["policy_optimization"]["method"]
    validate_runtime_contracts(config, method, session=session)

    sessions_cfg = get_sessions_config(config)
    sessions_enabled = _is_truthy(sessions_cfg.get("enabled", False))
    benchmark_cfg = get_benchmark_config(config)

    print("\n" + "=" * 72)
    print("  RLHF PIPELINE - EXECUTION PLAN (DRY RUN)")
    print("=" * 72)

    print(f"\n  Base Model:          {model_name}")
    print(f"  Active Method:       {method.upper()}")
    print(f"  Session Selector:    {session}")
    if sessions_enabled:
        print(
            "  Session Mode:        "
            f"{sessions_cfg.get('split_mode', 'single')}"
        )
    if _is_truthy(benchmark_cfg.get("enabled", False)):
        print("  Benchmark Harness:   ENABLED")

    # Stage 1: SFT
    sft_enabled = _is_truthy(pipeline.get("sft", {}).get("enabled", False))
    print(f"\n  [{'ON ' if sft_enabled else 'OFF'}] Stage 1: Supervised Fine-Tuning")
    if sft_enabled:
        sft_cfg = config.get("sft", {})
        print(f"         Data:    {sft_cfg.get('train_file', 'NOT SET')}")
        print(f"         Epochs:  {sft_cfg.get('num_epochs', 3)}")
        print(f"         LR:      {sft_cfg.get('learning_rate', 'default')}")
        print(f"         Output:  {sft_cfg.get('output_dir', 'default')}")
    else:
        print(f"         Using existing checkpoint: {model_name}")

    # Stage 2: Process Reward Model
    prm_pipeline = get_prm_pipeline_config(pipeline)
    prm_enabled = _is_truthy(prm_pipeline.get("enabled", False))
    prm_needed = method in METHODS_REQUIRING_REWARD
    prm_checkpoint = prm_pipeline.get("load_from_checkpoint")
    print(f"\n  [{'ON ' if prm_enabled else 'OFF'}] Stage 2: Process Reward Model")
    if prm_checkpoint:
        print(f"         Loading from checkpoint: {prm_checkpoint}")
    elif prm_enabled:
        prm_cfg = config.get("process_reward_model", config.get("reward_model", {}))
        print(f"         Data:    {prm_cfg.get('train_file', 'fallback to method data')}")
        print(f"         Epochs:  {prm_cfg.get('num_epochs', 2)}")
        print(f"         Output:  {prm_cfg.get('output_dir', 'default')}")
        print(f"         Step detection: {prm_pipeline.get('step_detection', 'newline')}")
        print(
            "         Process reward weight: "
            f"{prm_pipeline.get('process_reward_weight', 0.0)}"
        )
    if prm_needed and not prm_enabled and not prm_checkpoint:
        if method in STRICT_PRM_METHODS:
            print(
                f"    WARNING: {method.upper()} is configured without a live PRM. "
                "This run would fail loud by design."
            )
        else:
            artifact_reuse_cfg = get_artifact_reuse_config(config)
            if _is_truthy(artifact_reuse_cfg.get("enabled", False)) and _is_truthy(
                artifact_reuse_cfg.get("load_reward_model", False)
            ):
                print(
                    f"    NOTE: {method.upper()} may use artifact_reuse reward_model fallback "
                    "if load succeeds."
                )
            else:
                print(
                    f"    WARNING: {method.upper()} requires a reward signal but PRM is disabled"
                )

    # Stage 3: Policy Optimization
    po_enabled = _is_truthy(pipeline.get("policy_optimization", {}).get("enabled", True))
    print(f"\n  [{'ON ' if po_enabled else 'OFF'}] Stage 3: Policy Optimization ({method.upper()})")
    if po_enabled:
        method_cfg = _active_method_section(config, method)
        data_source = method_cfg.get("data_source", "").strip().lower()
        if data_source == "moonshine":
            moonshine_cfg = method_cfg.get("moonshine", {})
            print(f"         Data:    MOONSHINE ({moonshine_cfg.get('input_path', 'NOT SET')})")
            print(f"         Source:  moonshine_streamer.stream(mode='grpo')")
            if moonshine_cfg.get("provider"):
                print(f"         Filter:  provider={moonshine_cfg['provider']}")
            if moonshine_cfg.get("limit"):
                print(f"         Limit:   {moonshine_cfg['limit']}")
        else:
            print(f"         Data:    {method_cfg.get('train_file', 'NOT SET')}")
        print(f"         Epochs:  {method_cfg.get('num_epochs', 'default')}")
        print(f"         LR:      {method_cfg.get('learning_rate', 'default')}")
        print(f"         Output:  {method_cfg.get('output_dir', 'default')}")

        if method == "dpo":
            print(f"         Beta:    {method_cfg.get('beta', 0.1)}")
            print(f"         Loss:    {method_cfg.get('loss_type', 'sigmoid')}")
        elif method == "grpo":
            print(f"         Group:   {method_cfg.get('group_size', 4)}")
            print(f"         KL:      {method_cfg.get('kl_coeff', 0.1)}")
        elif method == "hidden_cot_sft":
            print(f"         Max seq: {method_cfg.get('max_seq_length', 2048)}")
            print("         Target:  prompt + <think>...</think>answer training warmstart")
        elif method == "tree_grpo":
            print(f"         Group:   {method_cfg.get('group_size', 4)}")
            print(f"         KL:      {method_cfg.get('kl_coeff', 0.1)}")
            print(f"         Source:  {method_cfg.get('rollout_source', 'mcts')}")
            print(f"         Replay:  {method_cfg.get('replay_buffer_size', 0)}")
            print(f"         CoT:     {method_cfg.get('cot_mode', False)}")
            print(f"         Answer-only reward: {method_cfg.get('reward_on_answer_only', True)}")
        elif method == "simpo":
            print(f"         Beta:    {method_cfg.get('beta', 2.0)}")
            print(f"         Gamma:   {method_cfg.get('gamma', 0.5)}")
        elif method == "kto":
            print(f"         Beta:    {method_cfg.get('beta', 0.1)}")
        elif method == "ppo":
            print(f"         Clip:    {method_cfg.get('clip_ratio', 0.2)}")
            print(f"         KL:      {method_cfg.get('kl_coeff', 0.02)}")

    # Inference optimization / Chiron runtime surfaces
    inf = config.get("inference_optimizations", {})
    inf_on = _is_truthy(inf.get("enabled", True))
    print(f"\n  [{'ON ' if inf_on else 'OFF'}] Inference Optimization Surface")
    if inf_on:
        print(f"         Search strategy: {inf.get('search_strategy', 'mcts')}")
        cot = inf.get("chain_of_thought", {})
        print(f"         Hidden CoT:      {_is_truthy(cot.get('enabled', False))}")
        if method == "tree_grpo":
            print("         NOTE: tree_grpo will use canon rollout collection in rlhf.py.")
            print("               capability_generate monkey-patching is skipped for this lane.")

    # Context Compression
    cc = config.get("context_compression", {})
    cc_on = _is_truthy(cc.get("enabled", False))
    print(f"\n  [{'ON ' if cc_on else 'OFF'}] Context Compression")
    if cc_on:
        print(f"         Ratio:   {cc.get('compression_ratio', 16)}")
        print(f"         Heads:   {cc.get('num_heads', 8)}")

    # Self-Play / Self-Improvement
    sp = config.get("self_play", {})
    sp_on = _is_truthy(sp.get("enabled", False))
    print(f"\n  [{'ON ' if sp_on else 'OFF'}] Self-Play / Self-Improvement")
    if sp_on:
        print(f"         Games:      {sp.get('n_games', 100)}")
        print(f"         Refine:     {sp.get('refinement_iterations', 3)} iterations")
        print(f"         Threshold:  {sp.get('quality_threshold', 0.85)}")

    # Benchmark harness
    bench_on = _is_truthy(benchmark_cfg.get("enabled", False))
    print(f"\n  [{'ON ' if bench_on else 'OFF'}] Benchmark Harness")
    if bench_on:
        print(f"         Max prompts: {benchmark_cfg.get('max_prompts', 8)}")
        print(f"         Search quality: {_is_truthy(benchmark_cfg.get('search_quality', {}).get('enabled', True))}")
        print(f"         Hidden CoT fidelity: {_is_truthy(benchmark_cfg.get('hidden_cot_fidelity', {}).get('enabled', False))}")
        print(f"         Tree-GRPO bench: {_is_truthy(benchmark_cfg.get('tree_grpo', {}).get('enabled', False))}")

    # Hardware
    hw = config.get("hardware", {})
    print(f"\n  Hardware:")
    print(f"         Device:    {config.get('training', {}).get('device', 'cpu')}")
    print(f"         Memory:    {hw.get('max_memory', 'not set')}")
    print(f"         Workers:   {hw.get('num_workers', 'default')}")
    print(f"         Threads:   {hw.get('torch_num_threads', 'default')}")

    # LoRA
    lora = config.get("lora", {})
    lora_on = _is_truthy(lora.get("use_lora", False))
    print(f"\n  [{'ON ' if lora_on else 'OFF'}] LoRA")
    if lora_on:
        print(f"         Rank:    {lora.get('r', 16)}")
        print(f"         Alpha:   {lora.get('alpha', 32)}")
        print(f"         QLoRA:   {lora.get('use_qlora', False)}")

    # Validation
    val = config.get("validation", {})
    print(f"\n  Validation:")
    print(f"         Capability tests: {val.get('run_capability_tests', True)}")
    print(f"         Auto-rollback:    {val.get('rollback_on_regression', True)}")
    print(f"         Threshold:        {val.get('regression_threshold', 0.15)}")

    # Post-training merge
    merge_cfg = config.get("merge", {})
    merge_on = _is_truthy(merge_cfg.get("enabled", False))
    print(f"\n  [{'ON ' if merge_on else 'OFF'}] Post-Training Intelligent Merge")
    if merge_on:
        print(f"         Method:     {merge_cfg.get('method', 'dare')}")
        print(f"         Density:    {merge_cfg.get('density', 0.7)}")
        print(f"         Weights:    {merge_cfg.get('weights', [0.7])}")
        print(f"         Base model: {merge_cfg.get('base_model_path', config['model']['name'])}")
        print(f"         Output:     {merge_cfg.get('output_dir', 'checkpoints/chiron_merged')}")
        print(f"         Conflict analytics: {_is_truthy(merge_cfg.get('conflict_report', True))}")
        if merge_cfg.get("layer_density_map"):
            print(f"         Per-layer density overrides: {len(merge_cfg['layer_density_map'])} rules")
        if sessions_enabled:
            print("         NOTE: merge runs after composed sessions complete (session=all only).")
        else:
            print("         NOTE: merge runs after training completes.")

    print("\n" + "=" * 72)
    print("  DRY RUN COMPLETE - no models loaded, no training executed")
    print("=" * 72 + "\n")


# ============================================================================
# POST-TRAINING INTELLIGENT MERGE
# ============================================================================

def run_post_training_merge(
    *,
    config: Dict[str, Any],
    trained_model: Any,
    output_base: Path,
) -> Dict[str, Any]:
    """
    Intelligently merge the RL-trained delta into the original base model.

    Uses model_merging.py methods (DARE, TIES, SLERP, task_arithmetic, etc.)
    instead of a naive overwrite. Produces a provenance-tracked merge artifact
    with SHA256 manifest and optional per-layer conflict analytics.

    Args:
        config: Full YAML config dict.
        trained_model: The RL-trained nn.Module (e.g. orchestrator.policy_model.model).
        output_base: Base output directory for the pipeline run.

    Returns:
        Merge result dict with paths, SHA256, and conflict report.
    """
    merge_cfg = config.get("merge", {})
    if not _is_truthy(merge_cfg.get("enabled", False)):
        return {"status": "skipped", "reason": "merge.enabled is false"}

    method = merge_cfg.get("method", "dare")
    density = float(merge_cfg.get("density", 0.7))
    weights = merge_cfg.get("weights")
    if isinstance(weights, list):
        weights = [float(w) for w in weights]
    seed = int(merge_cfg.get("seed", 42))
    layer_density_map = merge_cfg.get("layer_density_map")

    merge_output = merge_cfg.get("output_dir", "checkpoints/chiron_merged")
    merge_output_path = _resolve_output_path_from_config(config, merge_output)

    base_model_path = merge_cfg.get("base_model_path") or config["model"]["name"]
    base_model_resolved = _resolve_path_or_id_from_config(config, base_model_path) or base_model_path

    logger.info("=" * 72)
    logger.info("POST-TRAINING INTELLIGENT MERGE")
    logger.info("=" * 72)
    logger.info(f"Method: {method}")
    logger.info(f"Density: {density}")
    logger.info(f"Base model: {base_model_resolved}")
    logger.info(f"Output: {merge_output_path}")

    # Load the original base model for delta computation
    from rlhf import PolicyModel
    base_policy = PolicyModel(
        str(base_model_resolved),
        use_gradient_checkpointing=False,
    )
    base_nn = base_policy.model
    trained_nn = trained_model

    # Build MergeConfig
    mc = MergeConfig(
        method=method,
        weights=weights,
        density=density,
        seed=seed,
        layer_density_map=layer_density_map,
    )

    merger = ModelMerger(mc)

    # Optional conflict analytics
    conflict_report = None
    if _is_truthy(merge_cfg.get("conflict_report", True)):
        logger.info("Computing per-layer conflict analytics...")
        base_state = base_nn.state_dict()
        ft_states = [trained_nn.state_dict()]
        aligned_keys = sorted(set(base_state.keys()) & set(ft_states[0].keys()))
        conflict_report = merger.compute_conflict_report(base_state, ft_states, aligned_keys)

        # Log top conflicting layers
        if conflict_report:
            sorted_conflicts = sorted(
                conflict_report.items(),
                key=lambda kv: kv[1].get("drift_ratio", 0),
                reverse=True,
            )
            top_n = min(10, len(sorted_conflicts))
            logger.info("Top %d layers by drift ratio:", top_n)
            for key, stats in sorted_conflicts[:top_n]:
                logger.info(
                    "  %s — drift=%.4f, sign_disagree=%.4f",
                    key, stats.get("drift_ratio", 0), stats.get("sign_disagreement_ratio", 0),
                )

    # Run the merge
    logger.info("Running %s merge...", method)
    merged_nn = merger.merge(
        base_model=base_nn,
        fine_tuned_models=[trained_nn],
        config=mc,
    )

    # Save with provenance manifest
    input_state_dicts = [base_nn.state_dict(), trained_nn.state_dict()]
    artifact = save_merge_artifact(
        merged_model=merged_nn,
        output_dir=str(merge_output_path),
        metadata={
            "pipeline": "chiron",
            "base_model": str(base_model_resolved),
            "method": method,
            "density": density,
            "weights": weights,
            "seed": seed,
            "layer_density_map": layer_density_map,
        },
        input_state_dicts=input_state_dicts,
        merge_config=mc,
        conflict_report=conflict_report,
    )

    logger.info("Merge artifact saved: %s", artifact.get("model_path"))
    logger.info("Merge manifest: %s", artifact.get("manifest_path"))
    logger.info("SHA256 (merged): %s", artifact.get("sha256_merged", "")[:24] + "...")

    # Release base model memory
    del base_policy, base_nn, merged_nn
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "status": "ok",
        "method": method,
        "density": density,
        **artifact,
    }


# ============================================================================
# MAIN PIPELINE EXECUTION
# ============================================================================

def run(
    config_path: str,
    method_override: Optional[str] = None,
    dry_run: bool = False,
    force_sft: bool = False,
    session: str = "all",
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Execute the RLHF pipeline as defined by the YAML config.

    Args:
        config_path: Path to the YAML config file.
        method_override: Optional method override.
        dry_run: If True, print the execution plan and exit.
        force_sft: If True, run SFT even if disabled in YAML.
        session: Session selector for split/reload training ("1", "2", or "all").

    Returns:
        Run manifest dict, or None for dry run.
    """
    config = config or load_config(config_path)
    pipeline = config["pipeline"]

    method = method_override or pipeline["policy_optimization"].get("method", "dpo")
    method = method.lower()
    if method not in VALID_METHODS:
        raise ValueError(f"Invalid method: '{method}'. Valid: {sorted(VALID_METHODS)}")

    validate_runtime_contracts(config, method, session=session)

    sessions_cfg = get_sessions_config(config)
    sessions_enabled = _is_truthy(sessions_cfg.get("enabled", False))
    split_mode = str(sessions_cfg.get("split_mode", "single")).lower()

    if dry_run:
        print_execution_plan(config, method_override=method)
        return None

    if session == "all" and sessions_enabled and split_mode != "single":
        logger.info("=" * 72)
        logger.info("COMPOSED SESSION RUN: SESSION 1 -> SESSION 2")
        logger.info("=" * 72)
        session_1_results = run(
            config_path=config_path,
            method_override=method,
            dry_run=False,
            force_sft=force_sft,
            session="1",
            config=config,
        )

        # --- Inter-session cleanup: release all GPU/CPU memory ---
        gc_between = _is_truthy(sessions_cfg.get("gc_between", True))
        if gc_between:
            logger.info("=" * 72)
            logger.info("INTER-SESSION CLEANUP")
            logger.info("=" * 72)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info(
                    "CUDA memory released: allocated=%.1f MB",
                    torch.cuda.memory_allocated() / (1024 ** 2),
                )
            logger.info(
                "gc.collect() completed. Python garbage collector freed unreachable objects."
            )

        session_2_results = run(
            config_path=config_path,
            method_override=method,
            dry_run=False,
            force_sft=False,
            session="2",
            config=config,
        )
        combined = {
            "method": method,
            "session": "all",
            "composed_sessions": ["1", "2"],
            "session_1": session_1_results,
            "session_2": session_2_results,
            "config_path": str(config_path),
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        base_output_root = _resolve_output_path_from_config(
            config,
            config.get("training", {}).get("output_dir", "./checkpoints/pipeline_output"),
        )
        combined_manifest = (base_output_root / "combined_session_manifest.json").resolve()
        combined_manifest.parent.mkdir(parents=True, exist_ok=True)
        combined_manifest.write_text(json.dumps(combined, indent=2, default=str), encoding="utf-8")
        combined["combined_manifest"] = str(combined_manifest)

        # --- Post-training intelligent merge ---
        merge_cfg = config.get("merge", {})
        if _is_truthy(merge_cfg.get("enabled", False)):
            # Reload session 2's trained model for the merge
            session_2_output = _session_output_dir(base_output_root, config, "2")
            session_2_policy_path = session_2_output / "final_models" / "policy_model"
            if session_2_policy_path.exists():
                from rlhf import PolicyModel as _MergePM
                _trained_pm = _MergePM.from_pretrained(str(session_2_policy_path))
                merge_result = run_post_training_merge(
                    config=config,
                    trained_model=_trained_pm.model,
                    output_base=base_output_root,
                )
                combined["merge"] = merge_result
                del _trained_pm
                gc.collect()
            else:
                logger.warning(
                    "Post-training merge enabled but session 2 policy model not found at %s. "
                    "Merge skipped.",
                    session_2_policy_path,
                )
                combined["merge"] = {
                    "status": "skipped",
                    "reason": f"session_2 policy not found: {session_2_policy_path}",
                }

            # Re-write combined manifest with merge results
            combined_manifest.write_text(
                json.dumps(combined, indent=2, default=str), encoding="utf-8"
            )

        return combined

    sft_enabled = force_sft or _is_truthy(pipeline.get("sft", {}).get("enabled", False))
    prm_pipeline = get_prm_pipeline_config(pipeline)
    prm_enabled = _is_truthy(prm_pipeline.get("enabled", False))
    prm_checkpoint = prm_pipeline.get("load_from_checkpoint")
    prm_step_detection = prm_pipeline.get("step_detection", "newline")
    prm_weight = float(prm_pipeline.get("process_reward_weight", 0.0))
    po_enabled = _is_truthy(pipeline.get("policy_optimization", {}).get("enabled", True))

    if sessions_enabled and session == "2":
        if force_sft:
            logger.warning("force_sft ignored for session 2; reloading session-1 artifacts instead.")
        sft_enabled = False
        prm_enabled = False
        prm_checkpoint = None

    cc = config.get("context_compression", {})
    cc_enabled = _is_truthy(cc.get("enabled", True))

    sp = config.get("self_play", {})
    sp_enabled = _is_truthy(sp.get("enabled", True))

    val = config.get("validation", {})
    regression_threshold = val.get("regression_threshold", 0.15)

    model_name = _resolve_path_or_id_from_config(config, config["model"]["name"]) or config["model"]["name"]
    output_base_root = _resolve_output_path_from_config(
        config,
        config.get("training", {}).get("output_dir", "./checkpoints/pipeline_output"),
    )
    output_base = _session_output_dir(output_base_root, config, session)

    telemetry_runtime = _init_runtime_telemetry(
        config,
        output_base,
        method=method,
        session=session,
    )
    telemetry_recorder = telemetry_runtime["recorder"]

    device_manager = build_device_manager(config)
    _record_runtime_activation(
        telemetry_recorder,
        "device_manager",
        device=str(device_manager.device),
        dtype=str(device_manager.dtype),
        use_amp=bool(device_manager.use_amp),
    )
    telemetry_recorder.record_memory_snapshot("device_manager_online")

    logger.info("=" * 72)
    logger.info("INITIALIZING RLHF ORCHESTRATOR")
    logger.info("=" * 72)
    logger.info(f"Base model: {model_name}")
    logger.info(f"Method: {method.upper()}")
    logger.info(f"Session: {session}")
    logger.info(f"Output dir: {output_base}")
    logger.info(f"Context compression: {cc_enabled}")
    logger.info(f"Self-improvement: {sp_enabled}")

    orchestrator = RLHFOrchestrator(
        base_model=model_name,
        output_dir=str(output_base),
        use_self_improvement=sp_enabled,
        regression_threshold=regression_threshold,
        device_manager=device_manager,
        use_context_compressor=cc_enabled,
        context_compression_ratio=cc.get("compression_ratio", 16),
        context_compression_heads=cc.get("num_heads", 8),
        context_compression_dropout=cc.get("dropout", 0.1),
    )
    _record_runtime_activation(
        telemetry_recorder,
        "orchestrator",
        base_model=model_name,
        output_dir=str(output_base),
        context_compression=cc_enabled,
        self_play=sp_enabled,
    )
    telemetry_recorder.record_memory_snapshot("orchestrator_online")

    start_time = time.time()
    results = {
        "method": method,
        "session": session,
        "stages_executed": [],
        "config_path": str(config_path),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "output_dir": str(output_base),
        "telemetry": {
            "json_path": telemetry_runtime["json_path"],
            "tensorboard_dir": telemetry_runtime["tensorboard_dir"],
        },
        "runtime_bootstrap": {
            "model_name": model_name,
            "context_compression": cc_enabled,
            "self_play": sp_enabled,
            "session_mode": session,
        },
    }

    session_reload_applied = False
    if sessions_enabled and session == "2":
        reload_dir = _session_resume_dir(config, output_base_root)
        if not reload_dir.exists():
            raise FileNotFoundError(
                f"Session 2 requested, but resume directory was not found: {reload_dir}"
            )
        logger.info("Reloading session-1 artifacts from %s", reload_dir)
        orchestrator.load_models(str(reload_dir))
        session_reload_applied = True
        results["session_reload_dir"] = str(reload_dir)
        results["stages_executed"].append("session_reload")
        _record_runtime_activation(telemetry_recorder, "session_reload", source=str(reload_dir))
        telemetry_recorder.record_memory_snapshot("session_reload")

    # ==================================================================
    # STAGE 1: SFT
    # ==================================================================
    if sft_enabled:
        logger.info("=" * 72)
        logger.info("STAGE 1: SUPERVISED FINE-TUNING")
        logger.info("=" * 72)

        sft_config = build_sft_config(config)
        sft_data = load_sft_data(
            config,
            orchestrator.tokenizer,
            max_length=config.get("sft", {}).get("max_seq_length", 2048),
        )

        orchestrator.run_sft(sft_data, sft_config)
        results["stages_executed"].append("sft")
        _record_runtime_activation(telemetry_recorder, "policy_surface", source="sft")
        telemetry_recorder.record_memory_snapshot("post_sft")
        logger.info("SFT stage complete")
    elif session_reload_applied and orchestrator.policy_model is not None:
        logger.info("SFT skipped — session reload already supplied the policy model")
        results["stages_executed"].append("sft_skip_session_reload")
        _record_runtime_activation(telemetry_recorder, "policy_surface", source="session_reload")
    else:
        logger.info("SFT skipped — loading existing model as policy base")
        logger.info(f"Base model path: {model_name}")

        from rlhf import PolicyModel
        orchestrator.policy_model = PolicyModel(
            model_name,
            use_gradient_checkpointing=config.get("training", {}).get(
                "gradient_checkpointing", True
            ),
        )
        orchestrator._ensure_context_compressor()
        results["stages_executed"].append("sft_skip_load")
        _record_runtime_activation(telemetry_recorder, "policy_surface", source="existing_checkpoint", checkpoint=model_name)
        telemetry_recorder.record_memory_snapshot("policy_surface_online")
        logger.info("Policy model loaded from existing checkpoint")

    # Stage 0.5: torch.compile (YAML: inference_optimizations.compile_model)
    _inf_top = config.get("inference_optimizations", {})
    _compile_cfg = _inf_top.get("compile_model", {})
    _compile_mode = _compile_cfg.get("mode", "reduce-overhead")
    if _compile_cfg.get("enabled", False):
        logger.info(f"torch.compile policy model (mode={_compile_mode})")
        orchestrator.policy_model.model = compile_model(
            orchestrator.policy_model.model, mode=_compile_mode
        )
        logger.info("torch.compile applied to policy model")

    # Optional artifact reuse from previous runs (RM/PRM/context/self-improvement).
    artifact_reuse_summary = maybe_load_artifact_bundle(
        orchestrator=orchestrator,
        config=config,
        device_manager=device_manager,
    )
    results["artifact_reuse"] = artifact_reuse_summary
    if artifact_reuse_summary.get("loaded"):
        logger.info(
            f"Artifact reuse loaded components: {artifact_reuse_summary['loaded']}"
        )
        results["stages_executed"].append("artifact_reuse")
        _record_runtime_activation(telemetry_recorder, "artifact_reuse", loaded=list(artifact_reuse_summary.get("loaded", [])))
        telemetry_recorder.record_memory_snapshot("artifact_reuse")

    # ==================================================================
    # STAGE 2: PROCESS REWARD MODEL
    # ==================================================================
    if prm_enabled:
        if prm_checkpoint is not None:
            logger.info("=" * 72)
            logger.info("STAGE 2: LOADING PROCESS REWARD MODEL FROM CHECKPOINT")
            logger.info("=" * 72)
            logger.info(f"Checkpoint: {prm_checkpoint}")

            from rlhf import ProcessRewardModel

            checkpoint_path = _resolve_required_path_from_config(
                config,
                prm_checkpoint,
                label="Process reward model checkpoint",
            )

            orchestrator.process_reward_model = ProcessRewardModel.from_pretrained(
                str(checkpoint_path)
            )
            results["stages_executed"].append("prm_load")
            _record_runtime_activation(telemetry_recorder, "process_reward_model", source="checkpoint", checkpoint=str(checkpoint_path))
            telemetry_recorder.record_memory_snapshot("prm_online")
            logger.info("Process reward model loaded from checkpoint")
        elif artifact_reuse_summary.get("skip_prm_training", False):
            logger.info("=" * 72)
            logger.info("STAGE 2: PROCESS REWARD MODEL REUSE")
            logger.info("=" * 72)
            logger.info(
                "Skipping PRM training because artifact_reuse loaded an existing "
                "process reward model and skip_prm_training_if_loaded=true."
            )
            results["stages_executed"].append("prm_reuse")
            _record_runtime_activation(telemetry_recorder, "process_reward_model", source="artifact_reuse")
            telemetry_recorder.record_memory_snapshot("prm_reuse")
        else:
            logger.info("=" * 72)
            logger.info("STAGE 2: PROCESS REWARD MODEL TRAINING")
            logger.info("=" * 72)

            prm_config = build_process_reward_model_config(config)
            prm_data = load_prm_data(
                config,
                orchestrator.tokenizer,
                max_length=config.get("process_reward_model", {}).get("max_seq_length", 2048),
            )

            orchestrator.run_process_reward_model_training(
                prm_data,
                prm_config,
                step_detection=prm_step_detection,
                process_reward_weight=prm_weight,
            )
            results["stages_executed"].append("prm_train")
            _record_runtime_activation(telemetry_recorder, "process_reward_model", source="training")
            telemetry_recorder.record_memory_snapshot("post_prm_train")
            logger.info("Process reward model training complete")

    # Optional: compile PRM too
    if _compile_cfg.get("apply_to_prm", False) and orchestrator.process_reward_model is not None:
        logger.info(f"torch.compile process_reward_model (mode={_compile_mode})")
        orchestrator.process_reward_model = compile_model(
            orchestrator.process_reward_model, mode=_compile_mode
        )

    if method in STRICT_PRM_METHODS and orchestrator.process_reward_model is None:
        raise ValueError(
            f"{method.upper()} requires a live ProcessRewardModel. "
            "Refusing legacy reward_model fallback for this lane."
        )

    if not prm_enabled and method in METHODS_REQUIRING_REWARD:
        if orchestrator.process_reward_model is not None:
            logger.info(
                f"{method.upper()} reward path: using previously loaded process_reward_model."
            )
        elif method in STRICT_PRM_METHODS:
            raise ValueError(
                f"{method.upper()} requires a process_reward_model and none is available."
            )
        elif orchestrator.reward_models:
            logger.info(
                f"{method.upper()} reward path: process_reward_model disabled, "
                "falling back to loaded reward model."
            )
        else:
            logger.warning(
                f"{method.upper()} requires a reward signal but neither process_reward_model "
                "nor reward_model is available."
            )

    # ==================================================================
    # STAGE 3: POLICY OPTIMIZATION
    # ==================================================================
    method_config = None
    training_data = None
    policy_adapter = None
    generator_op = None
    cot_generator = None
    prm_adapter = None

    if po_enabled:
        logger.info("=" * 72)
        logger.info(f"STAGE 3: POLICY OPTIMIZATION ({method.upper()})")
        logger.info("=" * 72)

        method_config = build_method_config(method, config)
        method_data_cfg = _active_method_section(config, method)

        generator_op = None
        cot_generator = None
        prm_adapter = None
        policy_adapter = None

        # ==================================================================
        # INJECT ADVANCED INFERENCE OPTIMIZATIONS AS CAPABILITY FORMATION
        # ==================================================================
        # We wrap the policy model's generate method so that all RL training
        # trajectories, self-play rollouts, and testing loops automatically
        # use test-time compute (MCTS/BestOfN/A*/CoT). This bakes the search
        # capability into the eventual model weights via DPO/GRPO.

        try:
            inf_config = config.get("inference_optimizations", {})
            if not _is_truthy(inf_config.get("enabled", True)):
                raise RuntimeError("inference_optimizations disabled — skipping")

            logger.info("Wiring inference optimizations...")
            _record_runtime_activation(telemetry_recorder, "inference_bootstrap", requested=True)
            original_policy_generate = orchestrator.policy_model.generate

            # Protocol adapters (already imported from inference_protocols)
            policy_adapter = PolicyAdapter(
                orchestrator.policy_model,
                device=device_manager.device,
            )
            if orchestrator.process_reward_model is not None:
                prm_adapter = ProcessRewardModelAdapter(
                    orchestrator.process_reward_model,
                    orchestrator.tokenizer,
                    device_manager.device,
                    max_length=int(config.get("process_reward_model", {}).get("max_seq_length", 2048)),
                    process_weight=prm_weight,
                )

            # --- PagedKVCache ---
            kv_cache = None
            kv_cfg = inf_config.get("kv_cache", {})
            if kv_cfg.get("enabled", True):
                base_model_cfg = orchestrator.policy_model.model.config
                kv_cache = PagedKVCache(
                    num_layers=base_model_cfg.num_hidden_layers,
                    num_heads=base_model_cfg.num_attention_heads,
                    head_dim=base_model_cfg.hidden_size // base_model_cfg.num_attention_heads,
                    page_size=kv_cfg.get("page_size", 16),
                    max_pages=kv_cfg.get("max_pages", 128),
                    max_prefix_entries=kv_cfg.get("max_prefix_entries", 32),
                )
                logger.info(
                    f"PagedKVCache: layers={base_model_cfg.num_hidden_layers}, "
                    f"heads={base_model_cfg.num_attention_heads}, "
                    f"page_size={kv_cfg.get('page_size', 16)}, max_pages={kv_cfg.get('max_pages', 128)}"
                )

            # --- SpeculativeDecoder (optional, independent of search strategy) ---
            speculative_decoder = None
            spec_cfg_d = inf_config.get("speculative_decoding", {})
            if _is_truthy(spec_cfg_d.get("enabled", False)) and spec_cfg_d.get("draft_model_path"):
                from rlhf import PolicyModel as _PM
                draft_model_path = _resolve_path_or_id_from_config(
                    config,
                    spec_cfg_d["draft_model_path"],
                )
                _draft_pm = _PM(draft_model_path)
                spec_cfg = SpeculativeDecoderConfig(
                    gamma=spec_cfg_d.get("gamma", 4),
                    temperature=spec_cfg_d.get("temperature", 1.0),
                    adapt_gamma=spec_cfg_d.get("adapt_gamma", True),
                    gamma_min=spec_cfg_d.get("gamma_min", 2),
                    gamma_max=spec_cfg_d.get("gamma_max", 8),
                    adapt_window=spec_cfg_d.get("adapt_window", 50),
                )
                speculative_decoder = SpeculativeDecoder(
                    target_model=orchestrator.policy_model.model,
                    draft_model=_draft_pm.model,
                    config=spec_cfg,
                )
                logger.info("SpeculativeDecoder active (Chen et al. 2023 accept/resample)")

            # --- Primary search generator ---
            search_strategy = inf_config.get("search_strategy", "mcts")
            if search_strategy == "astar" and _is_truthy(inf_config.get("astar", {}).get("enabled", False)):
                astar_d = inf_config.get("astar", {})
                astar_cfg = AStarConfig(
                    max_nodes=astar_d.get("max_nodes", 50),
                    max_depth=astar_d.get("max_depth", 32),
                    n_actions=astar_d.get("n_actions", 4),
                    heuristic_weight=astar_d.get("heuristic_weight", 1.0),
                    temperature=astar_d.get("temperature", 0.8),
                    use_value_heuristic=astar_d.get("use_value_heuristic", True),
                )
                generator_op = AStarGenerator(
                    policy_model=policy_adapter,
                    tokenizer=orchestrator.tokenizer,
                    config=astar_cfg,
                    prm=prm_adapter,
                    kv_cache=kv_cache,
                )
                logger.info(
                    "Capability generator: AStarGenerator "
                    f"({'PRM-enabled' if prm_adapter is not None else 'no PRM'})"
                )
            else:
                mcts_d = inf_config.get("mcts", {})
                mcts_cfg = MCTSConfig(
                    n_simulations=mcts_d.get("n_simulations", 4),
                    max_rollout_depth=mcts_d.get("max_rollout_depth", 32),
                )
                generator_op = MCTSGenerator(
                    policy_model=policy_adapter,
                    value_model=prm_adapter,
                    tokenizer=orchestrator.tokenizer,
                    config=mcts_cfg,
                    kv_cache=kv_cache,
                )
                logger.info(
                    "Capability generator: MCTSGenerator "
                    f"({'PRM value heuristic' if prm_adapter is not None else 'reward_fn rollout'})"
                )

            # --- BestOfN (fallback when no PRM, or explicit strategy) ---
            bon_d = inf_config.get("best_of_n", {})
            bon_cfg = BestOfNConfig(
                n_samples=bon_d.get("n_samples", 4),
                temperature=bon_d.get("temperature", 1.0),
                top_p=bon_d.get("top_p", 0.95),
                reward_aggregation=bon_d.get("reward_aggregation", "mean"),
                use_diversity_bonus=bon_d.get("use_diversity_bonus", True),
                diversity_weight=bon_d.get("diversity_weight", 0.1),
                step_rerank=bon_d.get("step_rerank", False),
                step_prm=prm_adapter if bon_d.get("step_rerank", False) else None,
                prm_process_weight=bon_d.get("prm_process_weight", 0.0),
            )
            bon_sampler = BestOfNSampler(
                policy_model=policy_adapter,
                reward_model=prm_adapter,          # correct param: reward_model (not reward_scorer)
                config=bon_cfg,                    # correct param: config=BestOfNConfig (not n_candidates)
                tokenizer=orchestrator.tokenizer,
            )
            if generator_op is None:
                generator_op = bon_sampler
                logger.info("Capability generator: BestOfNSampler (no PRM)")

            # --- ChainOfThoughtGenerator (wraps chosen generator) ---
            cot_cfg_d = inf_config.get("chain_of_thought", {})
            if _is_truthy(cot_cfg_d.get("enabled", False)):
                cot_cfg = ChainOfThoughtConfig(
                    max_thinking_tokens=cot_cfg_d.get("max_thinking_tokens", 128),
                    max_answer_tokens=cot_cfg_d.get("max_answer_tokens", 256),
                    think_start_tag=cot_cfg_d.get("think_start_tag", "<think>"),
                    think_end_tag=cot_cfg_d.get("think_end_tag", "</think>"),
                    temperature=cot_cfg_d.get("temperature", 0.8),
                    strip_thinking=cot_cfg_d.get("strip_thinking", True),
                    prm_scorer=prm_adapter,
                )
                cot_generator = ChainOfThoughtGenerator(
                    policy_model=policy_adapter,
                    tokenizer=orchestrator.tokenizer,
                    config=cot_cfg,
                )
                logger.info("ChainOfThoughtGenerator active: full_sequence trains reasoning tokens; strip_thinking only affects the servable output surface.")

            # --- VerifiableRewards (GRPO/PPO reward_fn) ---
            vr_cfg = inf_config.get("verifiable_rewards", {})
            if vr_cfg.get("enabled", False) and method in METHODS_REQUIRING_REWARD:
                _verifiers = []
                if vr_cfg.get("format_enabled", False):
                    _verifiers.append(VerifiableRewardFactory.format_verifier(
                        required_tags=vr_cfg.get("format_required_tags", []),
                        forbidden_patterns=vr_cfg.get("format_forbidden_patterns", []),
                        min_steps=vr_cfg.get("format_min_steps", 0),
                    ))
                if vr_cfg.get("math_enabled", False):
                    logger.info("VerifiableRewardFactory.math_verifier available (bind gt at dataset time)")
                if vr_cfg.get("code_enabled", False):
                    logger.info("VerifiableRewardFactory.code_verifier available (bind test_cases at dataset time)")
                if _verifiers and hasattr(method_config, "reward_fn"):
                    method_config.reward_fn = _verifiers[0]
                    logger.info(f"VerifiableRewardFactory: {len(_verifiers)} verifier(s) wired to reward_fn")

            # --- TreeRolloutCollector (GRPO search-to-training bridge) ---
            rc_cfg = inf_config.get("rollout_collector", {})
            if rc_cfg.get("enabled", False) and method in {"grpo", "tree_grpo"}:
                _reward_fn = (
                    (lambda text: prm_adapter.score(text))
                    if prm_adapter is not None
                    else (lambda text: 0.0)
                )
                if isinstance(generator_op, MCTSGenerator):
                    rollout_collector = TreeRolloutCollector(
                        mcts_generator=generator_op,
                        reward_fn=_reward_fn,
                        min_reward_threshold=rc_cfg.get("min_reward_threshold", 0.0),
                    )
                    logger.info(
                        f"TreeRolloutCollector active: n_samples={rc_cfg.get('n_samples_per_prompt', 4)}, "
                        f"max_length={rc_cfg.get('max_length', 256)}"
                    )
                else:
                    logger.warning(
                        "rollout_collector.enabled requested, but current search generator is "
                        f"{type(generator_op).__name__}; TreeRolloutCollector currently requires MCTSGenerator."
                    )

            # --- Build capability_generate ---
            _use_search = isinstance(generator_op, (MCTSGenerator, AStarGenerator))

            def capability_generate(input_ids, **kwargs):
                from types import SimpleNamespace

                batch_texts = orchestrator.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                new_all_ids = []

                for prompt in batch_texts:
                    text = ""

                    if cot_generator is not None:
                        res = cot_generator.generate(prompt)
                        text = res.get("servable_sequence", "") or res.get("answer", "")
                    elif _use_search:
                        res = generator_op.generate(prompt)
                        text = res.get("text", "")
                    else:
                        res = bon_sampler.generate(
                            prompt,
                            tokenizer=orchestrator.tokenizer,
                            max_new_tokens=kwargs.get("max_new_tokens", 256),
                        )
                        text = res.get("best", "")

                    if not text:
                        return original_policy_generate(input_ids=input_ids, **kwargs)

                    new_ids = (
                        orchestrator.tokenizer(text, return_tensors="pt")
                        .input_ids
                        .to(input_ids.device)
                    )
                    new_all_ids.append(new_ids[0])

                max_len = max(t.size(0) for t in new_all_ids)
                padded = torch.full(
                    (len(new_all_ids), max_len),
                    orchestrator.tokenizer.pad_token_id,
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
                for i, t in enumerate(new_all_ids):
                    padded[i, : t.size(0)] = t

                if kwargs.get("return_dict_in_generate", False):
                    return SimpleNamespace(sequences=padded)
                return padded

            if method not in {"tree_grpo", "hidden_cot_sft"}:
                orchestrator.policy_model.generate = capability_generate
                logger.info(
                    "All inference optimizations wired successfully. capability_generate wrapper "
                    "applied to PolicyModel.generate without touching the raw HF model.generate surface."
                )
                _record_runtime_activation(
                    telemetry_recorder,
                    "inference_capability_surface",
                    search_generator=type(generator_op).__name__ if generator_op is not None else None,
                    cot_enabled=bool(cot_generator is not None),
                    prm_adapter=bool(prm_adapter is not None),
                    patched_policy_generate=True,
                )
                telemetry_recorder.record_memory_snapshot("inference_surface_online")
            else:
                logger.info(
                    "All inference optimizations wired successfully. %s uses canon rollout / hidden-CoT "
                    "collection paths, so PolicyModel.generate was left unpatched.",
                    method,
                )
                _record_runtime_activation(
                    telemetry_recorder,
                    "inference_capability_surface",
                    search_generator=type(generator_op).__name__ if generator_op is not None else None,
                    cot_enabled=bool(cot_generator is not None),
                    prm_adapter=bool(prm_adapter is not None),
                    patched_policy_generate=False,
                )
                telemetry_recorder.record_memory_snapshot("inference_surface_online")

        except Exception as e:
            if method in {"grpo", "tree_grpo"}:
                raise RuntimeError(
                    "Inference optimization wiring failed for a Chiron RL lane. "
                    "Refusing silent fallback to plain generation."
                ) from e
            logger.warning(
                f"Could not wire inference optimizations: {e}. Falling back to default generation."
            )
            _record_runtime_activation(telemetry_recorder, "inference_capability_surface", status="degraded", error=str(e))

        training_data = load_training_data(
            method,
            config,
            orchestrator.tokenizer,
            max_length=method_data_cfg.get("max_seq_length", 2048),
        )
        if isinstance(training_data, list):
            _record_runtime_activation(
                telemetry_recorder,
                "training_data_loaded",
                records=len(training_data),
                data_source=method_data_cfg.get("data_source", method_data_cfg.get("train_file", "in_memory")),
            )
        else:
            _record_runtime_activation(
                telemetry_recorder,
                "training_data_loaded",
                records=None,
                data_type=type(training_data).__name__,
            )
        telemetry_recorder.record_memory_snapshot("training_data_loaded")

        if sessions_enabled and session in {"1", "2"}:
            if not isinstance(training_data, list):
                raise ValueError(
                    "Session splitting requires list-backed training data for the active lane."
                )
            training_data, session_meta = _select_session_records(training_data, config, session)
            results["session_split"] = session_meta
            _record_runtime_activation(
                telemetry_recorder,
                "session_split",
                mode=session_meta.get("mode", "single"),
                selected=session_meta.get("selected", 0),
                total=session_meta.get("total", 0),
            )
            logger.info(
                "Session %s selected %d/%d training records (%s).",
                session,
                session_meta.get("selected", 0),
                session_meta.get("total", 0),
                session_meta.get("mode", "single"),
            )
            telemetry_recorder.record_memory_snapshot("session_split")

        use_process_reward_model = (
            method in METHODS_REQUIRING_REWARD
            and orchestrator.process_reward_model is not None
        )
        po_kwargs: Dict[str, Any] = {}
        if method == "tree_grpo":
            if isinstance(generator_op, MCTSGenerator):
                po_kwargs["mcts_generator"] = generator_op
            elif isinstance(generator_op, AStarGenerator):
                po_kwargs["astar_generator"] = generator_op
            if cot_generator is not None:
                po_kwargs["cot_generator"] = cot_generator
            if prm_adapter is not None:
                po_kwargs["prm_adapter"] = prm_adapter
        elif method == "hidden_cot_sft" and cot_generator is not None:
            po_kwargs["cot_generator"] = cot_generator

        orchestrator.run_policy_optimization(
            method=method,
            data=training_data,
            config=method_config,
            use_process_reward_model=use_process_reward_model,
            process_reward_weight=prm_weight,
            **po_kwargs,
        )
        results["stages_executed"].append(f"po_{method}")
        logger.info(f"{method.upper()} policy optimization complete")

    # ==================================================================
    # SAVE MODELS
    # ==================================================================
    logger.info("=" * 72)
    logger.info("SAVING FINAL MODELS")
    logger.info("=" * 72)

    orchestrator.save_models()
    results["stages_executed"].append("save")
    _record_runtime_activation(telemetry_recorder, "save_models", output_dir=str(output_base / "final_models"))
    telemetry_recorder.record_memory_snapshot("post_save")

    benchmark_cfg = get_benchmark_config(config)
    should_run_benchmarks = _is_truthy(benchmark_cfg.get("enabled", False)) and not (
        sessions_enabled and session == "1" and not _is_truthy(benchmark_cfg.get("run_after_session_1", False))
    )
    if should_run_benchmarks:
        logger.info("=" * 72)
        logger.info("BENCHMARK HARNESS")
        logger.info("=" * 72)
        benchmark_results = run_benchmark_suite(
            config=config,
            output_base=output_base,
            device_manager=device_manager,
            orchestrator=orchestrator,
            method=method,
            method_config=method_config,
            training_data=training_data,
            policy_adapter=policy_adapter,
            prm_adapter=prm_adapter,
            cot_generator=cot_generator,
            generator_op=generator_op,
            telemetry_recorder=telemetry_recorder,
        )
        results["benchmark_harness"] = benchmark_results
        results["stages_executed"].append("benchmark_harness")
        _record_runtime_activation(telemetry_recorder, "benchmark_harness", report_path=benchmark_results.get("report_path"))
        telemetry_recorder.record_memory_snapshot("post_benchmark")

    elapsed = time.time() - start_time
    results["elapsed_seconds"] = round(elapsed, 2)
    results["elapsed_human"] = f"{elapsed / 60:.1f} minutes"

    # Save run manifest
    manifest_path = Path(output_base) / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"Run manifest saved: {manifest_path}")

    logger.info("=" * 72)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Stages executed: {results['stages_executed']}")
    logger.info(f"Total time: {results['elapsed_human']}")
    logger.info("=" * 72)

    _record_runtime_activation(telemetry_recorder, "pipeline_complete", manifest_path=str(manifest_path))
    telemetry_recorder.record_memory_snapshot("pipeline_complete")
    _finalize_runtime_telemetry(telemetry_recorder, telemetry_runtime["json_path"])

    return results


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(config: Optional[Dict[str, Any]] = None, level: str = "INFO") -> None:
    """
    Configure logging for the pipeline run.

    Sets up both console and optional file handlers based on the
    logging section of the YAML config.
    """
    log_cfg = (config or {}).get("logging", {})
    log_level_str = log_cfg.get("log_level", level).upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Root logger + pipeline logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_fmt = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # File handler (if configured)
    log_file = log_cfg.get("log_file")
    if log_file:
        log_path = _resolve_output_path_from_config(config or {}, log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(console_fmt)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_path}")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main() -> None:
    """
    CLI entry point for the RLHF pipeline orchestrator.

    Parses command-line arguments, loads config, and runs the pipeline.
    """
    parser = argparse.ArgumentParser(
        description="RLHF Pipeline Orchestrator — YAML-driven entrypoint for rlhf.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (parse config, print plan, no model loading):
  .venv/bin/python run_pipeline.py --config config/qwen3_1.7b_model.yaml --dry-run

  # Run default method from config:
  .venv/bin/python run_pipeline.py --config config/qwen3_1.7b_model.yaml

  # Override method:
  .venv/bin/python run_pipeline.py --config config/qwen3_1.7b_model.yaml --method grpo

  # Run only session 1 or session 2 when session splitting is enabled:
  .venv/bin/python run_pipeline.py --config config/qwen3_1.7b_model.yaml --session 1
  .venv/bin/python run_pipeline.py --config config/qwen3_1.7b_model.yaml --session 2

  # Force SFT re-training:
  .venv/bin/python run_pipeline.py --config config/qwen3_1.7b_model.yaml --force-sft
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/qwen3_1.7b_model.yaml",
        help="Path to YAML config file (default: config/qwen3_1.7b_model.yaml)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=sorted(VALID_METHODS),
        help="Override the policy optimization method from config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and print execution plan without loading models",
    )
    parser.add_argument(
        "--force-sft",
        action="store_true",
        help="Force SFT stage even if disabled in config",
    )
    parser.add_argument(
        "--session",
        type=str,
        default="all",
        choices=sorted(SESSION_CHOICES),
        help="Run session 1, session 2, or all sessions when sessions.enabled=true",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging (pre-config load for early messages)
    setup_logging(level=args.log_level)

    # For dry run, we can print immediately
    if args.dry_run:
        config = load_config(args.config)
        print_execution_plan(config, method_override=args.method, session=args.session)
        return

    # Re-setup logging with config-aware settings
    config = load_config(args.config)
    # Clear existing handlers and reconfigure
    logging.getLogger().handlers.clear()
    setup_logging(config=config, level=args.log_level)

    # Run the pipeline
    run(
        config_path=args.config,
        method_override=args.method,
        dry_run=False,
        force_sft=args.force_sft,
        session=args.session,
        config=config,
    )


if __name__ == "__main__":
    main()
