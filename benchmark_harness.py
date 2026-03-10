"""
Benchmark Harness + Budget-Aware Profile Selector

Usage
-----
# Run from CLI:
python benchmark_harness.py --profile tiny_cpu --output bench.json

# Programmatic:
from benchmark_harness import BenchmarkHarness, BudgetProfileSelector, RegressionGate

harness = BenchmarkHarness()
results = harness.run_merge_benchmark(base, ft_models, strategies=["ties","dare"], eval_fn=eval_fn)
ranked  = harness.rank_strategies(results)

selector = BudgetProfileSelector()
merge_cfg, bon_cfg, mcts_cfg, spec_cfg = selector.select(
    {"max_ram_gb": 8, "has_gpu": False, "latency_budget_ms": 500, "quality_priority": True}
)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from model_merging import MergeConfig, ModelMerger
from inference_optimizations import (
    BestOfNConfig,
    BestOfNSampler,
    MCTSConfig,
    SpeculativeDecoderConfig,
)
from telemetry import TelemetryRecorder

try:
    import resource
except Exception:
    resource = None

logger = logging.getLogger("BenchmarkHarness")

# =============================================================================
# RESULT SCHEMA VERSION
# =============================================================================
SCHEMA_VERSION = "1.0"
DEFAULT_BASE_MODEL = os.getenv("RLHF_BASE_MODEL", "Qwen/Qwen3-1.7B")
DEFAULT_ADAPTER_PATH = os.getenv(
    "RLHF_SFT_ADAPTER",
    str(Path(__file__).resolve().parent / "checkpoints" / "checkpoints" / "full_pipeline" / "sft"),
)
DEFAULT_PROMPTS: List[str] = [
    "Explain what reinforcement learning is in simple terms.",
    "Write a Python function that computes Fibonacci numbers iteratively.",
    "What are the practical differences between DPO and PPO for alignment?",
    "Solve: If 3x + 7 = 22, what is x?",
    "Summarize why LoRA is useful for low-memory fine-tuning.",
]


def _get_process_peak_rss_mb() -> float:
    """
    Return process peak RSS in MB when available, otherwise NaN.
    Linux reports ru_maxrss in KiB, macOS reports bytes.
    """
    if resource is None:
        return float("nan")
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if usage <= 0:
        return float("nan")
    if sys.platform == "darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


def _resolve_path_or_id(value: Optional[str]) -> Optional[str]:
    """Resolve local paths to absolute strings while preserving hub IDs."""
    if not value:
        return None
    candidate = Path(value).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return value


def _parse_dtype(dtype_name: str) -> Optional[torch.dtype]:
    """Map CLI dtype string to torch dtype."""
    mapping = {
        "auto": None,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Use one of: {', '.join(mapping.keys())}")
    return mapping[dtype_name]


def _profile_default_bon_strategies(profile_name: str) -> List[str]:
    """Select default Best-of-N strategy set for each budget profile."""
    if profile_name == "tiny_cpu":
        return ["bon_n2", "bon_n4"]
    if profile_name == "balanced_cpu":
        return ["bon_n4", "bon_n8"]
    if profile_name == "gpu_lowlatency":
        return ["bon_n8", "bon_n12", "bon_n16"]
    if profile_name == "gpu_maxquality":
        return ["bon_n8", "bon_n16", "bon_n24"]
    return ["bon_n4", "bon_n8", "bon_n16"]


def _load_prompts(prompts_file: Optional[str], max_prompts: int = 8) -> List[str]:
    """
    Load prompts from txt/json/jsonl file, or return built-in defaults.
    JSON can be:
      - list[str]
      - list[{"prompt": "..."}]
      - {"prompts": [...]}
    """
    prompts: List[str] = []
    if prompts_file:
        path = Path(prompts_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Prompts file not found: {path}")
        suffix = path.suffix.lower()

        if suffix == ".txt":
            prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, str):
                    prompts.append(item)
                elif isinstance(item, dict):
                    val = item.get("prompt") or item.get("text")
                    if isinstance(val, str) and val.strip():
                        prompts.append(val.strip())
        elif suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("prompts", [])
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, str) and item.strip():
                        prompts.append(item.strip())
                    elif isinstance(item, dict):
                        val = item.get("prompt") or item.get("text")
                        if isinstance(val, str) and val.strip():
                            prompts.append(val.strip())
        else:
            raise ValueError(
                f"Unsupported prompts file extension '{suffix}'. Use .txt, .json, or .jsonl"
            )
    else:
        prompts = list(DEFAULT_PROMPTS)

    prompts = prompts[:max(1, max_prompts)]
    if not prompts:
        raise ValueError("No prompts loaded. Provide a non-empty prompts file or omit --prompts-file.")
    return prompts


class HeuristicRewardScorer:
    """
    Lightweight fallback reward scorer for benchmarking when no reward model is provided.
    Emphasizes coherent, non-repetitive completions with moderate length.
    """

    @staticmethod
    def _repeat_ratio(text: str) -> float:
        tokens = text.split()
        if len(tokens) < 3:
            return 0.0
        trigrams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
        unique = len(set(trigrams))
        return 1.0 - (unique / max(1, len(trigrams)))

    def score(self, prompt: str, completion: str = "") -> float:
        text = completion.strip() if completion else prompt.strip()
        if not text:
            return 0.0
        n_tokens = len(text.split())
        length_term = min(n_tokens / 96.0, 1.0)
        repeat_penalty = self._repeat_ratio(text)
        punctuation_bonus = 0.05 if text.endswith((".", "!", "?")) else 0.0
        return float(length_term - 0.5 * repeat_penalty + punctuation_bonus)

    def score_batch(self, texts: List[str]) -> List[float]:
        return [self.score(t) for t in texts]


def load_policy_with_optional_adapter(
    base_model: str,
    adapter_path: Optional[str] = None,
    merged_model_path: Optional[str] = None,
    merge_lora: bool = False,
    device: str = "cpu",
    dtype: str = "float32",
    trust_remote_code: bool = True,
) -> Tuple[nn.Module, Any]:
    """
    Load a policy model + tokenizer from:
      1) merged model path, OR
      2) base model + optional LoRA adapter path.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_merged = _resolve_path_or_id(merged_model_path)
    resolved_adapter = _resolve_path_or_id(adapter_path)
    resolved_base = _resolve_path_or_id(base_model) or base_model
    torch_dtype = _parse_dtype(dtype)
    load_kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }
    if torch_dtype is not None:
        load_kwargs["dtype"] = torch_dtype

    # Prefer tokenizer colocated with adapter/merged artifact if present
    tokenizer_source = resolved_merged or resolved_adapter or resolved_base
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code)

    if resolved_merged:
        logger.info(f"Loading merged policy model from: {resolved_merged}")
        try:
            model = AutoModelForCausalLM.from_pretrained(resolved_merged, **load_kwargs)
        except TypeError:
            # Backward compatibility with older transformers versions
            legacy_kwargs = dict(load_kwargs)
            legacy_kwargs.pop("dtype", None)
            if torch_dtype is not None:
                legacy_kwargs["torch_dtype"] = torch_dtype
            model = AutoModelForCausalLM.from_pretrained(resolved_merged, **legacy_kwargs)
    else:
        logger.info(f"Loading base policy model: {resolved_base}")
        try:
            model = AutoModelForCausalLM.from_pretrained(resolved_base, **load_kwargs)
        except TypeError:
            legacy_kwargs = dict(load_kwargs)
            legacy_kwargs.pop("dtype", None)
            if torch_dtype is not None:
                legacy_kwargs["torch_dtype"] = torch_dtype
            model = AutoModelForCausalLM.from_pretrained(resolved_base, **legacy_kwargs)

        if resolved_adapter:
            try:
                from peft import PeftModel
            except Exception as exc:
                raise RuntimeError(
                    "Adapter path provided but peft is not installed. "
                    "Install `peft` or provide --merged-model-path."
                ) from exc

            logger.info(f"Applying adapter from: {resolved_adapter}")
            model = PeftModel.from_pretrained(model, resolved_adapter)
            if merge_lora:
                logger.info("Merging adapter into base weights (merge_and_unload).")
                model = model.merge_and_unload()
            else:
                logger.info("Keeping adapter as PEFT wrapper (no merge_and_unload).")

    model = model.to(device)
    model.eval()
    return model, tokenizer


# =============================================================================
# BENCHMARK HARNESS
# =============================================================================

class BenchmarkHarness:
    """
    Standardized eval matrix for merge strategies and inference presets.

    All results are dicts keyed by strategy name, containing:
        - wall_time_s       — elapsed wall clock time
        - peak_rss_mb       — peak resident set size (MB)
        - eval_score        — quality proxy (from eval_fn, if provided)
        - conflict_summary  — from conflict_report (merge runs only)
        - latency_p50_s     — p50 latency (inference runs only)
        - latency_p95_s     — p95 latency (inference runs only)
        - tokens_per_sec    — throughput (inference runs only)
        - reward_score      — mean reward across prompts (inference runs only)
    """

    def __init__(self, telemetry: Optional[TelemetryRecorder] = None):
        self.telemetry = telemetry or TelemetryRecorder()

    # ------------------------------------------------------------------
    # Merge benchmark
    # ------------------------------------------------------------------

    def run_merge_benchmark(
        self,
        base_model: nn.Module,
        ft_models: List[nn.Module],
        strategies: List[str],
        eval_fn: Optional[Callable[[nn.Module], float]] = None,
        config_overrides: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Dict]:
        """
        Run each merge strategy and record timing, memory, eval score, conflict report.

        Args:
            base_model:       Pre-trained base model
            ft_models:        List of fine-tuned models
            strategies:       e.g. ["task_arithmetic", "ties", "dare", "slerp"]
            eval_fn:          Optional quality proxy; receives merged model
            config_overrides: Per-strategy MergeConfig kwargs override dict

        Returns:
            Dict[strategy_name -> result_dict]
        """
        results = {}
        config_overrides = config_overrides or {}

        for strategy in strategies:
            logger.info(f"[merge_benchmark] Running strategy: {strategy}")
            try:
                cfg_kwargs = {"method": strategy, **config_overrides.get(strategy, {})}
                # SLERP requires exactly 2 models
                effective_ft = ft_models[:2] if strategy == "slerp" else ft_models
                merge_cfg = MergeConfig(**cfg_kwargs)
                merger = ModelMerger(merge_cfg)

                # Measure time + memory
                tracemalloc.start()
                t0 = time.perf_counter()
                merged = merger.merge(base_model, effective_ft, merge_cfg)
                wall_time = time.perf_counter() - t0
                _, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_python_alloc_mb = peak_mem / 1024 / 1024
                peak_rss_mb = _get_process_peak_rss_mb()
                if peak_rss_mb != peak_rss_mb:  # NaN fallback
                    peak_rss_mb = peak_python_alloc_mb

                # Conflict report
                base_state = base_model.state_dict()
                ft_states  = [m.state_dict() for m in effective_ft]
                aligned    = merger._align_keys(base_state, ft_states)
                conflict   = merger.compute_conflict_report(base_state, ft_states, aligned)

                # Conflict summary scalars
                if conflict:
                    mean_cosine  = sum(v["cosine_conflict"] for v in conflict.values()) / len(conflict)
                    mean_sign_dis = sum(v["sign_disagreement_ratio"] for v in conflict.values()) / len(conflict)
                else:
                    mean_cosine = mean_sign_dis = float("nan")

                # Quality proxy
                eval_score = eval_fn(merged) if eval_fn is not None else float("nan")

                results[strategy] = {
                    "strategy": strategy,
                    "wall_time_s": wall_time,
                    "peak_rss_mb": peak_rss_mb,
                    "peak_python_alloc_mb": peak_python_alloc_mb,
                    "eval_score": eval_score,
                    "conflict_summary": {
                        "mean_cosine_conflict": mean_cosine,
                        "mean_sign_disagreement_ratio": mean_sign_dis,
                        "num_layers_analyzed": len(conflict),
                    },
                }
                self.telemetry.record_latency(f"merge_{strategy}", wall_time)

            except Exception as exc:
                logger.error(f"Strategy '{strategy}' failed: {exc}")
                results[strategy] = {"strategy": strategy, "error": str(exc)}

        return results

    # ------------------------------------------------------------------
    # Inference benchmark
    # ------------------------------------------------------------------

    def run_inference_benchmark(
        self,
        policy: Any,
        reward: Any,
        prompts: List[str],
        tokenizer: Any,
        strategies: Optional[List[str]] = None,
        bon_config: Optional[BestOfNConfig] = None,
        mcts_config: Optional[MCTSConfig] = None,
        max_new_tokens: int = 128,
    ) -> Dict[str, Dict]:
        """
        Benchmark Best-of-N sampling across different n_samples settings.

        Args:
            policy:     Policy model (PolicyLike)
            reward:     Reward model (RewardScorerLike)
            prompts:    List of input prompts
            tokenizer:  Tokenizer
            strategies: e.g. ["bon_n4", "bon_n8", "bon_n16"]
            bon_config: Base BestOfNConfig (n_samples overridden per strategy)

        Returns:
            Dict[strategy_name -> result_dict]
        """
        if strategies is None:
            strategies = ["bon_n4", "bon_n8", "bon_n16"]

        results = {}
        base_cfg = bon_config or BestOfNConfig()

        for strategy in strategies:
            rec = TelemetryRecorder()
            # Parse n_samples from strategy name if it matches bon_nN pattern
            n_samples = base_cfg.n_samples
            if strategy.startswith("bon_n"):
                try:
                    n_samples = int(strategy[5:])
                except ValueError:
                    pass

            cfg_dict = vars(base_cfg).copy()
            cfg_dict["n_samples"] = n_samples
            cfg = BestOfNConfig(**cfg_dict)
            sampler = BestOfNSampler(policy, reward, config=cfg, tokenizer=tokenizer)
            logger.info(
                f"[inference_benchmark] strategy={strategy} n_samples={n_samples} prompts={len(prompts)}"
            )

            latencies = []
            reward_scores = []
            total_tokens = 0

            for idx, prompt in enumerate(prompts, start=1):
                logger.info(
                    f"[inference_benchmark] strategy={strategy} prompt={idx}/{len(prompts)}"
                )
                t0 = time.perf_counter()
                try:
                    result = sampler.generate(prompt, tokenizer, max_new_tokens=max_new_tokens)
                    lat = time.perf_counter() - t0
                    latencies.append(lat)
                    rec.record_latency(strategy, lat)

                    score = result.get("best_score", float("nan"))
                    reward_scores.append(score)
                    tokens = len(result.get("best", "").split())
                    total_tokens += tokens
                    rec.record_tokens(tokens)
                except Exception as exc:
                    logger.warning(f"[{strategy}] prompt failed: {exc}")

            snap = rec.snapshot()
            lat_stats = snap["latency"].get(strategy, {})
            total_latency_s = sum(latencies)
            tokens_per_sec = (
                total_tokens / total_latency_s
                if total_latency_s > 0
                else float("nan")
            )

            results[strategy] = {
                "strategy": strategy,
                "n_samples": n_samples,
                "n_prompts": len(prompts),
                "n_success": len(latencies),
                "latency_p50_s": lat_stats.get("p50_s", float("nan")),
                "latency_p95_s": lat_stats.get("p95_s", float("nan")),
                "mean_latency_s": lat_stats.get("mean_s", float("nan")),
                "total_tokens": total_tokens,
                "tokens_per_sec": tokens_per_sec,
                "reward_score_mean": (
                    sum(reward_scores) / len(reward_scores) if reward_scores else float("nan")
                ),
            }
        return results

    def run_checkpoint_inference_benchmark(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        adapter_path: Optional[str] = DEFAULT_ADAPTER_PATH,
        merged_model_path: Optional[str] = None,
        prompts: Optional[List[str]] = None,
        profile_name: Optional[str] = None,
        bon_config: Optional[BestOfNConfig] = None,
        strategies: Optional[List[str]] = None,
        max_new_tokens: int = 128,
        merge_lora: bool = False,
        device: str = "cpu",
        dtype: str = "float32",
    ) -> Dict[str, Any]:
        """
        End-to-end checkpoint benchmark:
          load model/tokenizer -> run Best-of-N strategy sweep -> rank + summarize.
        """
        policy, tokenizer = load_policy_with_optional_adapter(
            base_model=base_model,
            adapter_path=adapter_path,
            merged_model_path=merged_model_path,
            merge_lora=merge_lora,
            device=device,
            dtype=dtype,
        )

        reward = HeuristicRewardScorer()
        active_profile = profile_name or "custom"
        active_strategies = strategies or _profile_default_bon_strategies(active_profile)
        active_prompts = prompts or list(DEFAULT_PROMPTS)
        active_bon = bon_config or BestOfNConfig()

        bench_results = self.run_inference_benchmark(
            policy=policy,
            reward=reward,
            prompts=active_prompts,
            tokenizer=tokenizer,
            strategies=active_strategies,
            bon_config=active_bon,
            max_new_tokens=max_new_tokens,
        )

        quality_rank = self.rank_strategies(
            bench_results, objective="reward_score_mean", higher_is_better=True
        )
        latency_rank = self.rank_strategies(
            bench_results, objective="latency_p95_s", higher_is_better=False
        )

        recommended = quality_rank[0] if quality_rank else None

        return {
            "profile": active_profile,
            "strategies": active_strategies,
            "prompts_evaluated": len(active_prompts),
            "max_new_tokens": max_new_tokens,
            "recommendation": {
                "best_quality": recommended,
                "quality_ranking": quality_rank,
                "latency_ranking": latency_rank,
            },
            "results": bench_results,
        }

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_strategies(
        self,
        results: Dict[str, Dict],
        objective: str = "eval_score",
        higher_is_better: bool = True,
    ) -> List[str]:
        """
        Sort strategies by a scalar objective.

        Args:
            results:          Output of run_merge_benchmark or run_inference_benchmark
            objective:        Key to sort by (e.g. "eval_score", "latency_p50_s")
            higher_is_better: True for quality metrics, False for latency/RAM

        Returns:
            Sorted list of strategy names (best first)
        """
        def get_score(name: str) -> float:
            val = results[name].get(objective, float("nan"))
            if val != val:  # nan check
                return float("-inf") if higher_is_better else float("inf")
            return float(val)

        names = [n for n in results if "error" not in results[n]]
        return sorted(names, key=get_score, reverse=higher_is_better)

    # ------------------------------------------------------------------
    # Report emit
    # ------------------------------------------------------------------

    def emit_report(self, results: Dict[str, Dict], path: str):
        """Write benchmark report JSON with schema version."""
        report = {
            "schema_version": SCHEMA_VERSION,
            "emitted_at_unix": time.time(),
            "results": results,
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info(f"Benchmark report written to {out}")


# =============================================================================
# BUDGET-AWARE PROFILE SELECTOR
# =============================================================================

_PROFILES: Dict[str, Dict[str, Any]] = {
    "tiny_cpu": {
        "merge": {"method": "task_arithmetic", "density": 0.8},
        "bon":   {"n_samples": 4, "temperature": 1.0, "batch_score": False},
        "mcts":  {"n_simulations": 20, "max_depth": 30, "max_rollout_depth": 20},
        "spec":  {"gamma": 3, "gamma_max": 5, "adapt_gamma": False},
    },
    "balanced_cpu": {
        "merge": {"method": "ties", "density": 0.6},
        "bon":   {"n_samples": 8, "temperature": 0.9, "batch_score": True},
        "mcts":  {"n_simulations": 50, "max_depth": 50, "max_rollout_depth": 30},
        "spec":  {"gamma": 5, "gamma_max": 8, "adapt_gamma": True},
    },
    "gpu_lowlatency": {
        "merge": {"method": "dare", "density": 0.5, "seed": 42},
        "bon":   {"n_samples": 8, "temperature": 0.8, "batch_score": True, "value_weight": 0.1},
        "mcts":  {"n_simulations": 50, "max_depth": 40, "max_rollout_depth": 25,
                  "progressive_widening_alpha": 0.4},
        "spec":  {"gamma": 7, "gamma_max": 10, "adapt_gamma": True, "adapt_window": 30},
    },
    "gpu_maxquality": {
        "merge": {"method": "ties", "density": 0.7, "seed": 0},
        "bon":   {"n_samples": 16, "temperature": 0.7, "batch_score": True,
                  "value_weight": 0.2, "length_penalty": 0.05},
        "mcts":  {"n_simulations": 100, "max_depth": 100, "max_rollout_depth": 50,
                  "progressive_widening_alpha": 0.5, "reward_value_blend": 0.5},
        "spec":  {"gamma": 10, "gamma_max": 12, "adapt_gamma": True, "adapt_window": 50},
    },
}


class BudgetProfileSelector:
    """
    Selects merge/inference config profiles based on hardware + latency budget.

    Optionally reads benchmark history JSON to pick empirically best config.
    """

    def __init__(self, history_path: Optional[str] = None):
        self._history_path = history_path
        self._history: Optional[Dict] = None
        self.last_selected_profile: Optional[str] = None
        if history_path and Path(history_path).exists():
            with open(history_path) as f:
                self._history = json.load(f)

    def select(
        self,
        constraints: Dict[str, Any],
        profile_override: Optional[str] = None,
    ) -> Tuple[MergeConfig, BestOfNConfig, MCTSConfig, SpeculativeDecoderConfig]:
        """
        Select configs based on constraints dict.

        Constraint keys (all optional):
            max_ram_gb       — available RAM in GB (default: 16)
            has_gpu          — bool (default: False)
            latency_budget_ms — target latency in ms (default: 1000)
            quality_priority — bool, prefer quality over speed (default: False)
            profile_override — force a specific profile name when not None

        Returns:
            (MergeConfig, BestOfNConfig, MCTSConfig, SpeculativeDecoderConfig)
        """
        if profile_override is not None:
            if profile_override not in _PROFILES:
                raise ValueError(
                    f"Unknown profile_override='{profile_override}'. "
                    f"Available: {', '.join(self.list_profiles())}"
                )
            profile_name = profile_override
        else:
            profile_name = self._pick_profile(constraints)

        # If history available, try to refine from empirical data
        if self._history is not None and profile_override is None:
            profile_name = self._refine_from_history(profile_name, constraints)
        self.last_selected_profile = profile_name
        logger.info(f"BudgetProfileSelector: selected profile '{profile_name}'")

        profile = _PROFILES[profile_name]

        merge_cfg = MergeConfig(**profile["merge"])
        bon_cfg   = BestOfNConfig(**profile["bon"])
        mcts_cfg  = MCTSConfig(**profile["mcts"])
        spec_cfg  = SpeculativeDecoderConfig(**profile["spec"])

        return merge_cfg, bon_cfg, mcts_cfg, spec_cfg

    def _pick_profile(self, constraints: Dict[str, Any]) -> str:
        max_ram   = constraints.get("max_ram_gb", 16)
        has_gpu   = constraints.get("has_gpu", False)
        lat_ms    = constraints.get("latency_budget_ms", 1000)
        quality   = constraints.get("quality_priority", False)

        if not has_gpu:
            if max_ram <= 4 or lat_ms <= 300:
                return "tiny_cpu"
            return "balanced_cpu"
        else:
            if quality or lat_ms > 500:
                return "gpu_maxquality"
            return "gpu_lowlatency"

    def _refine_from_history(self, profile_name: str, constraints: Dict) -> str:
        """
        If benchmark history contains a better empirical profile for similar constraints,
        switch to it. Simple heuristic: pick profile with best eval_score from history.
        """
        try:
            results = self._history.get("results", {})
            best_name = profile_name
            best_score = float("-inf")
            for name, data in results.items():
                if name in _PROFILES and data.get("eval_score", float("nan")) > best_score:
                    best_score = data["eval_score"]
                    best_name = name
            return best_name
        except Exception:
            return profile_name

    @staticmethod
    def list_profiles() -> List[str]:
        return list(_PROFILES.keys())


# =============================================================================
# REGRESSION GATE
# =============================================================================

class RegressionGate:
    """
    CI-style threshold checks to block promotion of regressed artifacts.

    Example:
        gate = RegressionGate(
            thresholds={"quality_floor": 0.7, "latency_ceiling_ms": 800, "ram_ceiling_gb": 12}
        )
        ok = gate.check(current_results, baseline_results)
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or {}

    def check(
        self,
        current: Dict[str, Any],
        baseline: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Returns True if all thresholds are met, False if any are violated.
        Logs a warning for each violation.
        """
        passed = True

        quality_floor = self.thresholds.get("quality_floor")
        if quality_floor is not None:
            score = current.get("eval_score", float("nan"))
            if score != score or score < quality_floor:
                logger.warning(
                    f"RegressionGate: quality_floor violated. "
                    f"Got {score:.4f}, floor={quality_floor}"
                )
                passed = False

        latency_ceil = self.thresholds.get("latency_ceiling_ms")
        if latency_ceil is not None:
            lat_ms = current.get("latency_p95_s", float("nan")) * 1000
            if lat_ms != lat_ms or lat_ms > latency_ceil:
                logger.warning(
                    f"RegressionGate: latency_ceiling_ms violated. "
                    f"p95={lat_ms:.1f}ms, ceiling={latency_ceil}ms"
                )
                passed = False

        ram_ceil = self.thresholds.get("ram_ceiling_gb")
        if ram_ceil is not None:
            ram_gb = current.get("peak_rss_mb", float("nan")) / 1024
            if ram_gb != ram_gb or ram_gb > ram_ceil:
                logger.warning(
                    f"RegressionGate: ram_ceiling_gb violated. "
                    f"Got {ram_gb:.2f}GB, ceiling={ram_ceil}GB"
                )
                passed = False

        # Relative regression vs baseline
        if baseline is not None:
            tol = self.thresholds.get("relative_regression_tolerance", 0.05)
            base_score = baseline.get("eval_score", float("nan"))
            curr_score = current.get("eval_score", float("nan"))
            if base_score == base_score and curr_score == curr_score and base_score > 0:
                drop = (base_score - curr_score) / base_score
                if drop > tol:
                    logger.warning(
                        f"RegressionGate: relative regression {drop:.2%} exceeds tolerance {tol:.2%}"
                    )
                    passed = False

        return passed


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def _cli():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="RLHF Benchmark Harness")
    parser.add_argument(
        "--mode",
        choices=["profile", "inference"],
        default="profile",
        help="`profile` emits selected configs only; `inference` runs checkpoint-backed benchmark.",
    )
    profile_choices = ["auto"] + BudgetProfileSelector.list_profiles()
    parser.add_argument(
        "--profile",
        choices=profile_choices,
        default="auto",
        help="Budget profile to select (or 'auto' to infer from constraints)",
    )
    parser.add_argument("--output", default="bench.json", help="Output JSON path")
    parser.add_argument("--max-ram-gb", type=float, default=8)
    parser.add_argument("--has-gpu", action="store_true")
    parser.add_argument("--latency-budget-ms", type=float, default=1000)
    parser.add_argument("--quality-priority", action="store_true")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model id/path")
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH, help="LoRA adapter path/id")
    parser.add_argument("--merged-model-path", default="", help="Optional merged model path/id")
    parser.add_argument("--prompts-file", default="", help="Optional prompts file (.txt/.json/.jsonl)")
    parser.add_argument("--max-prompts", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--merge-lora",
        action="store_true",
        help="Explicitly merge adapter into base weights (can be slow on CPU).",
    )
    # Backward-compatible no-op for older command examples.
    parser.add_argument(
        "--no-merge-lora",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=None,
        help="Optional inference strategies override, e.g. bon_n4 bon_n8 bon_n16",
    )
    args = parser.parse_args()

    selector = BudgetProfileSelector()
    constraints = {
        "max_ram_gb": args.max_ram_gb,
        "has_gpu": args.has_gpu,
        "latency_budget_ms": args.latency_budget_ms,
        "quality_priority": args.quality_priority,
    }
    profile_override = None if args.profile == "auto" else args.profile
    merge_cfg, bon_cfg, mcts_cfg, spec_cfg = selector.select(
        constraints,
        profile_override=profile_override,
    )
    selected_profile = selector.last_selected_profile or args.profile

    base_report = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "profile": selected_profile,
        "requested_profile": args.profile,
        "constraints": constraints,
        "selected_configs": {
            "merge": {
                "method": merge_cfg.method,
                "density": merge_cfg.density,
                "seed": merge_cfg.seed,
            },
            "best_of_n": {
                "n_samples": bon_cfg.n_samples,
                "temperature": bon_cfg.temperature,
                "value_weight": bon_cfg.value_weight,
            },
            "mcts": {
                "n_simulations": mcts_cfg.n_simulations,
                "max_depth": mcts_cfg.max_depth,
                "progressive_widening_alpha": mcts_cfg.progressive_widening_alpha,
            },
            "speculative": {
                "gamma": spec_cfg.gamma,
                "gamma_max": spec_cfg.gamma_max,
                "adapt_gamma": spec_cfg.adapt_gamma,
            },
        },
    }

    report = dict(base_report)

    if args.mode == "inference":
        merge_lora = (args.merge_lora and not args.no_merge_lora)
        if not merge_lora:
            logger.info(
                "Adapter merge is disabled (default) to avoid long CPU stalls. "
                "Pass --merge-lora to force merge_and_unload."
            )
        prompts = _load_prompts(args.prompts_file or None, max_prompts=args.max_prompts)
        harness = BenchmarkHarness()
        inference_summary = harness.run_checkpoint_inference_benchmark(
            base_model=args.base_model,
            adapter_path=(args.adapter_path or None),
            merged_model_path=(args.merged_model_path or None),
            prompts=prompts,
            profile_name=selected_profile,
            bon_config=bon_cfg,
            strategies=args.strategies,
            max_new_tokens=args.max_new_tokens,
            merge_lora=merge_lora,
            device=args.device,
            dtype=args.dtype,
        )
        report["inference_benchmark"] = inference_summary

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Profile '{selected_profile}' selected (requested='{args.profile}').")
    print(f"Merge method : {merge_cfg.method}")
    print(f"Best-of-N n  : {bon_cfg.n_samples}")
    print(f"MCTS sims    : {mcts_cfg.n_simulations}")
    print(f"Spec gamma   : {spec_cfg.gamma} (max {spec_cfg.gamma_max})")
    if args.mode == "inference":
        rec = report.get("inference_benchmark", {}).get("recommendation", {})
        print(f"Best quality strategy: {rec.get('best_quality')}")
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    _cli()
