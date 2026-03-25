"""
Unified Telemetry Recorder

Thread-safe metrics sink for model merging and inference optimization runs.
"""

import json
import logging
import os
import random
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

logger = logging.getLogger("Telemetry")

# Reservoir size for latency histogram
_RESERVOIR_SIZE = 1000

class _Reservoir:
    """Thread-safe reservoir sampler for streaming quantile estimation."""

    def __init__(self, size: int = _RESERVOIR_SIZE):
        self._size = size
        self._samples: List[float] = []
        self._count = 0
        self._lock = threading.Lock()

    def add(self, value: float):
        with self._lock:
            self._count += 1
            if len(self._samples) < self._size:
                self._samples.append(value)
            else:
                # Vitter's Algorithm R
                idx = random.randint(0, self._count - 1)
                if idx < self._size:
                    self._samples[idx] = value

    def quantile(self, q: float) -> float:
        with self._lock:
            if not self._samples:
                return float("nan")
            sorted_s = sorted(self._samples)
            k = q * (len(sorted_s) - 1)
            lo, hi = int(k), min(int(k) + 1, len(sorted_s) - 1)
            return sorted_s[lo] + (k - lo) * (sorted_s[hi] - sorted_s[lo])

    def count(self) -> int:
        with self._lock:
            return self._count

    def mean(self) -> float:
        with self._lock:
            return float(np.mean(self._samples)) if self._samples else float("nan")


class TelemetryRecorder:
    """
    Unified metrics sink for inference and merge runs.

    Thread-safe. All methods are safe to call from multiple workers.

    Usage
    -----
    recorder = TelemetryRecorder()

    with recorder.timer("best_of_n"):
        result = sampler.generate(prompt, tokenizer)

    recorder.record_tokens(len(result['best'].split()))
    snap = recorder.snapshot()
    recorder.emit_json("run_metrics.json")
    """

    def __init__(self, tb_writer: Optional[Any] = None):
        """
        Args:
            tb_writer: Optional TensorBoard SummaryWriter (same pattern as TrainingLogger).
        """
        self._lock = threading.Lock()
        self._latencies: Dict[str, _Reservoir] = {}
        self._counters: Dict[str, int] = {
            "tokens_generated": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "mcts_expansions": 0,
            "spec_accepted": 0,
            "spec_total": 0,
            "prm_calls": 0,
            "prm_steps_scored": 0,
            "prm_reranks": 0,
        }
        self._prm_score_reservoir: _Reservoir = _Reservoir()
        self._memory_snapshots: List[Dict[str, Any]] = []
        self._events: List[Dict[str, Any]] = []
        self._tb_writer = tb_writer
        self._global_step = 0

    # ------------------------------------------------------------------
    # Recording primitives
    # ------------------------------------------------------------------

    def record_latency(self, name: str, seconds: float):
        """Record a latency sample (in seconds) for the named operation."""
        with self._lock:
            if name not in self._latencies:
                self._latencies[name] = _Reservoir()
        self._latencies[name].add(seconds)
        if self._tb_writer is not None:
            try:
                self._tb_writer.add_scalar(f"latency/{name}_s", seconds, self._global_step)
            except Exception:
                pass

    def record_tokens(self, count: int):
        """Record number of tokens generated."""
        with self._lock:
            self._counters["tokens_generated"] += count

    def record_cache_event(self, hit: bool):
        """Record a KV-cache hit or miss."""
        with self._lock:
            key = "cache_hits" if hit else "cache_misses"
            self._counters[key] += 1

    def record_mcts_expansion(self):
        """Increment MCTS node expansion counter."""
        with self._lock:
            self._counters["mcts_expansions"] += 1

    def record_speculative_event(self, accepted: bool):
        """Record one speculative decoding token event."""
        with self._lock:
            self._counters["spec_total"] += 1
            if accepted:
                self._counters["spec_accepted"] += 1

    def increment(self, name: str, delta: int = 1):
        """Increment an arbitrary named counter."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + delta

    def set_step(self, step: int):
        """Update global step for TensorBoard logging."""
        with self._lock:
            self._global_step = step

    def record_event(self, name: str, payload: Optional[Dict[str, Any]] = None):
        """Record a timestamped event for phase-level traceability."""
        event = {
            "name": name,
            "timestamp_unix": time.time(),
            "step": self._global_step,
            "payload": payload or {},
        }
        with self._lock:
            self._events.append(event)

    def record_prm_event(
        self,
        mode: str,
        n_steps: int = 0,
        min_step_score: float = float("nan"),
        outcome_score: float = float("nan"),
    ):
        """
        Record one PRM scoring event.

        Args:
            mode: Scoring mode label — one of 'score', 'rerank', 'step_by_step'.
            n_steps: Number of reasoning steps scored in this call.
            min_step_score: Minimum per-step score (PRM-Min); NaN if not applicable.
            outcome_score: Outcome reward score; NaN if not applicable.
        """
        with self._lock:
            self._counters["prm_calls"] += 1
            self._counters["prm_steps_scored"] += n_steps
            if mode == "rerank":
                self._counters["prm_reranks"] += 1
        import math as _math
        if _math.isfinite(min_step_score):
            self._prm_score_reservoir.add(min_step_score)
        self.record_event("prm_score", {
            "mode": mode,
            "n_steps": n_steps,
            "min_step_score": min_step_score,
            "outcome_score": outcome_score,
        })

    def record_memory_snapshot(self, phase: str) -> Dict[str, Any]:
        """
        Capture process/device memory at a named phase.
        Works on CPU-only systems and enriches with CUDA stats when available.
        """
        snap: Dict[str, Any] = {
            "phase": phase,
            "timestamp_unix": time.time(),
            "step": self._global_step,
            "pid": os.getpid(),
        }

        if psutil is not None:
            try:
                proc = psutil.Process(os.getpid())
                mem = proc.memory_info()
                snap.update({
                    "rss_mb": round(mem.rss / (1024 ** 2), 2),
                    "vms_mb": round(mem.vms / (1024 ** 2), 2),
                })
                if hasattr(proc, "memory_percent"):
                    snap["rss_percent"] = round(float(proc.memory_percent()), 2)
            except Exception as exc:
                snap["process_mem_error"] = str(exc)
        else:
            snap["process_mem_error"] = "psutil_not_available"

        if torch.cuda.is_available():
            try:
                snap.update({
                    "cuda_allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
                    "cuda_reserved_mb": round(torch.cuda.memory_reserved() / (1024 ** 2), 2),
                    "cuda_max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
                })
            except Exception as exc:
                snap["cuda_mem_error"] = str(exc)

        with self._lock:
            self._memory_snapshots.append(snap)
        return snap

    # ------------------------------------------------------------------
    # Context manager timer
    # ------------------------------------------------------------------

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        """Context manager that records elapsed time as a latency sample."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.record_latency(name, elapsed)

    # ------------------------------------------------------------------
    # Snapshot / emit
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return a machine-readable summary dict."""
        with self._lock:
            counters = dict(self._counters)
            latency_names = list(self._latencies.keys())

        latency_stats: Dict[str, Any] = {}
        for name in latency_names:
            r = self._latencies[name]
            latency_stats[name] = {
                "count": r.count(),
                "mean_s": r.mean(),
                "p50_s": r.quantile(0.50),
                "p95_s": r.quantile(0.95),
                "p99_s": r.quantile(0.99),
            }

        # Derived metrics
        spec_total = counters.get("spec_total", 0)
        spec_accepted = counters.get("spec_accepted", 0)
        spec_rate = spec_accepted / spec_total if spec_total > 0 else float("nan")

        cache_hits = counters.get("cache_hits", 0)
        cache_total = cache_hits + counters.get("cache_misses", 0)
        cache_hit_rate = cache_hits / cache_total if cache_total > 0 else float("nan")

        prm_res = self._prm_score_reservoir
        return {
            "counters": counters,
            "latency": latency_stats,
            "derived": {
                "speculative_acceptance_rate": spec_rate,
                "cache_hit_rate": cache_hit_rate,
            },
            "prm": {
                "calls": counters.get("prm_calls", 0),
                "steps_scored": counters.get("prm_steps_scored", 0),
                "reranks": counters.get("prm_reranks", 0),
                "min_step_score_mean": prm_res.mean(),
                "min_step_score_p50": prm_res.quantile(0.5),
            },
            "memory": list(self._memory_snapshots),
            "events": list(self._events),
        }

    def emit_json(self, path: str):
        """Write snapshot to a JSON file."""
        snap = self.snapshot()
        snap["emitted_at_unix"] = time.time()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(snap, fh, indent=2, default=str)
        logger.info(f"Telemetry written to {out}")

    def reset(self):
        """Reset all counters and latency reservoirs."""
        with self._lock:
            self._latencies.clear()
            for k in self._counters:
                self._counters[k] = 0
            self._prm_score_reservoir = _Reservoir()
            self._memory_snapshots.clear()
            self._events.clear()


# Module-level default recorder (importable singleton)
default_recorder = TelemetryRecorder()
