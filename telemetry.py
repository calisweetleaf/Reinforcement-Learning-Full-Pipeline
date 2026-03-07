"""
Unified Telemetry Recorder

Thread-safe metrics sink for model merging and inference optimization runs.

Features
--------
- Latency histogram with reservoir sampling → p50 / p95 / p99
- Token throughput counter
- KV-cache hit/miss tracking
- MCTS node expansion counter
- Speculative-decoding acceptance tracking
- JSON snapshot emit
- Context manager: `with recorder.timer("label"):`
- Optional TensorBoard adapter (follows TrainingLogger pattern from rlhf.py)
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

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
        }
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

        return {
            "counters": counters,
            "latency": latency_stats,
            "derived": {
                "speculative_acceptance_rate": spec_rate,
                "cache_hit_rate": cache_hit_rate,
            },
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


# Module-level default recorder (importable singleton)
default_recorder = TelemetryRecorder()
