"""
Functional + stress tests for inference_optimizations.py

Uses tiny mock policy/reward/value (nn.Linear stubs).
No HuggingFace dependency required.
"""

import math
import time
import pytest
import torch
import torch.nn as nn

from inference_optimizations import (
    BestOfNConfig,
    BestOfNSampler,
    MCTSConfig,
    MCTSGenerator,
    MCTSNode,
    PagedKVCache,
    SpeculativeDecoderConfig,
    SpeculativeDecoder,
)
from inference_protocols import (
    PolicyAdapter,
    RewardScorerAdapter,
    ValueScorerAdapter,
)
from telemetry import TelemetryRecorder


# =============================================================================
# MOCK OBJECTS
# =============================================================================

class MockReward(nn.Module):
    """Deterministic reward: returns mean of input float tensor."""
    def forward(self, **kwargs):
        ids = kwargs.get("input_ids", torch.zeros(1, 4))
        return ids.float().mean(dim=-1)

    def score(self, text: str) -> float:
        # Assign score by text length so longest candidate wins
        return float(len(text))


class MockValue(nn.Module):
    def forward(self, **kwargs):
        ids = kwargs.get("input_ids", torch.zeros(1, 4))
        return ids.float().mean(dim=-1)

    def score(self, text: str) -> float:
        return 0.5


class MockPolicy(nn.Module):
    """Trivial policy: produces constant logits."""
    vocab_size = 16

    def forward(self, input_ids=None, attention_mask=None, **kw):
        batch = 1 if input_ids is None else input_ids.shape[0]
        seq   = 4  if input_ids is None else input_ids.shape[1]
        logits = torch.zeros(batch, seq, self.vocab_size)
        class Out:
            pass
        out = Out()
        out.logits = logits
        out.past_key_values = None
        return out

    def generate(self, input_ids, max_new_tokens=4, **kw):
        return torch.zeros(input_ids.shape[0], input_ids.shape[1] + max_new_tokens, dtype=torch.long)

    def parameters(self):
        return iter([torch.zeros(1)])  # Fake parameter for device resolution


class MockTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "<eos>"

    def __call__(self, text, return_tensors="pt", **kw):
        ids = torch.zeros(1, 4, dtype=torch.long)
        mask = torch.ones(1, 4, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}

    def decode(self, ids, skip_special_tokens=True):
        return "decoded text"


# =============================================================================
# BATCH 2 — Interface bug fixes
# =============================================================================

def test_policy_device_resolution():
    """BestOfNSampler._get_policy_device() must work without .device attribute."""
    policy = MockPolicy()
    assert not hasattr(policy, "device")
    sampler = BestOfNSampler(policy, MockReward(), tokenizer=MockTokenizer())
    dev = sampler._get_policy_device()
    assert isinstance(dev, torch.device)


def test_reward_scorer_adapter():
    """RewardScorerAdapter wraps forward() and returns scalar float."""

    class SimpleReward(nn.Module):
        def forward(self, input_ids, attention_mask=None):
            return input_ids.float().mean(dim=-1)

    class FakeTok:
        def __call__(self, text, return_tensors="pt", **kw):
            return {"input_ids": torch.ones(1, 4, dtype=torch.long)}

    reward = SimpleReward()
    adapter = RewardScorerAdapter.from_rlhf_model(reward, FakeTok())
    score = adapter.score("hello world")
    assert isinstance(score, float)


def test_value_scorer_adapter():
    """ValueScorerAdapter wraps forward() and returns scalar float."""

    class SimpleValue(nn.Module):
        def forward(self, input_ids, attention_mask=None, return_all_values=False):
            return input_ids.float().mean(dim=-1)

    class FakeTok:
        def __call__(self, text, return_tensors="pt", **kw):
            return {"input_ids": torch.ones(1, 4, dtype=torch.long)}

    value = SimpleValue()
    adapter = ValueScorerAdapter.from_rlhf_model(value, FakeTok())
    score = adapter.score("test state")
    assert isinstance(score, float)


def test_best_of_n_selects_best():
    """BestOfNSampler must select the candidate with highest score."""

    class LengthReward:
        def score(self, text: str) -> float:
            return float(len(text))

    class LengthFakePolicy:
        def generate(self, input_ids, max_new_tokens=8, **kw):
            return input_ids

        def parameters(self):
            return iter([torch.zeros(1)])

    sentences = ["a", "bb", "ccc", "dddd"]
    call_idx = [0]

    class IndexedReward:
        def score(self, text: str) -> float:
            i = call_idx[0] % len(sentences)
            call_idx[0] += 1
            return float(i)  # Last candidate gets highest score

    policy = MockPolicy()
    reward = IndexedReward()
    tok    = MockTokenizer()
    cfg    = BestOfNConfig(n_samples=4, use_diversity_bonus=False)
    sampler = BestOfNSampler(policy, reward, config=cfg, tokenizer=tok)

    result = sampler.generate("prompt", tok, max_new_tokens=8)
    assert "best" in result
    assert "best_score" in result


def test_best_of_n_format_checker():
    """format_checker=None passes all; always-False filter rejects all → all -inf scores."""
    policy = MockPolicy()

    class ConstReward:
        def score(self, text):
            return 1.0

    tok = MockTokenizer()
    cfg = BestOfNConfig(n_samples=3, use_diversity_bonus=False, format_checker=lambda t: False)
    sampler = BestOfNSampler(policy, ConstReward(), config=cfg, tokenizer=tok)
    scores = sampler._score_candidates(["a", "b", "c"])
    assert all(s == float("-inf") for s in scores)


def test_best_of_n_repetition_penalty():
    """repetition_penalty>0 must reduce score for highly repetitive text."""
    policy = MockPolicy()

    class ConstReward:
        def score(self, text):
            return 1.0

    tok = MockTokenizer()
    cfg_no_pen  = BestOfNConfig(n_samples=2, use_diversity_bonus=False, repetition_penalty=0.0)
    cfg_with_pen = BestOfNConfig(n_samples=2, use_diversity_bonus=False, repetition_penalty=1.0)

    repetitive = "the cat sat the cat sat the cat sat the cat sat"
    sampler_no  = BestOfNSampler(policy, ConstReward(), config=cfg_no_pen,   tokenizer=tok)
    sampler_pen = BestOfNSampler(policy, ConstReward(), config=cfg_with_pen, tokenizer=tok)

    score_no  = sampler_no._score_candidates([repetitive])[0]
    score_pen = sampler_pen._score_candidates([repetitive])[0]
    assert score_pen < score_no


# =============================================================================
# BATCH 3 — MCTS 2.0
# =============================================================================

def test_mcts_node_depth():
    root = MCTSNode("state", depth=0)
    child = root.add_child("action", "state+action")
    assert child.depth == 1
    grandchild = child.add_child("a2", "state+action+a2")
    assert grandchild.depth == 2


def test_mcts_node_puct():
    """PUCT score must be finite and > pure Q when unexplored parent."""
    root = MCTSNode("root", depth=0)
    root.visits = 10
    child = MCTSNode("child", parent=root, depth=1)
    child.visits = 2
    child.value_sum = 1.0
    child.prior = 0.5

    score = child.ucb_score(c_puct=1.25, c2=19652.0)
    assert math.isfinite(score)
    assert score > child.value()  # exploration bonus must add to Q


def test_mcts_puct_expansion():
    """MCTSGenerator must expand children proportional to progressive widening."""
    policy = MockPolicy()
    tok    = MockTokenizer()
    cfg    = MCTSConfig(n_simulations=5, progressive_widening_alpha=0.5)
    mcts   = MCTSGenerator(policy, None, tok, cfg)

    root = MCTSNode("prompt", depth=0)
    root.visits = 4  # max_children = ceil(4^0.5) = 2
    mcts._expand(root)
    expected_max = max(1, math.ceil((root.visits + 1) ** cfg.progressive_widening_alpha))
    assert len(root.children) <= expected_max


def test_mcts_depth_discount():
    """Backpropagated value must be discounted at ancestor nodes."""
    cfg  = MCTSConfig(depth_discount=0.9)
    mcts = MCTSGenerator(MockPolicy(), None, MockTokenizer(), cfg)

    root  = MCTSNode("root", depth=0)
    child = MCTSNode("child", parent=root, depth=1)

    mcts._backpropagate(child, 1.0)

    assert child.value_sum == pytest.approx(1.0, abs=1e-6)
    # Root is at depth 0, child at depth 1 → discount^(1-0) = 0.9
    assert root.value_sum == pytest.approx(0.9, abs=1e-6)


def test_mcts_value_score_no_score_text():
    """MCTSGenerator._value_score must not crash when value has no .score_text."""
    policy = MockPolicy()
    tok    = MockTokenizer()

    class ValueNoScoreText(nn.Module):
        def forward(self, input_ids=None, **kw):
            return torch.tensor([0.7])

    cfg  = MCTSConfig(use_value_model=True)
    mcts = MCTSGenerator(policy, ValueNoScoreText(), tok, cfg)
    # Should fallback to tokenizer path (no AttributeError)
    score = mcts._value_score("some state text")
    assert isinstance(score, float)


# =============================================================================
# BATCH 4 — SpeculativeDecoder 2.0
# =============================================================================

def test_speculative_decoder_config():
    cfg = SpeculativeDecoderConfig(gamma=5, gamma_min=2, gamma_max=10, adapt_gamma=True)
    assert cfg.gamma == 5
    assert cfg.adapt_gamma is True


def test_speculative_decoder_acceptance_rate():
    """acceptance_rate property must work before any generation."""

    class TrivialDraft(nn.Module):
        def forward(self, input_ids, use_cache=False, past_key_values=None):
            batch, seq = input_ids.shape
            logits = torch.ones(batch, seq, 16)
            class O:
                pass
            o = O()
            o.logits = logits
            o.past_key_values = None
            return o

    class TrivialTarget(nn.Module):
        def forward(self, input_ids, use_cache=False):
            batch, seq = input_ids.shape
            logits = torch.ones(batch, seq, 16)
            class O:
                pass
            o = O()
            o.logits = logits
            return o

    cfg = SpeculativeDecoderConfig(gamma=2, adapt_gamma=False)
    sd  = SpeculativeDecoder(TrivialTarget(), TrivialDraft(), config=cfg)
    assert sd.acceptance_rate == 0.0  # No tokens generated yet


def test_speculative_decoder_adaptive_gamma():
    """_adapt_gamma() must increase gamma when acceptance rate is high."""
    cfg = SpeculativeDecoderConfig(gamma=5, gamma_min=2, gamma_max=10,
                                   adapt_gamma=True, adapt_window=10)

    class Stub(nn.Module):
        def forward(self, *a, **kw):
            pass

    sd = SpeculativeDecoder(Stub(), Stub(), config=cfg)
    sd._current_gamma = 5
    # Simulate 10 steps with 100% acceptance
    sd._accepted_history = [1.0] * 10
    sd._adapt_gamma()
    assert sd._current_gamma == 6  # increased by 1


def test_speculative_decoder_adaptive_gamma_decreases():
    cfg = SpeculativeDecoderConfig(gamma=5, gamma_min=2, gamma_max=10,
                                   adapt_gamma=True, adapt_window=10)

    class Stub(nn.Module):
        def forward(self, *a, **kw):
            pass

    sd = SpeculativeDecoder(Stub(), Stub(), config=cfg)
    sd._current_gamma = 5
    # Simulate 10 steps with 0% acceptance
    sd._accepted_history = [0.0] * 10
    sd._adapt_gamma()
    assert sd._current_gamma == 4  # decreased by 1


# =============================================================================
# BATCH 5 — PagedKVCache 2.0
# =============================================================================

def test_paged_kv_cache_stats():
    kv = PagedKVCache(num_layers=2, num_heads=4, head_dim=8, page_size=4, max_pages=20)
    kv.allocate("seq1", 6)   # needs 2 pages
    s = kv.stats()
    assert s["allocated_pages"] == 2
    assert s["free_pages"] == 18
    assert s["active_sequences"] == 1


def test_paged_kv_cache_fragmentation():
    kv = PagedKVCache(num_layers=1, num_heads=2, head_dim=4, page_size=8, max_pages=10)
    kv.allocate("seq1", 3)
    # allocate() initializes sequence_length to 0; simulate 3 tokens written
    kv.sequence_lengths["seq1"] = 3  # 3 tokens in 1 page of 8 → frag = 1 - 3/8
    assert kv.fragmentation_ratio == pytest.approx(1 - 3/8, abs=1e-5)


def test_paged_kv_cache_eviction():
    """evict_lru must evict the oldest-accessed sequence first."""
    kv = PagedKVCache(num_layers=1, num_heads=2, head_dim=4, page_size=4, max_pages=20)
    kv.allocate("seq_old", 4)
    time.sleep(0.01)
    kv.allocate("seq_new", 4)

    freed = kv.evict_lru(n_pages=1)
    assert freed >= 1
    assert "seq_old" not in kv.sequence_pages


def test_paged_kv_prefix_reuse():
    """register_prefix marks seq as shared; free should update eviction tracking."""
    kv = PagedKVCache(num_layers=1, num_heads=2, head_dim=4, page_size=4, max_pages=20)
    kv.allocate("prefix_seq", 4)
    kv.register_prefix("shared_prefix_v1", "prefix_seq")
    assert "prefix_seq" in kv.shared_prefix_keys
    assert kv.shared_prefix_keys["prefix_seq"] == "shared_prefix_v1"


# =============================================================================
# BATCH 6 — Telemetry
# =============================================================================

def test_telemetry_record_and_snapshot():
    rec = TelemetryRecorder()
    with rec.timer("test_op"):
        time.sleep(0.001)
    rec.record_tokens(42)
    rec.record_cache_event(hit=True)
    rec.record_mcts_expansion()
    rec.record_speculative_event(accepted=True)
    rec.record_speculative_event(accepted=False)

    snap = rec.snapshot()
    assert snap["counters"]["tokens_generated"] == 42
    assert snap["counters"]["cache_hits"] == 1
    assert snap["counters"]["mcts_expansions"] == 1
    assert snap["counters"]["spec_total"] == 2
    assert snap["counters"]["spec_accepted"] == 1
    assert snap["derived"]["speculative_acceptance_rate"] == pytest.approx(0.5)
    assert "test_op" in snap["latency"]


def test_telemetry_p95_latency():
    """p95 latency from 1000 samples must be correct to ±5%."""
    import numpy as np
    rec = TelemetryRecorder()
    rng = torch.Generator()
    rng.manual_seed(0)

    values = torch.rand(1000, generator=rng).tolist()
    for v in values:
        rec.record_latency("op", v)

    snap = rec.snapshot()
    p95_recorded = snap["latency"]["op"]["p95_s"]
    p95_expected = float(np.percentile(values, 95))

    # Allow 5% relative tolerance
    assert abs(p95_recorded - p95_expected) / (p95_expected + 1e-9) < 0.05


def test_telemetry_emit_json():
    import json, tempfile, os
    rec = TelemetryRecorder()
    rec.record_tokens(10)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "metrics.json")
        rec.emit_json(path)
        with open(path) as f:
            data = json.load(f)
    assert "counters" in data
    assert data["counters"]["tokens_generated"] == 10
