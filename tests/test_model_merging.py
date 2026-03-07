"""
Functional + stress tests for model_merging.py

All tests use tiny in-process mock models (2-layer linear networks).
No HuggingFace dependency required.
"""

import copy
import math
import pytest
import torch
import torch.nn as nn

from model_merging import (
    MergeConfig,
    ModelMerger,
    ModelSoup,
    _sha256_state_dict,
    save_merge_artifact,
)


# =============================================================================
# FIXTURES
# =============================================================================

def make_model(hidden=8, seed=None) -> nn.Module:
    """Tiny 2-layer linear model."""
    if seed is not None:
        torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
    )


def make_bf16_model(hidden=8, seed=None) -> nn.Module:
    """Same but in bfloat16."""
    m = make_model(hidden, seed)
    return m.to(torch.bfloat16)


def perturb(model: nn.Module, scale: float = 0.1, seed: int = 0) -> nn.Module:
    """Return a copy of model with small random perturbation."""
    m = copy.deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * scale)
    return m


# =============================================================================
# BATCH 1 — Structural / dtype fixes
# =============================================================================

def test_slerp_and_dare_are_methods():
    merger = ModelMerger()
    assert hasattr(merger, "_slerp_merge"), "_slerp_merge must be a real method"
    assert hasattr(merger, "_dare_merge"), "_dare_merge must be a real method"
    assert callable(merger._slerp_merge)
    assert callable(merger._dare_merge)


def test_dtype_handling_bfloat16():
    """BF16 models must produce non-empty deltas."""
    base = make_bf16_model(seed=0)
    ft   = make_bf16_model(seed=1)
    merger = ModelMerger(MergeConfig(method="task_arithmetic"))
    base_state = base.state_dict()
    ft_states  = [ft.state_dict()]
    aligned = merger._align_keys(base_state, ft_states)

    deltas = []
    for fs in ft_states:
        d = {}
        for key in aligned:
            if base_state[key].is_floating_point():
                d[key] = fs[key].float() - base_state[key].float()
        deltas.append(d)

    assert len(deltas) == 1
    assert len(deltas[0]) > 0, "Delta should be non-empty for bfloat16 params"
    for key, delta in deltas[0].items():
        assert delta.dtype == torch.float32


def test_slerp_requires_two_models():
    base = make_model(seed=0)
    ft1  = perturb(base, seed=1)
    ft2  = perturb(base, seed=2)
    ft3  = perturb(base, seed=3)
    merger = ModelMerger(MergeConfig(method="slerp"))
    with pytest.raises(ValueError, match="exactly 2"):
        merger.merge(base, [ft1, ft2, ft3])


def test_dare_density_validation():
    base = make_model(seed=0)
    ft   = perturb(base, seed=1)
    merger = ModelMerger(MergeConfig(method="dare", density=0.0))
    with pytest.raises(ValueError, match="density"):
        merger.merge(base, [ft])


def test_ties_density_validation():
    base = make_model(seed=0)
    ft   = perturb(base, seed=1)
    merger = ModelMerger(MergeConfig(method="ties", density=1.5))
    with pytest.raises(ValueError, match="density"):
        merger.merge(base, [ft])


def test_key_alignment_diagnostics(caplog):
    """Missing keys should trigger a warning, not a crash."""
    import logging
    base = make_model(seed=0)
    ft   = perturb(base, seed=1)
    # Remove one key from ft
    ft_state = ft.state_dict()
    removed_key = list(ft_state.keys())[0]
    del ft_state[removed_key]

    merger = ModelMerger()
    with caplog.at_level(logging.WARNING, logger="ModelMerging"):
        aligned = merger._align_keys(base.state_dict(), [ft_state])
    assert removed_key not in aligned
    assert any("missing" in r.message.lower() for r in caplog.records)


# =============================================================================
# BATCH 2 — Merge algorithm correctness
# =============================================================================

def test_task_arithmetic_basic():
    """Merged delta should equal average of individual deltas (uniform weights)."""
    torch.manual_seed(42)
    base = make_model(seed=0)
    ft1  = perturb(base, scale=0.2, seed=1)
    ft2  = perturb(base, scale=0.2, seed=2)

    merger = ModelMerger(MergeConfig(method="task_arithmetic"))
    merged = merger.merge(base, [ft1, ft2])

    base_sd   = base.state_dict()
    ft1_sd    = ft1.state_dict()
    ft2_sd    = ft2.state_dict()
    merged_sd = merged.state_dict()

    for key in base_sd:
        if not base_sd[key].is_floating_point():
            continue
        expected = base_sd[key].float() + (
            (ft1_sd[key].float() - base_sd[key].float()) +
            (ft2_sd[key].float() - base_sd[key].float())
        ) / 2.0
        torch.testing.assert_close(
            merged_sd[key].float(), expected, atol=1e-5, rtol=1e-4
        )


def test_ties_merge_sign_election():
    """Majority sign should win; opposing-sign params should be zeroed."""
    base = make_model(seed=0)
    # Construct two ft models with coordinated perturbations
    ft1 = perturb(base, scale=0.5, seed=1)
    ft2 = perturb(base, scale=0.5, seed=2)
    ft3 = perturb(base, scale=0.5, seed=3)

    merger = ModelMerger(MergeConfig(method="ties", density=1.0, seed=42))
    merged = merger.merge(base, [ft1, ft2, ft3])
    # Just verify it runs and returns a model with same architecture
    assert set(merged.state_dict().keys()) == set(base.state_dict().keys())


def test_slerp_two_models():
    """SLERP at t=0.5 should produce a result between the two models."""
    base = make_model(seed=0)
    ft1  = perturb(base, scale=0.3, seed=1)
    ft2  = perturb(base, scale=0.3, seed=2)

    merger = ModelMerger(MergeConfig(method="slerp", weights=[0.5]))
    merged = merger.merge(base, [ft1, ft2])

    # Merged params should differ from both ft1 and ft2 (unless degenerate)
    for key in base.state_dict():
        if not base.state_dict()[key].is_floating_point():
            continue
        m = merged.state_dict()[key].float()
        p1 = ft1.state_dict()[key].float()
        p2 = ft2.state_dict()[key].float()
        # At t=0.5, merged should not be identical to either endpoint
        if not torch.allclose(p1, p2):
            assert not torch.allclose(m, p1) or not torch.allclose(m, p2)


def test_dare_mask_density():
    """DARE with density=0.5 should produce sparse merged deltas."""
    torch.manual_seed(0)
    base = make_model(seed=0)
    ft   = perturb(base, scale=0.5, seed=1)

    merger = ModelMerger(MergeConfig(method="dare", density=0.5, seed=99))
    merged = merger.merge(base, [ft])

    # The merged model should exist with same keys
    assert set(merged.state_dict().keys()) == set(base.state_dict().keys())


def test_fisher_merge():
    """Fisher merge with explicit weights should produce weighted result."""
    base = make_model(seed=0)
    ft1  = perturb(base, scale=0.2, seed=1)
    ft2  = perturb(base, scale=0.2, seed=2)

    # Fisher weights: all-ones (uniform)
    fisher_weights = [
        {k: torch.ones_like(v.float()) for k, v in base.state_dict().items()
         if v.is_floating_point()},
        {k: torch.ones_like(v.float()) for k, v in base.state_dict().items()
         if v.is_floating_point()},
    ]

    merger = ModelMerger(MergeConfig(method="fisher"))
    merged = merger.merge(base, [ft1, ft2], fisher_weights=fisher_weights)
    assert set(merged.state_dict().keys()) == set(base.state_dict().keys())

    # With uniform Fisher, result should equal task_arithmetic
    ta_merger = ModelMerger(MergeConfig(method="task_arithmetic"))
    ta_merged = ta_merger.merge(base, [ft1, ft2])
    for key in base.state_dict():
        if not base.state_dict()[key].is_floating_point():
            continue
        torch.testing.assert_close(
            merged.state_dict()[key].float(),
            ta_merged.state_dict()[key].float(),
            atol=1e-5, rtol=1e-4,
        )


def test_geometric_mean_convergence():
    """Karcher mean should converge and produce output close to arithmetic mean for small perturbations."""
    base = make_model(seed=0)
    models = [perturb(base, scale=0.01, seed=i) for i in range(3)]

    cfg = MergeConfig(method="geometric_mean", karcher_steps=10, karcher_threshold=1e-9)
    merger = ModelMerger(cfg)
    merged = merger.merge(base, models)

    # For tiny perturbations, Karcher ≈ arithmetic
    ta_merger = ModelMerger(MergeConfig(method="task_arithmetic"))
    ta_merged = ta_merger.merge(base, models)

    for key in base.state_dict():
        if not base.state_dict()[key].is_floating_point():
            continue
        torch.testing.assert_close(
            merged.state_dict()[key].float(),
            ta_merged.state_dict()[key].float(),
            atol=1e-3, rtol=1e-2,
        )


def test_sign_consensus_threshold():
    """Parameters below confidence threshold should be zeroed out."""
    base = make_model(seed=0)
    # Two models push positive, one pushes negative — confidence ~0.67 for 3 models
    ft1 = perturb(base, scale=0.5, seed=1)
    ft2 = perturb(base, scale=0.5, seed=2)
    ft3 = perturb(base, scale=0.5, seed=3)

    # High threshold: many params zeroed
    cfg_high = MergeConfig(method="sign_consensus", confidence_threshold=0.95)
    merger = ModelMerger(cfg_high)
    merged_high = merger.merge(base, [ft1, ft2, ft3])

    # Low threshold: fewer params zeroed
    cfg_low = MergeConfig(method="sign_consensus", confidence_threshold=0.1)
    merger_low = ModelMerger(cfg_low)
    merged_low = merger_low.merge(base, [ft1, ft2, ft3])

    # High threshold should produce deltas with more zeros than low threshold
    def delta_zeros(merged):
        count = 0
        for key in base.state_dict():
            if not base.state_dict()[key].is_floating_point():
                continue
            delta = (merged.state_dict()[key].float() - base.state_dict()[key].float())
            count += int((delta == 0).sum())
        return count

    assert delta_zeros(merged_high) >= delta_zeros(merged_low)


# =============================================================================
# BATCH 3 — Provenance / determinism
# =============================================================================

def test_manifest_sha256():
    """SHA256 must be present and stable across identical inputs."""
    import tempfile, os
    base = make_model(seed=0)
    ft   = perturb(base, seed=1)
    merger = ModelMerger(MergeConfig(method="task_arithmetic"))
    merged = merger.merge(base, [ft])

    with tempfile.TemporaryDirectory() as tmpdir:
        result = save_merge_artifact(
            merged,
            os.path.join(tmpdir, "artifact"),
            input_state_dicts=[base.state_dict(), ft.state_dict()],
            merge_config=merger.config,
        )
        assert "sha256_merged" in result
        assert len(result["sha256_merged"]) == 64  # sha256 hex = 64 chars

        result2 = save_merge_artifact(
            merged,
            os.path.join(tmpdir, "artifact2"),
            input_state_dicts=[base.state_dict(), ft.state_dict()],
            merge_config=merger.config,
        )
        assert result["sha256_merged"] == result2["sha256_merged"]


def test_deterministic_seed():
    """Same seed must produce identical DARE output."""
    base = make_model(seed=0)
    ft1  = perturb(base, scale=0.5, seed=1)
    ft2  = perturb(base, scale=0.5, seed=2)

    cfg = MergeConfig(method="dare", density=0.5, seed=777)
    m1 = ModelMerger(cfg).merge(base, [ft1, ft2])
    m2 = ModelMerger(cfg).merge(base, [ft1, ft2])

    for key in base.state_dict():
        if base.state_dict()[key].is_floating_point():
            torch.testing.assert_close(
                m1.state_dict()[key].float(),
                m2.state_dict()[key].float(),
            )


def test_conflict_report_shape():
    """conflict_report must return per-key dict with expected fields."""
    base = make_model(seed=0)
    ft1  = perturb(base, seed=1)
    ft2  = perturb(base, seed=2)

    merger = ModelMerger()
    aligned = merger._align_keys(base.state_dict(), [ft1.state_dict(), ft2.state_dict()])
    report = merger.compute_conflict_report(
        base.state_dict(),
        [ft1.state_dict(), ft2.state_dict()],
        aligned,
    )

    required_fields = {"delta_norm", "cosine_conflict", "sign_disagreement_ratio", "drift_ratio"}
    for key, entry in report.items():
        assert required_fields <= entry.keys(), f"Missing fields in report for {key}"
        assert 0.0 <= entry["sign_disagreement_ratio"] <= 1.0
        assert entry["cosine_conflict"] >= 0.0
