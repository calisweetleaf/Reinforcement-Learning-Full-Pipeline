"""
Model Merging and Ensembling for RLHF

Implements:
- Task Arithmetic merging
- TIES-Merging (Trim, Elect Sign & Merge)
- SLERP (Spherical Linear Interpolation)
- DARE (Drop And REscale)
- Fisher-Weighted merging
- RegMean merging
- Geometric (Karcher) mean merging
- Sign Consensus merging
- Model Soups
- Conflict analytics
- Deterministic SHA256 manifests
"""

import copy
import hashlib
import json
import logging
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__version__ = "2.0.0"

logger = logging.getLogger("ModelMerging")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MergeConfig:
    """Configuration for model merging."""
    method: str = "ties"  # task_arithmetic, ties, slerp, dare, fisher, regmean, geometric_mean, sign_consensus
    weights: Optional[List[float]] = None   # Per-model weights
    density: float = 0.6                    # For TIES/DARE: fraction of params to keep
    epsilon: float = 1e-8                   # Numerical stability
    normalize: bool = True                  # Normalize deltas
    seed: int = 42                          # For deterministic DARE masking

    # Per-layer density overrides: {regex_pattern: density}
    # e.g. {"layers\\.0\\..*": 0.9, "lm_head.*": 1.0}
    layer_density_map: Optional[Dict[str, float]] = None

    # Sign consensus confidence threshold (fraction of models that must agree)
    confidence_threshold: float = 0.5

    # Karcher mean parameters
    karcher_steps: int = 5
    karcher_threshold: float = 1e-6


# =============================================================================
# CORE MERGER
# =============================================================================

class ModelMerger:
    """
    Merge multiple models into one.

    All merge methods:
      - compute deltas in float32 (prevents silent skipping of bfloat16/float16 params)
      - cast merged parameters back to original dtype
      - emit warnings for key mismatches
    """

    def __init__(self, config: MergeConfig = None):
        self.config = config or MergeConfig()

    def merge(
        self,
        base_model: nn.Module,
        fine_tuned_models: List[nn.Module],
        config: MergeConfig = None,
        fisher_weights: Optional[List[Dict[str, torch.Tensor]]] = None,
        gram_matrices: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> nn.Module:
        """
        Merge fine-tuned models with base.

        Args:
            base_model: Pre-trained base model
            fine_tuned_models: List of fine-tuned models
            config: Optional merge configuration (overrides self.config)
            fisher_weights: Per-model Fisher info dicts (fisher method only)
            gram_matrices: Per-model activation Gram matrices (regmean method only)

        Returns:
            Merged model (copy of base_model with merged weights)
        """
        if config is None:
            config = self.config

        # Method-level pre-validation
        self._validate(fine_tuned_models, config)

        # Deterministic seed
        torch.manual_seed(config.seed)

        base_state = base_model.state_dict()
        ft_states = [m.state_dict() for m in fine_tuned_models]

        # Key alignment with diagnostics
        aligned_keys = self._align_keys(base_state, ft_states)

        # Original dtypes for cast-back
        orig_dtypes = {key: base_state[key].dtype for key in aligned_keys}

        # Compute float32 deltas (only for floating-point params)
        deltas = []
        for ft_state in ft_states:
            delta = {}
            for key in aligned_keys:
                if base_state[key].is_floating_point():
                    delta[key] = ft_state[key].float() - base_state[key].float()
            deltas.append(delta)

        # Apply merging method
        if config.method == "task_arithmetic":
            merged_delta = self._task_arithmetic(deltas, config.weights)
        elif config.method == "ties":
            merged_delta = self._ties_merge(deltas, config)
        elif config.method == "slerp":
            merged_delta = self._slerp_merge(base_state, ft_states, aligned_keys, config)
        elif config.method == "dare":
            merged_delta = self._dare_merge(deltas, config)
        elif config.method == "fisher":
            merged_delta = self._fisher_weighted_merge(deltas, fisher_weights, config)
        elif config.method == "regmean":
            merged_delta = self._regmean_merge(deltas, gram_matrices, config)
        elif config.method == "geometric_mean":
            merged_delta = self._geometric_mean_merge(deltas, config)
        elif config.method == "sign_consensus":
            merged_delta = self._sign_consensus_merge(deltas, config)
        else:
            raise ValueError(f"Unknown merge method: {config.method}")

        # Apply delta to base, casting back to original dtype
        merged_state = {}
        for key in base_state.keys():
            if key in merged_delta:
                merged_f32 = base_state[key].float() + merged_delta[key]
                merged_state[key] = merged_f32.to(orig_dtypes.get(key, torch.float32))
            else:
                merged_state[key] = base_state[key]

        merged_model = copy.deepcopy(base_model)
        merged_model.load_state_dict(merged_state)
        return merged_model

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, fine_tuned_models: List[nn.Module], config: MergeConfig):
        """Pre-validate method-specific requirements."""
        if config.method == "slerp":
            if len(fine_tuned_models) != 2:
                raise ValueError(
                    f"SLERP requires exactly 2 fine-tuned models, got {len(fine_tuned_models)}. "
                    "Use 'task_arithmetic' or 'ties' for 3+ models."
                )
        if config.method in ("dare", "ties"):
            if not (0 < config.density <= 1.0):
                raise ValueError(
                    f"density must be in (0, 1], got {config.density}. "
                    "Try density=0.6 as a starting point."
                )
        if config.method == "fisher":
            logger.debug("Fisher method: supply fisher_weights for best results.")

    # ------------------------------------------------------------------
    # Key alignment
    # ------------------------------------------------------------------

    def _align_keys(
        self,
        base_state: Dict[str, torch.Tensor],
        ft_states: List[Dict[str, torch.Tensor]],
    ) -> List[str]:
        """
        Compute the intersection of keys across base and all ft_states.
        Emits warnings for mismatches and returns sorted intersection list.
        """
        base_keys = set(base_state.keys())
        intersection = base_keys.copy()

        for i, ft_state in enumerate(ft_states):
            ft_keys = set(ft_state.keys())
            missing_in_ft = base_keys - ft_keys
            extra_in_ft = ft_keys - base_keys

            if missing_in_ft:
                preview = sorted(missing_in_ft)[:5]
                logger.warning(
                    f"Model {i}: {len(missing_in_ft)} base keys missing in fine-tuned model: "
                    f"{preview}{'...' if len(missing_in_ft) > 5 else ''}"
                )
            if extra_in_ft:
                preview = sorted(extra_in_ft)[:5]
                logger.warning(
                    f"Model {i}: {len(extra_in_ft)} extra keys in fine-tuned model (ignored): "
                    f"{preview}{'...' if len(extra_in_ft) > 5 else ''}"
                )
            intersection &= ft_keys

        dropped = base_keys - intersection
        if dropped:
            logger.warning(
                f"Key alignment: dropped {len(dropped)} keys not present in all models."
            )
        logger.info(
            f"Key alignment: {len(intersection)}/{len(base_keys)} keys used for merging."
        )
        return sorted(intersection)

    # ------------------------------------------------------------------
    # Per-layer density helper
    # ------------------------------------------------------------------

    def _get_layer_density(self, key: str, config: MergeConfig) -> float:
        """Return per-layer density override (regex match) or config.density."""
        if config.layer_density_map:
            for pattern, density in config.layer_density_map.items():
                if re.search(pattern, key):
                    return density
        return config.density

    # ------------------------------------------------------------------
    # Merge algorithms
    # ------------------------------------------------------------------

    def _task_arithmetic(
        self,
        deltas: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]],
    ) -> Dict[str, torch.Tensor]:
        """Weighted sum of deltas, normalized by total weight."""
        if weights is None:
            weights = [1.0 / len(deltas)] * len(deltas)

        total_w = sum(weights)
        merged = {}
        for key in deltas[0].keys():
            merged[key] = sum(w * d[key] for w, d in zip(weights, deltas)) / total_w
        return merged

    def _ties_merge(
        self,
        deltas: List[Dict[str, torch.Tensor]],
        config: MergeConfig,
    ) -> Dict[str, torch.Tensor]:
        """
        TIES-Merging: Trim, Elect Sign & Merge.

        1. Trim: keep top-k% by magnitude (per-layer density)
        2. Elect: majority vote on sign
        3. Merge: average only parameters agreeing with majority sign
        """
        # Step 1: Trim
        trimmed_deltas = []
        for delta in deltas:
            trimmed = {}
            for key, tensor in delta.items():
                density = self._get_layer_density(key, config)
                flat = tensor.abs().flatten()
                keep_count = max(1, int(density * flat.numel()))
                kth = max(1, flat.numel() - keep_count + 1)
                threshold = torch.kthvalue(flat, kth)[0]
                trimmed[key] = tensor * (tensor.abs() >= threshold)
            trimmed_deltas.append(trimmed)

        # Step 2: Elect sign
        elected_deltas = [{} for _ in trimmed_deltas]
        for key in trimmed_deltas[0].keys():
            sign_sum = sum(torch.sign(d[key]) for d in trimmed_deltas)
            majority_sign = torch.sign(sign_sum)
            for i, delta in enumerate(trimmed_deltas):
                mask = torch.sign(delta[key]) == majority_sign
                elected_deltas[i][key] = delta[key] * mask

        # Step 3: Average
        merged = {}
        for key in trimmed_deltas[0].keys():
            merged[key] = sum(d[key] for d in elected_deltas) / len(elected_deltas)
        return merged

    def _slerp_merge(
        self,
        base_state: Dict[str, torch.Tensor],
        ft_states: List[Dict[str, torch.Tensor]],
        aligned_keys: List[str],
        config: MergeConfig,
    ) -> Dict[str, torch.Tensor]:
        """
        Spherical Linear Interpolation between exactly 2 fine-tuned models.
        Returns delta dict (merged_position - base_position).
        """
        t = config.weights[0] if config.weights else 0.5
        merged_deltas = {}

        for key in aligned_keys:
            if not base_state[key].is_floating_point():
                continue

            base_f32 = base_state[key].float()
            v0 = ft_states[0][key].float() - base_f32
            v1 = ft_states[1][key].float() - base_f32

            norm0 = v0.flatten().norm()
            norm1 = v1.flatten().norm()

            if norm0 < config.epsilon or norm1 < config.epsilon:
                merged_deltas[key] = (1 - t) * v0 + t * v1
                continue

            cos_omega = (
                torch.dot(v0.flatten(), v1.flatten())
                / (norm0 * norm1 + config.epsilon)
            )
            omega = torch.acos(torch.clamp(cos_omega, -1.0, 1.0))
            sin_omega = torch.sin(omega)

            if sin_omega < config.epsilon:
                merged_deltas[key] = (1 - t) * v0 + t * v1
            else:
                coef0 = torch.sin((1 - t) * omega) / sin_omega
                coef1 = torch.sin(t * omega) / sin_omega
                merged_deltas[key] = coef0 * v0 + coef1 * v1

        return merged_deltas

    def _dare_merge(
        self,
        deltas: List[Dict[str, torch.Tensor]],
        config: MergeConfig,
    ) -> Dict[str, torch.Tensor]:
        """
        DARE: Drop And REscale with per-layer density support.
        Randomly masks parameters and rescales to maintain expected magnitude.
        """
        merged = {}
        for key in deltas[0].keys():
            density = self._get_layer_density(key, config)
            all_deltas = torch.stack([d[key] for d in deltas])
            mask = torch.rand_like(all_deltas[0]) < density
            masked = all_deltas * mask.unsqueeze(0)
            rescaled = masked / (density + config.epsilon)
            merged[key] = rescaled.mean(dim=0)
        return merged

    def _fisher_weighted_merge(
        self,
        deltas: List[Dict[str, torch.Tensor]],
        fisher_weights: Optional[List[Dict[str, torch.Tensor]]],
        config: MergeConfig,
    ) -> Dict[str, torch.Tensor]:
        """
        Fisher-weighted merge: weight each delta by diagonal Fisher information estimate.
        Falls back to parameter-magnitude approximation if fisher_weights is None.
        """
        if fisher_weights is None:
            logger.warning(
                "fisher_weights not provided; approximating Fisher via parameter magnitude."
            )
            fisher_weights = [
                {key: t.abs() + config.epsilon for key, t in d.items()}
                for d in deltas
            ]

        merged = {}
        for key in deltas[0].keys():
            stacked_deltas = torch.stack([d[key] for d in deltas], dim=0)   # (n, ...)
            stacked_fisher = torch.stack([fw[key] for fw in fisher_weights], dim=0)
            fisher_sum = stacked_fisher.sum(dim=0).clamp(min=config.epsilon)
            weights_normalized = stacked_fisher / fisher_sum.unsqueeze(0)
            merged[key] = (stacked_deltas * weights_normalized).sum(dim=0)
        return merged

    def _regmean_merge(
        self,
        deltas: List[Dict[str, torch.Tensor]],
        gram_matrices: Optional[List[Dict[str, torch.Tensor]]],
        config: MergeConfig,
    ) -> Dict[str, torch.Tensor]:
        """
        RegMean: activation-Gram-matrix-weighted interpolation.
        Falls back to task_arithmetic if gram_matrices is None.
        """
        if gram_matrices is None:
            logger.warning(
                "gram_matrices not provided for 'regmean'; falling back to task_arithmetic."
            )
            return self._task_arithmetic(deltas, config.weights)

        merged = {}
        for key in deltas[0].keys():
            if key not in gram_matrices[0]:
                merged[key] = sum(d[key] for d in deltas) / len(deltas)
                continue

            stacked_grams = [gram_matrices[m][key] for m in range(len(deltas))]
            gram_sum = sum(stacked_grams).clamp(min=config.epsilon)
            weighted_delta_sum = sum(
                gm * d[key] for gm, d in zip(stacked_grams, deltas)
            )
            merged[key] = weighted_delta_sum / gram_sum
        return merged

    def _geometric_mean_merge(
        self,
        deltas: List[Dict[str, torch.Tensor]],
        config: MergeConfig,
    ) -> Dict[str, torch.Tensor]:
        """
        Karcher (Frechet) mean in the delta tangent space.
        Iteratively refines the arithmetic mean towards the geometric mean.
        Converges within karcher_steps or when gradient norm < karcher_threshold.
        """
        if not deltas:
            return {}

        n = len(deltas)
        keys = list(deltas[0].keys())

        # Initialize with arithmetic mean
        mean_delta = {key: sum(d[key] for d in deltas) / n for key in keys}

        for step in range(config.karcher_steps):
            gradient = {}
            total_norm = 0.0
            for key in keys:
                residuals = torch.stack([d[key] - mean_delta[key] for d in deltas])
                grad = residuals.mean(dim=0)
                gradient[key] = grad
                total_norm += float(grad.norm())

            if total_norm < config.karcher_threshold:
                logger.debug(f"Karcher mean converged at step {step + 1}.")
                break

            for key in keys:
                mean_delta[key] = mean_delta[key] + gradient[key]

        return mean_delta

    def _sign_consensus_merge(
        self,
        deltas: List[Dict[str, torch.Tensor]],
        config: MergeConfig,
    ) -> Dict[str, torch.Tensor]:
        """
        Extended TIES: requires config.confidence_threshold fraction of models
        to agree on sign. Parameters below threshold are zeroed out.
        """
        merged = {}
        for key in deltas[0].keys():
            stacked = torch.stack([d[key] for d in deltas], dim=0)   # (n, ...)
            signs = torch.sign(stacked)

            pos_fraction = (signs > 0).float().mean(dim=0)
            neg_fraction = (signs < 0).float().mean(dim=0)
            confidence = torch.maximum(pos_fraction, neg_fraction)

            majority_sign = torch.sign(signs.sum(dim=0))
            mask = confidence >= config.confidence_threshold

            agreed = stacked * (torch.sign(stacked) == majority_sign.unsqueeze(0)).float()
            merged[key] = agreed.mean(dim=0) * mask
        return merged

    # ------------------------------------------------------------------
    # Conflict analytics
    # ------------------------------------------------------------------

    def compute_conflict_report(
        self,
        base_state: Dict[str, torch.Tensor],
        ft_states: List[Dict[str, torch.Tensor]],
        aligned_keys: List[str],
    ) -> Dict[str, Dict]:
        """
        Per-layer conflict analytics.

        Returns per key:
            delta_norm              — L2 norm of each model's delta
            cosine_conflict         — mean pairwise cosine distance between deltas
            sign_disagreement_ratio — fraction of params where models disagree on sign
            drift_ratio             — mean_delta_norm / base_param_norm
        """
        report = {}
        for key in aligned_keys:
            if not base_state[key].is_floating_point():
                continue

            base_f32 = base_state[key].float().flatten()
            key_deltas = []
            for ft_state in ft_states:
                if key in ft_state:
                    key_deltas.append((ft_state[key].float().flatten() - base_f32))

            if not key_deltas:
                continue

            stacked = torch.stack(key_deltas, dim=0)   # (n, d)
            norms = stacked.norm(dim=1)
            base_norm = base_f32.norm()

            # Pairwise cosine distance
            n = len(key_deltas)
            cosine_conflicts = []
            for i in range(n):
                for j in range(i + 1, n):
                    vi = key_deltas[i]
                    vj = key_deltas[j]
                    norm_i = vi.norm()
                    norm_j = vj.norm()
                    if norm_i > 1e-9 and norm_j > 1e-9:
                        sim = float(torch.dot(vi, vj) / (norm_i * norm_j))
                        cosine_conflicts.append(1.0 - sim)

            # Sign disagreement ratio
            signs = torch.sign(stacked)
            sign_agreement = (signs == signs[0:1]).all(dim=0)
            sign_disagreement_ratio = float((~sign_agreement).float().mean())

            report[key] = {
                "delta_norm": norms.tolist(),
                "cosine_conflict": float(np.mean(cosine_conflicts)) if cosine_conflicts else 0.0,
                "sign_disagreement_ratio": sign_disagreement_ratio,
                "drift_ratio": float(norms.mean() / (float(base_norm) + 1e-8)),
            }
        return report


# =============================================================================
# SHA256 / PROVENANCE
# =============================================================================

def _sha256_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    """Compute a stable SHA256 hash of a state_dict for provenance tracking."""
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        h.update(key.encode("utf-8"))
        if tensor.is_floating_point():
            h.update(tensor.float().detach().cpu().numpy().tobytes())
        else:
            h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


# =============================================================================
# CHECKPOINT UTILITIES
# =============================================================================

def list_stage_checkpoints(
    checkpoint_root: str, stage_name: Optional[str] = None
) -> List[Path]:
    """
    List checkpoint directories produced by rlhf.CheckpointManager.

    Example directory names: DPO_step_20, SFT_step_40
    """
    root = Path(checkpoint_root)
    pattern = f"{stage_name}_step_*" if stage_name else "*_step_*"
    return sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)


def load_checkpoint_state_dict(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    """Load model.pt state_dict from a checkpoint directory."""
    ckpt_dir = Path(checkpoint_dir)
    model_path = ckpt_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint model not found: {model_path}")
    return torch.load(model_path, map_location="cpu")


# =============================================================================
# ARTIFACT SAVING WITH DETERMINISTIC MANIFEST
# =============================================================================

def save_merge_artifact(
    merged_model: nn.Module,
    output_dir: str,
    metadata: Optional[Dict[str, Any]] = None,
    input_state_dicts: Optional[List[Dict[str, torch.Tensor]]] = None,
    merge_config: Optional[MergeConfig] = None,
    conflict_report: Optional[Dict] = None,
    key_alignment_summary: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Save merged model state_dict and a deterministic JSON manifest for provenance.

    Manifest includes:
        - SHA256 of merged model and each input state_dict
        - Full merge config
        - Conflict report
        - Key alignment summary
        - Code version
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "merged_model.pt"
    torch.save(merged_model.state_dict(), model_path)

    merged_sha256 = _sha256_state_dict(merged_model.state_dict())
    input_sha256s = (
        [_sha256_state_dict(sd) for sd in input_state_dicts]
        if input_state_dicts else []
    )

    config_dict: Dict[str, Any] = {}
    if merge_config is not None:
        config_dict = {
            "method": merge_config.method,
            "density": merge_config.density,
            "seed": merge_config.seed,
            "weights": merge_config.weights,
            "confidence_threshold": merge_config.confidence_threshold,
            "karcher_steps": merge_config.karcher_steps,
            "layer_density_map": merge_config.layer_density_map,
        }

    manifest = {
        "created_at_unix": time.time(),
        "code_version": __version__,
        "model_path": str(model_path),
        "sha256_merged": merged_sha256,
        "sha256_inputs": input_sha256s,
        "merge_config": config_dict,
        "key_alignment_summary": key_alignment_summary or {},
        "conflict_report": conflict_report or {},
        "metadata": metadata or {},
    }

    manifest_path = out / "merge_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    return {
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "sha256_merged": merged_sha256,
    }


# =============================================================================
# MODEL SOUPS
# =============================================================================

class ModelSoup:
    """
    Model Soups: Averaging weights of multiple fine-tuned models.
    Simpler than merging, often surprisingly effective.
    """

    @staticmethod
    def create_soup(
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        interpolation_weights: Optional[List[float]] = None,
    ) -> nn.Module:
        """Uniform or weighted parameter averaging."""
        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        states = [m.state_dict() for m in models]
        total_w = sum(weights)
        souped_state = {}
        for key in states[0].keys():
            souped_state[key] = (
                sum(w * state[key] for w, state in zip(weights, states)) / total_w
            )

        souped = copy.deepcopy(models[0])
        souped.load_state_dict(souped_state)
        return souped

    @staticmethod
    def greedy_soup(
        base_model: nn.Module,
        fine_tuned_models: List[nn.Module],
        eval_fn: Callable[[nn.Module], float],
    ) -> nn.Module:
        """
        Greedy soup: Add models one by one if they improve eval_fn performance.
        """
        scores = [(m, eval_fn(m)) for m in fine_tuned_models]
        scores.sort(key=lambda x: x[1], reverse=True)

        current_soup = [scores[0][0]]
        best_score = scores[0][1]
        best_soup = current_soup.copy()

        for model, _ in scores[1:]:
            trial_soup = current_soup + [model]
            souped = ModelSoup.create_soup(trial_soup)
            trial_score = eval_fn(souped)
            if trial_score > best_score:
                current_soup = trial_soup
                best_score = trial_score
                best_soup = current_soup.copy()

        return ModelSoup.create_soup(best_soup)


# =============================================================================
# ENSEMBLE POLICY
# =============================================================================

class EnsemblePolicy:
    """Ensemble multiple policies for robust generation."""

    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        method: str = "average",   # average, voting
    ):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.method = method

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        if self.method == "average":
            return self._average_generate(input_ids, max_new_tokens, temperature)
        elif self.method == "voting":
            return self._voting_generate(input_ids, max_new_tokens, temperature)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

    def _average_generate(
        self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float
    ) -> torch.Tensor:
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            all_logits = []
            for model, weight in zip(self.models, self.weights):
                outputs = model(generated)
                logits = outputs.logits[:, -1, :]
                all_logits.append(weight * logits)
            avg_logits = sum(all_logits) / sum(self.weights)
            probs = torch.softmax(avg_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == self.models[0].config.eos_token_id:
                break
        return generated

    def _voting_generate(
        self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float
    ) -> torch.Tensor:
        all_sequences = []
        for model in self.models:
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=1,
            )
            all_sequences.append(outputs)

        from collections import Counter
        sequences_str = [str(s.tolist()) for s in all_sequences]
        voted = Counter(sequences_str).most_common(1)[0][0]
        for seq in all_sequences:
            if str(seq.tolist()) == voted:
                return seq
        return all_sequences[0]


# =============================================================================
# LAYER-WISE INTERPOLATION
# =============================================================================

def layer_wise_interpolation(
    model1: nn.Module,
    model2: nn.Module,
    layer_weights: List[float],
) -> nn.Module:
    """
    Interpolate between models with different weights per layer.
    Useful for preserving lower layers while tuning upper layers.
    """
    state1 = model1.state_dict()
    state2 = model2.state_dict()

    layer_groups: Dict[int, Dict] = defaultdict(dict)
    for key in state1.keys():
        match = re.search(r'layer\.(\d+)', key)
        if match:
            layer_num = int(match.group(1))
            layer_groups[layer_num][key] = (state1[key], state2[key])
        else:
            layer_groups[-1][key] = (state1[key], state2[key])

    merged_state = {}
    for layer_num, params in layer_groups.items():
        if layer_num == -1:
            weight = 0.5
        else:
            idx = min(layer_num, len(layer_weights) - 1)
            weight = layer_weights[idx]
        for key, (p1, p2) in params.items():
            merged_state[key] = (1 - weight) * p1 + weight * p2

    merged = copy.deepcopy(model1)
    merged.load_state_dict(merged_state)
    return merged


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print(f"Model Merging and Ensembling Module v{__version__}")
    print("=" * 60)
    print("\nAvailable merge methods:")
    print("  task_arithmetic  — Weighted sum of deltas")
    print("  ties             — Trim, Elect Sign & Merge (recommended for 3+ models)")
    print("  slerp            — Spherical linear interpolation (2 models only)")
    print("  dare             — Drop And REscale")
    print("  fisher           — Fisher-information-weighted merging")
    print("  regmean          — Activation-gram-weighted merging")
    print("  geometric_mean   — Karcher mean in delta space")
    print("  sign_consensus   — Confidence-thresholded TIES")
    print("\nExample usage:")
    print("""
    merger = ModelMerger(MergeConfig(method="ties", density=0.6, seed=42))
    merged = merger.merge(base_model, [math_model, code_model, reasoning_model])

    # Save with provenance manifest
    save_merge_artifact(merged, "output/merged", merge_config=merger.config)

    # Greedy Soup
    best_soup = ModelSoup.greedy_soup(base_model, fine_tuned_models, eval_fn)

    # Ensemble for generation
    ensemble = EnsemblePolicy([model1, model2, model3], weights=[0.5, 0.3, 0.2])
    output = ensemble.generate(input_ids)
    """)
