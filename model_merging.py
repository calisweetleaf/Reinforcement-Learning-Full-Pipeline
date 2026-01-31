"""
Model Merging and Ensembling for RLHF

Implements:
- Task Arithmetic merging
- TIES-Merging (Trim, Elect Sign & Merge)
- SLERP (Spherical Linear Interpolation)
- DARE (Drop And REscale)
- Model Soups
"""

import torch
import torch.nn as nn
import copy
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass
import re


@dataclass
class MergeConfig:
    """Configuration for model merging."""
    method: str = "ties"  # task_arithmetic, ties, slerp, dare
    weights: Optional[List[float]] = None  # Per-model weights
    density: float = 0.6  # For TIES/DARE: fraction of params to keep
    epsilon: float = 1e-8  # Numerical stability
    normalize: bool = True  # Normalize deltas


class ModelMerger:
    """
    Merge multiple fine-tuned models into one.
    
    Based on research on model merging for multi-task learning.
    """
    
    def __init__(self, config: MergeConfig = None):
        self.config = config or MergeConfig()
    
    def merge(
        self,
        base_model: nn.Module,
        fine_tuned_models: List[nn.Module],
        config: MergeConfig = None
    ) -> nn.Module:
        """
        Merge fine-tuned models with base.
        
        Args:
            base_model: Pre-trained base model
            fine_tuned_models: List of fine-tuned models
            config: Optional merge configuration
        
        Returns:
            Merged model
        """
        if config is None:
            config = self.config
        
        # Get model weights as state dicts
        base_state = base_model.state_dict()
        ft_states = [m.state_dict() for m in fine_tuned_models]
        
        # Compute parameter deltas
        deltas = []
        for ft_state in ft_states:
            delta = {}
            for key in base_state.keys():
                if key in ft_state and base_state[key].dtype == torch.float32:
                    delta[key] = ft_state[key] - base_state[key]
            deltas.append(delta)
        
        # Apply merging method
        if config.method == "task_arithmetic":
            merged_delta = self._task_arithmetic(deltas, config.weights)
        elif config.method == "ties":
            merged_delta = self._ties_merge(deltas, config)
        elif config.method == "slerp":
            merged_delta = self._slerp_merge(base_state, ft_states, config)
        elif config.method == "dare":
            merged_delta = self._dare_merge(deltas, config)
        else:
            raise ValueError(f"Unknown merge method: {config.method}")
        
        # Apply merged delta to base
        merged_state = {}
        for key in base_state.keys():
            if key in merged_delta:
                merged_state[key] = base_state[key] + merged_delta[key]
            else:
                merged_state[key] = base_state[key]
        
        # Create merged model
        merged_model = copy.deepcopy(base_model)
        merged_model.load_state_dict(merged_state)
        
        return merged_model
    
    def _task_arithmetic(
        self,
        deltas: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]]
    ) -> Dict[str, torch.Tensor]:
        """
        Simple task arithmetic: weighted sum of deltas.
        """
        if weights is None:
            weights = [1.0 / len(deltas)] * len(deltas)
        
        merged = {}
        for key in deltas[0].keys():
            weighted_sum = sum(
                w * delta[key] for w, delta in zip(weights, deltas)
            )
            merged[key] = weighted_sum / sum(weights)
        
        return merged
    
    def _ties_merge(
        self,
        deltas: List[Dict[str, torch.Tensor]],
        config: MergeConfig
    ) -> Dict[str, torch.Tensor]:
        """
        TIES-Merging: Trim, Elect Sign & Merge.
        
        Steps:
        1. Trim: Keep only top-k% by magnitude
        2. Elect Sign: Majority vote on sign
        3. Merge: Average only parameters agreeing with majority
        """
        # Step 1: Trim (keep top-k% by magnitude)
        trimmed_deltas = []
        for delta in deltas:
            trimmed = {}
            for key, tensor in delta.items():
                # Flatten and find threshold
                flat = tensor.abs().flatten()
                k = int(config.density * flat.numel())
                threshold = torch.kthvalue(flat, k)[0] if k > 0 else float('inf')
                
                # Create mask for top-k%
                mask = tensor.abs() >= threshold
                trimmed[key] = tensor * mask
            trimmed_deltas.append(trimmed)
        
        # Step 2: Elect Sign (majority vote)
        elected_deltas = []
        for key in trimmed_deltas[0].keys():
            # Collect signs from all models
            signs = [torch.sign(d[key]) for d in trimmed_deltas]
            
            # Majority vote
            sign_sum = sum(signs)
            majority_sign = torch.sign(sign_sum)
            
            # Keep only parameters matching majority sign
            for i, delta in enumerate(trimmed_deltas):
                mask = torch.sign(delta[key]) == majority_sign
                elected = delta[key] * mask
                
                if key not in elected_deltas:
                    elected_deltas.append({})
                elected_deltas[i][key] = elected
        
        # Step 3: Merge (average)
        merged = {}
        for key in trimmed_deltas[0].keys():
            merged[key] = sum(d[key] for d in elected_deltas) / len(elected_deltas)
        
        return merged
    
    def _slerp_merge(
        self,
        base_state: Dict[str, torch.Tensor],
        ft_states: List[Dict[str, torch.Tensor]],
        config: MergeConfig
    ) -> Dict[str, torch.Tensor]:
        """
        Spherical Linear Interpolation.
        Better for merging in high-dimensional spaces.
        """
        if len(ft_states) != 2:
            raise ValueError("SLERP requires exactly 2 models")
        
        t = config.weights[0] if config.weights else 0.5
        
        merged = {}
        for key in base_state.keys():
            if key not in ft_states[0] or key not in ft_states[1]:
                merged[key] = base_state[key]
                continue
            
            # Get vectors from base to each FT model
            v0 = ft_states[0][key] - base_state[key]
            v1 = ft_states[1][key] - base_state[key]
            
            # Flatten for angle computation
            v0_flat = v0.flatten()
            v1_flat = v1.flatten()
            
            # Compute angle between vectors
            cos_omega = torch.dot(v0_flat, v1_flat) / (
                torch.norm(v0_flat) * torch.norm(v1_flat) + config.epsilon
            )
            omega = torch.acos(torch.clamp(cos_omega, -1, 1))
            
            # SLERP formula
            sin_omega = torch.sin(omega)
            if sin_omega < config.epsilon:
                # Linear interpolation for small angles
                result = base_state[key] + (1 - t) * v0 + t * v1
            else:
                # Spherical interpolation
                coef0 = torch.sin((1 - t) * omega) / sin_omega
                coef1 = torch.sin(t * omega) / sin_omega
                result = base_state[key] + coef0 * v0 + coef1 * v1
            
            merged[key] = result
        
        return {k: merged[k] - base_state[k] for k in merged.keys()}
    
    def _dare_merge(
        self,
        deltas: List[Dict[str, torch.Tensor]],
        config: MergeConfig
    ) -> Dict[str, torch.Tensor]:
        """
        DARE: Drop And REscale.
        Randomly drop parameters and rescale remaining.
        """
        merged = {}
        
        for key in deltas[0].keys():
            all_deltas = torch.stack([d[key] for d in deltas])
            
            # Random mask (keep density fraction)
            mask = torch.rand_like(all_deltas[0]) < config.density
            
            # Apply mask and rescale
            masked = all_deltas * mask.unsqueeze(0)
            rescaled = masked / config.density  # Rescale to maintain magnitude
            
            # Average
            merged[key] = rescaled.mean(dim=0)
        
        return merged


class ModelSoup:
    """
    Model Soups: Averaging weights of multiple fine-tuned models.
    Simpler than merging, often surprisingly effective.
    """
    
    @staticmethod
    def create_soup(
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        interpolation_weights: Optional[List[float]] = None
    ) -> nn.Module:
        """
        Create uniform or weighted soup.
        
        Args:
            models: List of models to soup
            weights: Per-model weights (uniform if None)
            interpolation_weights: For layer-wise interpolation
        """
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        # Get state dicts
        states = [m.state_dict() for m in models]
        
        # Weighted average
        souped_state = {}
        for key in states[0].keys():
            weighted_sum = sum(
                w * state[key] for w, state in zip(weights, states)
            )
            souped_state[key] = weighted_sum / sum(weights)
        
        # Load into new model
        souped = copy.deepcopy(models[0])
        souped.load_state_dict(souped_state)
        
        return souped
    
    @staticmethod
    def greedy_soup(
        base_model: nn.Module,
        fine_tuned_models: List[nn.Module],
        eval_fn: Callable[[nn.Module], float]
    ) -> nn.Module:
        """
        Greedy soup: Add models one by one if they improve performance.
        
        Args:
            base_model: Starting point
            fine_tuned_models: Models to potentially include
            eval_fn: Function to evaluate model quality
        
        Returns:
            Best performing soup
        """
        # Sort models by individual performance
        scores = [(m, eval_fn(m)) for m in fine_tuned_models]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Start with best model
        current_soup = [scores[0][0]]
        best_score = scores[0][1]
        best_soup = current_soup.copy()
        
        # Try adding each remaining model
        for model, score in scores[1:]:
            trial_soup = current_soup + [model]
            souped = ModelSoup.create_soup(trial_soup)
            
            trial_score = eval_fn(souped)
            
            if trial_score > best_score:
                current_soup = trial_soup
                best_score = trial_score
                best_soup = current_soup.copy()
        
        return ModelSoup.create_soup(best_soup)


class EnsemblePolicy:
    """
    Ensemble multiple policies for robust generation.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        method: str = "average"  # average, voting, weighted_logits
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
        **kwargs
    ) -> torch.Tensor:
        """Generate with ensemble."""
        if self.method == "average":
            return self._average_generate(input_ids, max_new_tokens, temperature)
        elif self.method == "voting":
            return self._voting_generate(input_ids, max_new_tokens, temperature)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _average_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float
    ) -> torch.Tensor:
        """Generate by averaging logits at each step."""
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Get logits from all models
            all_logits = []
            for model, weight in zip(self.models, self.weights):
                outputs = model(generated)
                logits = outputs.logits[:, -1, :]  # Last token
                all_logits.append(weight * logits)
            
            # Average logits
            avg_logits = sum(all_logits) / sum(self.weights)
            
            # Sample
            probs = torch.softmax(avg_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            if next_token.item() == self.models[0].config.eos_token_id:
                break
        
        return generated
    
    def _voting_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float
    ) -> torch.Tensor:
        """Generate full sequences and vote on best."""
        all_sequences = []
        
        for model in self.models:
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=1
            )
            all_sequences.append(outputs)
        
        # Vote: Use most common sequence
        # (simplified - could use embedding similarity)
        sequences_str = [str(s.tolist()) for s in all_sequences]
        from collections import Counter
        voted = Counter(sequences_str).most_common(1)[0][0]
        
        # Find corresponding tensor
        for seq in all_sequences:
            if str(seq.tolist()) == voted:
                return seq
        
        return all_sequences[0]  # Fallback


def layer_wise_interpolation(
    model1: nn.Module,
    model2: nn.Module,
    layer_weights: List[float]
) -> nn.Module:
    """
    Interpolate between models with different weights per layer.
    
    Useful for:
    - Preserving lower layers (syntax) while tuning upper layers (semantics)
    - Smooth transitions between capabilities
    """
    state1 = model1.state_dict()
    state2 = model2.state_dict()
    
    # Group parameters by layer
    layer_groups = defaultdict(dict)
    for key in state1.keys():
        # Extract layer number from key (e.g., "layer.0.attention.q_proj.weight")
        match = re.search(r'layer\.(\d+)', key)
        if match:
            layer_num = int(match.group(1))
            layer_groups[layer_num][key] = (state1[key], state2[key])
        else:
            # Non-layer parameters (embeddings, LM head, etc.)
            layer_groups[-1][key] = (state1[key], state2[key])
    
    # Interpolate each layer with its weight
    merged_state = {}
    max_layer = max(layer_groups.keys())
    
    for layer_num, params in layer_groups.items():
        if layer_num == -1:
            weight = 0.5  # Default for non-layer params
        else:
            # Map layer number to weight index
            idx = min(layer_num, len(layer_weights) - 1)
            weight = layer_weights[idx]
        
        for key, (p1, p2) in params.items():
            merged_state[key] = (1 - weight) * p1 + weight * p2
    
    # Load merged state
    merged = copy.deepcopy(model1)
    merged.load_state_dict(merged_state)
    
    return merged


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("Model Merging and Ensembling Module")
    print("=" * 60)
    
    print("\nAvailable methods:")
    print("  - Task Arithmetic: Simple weighted sum of deltas")
    print("  - TIES-Merging: Trim, Elect Sign & Merge (recommended)")
    print("  - SLERP: Spherical linear interpolation")
    print("  - DARE: Drop And REscale")
    print("  - Model Soups: Uniform or greedy averaging")
    print("  - Ensemble: Multi-model voting/averaging")
    
    print("\nExample usage:")
    print("""
    # TIES-Merging (recommended)
    merger = ModelMerger(MergeConfig(method="ties", density=0.6))
    merged = merger.merge(
        base_model,
        [math_model, code_model, reasoning_model]
    )
    
    # Greedy Soup
    best_soup = ModelSoup.greedy_soup(
        base_model,
        fine_tuned_models,
        eval_fn=lambda m: evaluate_on_val_set(m)
    )
    
    # Ensemble for generation
    ensemble = EnsemblePolicy(
        [model1, model2, model3],
        weights=[0.5, 0.3, 0.2],
        method="average"
    )
    output = ensemble.generate(input_ids)
    """)
