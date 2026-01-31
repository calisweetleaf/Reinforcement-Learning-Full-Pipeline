"""
Inference-Time Compute Optimizations for SOTA++ RLHF

This module provides:
- Best-of-N sampling with reranking
- Monte Carlo Tree Search (MCTS) for reasoning
- Speculative decoding
- KV-cache optimization
- Flash Attention 2 integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Callable, Any
import numpy as np
from dataclasses import dataclass
import math
import copy
from collections import defaultdict
import heapq


# =============================================================================
# FLASH ATTENTION 2 INTEGRATION
# =============================================================================

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func = None


class OptimizedAttention(nn.Module):
    """
    Attention with automatic Flash Attention 2 / SDPA fallback.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self._use_flash = FLASH_ATTN_AVAILABLE
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Try Flash Attention 2 first
        if self._use_flash and hidden_states.is_cuda and is_causal:
            try:
                # Flash attention expects (batch, seqlen, nheads, headdim)
                out = flash_attn_func(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=is_causal,
                    softmax_scale=None  # Use default 1/sqrt(head_dim)
                )
            except Exception:
                # Fallback to SDPA
                out = self._sdpa_attention(q, k, v, attention_mask, is_causal)
        else:
            # Use PyTorch's SDPA
            out = self._sdpa_attention(q, k, v, attention_mask, is_causal)
        
        # Reshape and output projection
        out = out.view(batch_size, seq_len, self.embed_dim)
        out = self.o_proj(out)
        
        return out
    
    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        is_causal: bool
    ) -> torch.Tensor:
        """Scaled Dot Product Attention (native PyTorch)."""
        # Transpose for attention: (batch, nheads, seqlen, headdim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and attention_mask is None
        )
        
        # Transpose back: (batch, seqlen, nheads, headdim)
        out = out.transpose(1, 2)
        return out


# =============================================================================
# KV-CACHE OPTIMIZATION (PagedAttention-style)
# =============================================================================

class PagedKVCache:
    """
    Efficient KV cache management with paging.
    Reduces memory fragmentation and enables efficient batching.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int = 16,
        max_pages: int = 10000,
        dtype: torch.dtype = torch.bfloat16
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages
        
        # Allocate paginated cache
        # Shape: (max_pages, 2 (k/v), page_size, num_heads, head_dim)
        self.cache = torch.zeros(
            (max_pages, 2, page_size, num_heads, head_dim),
            dtype=dtype
        )
        
        # Track allocation
        self.free_pages = list(range(max_pages))
        self.sequence_pages = {}  # seq_id -> list of page indices
    
    def allocate(self, seq_id: str, num_tokens: int) -> List[int]:
        """Allocate pages for a sequence."""
        num_pages_needed = (num_tokens + self.page_size - 1) // self.page_size
        
        if len(self.free_pages) < num_pages_needed:
            raise RuntimeError(f"Out of KV cache memory. Need {num_pages_needed}, have {len(self.free_pages)}")
        
        pages = [self.free_pages.pop() for _ in range(num_pages_needed)]
        self.sequence_pages[seq_id] = pages
        return pages
    
    def append_token(self, seq_id: str, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Append a single token's KV to cache."""
        pages = self.sequence_pages[seq_id]
        seq_len = self.get_sequence_length(seq_id)
        
        page_idx = seq_len // self.page_size
        offset_in_page = seq_len % self.page_size
        
        # Allocate new page if needed
        if page_idx >= len(pages):
            if not self.free_pages:
                raise RuntimeError("Out of KV cache memory")
            pages.append(self.free_pages.pop())
        
        physical_page = pages[page_idx]
        self.cache[physical_page, 0, offset_in_page] = k  # Key
        self.cache[physical_page, 1, offset_in_page] = v  # Value
    
    def get_kv(self, seq_ids: List[str], layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather KV tensors for a batch of sequences."""
        all_k = []
        all_v = []
        
        for seq_id in seq_ids:
            pages = self.sequence_pages[seq_id]
            seq_k = []
            seq_v = []
            
            for page in pages:
                seq_k.append(self.cache[page, 0])  # Keys
                seq_v.append(self.cache[page, 1])  # Values
            
            # Concatenate pages
            all_k.append(torch.cat(seq_k, dim=0))
            all_v.append(torch.cat(seq_v, dim=0))
        
        # Pad to same length for batching
        max_len = max(k.size(0) for k in all_k)
        
        padded_k = torch.stack([
            F.pad(k, (0, 0, 0, 0, 0, max_len - k.size(0)))
            for k in all_k
        ])
        padded_v = torch.stack([
            F.pad(v, (0, 0, 0, 0, 0, max_len - v.size(0)))
            for v in all_v
        ])
        
        return padded_k, padded_v
    
    def get_sequence_length(self, seq_id: str) -> int:
        """Get current length of cached sequence."""
        if seq_id not in self.sequence_pages:
            return 0
        return len(self.sequence_pages[seq_id]) * self.page_size
    
    def free(self, seq_id: str):
        """Free pages for a sequence."""
        if seq_id in self.sequence_pages:
            pages = self.sequence_pages.pop(seq_id)
            self.free_pages.extend(pages)


# =============================================================================
# SPECULATIVE DECODING
# =============================================================================

class SpeculativeDecoder:
    """
    Speculative decoding for 2-3× faster generation.
    Uses small draft model to predict tokens, large model verifies.
    """
    
    def __init__(
        self,
        target_model: nn.Module,  # Large model
        draft_model: nn.Module,   # Small model (can be quantized)
        gamma: int = 5,           # Number of draft tokens to generate
        temperature: float = 1.0
    ):
        self.target = target_model
        self.draft = draft_model
        self.gamma = gamma
        self.temperature = temperature
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        **kwargs
    ) -> torch.Tensor:
        """Generate with speculative decoding."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        generated = input_ids.clone()
        
        while generated.shape[1] < input_ids.shape[1] + max_new_tokens:
            # Step 1: Draft model generates gamma tokens
            draft_tokens = self._draft_generate(generated, self.gamma)
            
            # Step 2: Target model verifies in parallel
            # Run target model on [original + draft_tokens]
            full_input = torch.cat([generated, draft_tokens], dim=1)
            target_logits = self.target(full_input).logits
            
            # Step 3: Accept/reject draft tokens
            accepted = 0
            for i in range(draft_tokens.shape[1]):
                pos = generated.shape[1] + i
                
                # Get target probability for this position
                target_probs = F.softmax(
                    target_logits[:, pos, :] / self.temperature,
                    dim=-1
                )
                
                # Get draft probability
                draft_token = draft_tokens[:, i]
                target_token_prob = target_probs.gather(-1, draft_token.unsqueeze(-1))
                
                # Acceptance criterion (simplified)
                if torch.rand(1, device=device) < target_token_prob:
                    accepted += 1
                    generated = torch.cat([generated, draft_token.unsqueeze(1)], dim=1)
                else:
                    # Rejection: resample from adjusted distribution
                    adjusted_probs = target_probs
                    new_token = torch.multinomial(adjusted_probs, num_samples=1)
                    generated = torch.cat([generated, new_token], dim=1)
                    break
            
            # If all accepted, add one more token from target
            if accepted == self.gamma:
                next_token = torch.multinomial(
                    F.softmax(target_logits[:, -1, :], dim=-1),
                    num_samples=1
                )
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def _draft_generate(self, input_ids: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """Generate draft tokens with small model."""
        draft_tokens = []
        current = input_ids.clone()
        
        for _ in range(num_tokens):
            logits = self.draft(current).logits[:, -1, :]
            probs = F.softmax(logits / self.temperature, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
            draft_tokens.append(token)
            current = torch.cat([current, token], dim=1)
        
        return torch.cat(draft_tokens, dim=1)


# =============================================================================
# BEST-OF-N SAMPLING
# =============================================================================

@dataclass
class BestOfNConfig:
    """Configuration for Best-of-N sampling."""
    n_samples: int = 16
    temperature: float = 1.0
    top_p: float = 0.95
    reward_aggregation: str = "mean"  # mean, max, min
    use_diversity_bonus: bool = True
    diversity_weight: float = 0.1


class BestOfNSampler:
    """
    Best-of-N sampling with reranking.
    Generates N candidates and selects best by reward model.
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        reward_model: nn.Module,
        config: BestOfNConfig = None
    ):
        self.policy = policy_model
        self.reward = reward_model
        self.config = config or BestOfNConfig()
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        tokenizer: Any,
        max_new_tokens: int = 256,
        return_all: bool = False
    ) -> Dict[str, Any]:
        """
        Generate with Best-of-N selection.
        
        Returns:
            Dictionary with 'best', 'all_candidates', 'scores', etc.
        """
        # Generate N candidates
        candidates = []
        for _ in range(self.config.n_samples):
            output = self._generate_single(
                prompt, tokenizer, max_new_tokens
            )
            candidates.append(output)
        
        # Score all candidates
        scores = self._score_candidates(candidates)
        
        # Add diversity bonus
        if self.config.use_diversity_bonus:
            diversity_scores = self._compute_diversity(candidates)
            scores = [s + self.config.diversity_weight * d 
                     for s, d in zip(scores, diversity_scores)]
        
        # Select best
        best_idx = int(np.argmax(scores))
        best_candidate = candidates[best_idx]
        
        result = {
            'best': best_candidate,
            'best_score': scores[best_idx],
            'all_candidates': candidates if return_all else None,
            'all_scores': scores if return_all else None,
            'mean_score': np.mean(scores),
            'max_score': np.max(scores),
            'min_score': np.min(scores),
        }
        
        return result
    
    def _generate_single(
        self,
        prompt: str,
        tokenizer: Any,
        max_new_tokens: int
    ) -> str:
        """Generate single candidate."""
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.policy.device) for k, v in inputs.items()}
        
        outputs = self.policy.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return generated_text
    
    def _score_candidates(self, candidates: List[str]) -> List[float]:
        """Score candidates with reward model."""
        scores = []
        for candidate in candidates:
            score = self.reward.score_text(candidate)
            scores.append(score)
        return scores
    
    def _compute_diversity(self, candidates: List[str]) -> List[float]:
        """Compute diversity scores based on pairwise differences."""
        # Simple diversity: average edit distance to other candidates
        diversity_scores = []
        
        for i, c1 in enumerate(candidates):
            distances = []
            for j, c2 in enumerate(candidates):
                if i != j:
                    # Simple token-based distance
                    dist = self._levenshtein_distance(c1, c2)
                    distances.append(dist)
            
            diversity_scores.append(np.mean(distances) if distances else 0)
        
        # Normalize
        max_div = max(diversity_scores) if diversity_scores else 1
        return [d / max_div for d in diversity_scores]
    
    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Compute Levenshtein distance between strings."""
        if len(s1) < len(s2):
            return BestOfNSampler._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


# =============================================================================
# MONTE CARLO TREE SEARCH (MCTS)
# =============================================================================

@dataclass 
class MCTSConfig:
    """Configuration for MCTS."""
    n_simulations: int = 100
    c_puct: float = 2.0  # Exploration constant
    temperature: float = 1.0
    max_depth: int = 100
    n_actions: int = 10  # Number of actions to consider per node
    use_value_model: bool = True


class MCTSNode:
    """Node in the MCTS tree."""
    
    def __init__(self, state: str, parent: Optional['MCTSNode'] = None, action: str = ""):
        self.state = state
        self.parent = parent
        self.action = action  # Action taken to reach this node
        
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value_sum = 0.0
        self.prior = 1.0  # Prior probability from policy
        
        self.is_expanded = False
        self.is_terminal = False
    
    def value(self) -> float:
        """Mean value of this node."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    def ucb_score(self, c_puct: float) -> float:
        """UCB score for node selection."""
        if self.visits == 0:
            return float('inf')
        
        # Q-value (exploitation)
        q_value = self.value()
        
        # U-value (exploration)
        if self.parent:
            u_value = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        else:
            u_value = 0
        
        return q_value + u_value
    
    def best_child(self, c_puct: float) -> 'MCTSNode':
        """Select best child by UCB."""
        return max(self.children, key=lambda c: c.ucb_score(c_puct))
    
    def add_child(self, action: str, state: str) -> 'MCTSNode':
        """Add child node."""
        child = MCTSNode(state, parent=self, action=action)
        self.children.append(child)
        return child


class MCTSGenerator:
    """
    Monte Carlo Tree Search for language generation.
    Useful for reasoning tasks with sparse rewards.
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        value_model: Optional[nn.Module],
        tokenizer: Any,
        config: MCTSConfig = None
    ):
        self.policy = policy_model
        self.value = value_model
        self.tokenizer = tokenizer
        self.config = config or MCTSConfig()
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 256,
        reward_fn: Optional[Callable[[str], float]] = None
    ) -> Dict[str, Any]:
        """
        Generate with MCTS.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            reward_fn: Optional reward function for terminal states
        
        Returns:
            Dictionary with 'text', 'tree_stats', etc.
        """
        root = MCTSNode(prompt)
        
        # Run simulations
        for sim in range(self.config.n_simulations):
            # Selection
            node = self._select(root)
            
            # Expansion
            if not node.is_terminal and (node.visits > 0 or node == root):
                self._expand(node)
                if node.children:
                    node = node.children[0]
            
            # Simulation/Evaluation
            value = self._evaluate(node, max_length, reward_fn)
            
            # Backpropagation
            self._backpropagate(node, value)
        
        # Select best sequence
        best_sequence = self._get_best_sequence(root)
        
        return {
            'text': best_sequence,
            'root': root,
            'visit_counts': self._get_visit_distribution(root),
            'best_child_values': [c.value() for c in root.children]
        }
    
    def _select(self, root: MCTSNode) -> MCTSNode:
        """Select node to expand using UCB."""
        node = root
        while node.children and not node.is_terminal:
            node = node.best_child(self.config.c_puct)
        return node
    
    def _expand(self, node: MCTSNode):
        """Expand node by adding children."""
        # Generate candidate actions (next tokens/chunks)
        actions = self._generate_actions(node.state, self.config.n_actions)
        
        for action, prob in actions:
            new_state = node.state + action
            child = node.add_child(action, new_state)
            child.prior = prob
            
            # Check if terminal
            if self._is_terminal(new_state):
                child.is_terminal = True
        
        node.is_expanded = True
    
    def _generate_actions(self, state: str, n: int) -> List[Tuple[str, float]]:
        """Generate candidate next actions with probabilities."""
        inputs = self.tokenizer(state, return_tensors="pt")
        inputs = {k: v.to(self.policy.device) for k, v in inputs.items()}
        
        outputs = self.policy(**inputs)
        logits = outputs.logits[0, -1, :]
        
        # Sample top-k actions
        probs = F.softmax(logits / self.config.temperature, dim=-1)
        top_k = torch.topk(probs, min(n, probs.shape[0]))
        
        actions = []
        for token_id, prob in zip(top_k.indices, top_k.values):
            token = self.tokenizer.decode([token_id])
            actions.append((token, prob.item()))
        
        return actions
    
    def _evaluate(
        self,
        node: MCTSNode,
        max_length: int,
        reward_fn: Optional[Callable]
    ) -> float:
        """Evaluate node with value model or rollout."""
        # Terminal state: use reward function
        if node.is_terminal and reward_fn:
            return reward_fn(node.state)
        
        # Use value model if available
        if self.value and self.config.use_value_model:
            return self.value.score_text(node.state)
        
        # Otherwise: rollout with policy
        return self._rollout(node.state, max_length, reward_fn)
    
    def _rollout(
        self,
        state: str,
        max_length: int,
        reward_fn: Optional[Callable]
    ) -> float:
        """Simulate rollout to terminal state."""
        current = state
        
        for _ in range(max_length - len(state.split())):
            if self._is_terminal(current):
                break
            
            # Sample next token
            actions = self._generate_actions(current, n=1)
            if actions:
                current += actions[0][0]
        
        if reward_fn:
            return reward_fn(current)
        return 0.0
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent
    
    def _is_terminal(self, state: str) -> bool:
        """Check if state is terminal."""
        # Simple checks
        if len(state) > 2000:  # Max length
            return True
        if state.endswith(self.tokenizer.eos_token):
            return True
        return False
    
    def _get_best_sequence(self, root: MCTSNode) -> str:
        """Get best sequence by visit count."""
        node = root
        sequence = node.state
        
        while node.children:
            # Select most visited child
            node = max(node.children, key=lambda c: c.visits)
            sequence += node.action
        
        return sequence
    
    def _get_visit_distribution(self, root: MCTSNode) -> List[int]:
        """Get visit counts for root children."""
        return [c.visits for c in root.children]


# =============================================================================
# TORCH.COMPILE INTEGRATION
# =============================================================================

def compile_model(model: nn.Module, mode: str = "reduce-overhead") -> nn.Module:
    """
    Compile model with torch.compile for faster training/inference.
    
    Args:
        model: PyTorch model
        mode: "default", "reduce-overhead", "max-autotune"
    
    Returns:
        Compiled model
    """
    if not hasattr(torch, 'compile'):
        print("torch.compile not available (requires PyTorch 2.0+)")
        return model
    
    try:
        compiled = torch.compile(model, mode=mode, fullgraph=False)
        print(f"Model compiled with mode='{mode}'")
        return compiled
    except Exception as e:
        print(f"Failed to compile model: {e}")
        return model


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("Inference Optimizations Module")
    print("=" * 60)
    
    print("\nAvailable optimizations:")
    print("  - OptimizedAttention: Flash Attention 2 / SDPA")
    print("  - PagedKVCache: Efficient KV cache management")
    print("  - SpeculativeDecoder: 2-3× faster generation")
    print("  - BestOfNSampler: Quality improvement via reranking")
    print("  - MCTSGenerator: Tree search for reasoning")
    print("  - compile_model: torch.compile integration")
    
    print("\nExample usage:")
    print("""
    # Best-of-N sampling
    sampler = BestOfNSampler(policy, reward_model)
    result = sampler.generate(prompt, tokenizer)
    best_output = result['best']
    
    # MCTS for reasoning
    mcts = MCTSGenerator(policy, value_model, tokenizer)
    result = mcts.generate(
        prompt="Solve: 2x + 5 = 13",
        reward_fn=lambda x: 1.0 if check_answer(x) else 0.0
    )
    solution = result['text']
    
    # Compile for speed
    fast_policy = compile_model(policy, mode="max-autotune")
    """)
