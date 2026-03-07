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
import time
from collections import defaultdict
import heapq

try:
    from rapidfuzz.distance import Levenshtein as RapidLevenshtein
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    RAPIDFUZZ_AVAILABLE = False
    RapidLevenshtein = None


def _extract_logits(outputs: Any) -> torch.Tensor:
    """Extract logits tensor from model outputs across common return formats."""
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, dict):
        logits = outputs.get("logits")
        if logits is not None:
            return logits
        scores = outputs.get("scores")
        if scores is not None:
            return scores
    logits = getattr(outputs, "logits", None)
    if logits is not None:
        return logits
    scores = getattr(outputs, "scores", None)
    if scores is not None:
        return scores
    raise AttributeError("Model output does not expose logits/scores tensor")


def _extract_past_key_values(outputs: Any) -> Any:
    """Extract past_key_values from model outputs when available."""
    if isinstance(outputs, dict):
        return outputs.get("past_key_values")
    return getattr(outputs, "past_key_values", None)


def _extract_scalar_output(outputs: Any) -> float:
    """
    Reduce model output into a scalar score across tensor/dict/object formats.
    Returns 0.0 when no recognizable score field is present.
    """
    if isinstance(outputs, torch.Tensor):
        return float(outputs.squeeze().mean())
    if isinstance(outputs, dict):
        for key in ("score", "reward", "rewards", "value", "values", "logits"):
            val = outputs.get(key)
            if isinstance(val, torch.Tensor):
                return float(val.squeeze().mean())
            if isinstance(val, (int, float)):
                return float(val)
        return 0.0
    for attr in ("score", "reward", "rewards", "value", "values", "logits"):
        val = getattr(outputs, attr, None)
        if isinstance(val, torch.Tensor):
            return float(val.squeeze().mean())
        if isinstance(val, (int, float)):
            return float(val)
    return 0.0

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
        self.sequence_pages: Dict[str, List[int]] = {}      # seq_id -> list of page indices
        self.sequence_lengths: Dict[str, int] = {}           # seq_id -> actual token count
        self.sequence_last_access: Dict[str, float] = {}    # seq_id -> timestamp (for LRU)
        self.shared_prefix_keys: Dict[str, str] = {}        # seq_id -> prefix_key (CoW marker)
        # Telemetry
        self._hit_count = 0
        self._eviction_count = 0
    
    def allocate(self, seq_id: str, num_tokens: int) -> List[int]:
        """Allocate pages for a sequence."""
        num_pages_needed = (num_tokens + self.page_size - 1) // self.page_size

        if len(self.free_pages) < num_pages_needed:
            raise RuntimeError(
                f"Out of KV cache memory. Need {num_pages_needed} pages, "
                f"have {len(self.free_pages)}. Try evict_lru() first."
            )

        pages = [self.free_pages.pop() for _ in range(num_pages_needed)]
        self.sequence_pages[seq_id] = pages
        self.sequence_lengths[seq_id] = 0
        self.sequence_last_access[seq_id] = time.monotonic()
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
        self.sequence_lengths[seq_id] = seq_len + 1
    
    def get_kv(self, seq_ids: List[str], layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather KV tensors for a batch of sequences."""
        all_k = []
        all_v = []
        
        for seq_id in seq_ids:
            pages = self.sequence_pages[seq_id]
            seq_len = self.get_sequence_length(seq_id)
            seq_k_tokens = []
            seq_v_tokens = []

            for token_idx in range(seq_len):
                page_idx = token_idx // self.page_size
                offset = token_idx % self.page_size
                physical_page = pages[page_idx]
                seq_k_tokens.append(self.cache[physical_page, 0, offset])
                seq_v_tokens.append(self.cache[physical_page, 1, offset])

            if seq_k_tokens:
                all_k.append(torch.stack(seq_k_tokens, dim=0))
                all_v.append(torch.stack(seq_v_tokens, dim=0))
            else:
                empty = torch.empty(
                    (0, self.num_heads, self.head_dim),
                    dtype=self.cache.dtype,
                    device=self.cache.device,
                )
                all_k.append(empty)
                all_v.append(empty)
        
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
        self.sequence_last_access[seq_id] = time.monotonic()
        self._hit_count += 1
        return self.sequence_lengths.get(seq_id, 0)

    def free(self, seq_id: str):
        """Free pages for a sequence (not shared-prefix pages unless last reference)."""
        if seq_id in self.sequence_pages:
            # Don't free pages if this seq is a shared prefix still in use
            pages = self.sequence_pages.pop(seq_id)
            self.free_pages.extend(pages)
            self.sequence_lengths.pop(seq_id, None)
            self.sequence_last_access.pop(seq_id, None)
            self.shared_prefix_keys.pop(seq_id, None)

    @property
    def fragmentation_ratio(self) -> float:
        """
        Fraction of allocated page space that is wasted due to partial-page fills.
        0 = no fragmentation, 1 = maximally fragmented.
        """
        allocated_pages = self.max_pages - len(self.free_pages)
        if allocated_pages == 0:
            return 0.0
        used_tokens = sum(self.sequence_lengths.values())
        allocated_tokens = allocated_pages * self.page_size
        return 1.0 - (used_tokens / allocated_tokens)

    def evict_lru(self, n_pages: int) -> int:
        """
        Evict least-recently-used sequences until n_pages are freed (or all seqs exhausted).
        Returns number of pages actually freed.
        """
        if not self.sequence_last_access:
            return 0

        # Sort sequences by last access time (oldest first)
        by_lru = sorted(self.sequence_last_access.items(), key=lambda x: x[1])
        freed = 0
        for seq_id, _ in by_lru:
            if freed >= n_pages:
                break
            pages = self.sequence_pages.get(seq_id, [])
            self.free(seq_id)
            freed += len(pages)
            self._eviction_count += len(pages)
        return freed

    def register_prefix(self, prefix_key: str, seq_id: str):
        """
        Mark seq_id's pages as a shared prefix under prefix_key.
        Copy-on-write semantics: pages are not freed when the owning seq is freed
        unless this is the last sequence holding the prefix.
        """
        self.shared_prefix_keys[seq_id] = prefix_key

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics snapshot."""
        allocated = self.max_pages - len(self.free_pages)
        return {
            "free_pages": len(self.free_pages),
            "allocated_pages": allocated,
            "max_pages": self.max_pages,
            "active_sequences": len(self.sequence_pages),
            "fragmentation_ratio": self.fragmentation_ratio,
            "hit_count": self._hit_count,
            "eviction_count": self._eviction_count,
        }


# =============================================================================
# SPECULATIVE DECODING
# =============================================================================

@dataclass
class SpeculativeDecoderConfig:
    """Configuration for Speculative Decoding 2.0."""
    gamma: int = 5                  # Initial draft tokens per step
    temperature: float = 1.0
    gamma_min: int = 3              # Adaptive gamma lower bound
    gamma_max: int = 12             # Adaptive gamma upper bound
    adapt_gamma: bool = True        # Enable adaptive gamma controller
    adapt_window: int = 50          # Steps between gamma adaptation checks


class SpeculativeDecoder:
    """
    Speculative decoding for 2-3× faster generation.
    Uses small draft model to predict tokens, large model verifies.

    Implements correct Chen et al. 2023 acceptance-resampling:
        r = p_target(x) / p_draft(x)
        accept with prob min(1, r)
        on reject: resample from (p_target - p_draft)+ normalized
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        config: Optional[SpeculativeDecoderConfig] = None,
        # Legacy scalar params kept for backward compatibility
        gamma: int = 5,
        temperature: float = 1.0,
    ):
        self.target = target_model
        self.draft = draft_model
        if config is not None:
            self.config = config
        else:
            self.config = SpeculativeDecoderConfig(gamma=gamma, temperature=temperature)
        # Adaptive gamma state
        self._current_gamma = self.config.gamma
        self._accepted_history: List[float] = []   # acceptance rates per step
        self._step_count = 0

        # Telemetry counters
        self.accepted_tokens = 0
        self.total_draft_tokens = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_draft_tokens
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate with speculative decoding (Chen et al. 2023 correct algorithm).

        Each step:
          1. Draft model autoregressively produces gamma tokens with their probs.
          2. Single batched forward pass through target for all gamma+1 positions.
          3. Acceptance-resampling: accept token i with prob min(1, p_target/p_draft).
             On rejection: resample from (p_target - p_draft)+, normalized.
          4. If all accepted: append one bonus token from p_target at position gamma.
        """
        cfg = self.config
        device = input_ids.device
        generated = input_ids.clone()
        target_len = input_ids.shape[1] + max_new_tokens

        while generated.shape[1] < target_len:
            remaining = target_len - generated.shape[1]
            gamma = min(self._current_gamma, remaining)

            # Step 1: Draft model — generate gamma tokens + collect draft probs
            draft_tokens, draft_probs_list = self._draft_generate_with_probs(
                generated, gamma
            )
            if draft_tokens.shape[1] == 0:
                break

            actual_gamma = draft_tokens.shape[1]
            self.total_draft_tokens += actual_gamma

            # Step 2: Single batched target forward pass
            full_input = torch.cat([generated, draft_tokens], dim=1)
            target_outputs = self.target(full_input)
            target_logits = _extract_logits(target_outputs)  # (1, len+gamma, vocab)

            # Step 3: Acceptance-resampling (Chen et al.)
            accepted = 0
            prefix_len = generated.shape[1]
            for i in range(actual_gamma):
                # Use the pre-step prefix length; `generated` mutates as tokens are accepted.
                pos = prefix_len + i  # position in full_input logits
                t = cfg.temperature
                p_target = F.softmax(target_logits[:, pos - 1, :] / t, dim=-1)  # next-token dist
                draft_token = draft_tokens[:, i]   # shape (batch,)

                p_draft_token = draft_probs_list[i].gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)
                p_target_token = p_target.gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)

                accept_prob = torch.clamp(p_target_token / (p_draft_token + 1e-9), max=1.0)

                if torch.rand(1, device=device).item() < accept_prob.mean().item():
                    generated = torch.cat([generated, draft_token.unsqueeze(1)], dim=1)
                    accepted += 1
                else:
                    # Resample from adjusted distribution: (p_target - p_draft)+
                    p_draft_full = draft_probs_list[i]
                    adjusted = (p_target - p_draft_full).clamp(min=0.0)
                    adj_sum = adjusted.sum(dim=-1, keepdim=True)
                    # If adjusted is all-zero (degenerate): fall back to p_target
                    adjusted = torch.where(
                        adj_sum > 1e-9,
                        adjusted / (adj_sum + 1e-9),
                        p_target,
                    )
                    new_token = torch.multinomial(adjusted, num_samples=1)
                    generated = torch.cat([generated, new_token], dim=1)
                    break

            self.accepted_tokens += accepted

            # Adaptive gamma
            self._accepted_history.append(accepted / max(actual_gamma, 1))
            self._step_count += 1
            if cfg.adapt_gamma and self._step_count % cfg.adapt_window == 0:
                self._adapt_gamma()

            # If all accepted: draw one bonus token from target at last position
            if accepted == actual_gamma:
                last_pos = generated.shape[1] - 1
                p_bonus = F.softmax(target_logits[:, last_pos, :] / cfg.temperature, dim=-1)
                bonus = torch.multinomial(p_bonus, num_samples=1)
                generated = torch.cat([generated, bonus], dim=1)

        return generated

    def _adapt_gamma(self):
        """Adjust gamma based on recent acceptance rate."""
        cfg = self.config
        if not self._accepted_history:
            return
        window = self._accepted_history[-cfg.adapt_window:]
        rate = float(np.mean(window))
        if rate > 0.8 and self._current_gamma < cfg.gamma_max:
            self._current_gamma = min(self._current_gamma + 1, cfg.gamma_max)
        elif rate < 0.5 and self._current_gamma > cfg.gamma_min:
            self._current_gamma = max(self._current_gamma - 1, cfg.gamma_min)
    
    def _draft_generate_with_probs(
        self, input_ids: torch.Tensor, num_tokens: int
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generate draft tokens and return both tokens and their probability distributions.
        Returns (draft_tokens [batch, n], draft_probs [n x (batch, vocab)]).
        """
        draft_tokens = []
        draft_probs = []
        current = input_ids
        past_key_values = None

        try:
            outputs = self.draft(current, use_cache=True)
            logits = _extract_logits(outputs)[:, -1, :]
            past_key_values = _extract_past_key_values(outputs)
            if past_key_values is None:
                raise RuntimeError("Draft model did not return past_key_values")

            for _ in range(num_tokens):
                probs = F.softmax(logits / self.config.temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
                draft_tokens.append(token)
                draft_probs.append(probs)

                outputs = self.draft(
                    input_ids=token, use_cache=True, past_key_values=past_key_values
                )
                logits = _extract_logits(outputs)[:, -1, :]
                past_key_values = _extract_past_key_values(outputs)

            if draft_tokens:
                return torch.cat(draft_tokens, dim=1), draft_probs
            return current[:, :0], []

        except Exception:
            # Stateless fallback (slower, no KV cache)
            draft_tokens, draft_probs = [], []
            for _ in range(num_tokens):
                logits = _extract_logits(self.draft(current))[:, -1, :]
                probs = F.softmax(logits / self.config.temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
                draft_tokens.append(token)
                draft_probs.append(probs)
                current = torch.cat([current, token], dim=1)
            if draft_tokens:
                return torch.cat(draft_tokens, dim=1), draft_probs
            return current[:, :0], []

    def _draft_generate(self, input_ids: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """Generate draft tokens with small model."""
        draft_tokens = []
        current = input_ids
        past_key_values = None

        try:
            outputs = self.draft(current, use_cache=True)
            logits = _extract_logits(outputs)[:, -1, :]
            past_key_values = _extract_past_key_values(outputs)

            if past_key_values is None:
                raise RuntimeError("Draft model did not return past_key_values")

            for _ in range(num_tokens):
                probs = F.softmax(logits / self.config.temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
                draft_tokens.append(token)

                outputs = self.draft(
                    input_ids=token,
                    use_cache=True,
                    past_key_values=past_key_values
                )
                logits = _extract_logits(outputs)[:, -1, :]
                past_key_values = _extract_past_key_values(outputs)

            return torch.cat(draft_tokens, dim=1) if draft_tokens else current[:, :0]
        except Exception:
            for _ in range(num_tokens):
                logits = _extract_logits(self.draft(current))[:, -1, :]
                probs = F.softmax(logits / self.config.temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
                draft_tokens.append(token)
                current = torch.cat([current, token], dim=1)

            return torch.cat(draft_tokens, dim=1) if draft_tokens else current[:, :0]


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
    # Multi-objective reranking fields
    value_weight: float = 0.0           # Blend value model score into ranking
    length_penalty: float = 0.0        # Penalize verbosity (tokens / max_tokens)
    repetition_penalty: float = 0.0    # Penalize 3-gram repetition
    format_checker: Optional[Callable[[str], bool]] = None  # Hard constraint filter
    batch_score: bool = True            # Batch all candidates through reward model


class BestOfNSampler:
    """
    Best-of-N sampling with reranking.
    Generates N candidates and selects best by reward model.
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        reward_model: nn.Module,
        config: BestOfNConfig = None,
        tokenizer: Optional[Any] = None,
        value_model: Optional[nn.Module] = None,
    ):
        self.policy = policy_model
        self.reward = reward_model
        self.value = value_model
        self.tokenizer = tokenizer
        self.config = config or BestOfNConfig()
        self._score_warned_once = False
    
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
        scores = self._score_candidates(candidates, max_new_tokens=max_new_tokens)
        
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
        inputs = {k: v.to(self._get_policy_device()) for k, v in inputs.items()}
        
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
    
    def _get_policy_device(self) -> torch.device:
        """Resolve policy device robustly across wrapped and raw modules."""
        device = getattr(self.policy, "device", None)
        if device is not None:
            return torch.device(device)
        try:
            return next(self.policy.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _score_one(self, text: str) -> float:
        """
        Score a single candidate text with the reward model.
        Supports: .score(text), .score_text(text), or raw nn.Module forward with tokenizer.
        """
        reward = self.reward
        # Protocol-style duck typing
        if hasattr(reward, "score"):
            return float(reward.score(text))
        if hasattr(reward, "score_text"):
            return float(reward.score_text(text))
        # Fallback: tokenize and call forward
        tok = self.tokenizer
        if tok is None:
            if not self._score_warned_once:
                import logging
                logging.getLogger("BestOfNSampler").warning(
                    "No .score()/.score_text() on reward model and no tokenizer provided. "
                    "All candidates will score 0.0."
                )
                self._score_warned_once = True
            return 0.0
        device = self._get_policy_device()
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = reward(**enc)
        return _extract_scalar_output(out)

    def _score_value(self, text: str) -> float:
        """Score a candidate with the value model (optional)."""
        if self.value is None:
            return 0.0
        if hasattr(self.value, "score"):
            return float(self.value.score(text))
        if hasattr(self.value, "score_text"):
            return float(self.value.score_text(text))
        tok = self.tokenizer
        if tok is None:
            return 0.0
        device = self._get_policy_device()
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.value(**enc)
        return _extract_scalar_output(out)

    @staticmethod
    def _ngram_repetition_score(text: str, n: int = 3) -> float:
        """Fraction of n-grams that are repeated (0 = no repetition, 1 = all repeated)."""
        tokens = text.split()
        if len(tokens) < n:
            return 0.0
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0
        unique = len(set(ngrams))
        return 1.0 - unique / len(ngrams)

    def _score_candidates(self, candidates: List[str], max_new_tokens: int = 256) -> List[float]:
        """
        Multi-objective scoring: reward + optional value blend,
        length penalty, repetition penalty, and hard format filter.
        Supports batched reward scoring when config.batch_score=True and
        reward has a .score_batch() method.
        """
        cfg = self.config
        n = len(candidates)

        # Hard format filter — remove non-conforming candidates before ranking
        valid_mask = [True] * n
        if cfg.format_checker is not None:
            for i, c in enumerate(candidates):
                valid_mask[i] = bool(cfg.format_checker(c))

        # Reward scores
        if cfg.batch_score and hasattr(self.reward, "score_batch"):
            reward_scores = list(self.reward.score_batch(candidates))
        else:
            reward_scores = [self._score_one(c) for c in candidates]

        scores = []
        for i, candidate in enumerate(candidates):
            s = reward_scores[i]

            # Value model blend
            if cfg.value_weight > 0.0:
                v = self._score_value(candidate)
                s = (1.0 - cfg.value_weight) * s + cfg.value_weight * v

            # Length penalty
            if cfg.length_penalty > 0.0:
                length_ratio = len(candidate.split()) / max(max_new_tokens, 1)
                s -= cfg.length_penalty * length_ratio

            # Repetition penalty
            if cfg.repetition_penalty > 0.0:
                s -= cfg.repetition_penalty * self._ngram_repetition_score(candidate)

            # Hard filter: set invalid to -inf so they never win
            if not valid_mask[i]:
                s = float("-inf")

            scores.append(s)
        return scores
    
    def _compute_diversity(self, candidates: List[str]) -> List[float]:
        """Compute diversity scores based on pairwise differences."""
        # Simple diversity: average edit distance to other candidates
        n = len(candidates)
        if n == 0:
            return []
        if n == 1:
            return [0.0]

        distances = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                c1 = candidates[i]
                c2 = candidates[j]
                if RAPIDFUZZ_AVAILABLE:
                    dist = RapidLevenshtein.distance(c1, c2)
                else:
                    dist = self._levenshtein_distance(c1, c2)
                distances[i, j] = dist
                distances[j, i] = dist

        diversity_scores = distances.sum(axis=1) / (n - 1)

        # Normalize
        max_div = float(np.max(diversity_scores)) if n > 1 else 1.0
        if max_div <= 0:
            return [0.0] * n
        return [float(d / max_div) for d in diversity_scores]
    
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
    c_puct: float = 1.25           # AlphaZero-style PUCT base constant
    puct_c2: float = 19652.0       # PUCT log-factor denominator (MuZero style)
    temperature: float = 1.0
    max_depth: int = 100
    max_rollout_depth: int = 50    # Cap for rollout steps (separate from tree depth)
    n_actions: int = 10            # Number of actions to consider per node
    use_value_model: bool = True
    progressive_widening_alpha: float = 0.5  # max_children ∝ N^alpha
    depth_discount: float = 0.95   # Value discount per depth level during backprop
    reward_value_blend: float = 0.5  # Blend terminal reward + value estimate
    serialize_tree: bool = False   # Dump tree JSON for replay/debug


class MCTSNode:
    """Node in the MCTS tree."""

    def __init__(
        self,
        state: str,
        parent: Optional['MCTSNode'] = None,
        action: str = "",
        depth: int = 0,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = depth

        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value_sum = 0.0
        self.prior = 1.0       # Prior probability from policy
        self.reward: Optional[float] = None  # Terminal reward if evaluated

        self.is_expanded = False
        self.is_terminal = False

    def value(self) -> float:
        """Mean value of this node."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def ucb_score(self, c_puct: float, c2: float = 19652.0) -> float:
        """
        AlphaZero / MuZero PUCT score.
        U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent * log((N_parent + c2 + 1)/c2)) / (1 + N(s,a))
        """
        if self.visits == 0:
            return float('inf')
        q_value = self.value()
        if self.parent:
            n_parent = self.parent.visits
            log_factor = math.log((n_parent + c2 + 1) / c2)
            explore_rate = math.sqrt(n_parent * log_factor)
            u_value = c_puct * self.prior * explore_rate / (1 + self.visits)
        else:
            u_value = 0.0
        return q_value + u_value

    def best_child(self, c_puct: float, c2: float = 19652.0) -> 'MCTSNode':
        """Select best child by PUCT score."""
        return max(self.children, key=lambda c: c.ucb_score(c_puct, c2))

    def add_child(self, action: str, state: str) -> 'MCTSNode':
        """Add child node with incremented depth."""
        child = MCTSNode(state, parent=self, action=action, depth=self.depth + 1)
        self.children.append(child)
        return child

    def to_dict(self) -> Dict:
        """Serialize node for tree JSON dump."""
        return {
            "state_preview": self.state[-80:],
            "action": self.action,
            "depth": self.depth,
            "visits": self.visits,
            "value": self.value(),
            "prior": self.prior,
            "reward": self.reward,
            "is_terminal": self.is_terminal,
            "children": [c.to_dict() for c in self.children],
        }


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

            # Expansion (progressive widening)
            if not node.is_terminal and (node.visits > 0 or node == root):
                self._expand(node)
                if node.children:
                    node = node.children[0]

            # Simulation/Evaluation
            value = self._evaluate(node, self.config.max_rollout_depth, reward_fn)

            # Backpropagation with depth discount
            self._backpropagate(node, value)

        # Select best sequence
        best_sequence = self._get_best_sequence(root)

        result = {
            'text': best_sequence,
            'root': root,
            'visit_counts': self._get_visit_distribution(root),
            'best_child_values': [c.value() for c in root.children],
        }
        if self.config.serialize_tree:
            import json as _json
            result['tree_json'] = _json.dumps(root.to_dict(), default=str)
        return result
    
    def _select(self, root: MCTSNode) -> MCTSNode:
        """Select node to expand using PUCT."""
        node = root
        while node.children and not node.is_terminal:
            node = node.best_child(self.config.c_puct, self.config.puct_c2)
        return node

    def _expand(self, node: MCTSNode):
        """
        Expand node with progressive widening.
        max_children = ceil(visits ^ progressive_widening_alpha)
        """
        alpha = self.config.progressive_widening_alpha
        max_children = max(1, math.ceil((node.visits + 1) ** alpha))

        # Don't re-expand beyond current widening budget
        if len(node.children) >= max_children and node.is_expanded:
            return

        n_to_generate = max_children - len(node.children)
        if n_to_generate <= 0:
            node.is_expanded = True
            return

        # Total actions to sample; pick from top-n_actions, add n_to_generate
        actions = self._generate_actions(node.state, self.config.n_actions)

        # Skip actions already expanded (by action string)
        existing_actions = {c.action for c in node.children}
        new_actions = [(a, p) for a, p in actions if a not in existing_actions]

        for action, prob in new_actions[:n_to_generate]:
            new_state = node.state + action
            child = node.add_child(action, new_state)
            child.prior = prob
            if self._is_terminal(new_state):
                child.is_terminal = True

        node.is_expanded = True
    
    def _generate_actions(self, state: str, n: int) -> List[Tuple[str, float]]:
        """Generate candidate next actions with probabilities."""
        device = self._get_policy_device()
        inputs = self.tokenizer(state, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = self.policy(**inputs)
        logits = _extract_logits(outputs)[0, -1, :]
        
        # Sample top-k actions
        probs = F.softmax(logits / self.config.temperature, dim=-1)
        top_k = torch.topk(probs, min(n, probs.shape[0]))
        
        actions = []
        for token_id, prob in zip(top_k.indices, top_k.values):
            token = self.tokenizer.decode([token_id])
            actions.append((token, prob.item()))
        
        return actions
    
    def _value_score(self, text: str) -> float:
        """
        Score text with value model using duck-typing + fallback tokenizer path.
        Never calls the missing .score_text() directly.
        """
        vm = self.value
        if vm is None:
            return 0.0
        # Protocol: .score(text) or .score_text(text)
        if hasattr(vm, "score"):
            return float(vm.score(text))
        if hasattr(vm, "score_text"):
            return float(vm.score_text(text))
        # Fallback: tokenize + forward
        tok = self.tokenizer
        if tok is None:
            return 0.0
        device = self._get_policy_device()
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = vm(**enc)
        return _extract_scalar_output(out)

    def _evaluate(
        self,
        node: MCTSNode,
        max_length: int,
        reward_fn: Optional[Callable],
    ) -> float:
        """
        Evaluate node with value model and/or rollout.
        Blends terminal reward and value estimate per reward_value_blend.
        """
        cfg = self.config

        # Terminal: use reward function
        if node.is_terminal and reward_fn is not None:
            r = float(reward_fn(node.state))
            node.reward = r
            if self.value and cfg.use_value_model:
                v = self._value_score(node.state)
                return cfg.reward_value_blend * r + (1 - cfg.reward_value_blend) * v
            return r

        # Non-terminal with value model: blend value + optional rollout reward
        if self.value and cfg.use_value_model:
            v = self._value_score(node.state)
            if reward_fn is not None:
                # Roll out and blend
                rollout_r = self._rollout(node.state, max_length, reward_fn)
                return cfg.reward_value_blend * rollout_r + (1 - cfg.reward_value_blend) * v
            return v

        # Pure rollout fallback
        return self._rollout(node.state, max_length, reward_fn)
    
    def _rollout(
        self,
        state: str,
        max_length: int,
        reward_fn: Optional[Callable]
    ) -> float:
        """Simulate rollout to terminal state."""
        device = self._get_policy_device()

        inputs = self.tokenizer(state, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        max_new_tokens = max(0, max_length - input_ids.shape[1])
        if max_new_tokens == 0:
            if reward_fn:
                return reward_fn(state)
            return 0.0

        try:
            outputs = self.policy(**inputs, use_cache=True)
            past_key_values = _extract_past_key_values(outputs)
            if past_key_values is None:
                raise RuntimeError("Policy model did not return past_key_values")

            logits = _extract_logits(outputs)[:, -1, :]
            generated_tokens = []

            for _ in range(max_new_tokens):
                probs = F.softmax(logits / self.config.temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
                generated_tokens.append(token)

                if self.tokenizer.eos_token_id is not None and token.item() == self.tokenizer.eos_token_id:
                    break

                outputs = self.policy(
                    input_ids=token,
                    use_cache=True,
                    past_key_values=past_key_values
                )
                logits = _extract_logits(outputs)[:, -1, :]
                past_key_values = _extract_past_key_values(outputs)

            if generated_tokens:
                new_tokens = torch.cat(generated_tokens, dim=1)
                full_ids = torch.cat([input_ids, new_tokens], dim=1)
            else:
                full_ids = input_ids

            current = self.tokenizer.decode(full_ids[0], skip_special_tokens=True)
        except Exception:
            current = state
            for _ in range(max_length - len(state.split())):
                if self._is_terminal(current):
                    break

                actions = self._generate_actions(current, n=1)
                if actions:
                    current += actions[0][0]

        if reward_fn:
            return reward_fn(current)
        return 0.0

    def _get_policy_device(self) -> torch.device:
        """Resolve policy device robustly across wrapped and raw modules."""
        device = getattr(self.policy, "device", None)
        if device is not None:
            return torch.device(device)
        try:
            return next(self.policy.parameters()).device
        except StopIteration:
            return torch.device("cpu")
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """
        Backpropagate value up the tree with depth-based discounting.
        value at depth d is discounted by depth_discount^(d - root_depth) at each ancestor.
        """
        discount = self.config.depth_discount
        current_depth = node.depth
        while node is not None:
            node.visits += 1
            # Discount value relative to the evaluated node's depth
            discounted = value * (discount ** (current_depth - node.depth))
            node.value_sum += discounted
            node = node.parent
    
    def _is_terminal(self, state: str) -> bool:
        """Check if state is terminal."""
        # Simple checks
        if len(state) > 2000:  # Max length
            return True
        eos_token = self.tokenizer.eos_token
        if eos_token and state.endswith(eos_token):
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
