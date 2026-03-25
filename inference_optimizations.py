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
from dataclasses import dataclass, field
import math
import copy
import time
import re
import os
import sys
import tempfile
import logging
import json
import ast
from collections import defaultdict
import heapq
import inspect

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


def _resolve_model_device(model: Any) -> torch.device:
    """Resolve device from any model supporting .device attr or .parameters()."""
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

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
        dtype: torch.dtype = torch.bfloat16,
        max_prefix_entries: int = 64,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages
        self.dtype = dtype

        # Track allocation
        self.free_pages = list(range(max_pages))
        self.sequence_pages: Dict[str, List[int]] = {}      # seq_id -> list of page indices
        self.sequence_lengths: Dict[str, int] = {}           # seq_id -> actual token count
        self.sequence_layer_lengths: Dict[str, Dict[int, int]] = {}
        self.sequence_last_access: Dict[str, float] = {}    # seq_id -> timestamp (for LRU)
        self.shared_prefix_keys: Dict[str, str] = {}        # seq_id -> logical prefix tag only

        # Per-layer page blocks are allocated lazily to avoid an impossible
        # eager tensor of shape (num_layers, max_pages, ...).
        self._page_blocks: Dict[Tuple[int, int], torch.Tensor] = {}

        # Prefix KV cache is separate from paged blocks. Entries contain the
        # reusable cache state plus the last logits so generation can resume
        # without replaying the prefix tokens.
        self._prefix_cache: Dict[int, Dict[str, Any]] = {}
        self._max_prefix_entries = max_prefix_entries
        # Telemetry
        self._hit_count = 0
        self._eviction_count = 0

    @staticmethod
    def _prefix_cache_key(prefix_text: str) -> Tuple[int, int]:
        """Primary cache key; stored prefix text is used to verify collisions."""
        return (hash(prefix_text), len(prefix_text))

    def _validate_layer_idx(self, layer_idx: int) -> None:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range for num_layers={self.num_layers}"
            )

    def _page_key(self, layer_idx: int, physical_page: int) -> Tuple[int, int]:
        self._validate_layer_idx(layer_idx)
        return (layer_idx, physical_page)

    def _get_or_create_page_block(
        self,
        layer_idx: int,
        physical_page: int,
        device: torch.device,
    ) -> torch.Tensor:
        key = self._page_key(layer_idx, physical_page)
        block = self._page_blocks.get(key)
        if block is None or block.device != device:
            block = torch.zeros(
                (2, self.page_size, self.num_heads, self.head_dim),
                dtype=self.dtype,
                device=device,
            )
            self._page_blocks[key] = block
        return block
    
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
        self.sequence_layer_lengths[seq_id] = {layer_idx: 0 for layer_idx in range(self.num_layers)}
        self.sequence_last_access[seq_id] = time.monotonic()
        return pages
    
    def append_token(self, seq_id: str, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Append a single token's KV to cache."""
        self._validate_layer_idx(layer_idx)
        pages = self.sequence_pages[seq_id]
        layer_lengths = self.sequence_layer_lengths.setdefault(
            seq_id, {idx: 0 for idx in range(self.num_layers)}
        )
        seq_len = layer_lengths[layer_idx]
        
        page_idx = seq_len // self.page_size
        offset_in_page = seq_len % self.page_size
        
        # Allocate new page if needed
        if page_idx >= len(pages):
            if not self.free_pages:
                raise RuntimeError("Out of KV cache memory")
            pages.append(self.free_pages.pop())

        physical_page = pages[page_idx]
        page_block = self._get_or_create_page_block(layer_idx, physical_page, k.device)
        page_block[0, offset_in_page] = k.to(dtype=self.dtype, device=page_block.device)
        page_block[1, offset_in_page] = v.to(dtype=self.dtype, device=page_block.device)
        layer_lengths[layer_idx] = seq_len + 1
        self.sequence_lengths[seq_id] = max(layer_lengths.values(), default=0)

    def get_kv(self, seq_ids: List[str], layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather KV tensors for a batch of sequences."""
        self._validate_layer_idx(layer_idx)
        all_k = []
        all_v = []
        target_device: Optional[torch.device] = None
        target_dtype = self.dtype

        for seq_id in seq_ids:
            pages = self.sequence_pages[seq_id]
            self.sequence_last_access[seq_id] = time.monotonic()
            self._hit_count += 1
            seq_len = self.sequence_layer_lengths.get(seq_id, {}).get(layer_idx, 0)
            seq_k_tokens = []
            seq_v_tokens = []

            for token_idx in range(seq_len):
                page_idx = token_idx // self.page_size
                offset = token_idx % self.page_size
                physical_page = pages[page_idx]
                block = self._page_blocks.get(self._page_key(layer_idx, physical_page))
                if block is None:
                    continue
                target_device = block.device
                target_dtype = block.dtype
                seq_k_tokens.append(block[0, offset])
                seq_v_tokens.append(block[1, offset])

            if seq_k_tokens:
                all_k.append(torch.stack(seq_k_tokens, dim=0))
                all_v.append(torch.stack(seq_v_tokens, dim=0))
            else:
                empty = torch.empty(
                    (0, self.num_heads, self.head_dim),
                    dtype=target_dtype,
                    device=target_device or torch.device("cpu"),
                )
                all_k.append(empty)
                all_v.append(empty)

        if not all_k:
            empty = torch.empty(
                (0, 0, self.num_heads, self.head_dim),
                dtype=target_dtype,
                device=target_device or torch.device("cpu"),
            )
            return empty, empty

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
        """Free pages for a sequence and drop all per-layer page blocks."""
        if seq_id in self.sequence_pages:
            pages = self.sequence_pages.pop(seq_id)
            for physical_page in pages:
                for layer_idx in range(self.num_layers):
                    self._page_blocks.pop((layer_idx, physical_page), None)
            self.free_pages.extend(pages)
            self.sequence_lengths.pop(seq_id, None)
            self.sequence_layer_lengths.pop(seq_id, None)
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
        Attach a logical prefix tag to a sequence id.
        This does not alter physical page retention; reusable prompt-prefix
        acceleration is handled by the raw `past_key_values` prefix cache.
        """
        self.shared_prefix_keys[seq_id] = prefix_key

    def get_or_compute_prefix(self, model: Any, tokenizer: Any, prefix_text: str) -> Optional[Dict[str, Any]]:
        """
        Cache-or-compute reusable prefix state for a shared prompt prefix.

        Returned entries contain:
            {
                "past_key_values": ...,
                "last_logits": torch.Tensor shape [1, vocab],
                "prefix_len": int,
            }
        """
        key = self._prefix_cache_key(prefix_text)
        cached = self._prefix_cache.get(key)
        if cached is not None and cached.get("prefix_text") == prefix_text:
            return cached
        device = _resolve_model_device(model)
        inputs = tokenizer(prefix_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            try:
                outputs = model(**inputs, use_cache=True)
            except TypeError:
                outputs = model(**inputs)
        pkv = _extract_past_key_values(outputs)
        logits = _extract_logits(outputs)[:, -1, :]
        if pkv is not None:
            if len(self._prefix_cache) >= self._max_prefix_entries:
                # FIFO eviction: remove oldest key
                oldest = next(iter(self._prefix_cache))
                del self._prefix_cache[oldest]
            self._prefix_cache[key] = {
                "prefix_text": prefix_text,
                "past_key_values": pkv,
                "last_logits": logits.detach(),
                "prefix_len": int(inputs["input_ids"].shape[1]),
            }
        return self._prefix_cache.get(key)

    def evict_prefix(self, prefix_text: str) -> None:
        """Manually evict a prefix entry from the cache."""
        self._prefix_cache.pop(self._prefix_cache_key(prefix_text), None)

    def clear_prefix_cache(self) -> None:
        """Clear all cached prefix KV states."""
        self._prefix_cache.clear()

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
            "prefix_cache_entries": len(self._prefix_cache),
            "materialized_layer_pages": len(self._page_blocks),
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

    def stats(self) -> Dict[str, Any]:
        """Return decoding telemetry snapshot."""
        return {
            "accepted_tokens": self.accepted_tokens,
            "total_draft_tokens": self.total_draft_tokens,
            "acceptance_rate": self.acceptance_rate,
            "current_gamma": self._current_gamma,
            "step_count": self._step_count,
        }

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
    step_rerank: bool = False          # Enable PRM-Min step-level reranking
    step_prm: Optional[Any] = None     # PRM with .score(text)->float interface
    step_delimiter: str = "\n\n"       # Step boundary delimiter
    prm_process_weight: float = 0.0   # process/outcome blend when step_prm is a ProcessRewardModelAdapter


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
        scores = self._score_candidates(
            candidates,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        
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

    def _score_one(self, completion: str, prompt: str = "") -> float:
        """
        Score a single candidate with the reward model.
        Supports: .score(text), .score_text(text), or raw nn.Module forward with tokenizer.
        """
        reward = self.reward
        full_text = ((prompt + " " + completion).strip() if prompt else completion).strip()

        # Protocol-style duck typing. Prefer the protocol contract
        # score(prompt, completion) when available so RewardScorerAdapter
        # and native reward models see the full pair instead of an isolated
        # completion string.
        if hasattr(reward, "score"):
            try:
                score_sig = inspect.signature(reward.score)
                if len(score_sig.parameters) >= 2:
                    return float(reward.score(prompt, completion))
            except (TypeError, ValueError):
                # Builtins / C-backed callables may not expose inspectable signatures.
                if prompt:
                    try:
                        return float(reward.score(prompt, completion))
                    except TypeError:
                        pass
            return float(reward.score(full_text))
        if hasattr(reward, "score_text"):
            return float(reward.score_text(full_text))
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
        enc = tok(full_text, return_tensors="pt", truncation=True, max_length=512)
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

    def _score_candidates(
        self,
        candidates: List[str],
        prompt: str = "",
        max_new_tokens: int = 256,
    ) -> List[float]:
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
        if cfg.batch_score and hasattr(self.reward, "score_batch") and not prompt:
            reward_scores = list(self.reward.score_batch(candidates))
        else:
            reward_scores = [self._score_one(c, prompt=prompt) for c in candidates]

        # Step-level reranking overrides outcome scores when PRM is provided
        if cfg.step_rerank and cfg.step_prm is not None:
            reward_scores = [
                self.score_step_by_step(
                    c,
                    cfg.step_prm,
                    cfg.step_delimiter,
                    prompt=prompt,
                )
                for c in candidates
            ]

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

    def score_step_by_step(
        self,
        completion: str,
        prm: Any,
        delimiter: str = "\n\n",
        prompt: str = "",
    ) -> float:
        """PRM-Min aggregation: score each reasoning step, return the minimum.

        The weakest step defines trajectory quality (weakest-link principle).
        Fast path: if prm exposes score_steps() (StepLevelScorerLike /
        ProcessRewardModelAdapter), uses PRM's internal token-aligned boundary
        detection instead of naive delimiter splitting.
        """
        if prm is None:
            return self._score_one(completion, prompt=prompt)

        # Fast path: StepLevelScorerLike / ProcessRewardModelAdapter.
        # Uses PRM's internal token-aligned boundary detection.
        if hasattr(prm, "score_steps"):
            try:
                step_scores = prm.score_steps(completion)
                if step_scores:
                    return min(step_scores)
            except Exception:
                pass  # fall through to legacy path on adapter error

        # Legacy path: naive delimiter split
        steps = [s.strip() for s in completion.split(delimiter) if s.strip()]
        if not steps:
            return self._score_one(completion, prompt=prompt)
        step_scores = []
        for step in steps:
            if hasattr(prm, "score"):
                step_scores.append(float(prm.score(step)))
            elif hasattr(prm, "score_text"):
                step_scores.append(float(prm.score_text(step)))
            else:
                step_scores.append(0.0)
        return min(step_scores)

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
        config: MCTSConfig = None,
        kv_cache: Optional[PagedKVCache] = None,
        mdp: Optional[Any] = None,
    ):
        self.policy = policy_model
        self.value = value_model
        self.tokenizer = tokenizer
        self.config = config or MCTSConfig()
        self.kv_cache = kv_cache
        self.mdp = mdp
    
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
        shared_prefix_state = prompt
        
        # Run simulations
        for sim in range(self.config.n_simulations):
            # Selection
            node = self._select(root)

            # Expansion (progressive widening)
            if not node.is_terminal and (node.visits > 0 or node == root):
                self._expand(node)
                if node.children:
                    node = self._pick_expansion_child(node)

            # Simulation/Evaluation
            value = self._evaluate(
                node,
                self.config.max_rollout_depth,
                reward_fn,
                prefix_state=shared_prefix_state,
            )

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

    @staticmethod
    def _pick_expansion_child(node: MCTSNode) -> MCTSNode:
        """Prefer the strongest prior among equally unvisited newly expanded children."""
        return max(node.children, key=lambda child: (child.visits, child.prior))
    
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
            if self.mdp is not None:
                new_state = self.mdp.transition(node.state, action)
            else:
                new_state = node.state + action
            child = node.add_child(action, new_state)
            child.prior = prob
            if self._is_terminal(new_state):
                child.is_terminal = True

        node.is_expanded = True
    
    def _generate_actions(self, state: str, n: int) -> List[Tuple[str, float]]:
        """Generate candidate next actions with probabilities."""
        if self.mdp is not None:
            return self.mdp.legal_actions(state, n=n, temperature=self.config.temperature)
        device = self._get_policy_device()
        inputs = self.tokenizer(
            state, return_tensors="pt", truncation=True,
            max_length=getattr(self.tokenizer, "model_max_length", 2048),
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.policy(**inputs)
        logits = _extract_logits(outputs)[0, -1, :]
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
        rollout_budget: int,
        reward_fn: Optional[Callable],
        prefix_state: Optional[str] = None,
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
                rollout_r = self._rollout(
                    node.state,
                    rollout_budget,
                    reward_fn,
                    prefix_state=prefix_state,
                )
                return cfg.reward_value_blend * rollout_r + (1 - cfg.reward_value_blend) * v
            return v

        # Pure rollout fallback
        return self._rollout(
            node.state,
            rollout_budget,
            reward_fn,
            prefix_state=prefix_state,
        )
    
    def _rollout(
        self,
        state: str,
        rollout_budget: int,
        reward_fn: Optional[Callable],
        prefix_state: Optional[str] = None,
    ) -> float:
        """Simulate rollout to terminal state with a new-token budget."""
        device = self._get_policy_device()

        inputs = self.tokenizer(state, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        max_new_tokens = max(0, int(rollout_budget))
        if max_new_tokens == 0:
            if reward_fn:
                return reward_fn(state)
            return 0.0

        prefix_entry = None
        prefix_anchor = None
        if self.kv_cache is not None:
            # Prefer shared root-prefix reuse when the current state extends it.
            if prefix_state and state.startswith(prefix_state):
                prefix_anchor = prefix_state
            else:
                # Fallback to state-local cache when no shared anchor applies.
                prefix_anchor = state
            prefix_entry = self.kv_cache.get_or_compute_prefix(
                self.policy,
                self.tokenizer,
                prefix_anchor,
            )

        try:
            past_key_values = None
            logits = None
            if prefix_entry is not None:
                past_key_values = prefix_entry.get("past_key_values")
                logits = prefix_entry.get("last_logits")
                if logits is not None:
                    logits = logits.to(device)

                # If state extends the cached prefix, replay only the suffix
                # tokens to recover exact per-node context without recomputing
                # the full shared prefix.
                if (
                    prefix_anchor is not None
                    and len(state) > len(prefix_anchor)
                    and past_key_values is not None
                    and logits is not None
                ):
                    suffix_text = state[len(prefix_anchor):]
                    try:
                        suffix_inputs = self.tokenizer(
                            suffix_text,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )
                    except TypeError:
                        suffix_inputs = self.tokenizer(suffix_text, return_tensors="pt")
                    suffix_ids = suffix_inputs.get("input_ids")
                    if suffix_ids is not None and suffix_ids.numel() > 0:
                        suffix_ids = suffix_ids.to(device)
                        for i in range(suffix_ids.shape[1]):
                            next_token = suffix_ids[:, i:i + 1]
                            outputs = self.policy(
                                input_ids=next_token,
                                use_cache=True,
                                past_key_values=past_key_values,
                            )
                            logits = _extract_logits(outputs)[:, -1, :]
                            past_key_values = _extract_past_key_values(outputs)

            if past_key_values is None or logits is None:
                outputs = self.policy(**inputs, use_cache=True)
                past_key_values = _extract_past_key_values(outputs)
                logits = _extract_logits(outputs)[:, -1, :]

            if past_key_values is None:
                raise RuntimeError("Policy model did not return past_key_values")
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
            for _ in range(max_new_tokens):
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
        if self.mdp is not None:
            return self.mdp.is_terminal(state)
        if len(state) > 2000:
            return True
        eos_token = self.tokenizer.eos_token
        if eos_token and state.endswith(eos_token):
            return True
        return False
    
    def _get_best_sequence(self, root: MCTSNode) -> str:
        """Get best sequence by visit count."""
        node = root
        while node.children:
            # Select most visited child
            node = max(node.children, key=lambda c: c.visits)
        return node.state
    
    def _get_visit_distribution(self, root: MCTSNode) -> List[int]:
        """Get visit counts for root children."""
        return [c.visits for c in root.children]


# =============================================================================
# LEXICAL MDP FORMALIZATION
# =============================================================================

@dataclass
class MDPConfig:
    """Configuration for the lexical MDP abstraction."""
    max_depth: int = 100
    terminal_strings: List[str] = field(default_factory=lambda: ["</s>"])
    terminal_max_len: int = 2000
    step_delimiter: str = "\n\n"


class LexicalMDP:
    """
    Formalizes text generation as a Markov Decision Process:
    τ = ⟨s₀, a₀, s₁, a₁, ...⟩ where states are strings and actions are token strings.

    Both MCTSGenerator and AStarDecoder can operate as clients of this MDP.
    """

    def __init__(
        self,
        policy_model: Any,
        tokenizer: Any,
        config: Optional[MDPConfig] = None,
    ):
        self.policy = policy_model
        self.tokenizer = tokenizer
        self.config = config or MDPConfig()

    def initial_state(self, prompt: str) -> str:
        """Return the initial MDP state from a prompt string."""
        return prompt

    def transition(self, state: str, action: str) -> str:
        """Apply action to state: concatenate action string onto state."""
        return state + action

    def is_terminal(self, state: str) -> bool:
        """Check if state is terminal by length or terminal string membership."""
        cfg = self.config
        if len(state) >= cfg.terminal_max_len:
            return True
        for t in cfg.terminal_strings:
            if state.endswith(t):
                return True
        eos_token = getattr(self.tokenizer, "eos_token", None)
        if eos_token and state.endswith(eos_token):
            return True
        return False

    def legal_actions(
        self,
        state: str,
        n: int = 10,
        temperature: float = 1.0,
    ) -> List[Tuple[str, float]]:
        """
        Sample top-n legal actions (token strings) with their softmax probabilities.
        Returns list of (action_str, probability) tuples.
        """
        device = _resolve_model_device(self.policy)
        inputs = self.tokenizer(
            state, return_tensors="pt", truncation=True,
            max_length=getattr(self.tokenizer, "model_max_length", 2048),
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.policy(**inputs)
        logits = _extract_logits(outputs)[0, -1, :]
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        top_k = torch.topk(probs, min(n, probs.shape[0]))
        actions = []
        for token_id, prob in zip(top_k.indices, top_k.values):
            token_str = self.tokenizer.decode([token_id])
            actions.append((token_str, float(prob)))
        return actions

    def reward(
        self,
        state: str,
        reward_fn: Optional[Callable[[str], float]] = None,
    ) -> float:
        """Delegate to reward_fn if provided, else return 0.0."""
        if reward_fn is not None:
            return float(reward_fn(state))
        return 0.0

    def _get_policy_device(self) -> torch.device:
        return _resolve_model_device(self.policy)


# =============================================================================
# VERIFIABLE REWARD FUNCTIONS
# =============================================================================

class VerifiableRewardFactory:
    """
    Static factory methods for verifiable reward functions.
    All returned callables map completion strings to float in [-1.0, 1.0].
    """

    @staticmethod
    def math_verifier(ground_truth: str) -> Callable[[str], float]:
        """
        Returns a reward function that checks math answers.
        Extracts answer from \\boxed{}, 'answer is X', or last number.
        Returns 1.0 (correct), -1.0 (wrong), 0.0 (could not extract).
        """
        gt = ground_truth.strip()

        def _norm(s: str) -> str:
            """Normalise to float string. Handles fractions, percentages, LaTeX."""
            s = s.strip()
            # Percentage: check first, before any other stripping
            if s.endswith("%"):
                try:
                    return str(float(s[:-1].strip()) / 100.0)
                except ValueError:
                    pass
            # LaTeX fraction: \frac{a}{b}
            frac = re.match(r'\\frac\{([^}]+)\}\{([^}]+)\}', s)
            if frac:
                try:
                    return str(float(frac.group(1)) / float(frac.group(2)))
                except (ValueError, ZeroDivisionError):
                    pass
            # Plain fraction: a/b
            slash = re.match(r'^(-?\d+)\s*/\s*(\d+)$', s)
            if slash:
                try:
                    return str(float(slash.group(1)) / float(slash.group(2)))
                except (ValueError, ZeroDivisionError):
                    pass
            # Scientific notation, plain float, int
            try:
                return str(float(s))
            except ValueError:
                return s.lower()

        def _extract_answer(text: str) -> Optional[str]:
            # Try \boxed{...} — last occurrence wins (final answer)
            # Pattern handles one level of nested braces e.g. \boxed{\frac{1}{2}}
            boxed = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', text)
            if boxed:
                return boxed[-1].strip()
            # Try "answer is X", "answer: X", "result is X"
            ans_match = re.search(
                r'(?:answer|result)\s*(?:is|=|:)\s*([^\s,\.\n]+)', text, re.IGNORECASE
            )
            if ans_match:
                return ans_match.group(1).strip().rstrip(".,;")
            # Try "= X" at end of line (common in math solutions)
            eq_match = re.search(r'=\s*(-?[\d\/\.e\+\-]+)\s*$', text, re.MULTILINE)
            if eq_match:
                return eq_match.group(1).strip()
            # Fallback: last number (int, float, scientific notation)
            numbers = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', text)
            if numbers:
                return numbers[-1]
            return None

        def verifier(completion: str) -> float:
            pred = _extract_answer(completion)
            if pred is None:
                return 0.0
            if _norm(pred) == _norm(gt):
                return 1.0
            return -1.0

        return verifier

    @staticmethod
    def code_verifier(
        test_cases: List[Dict],
        timeout: float = 5.0,
    ) -> Callable[[str], float]:
        """
        Returns a reward function that tests code completions against test cases.
        Preferred test case format:
            {"fn": "solve", "args": [5], "kwargs": {}, "expected_output": "10"}

        Legacy format:
            {"call": "solve(5)", "expected_output": "10"}

        Returns pass_rate mapped to [-1, 1]: 2*pass_rate - 1.
        Returns -1.0 if no code block found, 0.0 if test_cases is empty.
        """
        if not test_cases:
            return lambda _: 0.0

        import subprocess as _subprocess  # imported once per factory call, not per reward call
        _log = logging.getLogger("VerifiableRewardFactory.code_verifier")
        blocked_imports = {
            "os", "sys", "subprocess", "socket", "pathlib", "shutil",
            "importlib", "ctypes", "resource", "threading", "multiprocessing",
        }
        blocked_calls = {
            "open", "exec", "eval", "compile", "__import__", "input",
            "breakpoint", "globals", "locals", "vars",
        }

        def _is_safe_code(code: str) -> bool:
            try:
                tree = ast.parse(code)
            except SyntaxError:
                return False
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    if any(alias.name.split(".")[0] in blocked_imports for alias in node.names):
                        return False
                elif isinstance(node, ast.ImportFrom):
                    base = (node.module or "").split(".")[0]
                    if base in blocked_imports:
                        return False
                elif isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Name) and func.id in blocked_calls:
                        return False
                    if isinstance(func, ast.Attribute) and func.attr.startswith("__"):
                        return False
                elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                    return False
            return True

        def verifier(completion: str) -> float:
            # Extract first ```python ... ``` or ``` ... ``` block; bare code is too
            # ambiguous (natural language would be executed) — require fenced block.
            code_match = re.search(r'```(?:python)?\s*\n(.*?)```', completion, re.DOTALL)
            if code_match is None:
                _log.debug("No fenced code block found in completion; returning -1.0")
                return -1.0
            code = code_match.group(1)
            if not _is_safe_code(code):
                _log.warning("Code verifier rejected completion due to blocked syntax/imports")
                return -1.0

            passed = 0
            for tc in test_cases:
                call = tc.get("call", "")
                expected = str(tc.get("expected_output", "")).strip()
                fname = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".py", delete=False
                    ) as f:
                        fname = f.name
                        f.write("import json\n")
                        f.write(code)
                        f.write("\n")
                        if "fn" in tc:
                            fn_name = tc["fn"]
                            args_blob = json.dumps(tc.get("args", []))
                            kwargs_blob = json.dumps(tc.get("kwargs", {}))
                            f.write(
                                f"_args = json.loads({args_blob!r})\n"
                                f"_kwargs = json.loads({kwargs_blob!r})\n"
                                f"_result = str({fn_name}(*_args, **_kwargs)).strip()\n"
                                f"print('PASS' if _result == {repr(expected)} else 'FAIL')\n"
                            )
                        else:
                            f.write(
                                f"_result = str({call}).strip()\n"
                                f"print('PASS' if _result == {repr(expected)} else 'FAIL')\n"
                            )
                    proc = _subprocess.run(
                        [sys.executable, fname],
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                    if proc.stdout.strip() == "PASS":
                        passed += 1
                    elif proc.returncode != 0 and proc.stderr:
                        _log.debug("Test case stderr: %s", proc.stderr[:200])
                except _subprocess.TimeoutExpired:
                    _log.warning("Code verifier timed out after %.1fs on test case", timeout)
                except Exception as exc:
                    _log.warning("Code verifier execution error: %s", exc)
                finally:
                    if fname is not None:
                        try:
                            os.unlink(fname)
                        except OSError:
                            pass

            pass_rate = passed / len(test_cases)
            return 2.0 * pass_rate - 1.0

        return verifier

    @staticmethod
    def format_verifier(
        required_tags: Optional[List[str]] = None,
        forbidden_patterns: Optional[List[str]] = None,
        min_steps: int = 0,
        step_marker: str = "\n\n",
    ) -> Callable[[str], float]:
        """
        Returns a reward function that checks structural compliance.
        Maps pass ratio to [-1, 1]: 2*ratio - 1.
        """
        required_tags = required_tags or []
        forbidden_patterns = forbidden_patterns or []

        def verifier(completion: str) -> float:
            checks: List[bool] = []
            for tag in required_tags:
                checks.append(tag in completion)
            for pat in forbidden_patterns:
                checks.append(not re.search(pat, completion))
            if min_steps > 0:
                steps = [s for s in completion.split(step_marker) if s.strip()]
                checks.append(len(steps) >= min_steps)
            if not checks:
                return 0.0
            pass_ratio = sum(checks) / len(checks)
            return 2.0 * pass_ratio - 1.0

        return verifier


# =============================================================================
# CHAIN-OF-THOUGHT GENERATION (R1-STYLE)
# =============================================================================

@dataclass
class ChainOfThoughtConfig:
    """Configuration for two-phase R1/o1-style chain-of-thought generation."""
    max_thinking_tokens: int = 512
    max_answer_tokens: int = 256
    think_start_tag: str = "<think>"
    think_end_tag: str = "</think>"
    temperature: float = 0.8
    strip_thinking: bool = True
    prm_scorer: Optional[Any] = None


class ChainOfThoughtGenerator:
    """Two-phase R1/o1-style generation."""

    _log = logging.getLogger("ChainOfThoughtGenerator")

    def __init__(
        self,
        policy_model: Any,
        tokenizer: Any,
        config: Optional[ChainOfThoughtConfig] = None,
    ):
        self.policy = policy_model
        self.tokenizer = tokenizer
        self.config = config or ChainOfThoughtConfig()

    @torch.no_grad()
    def generate(self, prompt: str) -> Dict[str, Any]:
        cfg = self.config

        phase1_prompt = prompt + cfg.think_start_tag
        thinking_text = self._generate_phase(
            phase1_prompt, cfg.max_thinking_tokens, cfg.think_end_tag
        ).strip()

        full_context = (
            prompt + cfg.think_start_tag + thinking_text + cfg.think_end_tag
        )
        answer_text = self._generate_phase(
            full_context, cfg.max_answer_tokens, stop_string=None
        )

        full_sequence = full_context + answer_text
        servable_sequence = prompt + answer_text if cfg.strip_thinking else full_sequence

        thinking_score: Optional[float] = None
        prm = cfg.prm_scorer
        if prm is not None:
            if hasattr(prm, "score"):
                thinking_score = float(prm.score(thinking_text))
            elif hasattr(prm, "score_text"):
                thinking_score = float(prm.score_text(thinking_text))

        return {
            "answer": answer_text,
            "thinking": thinking_text,
            "full_sequence": full_sequence,
            "servable_sequence": servable_sequence,
            "thinking_score": thinking_score,
        }

    def _generate_phase(
        self,
        prompt: str,
        max_new_tokens: int,
        stop_string: Optional[str],
    ) -> str:
        cfg = self.config
        device = _resolve_model_device(self.policy)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        try:
            output_ids = self.policy.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=cfg.temperature,
                do_sample=True,
                pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            )
        except Exception as exc:
            self._log.error("Generation phase failed: %s", exc)
            return ""

        prompt_len = inputs["input_ids"].shape[1]
        new_ids = output_ids[0][prompt_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=False)

        if stop_string is not None:
            idx = text.find(stop_string)
            if idx >= 0:
                text = text[:idx]

        return text


# =============================================================================
# A* SEARCH DECODING
# =============================================================================

@dataclass
class AStarConfig:
    """Configuration for A* search decoding."""
    max_nodes: int = 200
    max_depth: int = 50
    n_actions: int = 8
    heuristic_weight: float = 1.0
    temperature: float = 0.8
    use_value_heuristic: bool = True


@dataclass(eq=False)
class AStarNode:
    """
    Node in the A* search frontier.
    eq=False prevents auto-generated __eq__ from traversing the parent chain.
    """
    state: str
    g_score: float
    h_score: float
    parent: Optional['AStarNode']
    action: str
    depth: int

    @property
    def f_score(self) -> float:
        return self.g_score + self.h_score


class AStarDecoder:
    """
    Best-first (A*) search decoding over a LexicalMDP.

    g-score: cumulative PRM score along the path.
    h-score: value model estimate of future reward (heuristic).
    Falls back to greedy best-first (h=0) when no value model is provided
    or use_value_heuristic=False.

    When kv_cache is provided, each node expansion uses prefix KV reuse:
    the shared prompt prefix is computed once and cached; subsequent node
    expansions that extend that prefix replay only suffix tokens, avoiding
    O(full-sequence) forward passes for every node in the search tree.
    """

    def __init__(
        self,
        mdp: LexicalMDP,
        config: Optional[AStarConfig] = None,
        prm: Optional[Any] = None,
        value_model: Optional[Any] = None,
        kv_cache: Optional["PagedKVCache"] = None,
    ):
        """
        Args:
            mdp: LexicalMDP that provides transition, legal_actions, and is_terminal.
            config: AStarConfig controlling search parameters.
            prm: Optional PRM scorer; provides g-score via _prm_score().
            value_model: Optional value head; provides h-score via _heuristic().
            kv_cache: Optional PagedKVCache for prefix KV reuse across node expansions.
        """
        self.mdp = mdp
        self.config = config or AStarConfig()
        self.prm = prm
        self.value_model = value_model
        self.kv_cache = kv_cache

    def decode(
        self,
        prompt: str,
        reward_fn: Optional[Callable[[str], float]] = None,
    ) -> Dict[str, Any]:
        """
        Run A* search from prompt.

        Returns:
            {
                'text': str,
                'g_score': float,
                'depth': int,
                'nodes_expanded': int,
                'path': List[str],   # action sequence
            }
        """
        cfg = self.config
        mdp = self.mdp

        start_state = mdp.initial_state(prompt)
        h0 = self._heuristic(start_state)
        root = AStarNode(
            state=start_state,
            g_score=0.0,
            h_score=h0,
            parent=None,
            action="",
            depth=0,
        )

        frontier: List[Tuple[float, int, AStarNode]] = []
        push_index = 0
        heapq.heappush(frontier, (root.f_score, push_index, root))
        best_g_by_state: Dict[str, float] = {start_state: 0.0}
        nodes_expanded = 0
        best_terminal: Optional[Tuple[AStarNode, float]] = None
        best_nonterminal: Optional[AStarNode] = root

        while frontier and nodes_expanded < cfg.max_nodes:
            _priority, _order, node = heapq.heappop(frontier)

            if node.g_score < best_g_by_state.get(node.state, float("-inf")):
                continue
            nodes_expanded += 1

            if mdp.is_terminal(node.state) or node.depth >= cfg.max_depth:
                terminal_bonus = float(reward_fn(node.state)) if reward_fn is not None else 0.0
                effective_g = node.g_score + terminal_bonus
                if best_terminal is None or effective_g > best_terminal[1]:
                    best_terminal = (node, effective_g)
                continue

            if best_nonterminal is None or node.g_score > best_nonterminal.g_score:
                best_nonterminal = node

            actions = self._cached_legal_actions(
                node.state, prompt, cfg.n_actions, cfg.temperature
            )
            for action_str, _prior in actions:
                new_state = mdp.transition(node.state, action_str)
                step_reward = self._prm_score(new_state)
                g_new = node.g_score + step_reward
                if g_new <= best_g_by_state.get(new_state, float("-inf")):
                    continue
                best_g_by_state[new_state] = g_new
                h_new = self._heuristic(new_state)
                child = AStarNode(
                    state=new_state,
                    g_score=g_new,
                    h_score=h_new,
                    parent=node,
                    action=action_str,
                    depth=node.depth + 1,
                )
                push_index += 1
                heapq.heappush(frontier, (child.f_score, push_index, child))

        if best_terminal is not None:
            result_node, result_g = best_terminal
        else:
            result_node = best_nonterminal if best_nonterminal is not None else root
            result_g = result_node.g_score

        return {
            "text": result_node.state,
            "g_score": result_g,
            "depth": result_node.depth,
            "nodes_expanded": nodes_expanded,
            "path": self._extract_path(result_node),
        }

    def _heuristic(self, state: str) -> float:
        """Compute A* heuristic via value model. Returns 0.0 when disabled."""
        cfg = self.config
        if not cfg.use_value_heuristic or self.value_model is None:
            return 0.0
        vm = self.value_model
        if hasattr(vm, "score"):
            return float(vm.score(state)) * cfg.heuristic_weight
        if hasattr(vm, "score_text"):
            return float(vm.score_text(state)) * cfg.heuristic_weight
        return 0.0

    def _prm_score(self, state: str) -> float:
        """Score a state with the PRM. Returns 0.0 if no PRM."""
        if self.prm is None:
            return 0.0
        if hasattr(self.prm, "score"):
            return float(self.prm.score(state))
        if hasattr(self.prm, "score_text"):
            return float(self.prm.score_text(state))
        return 0.0

    def _extract_path(self, node: AStarNode) -> List[str]:
        """Iteratively extract the action path from root to node."""
        path: List[str] = []
        current = node
        while current is not None and current.action:
            path.append(current.action)
            current = current.parent
        path.reverse()
        return path

    def _cached_legal_actions(
        self,
        state: str,
        prompt: str,
        n: int,
        temperature: float,
    ) -> List[Tuple[str, float]]:
        """
        Wrap mdp.legal_actions() with optional PagedKVCache prefix reuse.

        When kv_cache is available and state extends the prompt prefix, computes
        the shared-prefix KV state once (via get_or_compute_prefix), then replays
        only the suffix tokens to obtain the next-token logits for this node.
        This avoids re-tokenizing and re-computing the full prompt on every
        node expansion during the A* search loop.

        Falls back to mdp.legal_actions() when:
          - kv_cache is None
          - state does not extend the cached prompt prefix
          - the model does not return past_key_values
          - any error occurs during suffix replay

        Args:
            state: Current node state string.
            prompt: Root prompt (shared prefix for all nodes in this search).
            n: Number of candidate actions to return.
            temperature: Softmax temperature for action sampling.

        Returns:
            List of (action_str, probability) tuples, same format as legal_actions().
        """
        if self.kv_cache is None or not state.startswith(prompt) or len(state) <= len(prompt):
            return self.mdp.legal_actions(state, n=n, temperature=temperature)

        try:
            policy = self.mdp.policy
            tokenizer = self.mdp.tokenizer
            device = _resolve_model_device(policy)

            prefix_entry = self.kv_cache.get_or_compute_prefix(policy, tokenizer, prompt)
            if prefix_entry is None:
                return self.mdp.legal_actions(state, n=n, temperature=temperature)

            past_key_values = prefix_entry.get("past_key_values")
            logits = prefix_entry.get("last_logits")
            if past_key_values is None or logits is None:
                return self.mdp.legal_actions(state, n=n, temperature=temperature)

            # Replay only the suffix tokens that extend the cached prompt prefix.
            suffix_text = state[len(prompt):]
            try:
                suffix_inputs = tokenizer(
                    suffix_text, return_tensors="pt", add_special_tokens=False
                )
            except TypeError:
                suffix_inputs = tokenizer(suffix_text, return_tensors="pt")

            suffix_ids = suffix_inputs.get("input_ids")
            if suffix_ids is not None and suffix_ids.numel() > 0:
                suffix_ids = suffix_ids.to(device)
                with torch.no_grad():
                    for i in range(suffix_ids.shape[1]):
                        next_token = suffix_ids[:, i:i + 1]
                        outputs = policy(
                            input_ids=next_token,
                            use_cache=True,
                            past_key_values=past_key_values,
                        )
                        logits = _extract_logits(outputs)[:, -1, :]
                        past_key_values = _extract_past_key_values(outputs)

            logits = logits.to(device)
            probs = F.softmax(logits[0] / max(temperature, 1e-6), dim=-1)
            top_k = torch.topk(probs, min(n, probs.shape[0]))
            actions = []
            for token_id, prob in zip(top_k.indices, top_k.values):
                token_str = tokenizer.decode([token_id])
                actions.append((token_str, float(prob)))
            return actions

        except Exception:
            # Graceful fallback: never let a cache error abort the search.
            return self.mdp.legal_actions(state, n=n, temperature=temperature)


# =============================================================================
# TREE ROLLOUT COLLECTION (SEARCH → TRAINING DATA BRIDGE)
# =============================================================================

@dataclass
class RolloutSample:
    """A single training-ready sample collected from an MCTS rollout."""
    prompt: str
    completion: str
    reward: float
    depth: int
    visit_count: int
    full_state: str
    path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AStarGenerator:
    """
    Unified A* search generator wrapping LexicalMDP + AStarDecoder.

    Mirrors MCTSGenerator's generate(prompt, reward_fn) -> dict interface.
    A* is the most PRM-native search path: the PRM provides both the step-level
    g-score (via _prm_score) and the value heuristic (via _heuristic).

    Usage::

        gen = AStarGenerator(policy, tokenizer, prm=prm_adapter, value_model=val_adapter)
        result = gen.generate("Solve: 2x + 5 = 13", reward_fn=math_verifier)
        print(result["text"], result["g_score"])
    """

    def __init__(
        self,
        policy_model: Any,
        tokenizer: Any,
        config: Optional[AStarConfig] = None,
        mdp_config: Optional[MDPConfig] = None,
        prm: Optional[Any] = None,
        value_model: Optional[Any] = None,
        kv_cache: Optional[PagedKVCache] = None,
    ):
        """
        Args:
            policy_model: Language model used as the A* expansion policy.
            tokenizer: HuggingFace-compatible tokenizer.
            config: AStarConfig controlling beam width, max depth, etc.
            mdp_config: MDPConfig for the underlying LexicalMDP.
            prm: Optional ProcessRewardModelAdapter or compatible scorer
                 that acts as both g-score accumulator and heuristic.
            value_model: Optional value head for heuristic estimation.
            kv_cache: Optional PagedKVCache for prompt-prefix KV reuse across
                      node expansions. Mirrors MCTSGenerator's kv_cache param.
        """
        self.policy = policy_model
        self.tokenizer = tokenizer
        self.config = config or AStarConfig()
        self._mdp_config = mdp_config
        self.prm = prm
        self.value_model = value_model
        self.kv_cache = kv_cache

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        reward_fn: Optional[Callable[[str], float]] = None,
    ) -> Dict[str, Any]:
        """
        Run A* search from prompt.

        Returns a dict with keys matching MCTSGenerator output where possible:
            text           — best decoded text
            g_score        — cumulative PRM score of winning path
            depth          — depth of result node
            nodes_expanded — nodes expanded during search
            path           — action sequence from root to result

        Args:
            prompt: Input prompt to search from.
            reward_fn: Optional terminal reward function; passed to AStarDecoder.decode().

        Returns:
            Dict with keys: text, g_score, depth, nodes_expanded, path.
        """
        mdp = LexicalMDP(
            policy_model=self.policy,
            tokenizer=self.tokenizer,
            config=self._mdp_config,
        )
        decoder = AStarDecoder(
            mdp=mdp,
            config=self.config,
            prm=self.prm,
            value_model=self.value_model,
            kv_cache=self.kv_cache,
        )
        return decoder.decode(prompt, reward_fn=reward_fn)


class TreeRolloutCollector:
    """
    Bridge between MCTSGenerator and GRPO training data collection.

    Runs MCTS over a list of prompts, extracts the top-n leaf paths by
    visit count, and returns RolloutSample objects ready for training.
    Callers pass the result to GRPOTrainer — this class does NOT call it.
    """

    def __init__(
        self,
        mcts_generator: MCTSGenerator,
        reward_fn: Callable[[str], float],
        min_reward_threshold: float = 0.0,
    ):
        self.mcts = mcts_generator
        self.reward_fn = reward_fn
        self.min_reward_threshold = min_reward_threshold

    def collect(
        self,
        prompts: List[str],
        n_samples_per_prompt: int = 8,
        max_length: int = 512,
    ) -> List[RolloutSample]:
        """
        Run MCTS for each prompt, collect top-n paths, return RolloutSamples.
        """
        samples: List[RolloutSample] = []
        for prompt in prompts:
            result = self.mcts.generate(prompt, max_length, self.reward_fn)
            root = result.get("root")
            if root is None:
                continue
            paths = self._extract_leaf_paths(root, n_samples_per_prompt)
            for node, visit_count, depth in paths:
                full_state = node.state
                completion = full_state[len(prompt):]
                reward = node.reward if node.reward is not None else float(self.reward_fn(full_state))
                if reward < self.min_reward_threshold:
                    continue
                samples.append(RolloutSample(
                    prompt=prompt,
                    completion=completion,
                    reward=reward,
                    depth=depth,
                    visit_count=visit_count,
                    full_state=full_state,
                    path=self._extract_path(node),
                    metadata={"node_reward_present": node.reward is not None},
                ))
        return samples

    def _extract_leaf_paths(
        self,
        root: MCTSNode,
        n: int,
    ) -> List[Tuple[Any, int, int]]:
        leaves: List[Tuple[Any, int, int]] = []
        stack: List[MCTSNode] = [root]
        while stack:
            node = stack.pop()
            if not node.children:
                leaves.append((node, node.visits, node.depth))
            else:
                stack.extend(node.children)
        leaves.sort(key=lambda x: x[1], reverse=True)
        return leaves[:n]

    @staticmethod
    def _extract_path(node: MCTSNode) -> List[str]:
        path: List[str] = []
        current: Optional[MCTSNode] = node
        while current is not None and current.parent is not None:
            path.append(current.action)
            current = current.parent
        path.reverse()
        return path


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
