"""
Inference Protocols and Adapter Layer

Defines Protocol-based contracts for policy, reward, and value interfaces
so that inference_optimizations.py components never depend on ad-hoc
model attributes (.device, .score_text) that may not exist.

Usage
-----
from inference_protocols import PolicyAdapter, RewardScorerAdapter, ValueScorerAdapter

policy  = PolicyAdapter.from_rlhf_model(rlhf_policy, tokenizer, device)
reward  = RewardScorerAdapter.from_rlhf_model(rlhf_reward, tokenizer, device)
value   = ValueScorerAdapter.from_rlhf_model(rlhf_value, tokenizer, device)

# Pass to inference_optimizations classes:
sampler = BestOfNSampler(policy, reward, tokenizer=tokenizer)
mcts    = MCTSGenerator(policy, value, tokenizer)
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn

logger = logging.getLogger("InferenceProtocols")


# =============================================================================
# PROTOCOLS
# =============================================================================

@runtime_checkable
class PolicyLike(Protocol):
    """Minimal interface expected of a policy model by inference components."""

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kw) -> Any:
        ...

    def generate(self, input_ids: torch.Tensor, **kw) -> torch.Tensor:
        ...

    def get_device(self) -> torch.device:
        ...


@runtime_checkable
class RewardScorerLike(Protocol):
    """Interface for a reward model that can score (prompt, completion) text pairs."""

    def score(self, prompt: str, completion: str = "") -> float:
        ...


@runtime_checkable
class ValueScorerLike(Protocol):
    """Interface for a value model that can score a state text."""

    def score(self, text: str) -> float:
        ...


# =============================================================================
# ADAPTERS — wrap rlhf.py objects to satisfy protocols
# =============================================================================

class PolicyAdapter:
    """
    Wraps rlhf.PolicyModel (or any nn.Module) and exposes:
      - forward() / generate() delegating to the underlying model
      - get_device() via parameter iteration (never assumes .device attr)
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self._model = model
        self._device = device

    def _inner_model(self) -> nn.Module:
        """Resolve raw HF model when wrapped by rlhf.PolicyModel."""
        return getattr(self._model, "model", self._model)

    @classmethod
    def from_rlhf_model(
        cls,
        model: nn.Module,
        tokenizer: Any = None,
        device: Optional[torch.device] = None,
        max_length: int = 512,
    ) -> "PolicyAdapter":
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        return cls(model, device)

    def get_device(self) -> torch.device:
        if self._device is not None:
            return self._device
        try:
            return next(self._inner_model().parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def device(self) -> torch.device:
        """Backwards-compatible attribute expected by legacy callers."""
        return self.get_device()

    @staticmethod
    def _normalize_output(output: Any) -> Any:
        """
        Normalize outputs to an attribute-style object.

        rlhf.PolicyModel.forward returns dict {'logits': ..., 'loss': ...}.
        HuggingFace models typically return dataclass-like objects with attributes.
        """
        if isinstance(output, dict):
            return SimpleNamespace(**output)
        return output

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kw) -> Any:
        inner = self._inner_model()
        output = inner(input_ids=input_ids, attention_mask=attention_mask, **kw)
        return self._normalize_output(output)

    def generate(self, input_ids: torch.Tensor, **kw) -> torch.Tensor:
        inner = self._inner_model()
        return inner.generate(input_ids, **kw)

    def parameters(self):
        return self._inner_model().parameters()

    def __call__(self, *args, **kwargs):
        if args:
            # Preserve generic nn.Module call semantics when positional args are used.
            output = self._inner_model()(*args, **kwargs)
            return self._normalize_output(output)
        return self.forward(**kwargs)


class RewardScorerAdapter:
    """
    Wraps rlhf.RewardModel (or any nn.Module) and exposes score(prompt, completion).

    Tokenizes the concatenated text, calls model.forward(), and returns a scalar.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: torch.device,
        max_length: int = 512,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._max_length = max_length

    @classmethod
    def from_rlhf_model(
        cls,
        model: nn.Module,
        tokenizer: Any,
        device: Optional[torch.device] = None,
        max_length: int = 512,
    ) -> "RewardScorerAdapter":
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        return cls(model, tokenizer, device, max_length)

    def score(self, prompt: str, completion: str = "") -> float:
        text = (prompt + " " + completion).strip() if completion else prompt
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
            padding=False,
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model(**enc)
        # rlhf.RewardModel returns a scalar tensor of shape (batch,)
        if isinstance(out, torch.Tensor):
            return float(out.squeeze().mean())
        # Handle dict-like outputs
        for attr in ("logits", "rewards", "score"):
            if hasattr(out, attr):
                return float(getattr(out, attr).squeeze().mean())
        return 0.0

    def score_batch(self, texts: list) -> list:
        """Batch-score a list of texts. Used by BestOfNSampler batch_score path."""
        return [self.score(t) for t in texts]

    def score_text(self, text: str) -> float:
        """Backwards-compatible alias expected by legacy inference code."""
        return self.score(text, "")


class ValueScorerAdapter:
    """
    Wraps rlhf.ValueModel (or any nn.Module) and exposes score(text).

    Tokenizes text, calls model.forward(return_all_values=False), returns scalar.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: torch.device,
        max_length: int = 512,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._max_length = max_length

    @classmethod
    def from_rlhf_model(
        cls,
        model: nn.Module,
        tokenizer: Any,
        device: Optional[torch.device] = None,
        max_length: int = 512,
    ) -> "ValueScorerAdapter":
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        return cls(model, tokenizer, device, max_length)

    def score(self, text: str) -> float:
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
            padding=False,
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            # rlhf.ValueModel supports return_all_values flag
            try:
                out = self._model(**enc, return_all_values=False)
            except TypeError:
                out = self._model(**enc)
        if isinstance(out, torch.Tensor):
            return float(out.squeeze().mean())
        return 0.0

    def score_text(self, text: str) -> float:
        """Backwards-compatible alias expected by legacy inference code."""
        return self.score(text)
