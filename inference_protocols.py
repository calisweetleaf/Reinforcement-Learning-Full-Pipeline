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
import math
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import torch
import torch.nn as nn

logger = logging.getLogger("InferenceProtocols")
DEFAULT_THINK_START_TAG = "<think>"
DEFAULT_THINK_END_TAG = "</think>"


class InferenceProtocolError(RuntimeError):
    """Raised when protocol contracts are violated."""


class ProtocolValidationError(ValueError):
    """Raised on invalid adapter inputs/outputs."""


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


@runtime_checkable
class HiddenCoTLike(Protocol):
    """
    Contract for components that MUST emit hidden reasoning and answer-only text.

    Returns:
        (full_sequence_with_hidden_cot, answer_only_without_hidden_cot)
    """

    def generate_with_hidden_cot(
        self,
        prompt: str,
        max_thinking_tokens: int = 512,
        max_answer_tokens: int = 256,
    ) -> Tuple[str, str]:
        ...


@runtime_checkable
class VerifiableRewardLike(Protocol):
    """Contract for deterministic/rule-based verifiers."""

    def verify(self, completion: str, **kwargs) -> float:
        ...


@runtime_checkable
class StepLevelScorerLike(Protocol):
    """Contract for step-level scorers (returns one score per reasoning step)."""

    def score_steps(self, completion: str) -> List[float]:
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
        """
        Resolve the callable policy surface.

        For rlhf.PolicyModel wrappers, unwrap exactly one level to reach the
        HF causal-lm module. For already-raw HF causal-lm models, keep the
        top-level module so logits/lm_head remain available.
        """
        candidate = getattr(self._model, "model", None)
        if candidate is None:
            return self._model

        # rlhf.PolicyModel exposes get_log_probs and should be unwrapped once.
        if hasattr(self._model, "get_log_probs"):
            return candidate

        # Raw HF causal LM modules also expose `.model`, but unwrapping them
        # drops lm_head/logits and breaks search scorers.
        if hasattr(self._model, "lm_head"):
            return self._model

        return candidate

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


class HiddenCoTAdapter:
    """
    Enforces a strict hidden-CoT contract over generation components.

    This adapter is intended for Q* lanes where hidden chain-of-thought is
    mandatory and answer-only serving must be guaranteed.
    """

    def __init__(
        self,
        generator: Any,
        *,
        think_start_tag: str = DEFAULT_THINK_START_TAG,
        think_end_tag: str = DEFAULT_THINK_END_TAG,
    ):
        if generator is None:
            raise ProtocolValidationError("HiddenCoTAdapter requires a non-null generator.")
        if not isinstance(think_start_tag, str) or not think_start_tag.strip():
            raise ProtocolValidationError("think_start_tag must be a non-empty string.")
        if not isinstance(think_end_tag, str) or not think_end_tag.strip():
            raise ProtocolValidationError("think_end_tag must be a non-empty string.")
        if think_start_tag == think_end_tag:
            raise ProtocolValidationError("think_start_tag and think_end_tag must differ.")
        self._generator = generator
        self._think_start_tag = think_start_tag
        self._think_end_tag = think_end_tag

    def generate_with_hidden_cot(
        self,
        prompt: str,
        max_thinking_tokens: int = 512,
        max_answer_tokens: int = 256,
    ) -> Tuple[str, str]:
        self._validate_prompt(prompt)
        self._validate_token_budget("max_thinking_tokens", max_thinking_tokens)
        self._validate_token_budget("max_answer_tokens", max_answer_tokens)

        full_sequence: str
        answer_only: str
        generator = self._generator

        if hasattr(generator, "generate_with_hidden_cot"):
            logger.debug("Using native generate_with_hidden_cot path.")
            full_sequence, answer_only = self._call_native_hidden_cot(
                prompt, max_thinking_tokens, max_answer_tokens
            )
        elif hasattr(generator, "generate"):
            logger.debug("Using generic generate() path and normalizing output.")
            full_sequence, answer_only = self._call_generic_generate(prompt)
        else:
            raise InferenceProtocolError(
                "Generator does not implement generate_with_hidden_cot() or generate()."
            )

        if not isinstance(full_sequence, str):
            raise InferenceProtocolError("Hidden CoT full_sequence must be a string.")
        if not isinstance(answer_only, str):
            raise InferenceProtocolError("Hidden CoT answer_only must be a string.")

        full_sequence = full_sequence.strip()
        answer_only = answer_only.strip()
        if not full_sequence:
            raise InferenceProtocolError("Hidden CoT generation returned empty full_sequence.")
        if not answer_only:
            answer_only = self._derive_answer_only(full_sequence)

        self._validate_hidden_cot(full_sequence)
        answer_only = self._sanitize_answer_only(answer_only)
        if not answer_only:
            raise InferenceProtocolError("Answer-only output is empty after sanitization.")

        logger.info(
            "HiddenCoTAdapter generated output successfully (full_len=%d, answer_len=%d).",
            len(full_sequence),
            len(answer_only),
        )
        return full_sequence, answer_only

    def _call_native_hidden_cot(
        self,
        prompt: str,
        max_thinking_tokens: int,
        max_answer_tokens: int,
    ) -> Tuple[str, str]:
        fn = self._generator.generate_with_hidden_cot
        try:
            result = fn(
                prompt=prompt,
                max_thinking_tokens=max_thinking_tokens,
                max_answer_tokens=max_answer_tokens,
            )
        except TypeError:
            # Backward-compatible call shape.
            result = fn(prompt, max_thinking_tokens=max_thinking_tokens)
        return self._normalize_pair_result(result, source="generate_with_hidden_cot")

    def _call_generic_generate(self, prompt: str) -> Tuple[str, str]:
        result = self._generator.generate(prompt)
        if isinstance(result, dict):
            full_sequence = self._pick_first_non_empty(
                result,
                ("full_sequence", "full_text", "raw_text", "sequence"),
            )
            answer_only = self._pick_first_non_empty(
                result,
                ("answer_only", "servable_sequence", "answer"),
            )
            if not full_sequence and answer_only:
                full_sequence = prompt + " " + answer_only
            if not answer_only and full_sequence:
                answer_only = self._derive_answer_only(full_sequence)
            return full_sequence, answer_only
        return self._normalize_pair_result(result, source="generate")

    @staticmethod
    def _normalize_pair_result(result: Any, source: str) -> Tuple[str, str]:
        if not isinstance(result, tuple) or len(result) != 2:
            raise InferenceProtocolError(
                f"{source} must return a 2-tuple (full_sequence, answer_only)."
            )
        full_sequence, answer_only = result
        if not isinstance(full_sequence, str) or not isinstance(answer_only, str):
            raise InferenceProtocolError(
                f"{source} returned non-string outputs; expected (str, str)."
            )
        return full_sequence, answer_only

    @staticmethod
    def _pick_first_non_empty(payload: Dict[str, Any], keys: Tuple[str, ...]) -> str:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _validate_prompt(prompt: str) -> None:
        if not isinstance(prompt, str):
            raise ProtocolValidationError("prompt must be a string.")
        if not prompt.strip():
            raise ProtocolValidationError("prompt must be non-empty.")

    @staticmethod
    def _validate_token_budget(name: str, value: int) -> None:
        if not isinstance(value, int):
            raise ProtocolValidationError(f"{name} must be an integer.")
        if value <= 0:
            raise ProtocolValidationError(f"{name} must be positive.")
        if value > 16384:
            raise ProtocolValidationError(f"{name}={value} exceeds safety ceiling 16384.")

    def _validate_hidden_cot(self, full_sequence: str) -> None:
        has_start = self._think_start_tag in full_sequence
        has_end = self._think_end_tag in full_sequence
        if not has_start or not has_end:
            raise InferenceProtocolError(
                "Hidden CoT contract violation: missing required thinking tags."
            )
        if has_start != has_end:
            raise InferenceProtocolError("Malformed hidden CoT tags: missing opening/closing pair.")
        if has_start and full_sequence.find(self._think_start_tag) > full_sequence.find(self._think_end_tag):
            raise InferenceProtocolError("Malformed hidden CoT tags: closing tag appears before opening tag.")

    def _derive_answer_only(self, full_sequence: str) -> str:
        if self._think_start_tag not in full_sequence:
            return full_sequence.strip()
        pattern = (
            re.escape(self._think_start_tag)
            + r".*?"
            + re.escape(self._think_end_tag)
        )
        stripped = re.sub(pattern, "", full_sequence, flags=re.DOTALL).strip()
        return stripped

    def _sanitize_answer_only(self, answer_only: str) -> str:
        if self._think_start_tag in answer_only or self._think_end_tag in answer_only:
            logger.warning("Answer-only output still contains hidden-CoT tags; stripping tags defensively.")
            answer_only = self._derive_answer_only(answer_only)
        return answer_only.strip()


class VerifiableRewardAdapter:
    """
    Normalizes deterministic verifier interfaces to verify(completion, **kwargs).
    """

    def __init__(self, verifier: Any, *, clamp_to_unit: bool = True):
        if verifier is None:
            raise ProtocolValidationError("VerifiableRewardAdapter requires a non-null verifier.")
        self._verifier = verifier
        self._clamp_to_unit = bool(clamp_to_unit)

    def verify(self, completion: str, **kwargs) -> float:
        if not isinstance(completion, str):
            raise ProtocolValidationError("completion must be a string.")
        if not completion.strip():
            logger.warning("Verifier received empty completion text.")

        raw_score = self._call_verifier(completion, **kwargs)
        if not isinstance(raw_score, (int, float)):
            raise InferenceProtocolError("Verifier returned non-numeric score.")
        score = float(raw_score)
        if not math.isfinite(score):
            raise InferenceProtocolError("Verifier returned non-finite score.")

        if self._clamp_to_unit:
            if score < -1.0 or score > 1.0:
                logger.warning("Verifier score %.4f out of range [-1,1]; clamping.", score)
            return max(-1.0, min(1.0, score))
        return score

    def _call_verifier(self, completion: str, **kwargs) -> float:
        verifier = self._verifier
        if hasattr(verifier, "verify"):
            return float(verifier.verify(completion, **kwargs))
        if callable(verifier):
            return float(verifier(completion, **kwargs))
        if hasattr(verifier, "score"):
            return float(verifier.score(completion))
        raise InferenceProtocolError(
            "Verifier must implement verify(), score(), or be callable."
        )


class StepLevelScorerAdapter:
    """
    Adapts step-level scoring for PRM-style reranking with strict score hygiene.
    """

    def __init__(
        self,
        scorer: Any,
        *,
        step_delimiter: str = "\n\n",
        clamp_min: float = -1.0,
        clamp_max: float = 1.0,
    ):
        if scorer is None:
            raise ProtocolValidationError("StepLevelScorerAdapter requires a non-null scorer.")
        if not isinstance(step_delimiter, str) or not step_delimiter:
            raise ProtocolValidationError("step_delimiter must be a non-empty string.")
        if not isinstance(clamp_min, (int, float)) or not isinstance(clamp_max, (int, float)):
            raise ProtocolValidationError("clamp_min/clamp_max must be numeric.")
        if float(clamp_min) >= float(clamp_max):
            raise ProtocolValidationError("clamp_min must be strictly less than clamp_max.")

        self._scorer = scorer
        self._step_delimiter = step_delimiter
        self._clamp_min = float(clamp_min)
        self._clamp_max = float(clamp_max)

    def score_steps(self, completion: str) -> List[float]:
        if not isinstance(completion, str):
            raise ProtocolValidationError("completion must be a string.")

        scorer = self._scorer
        if hasattr(scorer, "score_steps"):
            raw_scores = scorer.score_steps(completion)
            return self._normalize_score_list(raw_scores)

        steps = [s.strip() for s in completion.split(self._step_delimiter) if s.strip()]
        if not steps:
            return [self._normalize_score(self._score_single(completion))]

        return [self._normalize_score(self._score_single(step)) for step in steps]

    def _score_single(self, text: str) -> float:
        scorer = self._scorer
        if hasattr(scorer, "score"):
            return float(scorer.score(text))
        if hasattr(scorer, "score_text"):
            return float(scorer.score_text(text))
        if callable(scorer):
            return float(scorer(text))
        raise InferenceProtocolError(
            "Step scorer must implement score_steps(), score(), score_text(), or be callable."
        )

    def _normalize_score_list(self, raw_scores: Any) -> List[float]:
        if not isinstance(raw_scores, list):
            raise InferenceProtocolError("score_steps() must return a list of numeric scores.")
        if not raw_scores:
            logger.warning("score_steps() returned an empty list; injecting neutral score 0.0.")
            return [0.0]
        return [self._normalize_score(v) for v in raw_scores]

    def _normalize_score(self, value: Any) -> float:
        if not isinstance(value, (int, float)):
            raise InferenceProtocolError("Step scorer produced non-numeric score.")
        score = float(value)
        if not math.isfinite(score):
            raise InferenceProtocolError("Step scorer produced non-finite score.")
        if score < self._clamp_min or score > self._clamp_max:
            logger.warning(
                "Step score %.4f outside [%.4f, %.4f]; clamping.",
                score,
                self._clamp_min,
                self._clamp_max,
            )
        return max(self._clamp_min, min(self._clamp_max, score))


class ProcessRewardModelAdapter:
    """
    Wraps rlhf.ProcessRewardModel and exposes RewardScorerLike + StepLevelScorerLike contracts.

    score_steps() uses the PRM's internal token-aligned boundary detection (newline token IDs
    13/198/271 by default, or learned detector) instead of naive text splitting — the key
    correctness improvement over StepLevelScorerAdapter.

    Blending contract matches RewardFunctionFactory.from_process_reward_model() in rlhf.py:
      reward = outcome_reward + process_weight * mean(process_rewards[:num_steps])
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: torch.device,
        max_length: int = 2048,
        process_weight: float = 0.0,
    ):
        """
        Args:
            model: ProcessRewardModel instance from rlhf.py.
            tokenizer: HuggingFace-compatible tokenizer.
            device: Target device for inference.
            max_length: Tokenizer truncation limit.
            process_weight: Blend weight for process rewards vs outcome reward.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._max_length = max_length
        self._process_weight = float(process_weight)

    @classmethod
    def from_rlhf_model(
        cls,
        prm: nn.Module,
        tokenizer: Any,
        device: Optional[torch.device] = None,
        max_length: int = 2048,
        process_weight: float = 0.0,
    ) -> "ProcessRewardModelAdapter":
        """
        Construct adapter from a ProcessRewardModel instance.

        Args:
            prm: ProcessRewardModel from rlhf.py.
            tokenizer: HuggingFace-compatible tokenizer.
            device: Target device; inferred from model parameters if None.
            max_length: Tokenizer truncation limit.
            process_weight: Blend weight for process rewards vs outcome reward.

        Returns:
            Configured ProcessRewardModelAdapter.
        """
        if device is None:
            try:
                device = next(prm.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        return cls(prm, tokenizer, device, max_length, process_weight)

    def _forward(self, text: str) -> Dict[str, Any]:
        """Run one tokenize + model forward pass; return raw output dict."""
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
            padding=False,
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            return self._model(
                enc["input_ids"],
                enc["attention_mask"],
                return_process_rewards=True,
            )

    def score_outcome(self, prompt: str, completion: str = "") -> float:
        """
        Score the final outcome reward for a prompt + completion pair.

        Args:
            prompt: Input prompt or full text.
            completion: Optional completion to append.

        Returns:
            Scalar outcome reward as float.
        """
        text = (prompt + " " + completion).strip() if completion else prompt
        return float(self._forward(text)["outcome_reward"].squeeze().mean())

    def score_steps(self, completion: str) -> List[float]:
        """
        Per-step process_rewards from actual PRM forward pass, not delimiter splitting.

        Uses the PRM's internal token-aligned boundary detection for correctness.

        Args:
            completion: Full completion text to score step-by-step.

        Returns:
            List of per-step reward floats.
        """
        if not isinstance(completion, str):
            raise ProtocolValidationError("completion must be a string.")
        out = self._forward(completion)
        process_rewards = out.get("process_rewards")
        num_steps = out.get("num_steps")
        if process_rewards is None:
            return [float(out["outcome_reward"].squeeze().mean())]
        n = max(
            int(num_steps[0].item()) if num_steps is not None else process_rewards.shape[1],
            1,
        )
        step_scores = process_rewards[0, :n].tolist()
        return [float(s) for s in step_scores] if step_scores else [float(out["outcome_reward"].squeeze().mean())]

    def score_combined(
        self,
        prompt: str,
        completion: str = "",
        process_weight: Optional[float] = None,
    ) -> float:
        """
        Blended outcome + process reward score.

        Mirrors RewardFunctionFactory.from_process_reward_model() in rlhf.py:
          reward = outcome_reward + process_weight * mean(process_rewards[:num_steps])

        Args:
            prompt: Input prompt or full text.
            completion: Optional completion to append.
            process_weight: Override instance process_weight if provided.

        Returns:
            Blended scalar reward as float.
        """
        w = self._process_weight if process_weight is None else float(process_weight)
        text = (prompt + " " + completion).strip() if completion else prompt
        out = self._forward(text)
        outcome = float(out["outcome_reward"].squeeze().mean())
        if w <= 0.0:
            return outcome
        process_rewards = out.get("process_rewards")
        num_steps = out.get("num_steps")
        if process_rewards is None:
            return outcome
        n = max(
            int(num_steps[0].item()) if num_steps is not None else process_rewards.shape[1],
            1,
        )
        process_mean = float(process_rewards[0, :n].mean())
        return outcome + w * process_mean  # mirrors RewardFunctionFactory contract

    def score(self, prompt: str, completion: str = "") -> float:
        """
        Primary scoring entry point; delegates to score_combined().

        Args:
            prompt: Input prompt or full text.
            completion: Optional completion to append.

        Returns:
            Blended scalar reward as float.
        """
        return self.score_combined(prompt, completion)

    def score_batch(self, texts: List[str]) -> List[float]:
        """
        Score a list of texts with a single batched tokenizer call + model forward.

        Uses padded batch tokenization so all texts are processed in one pass.
        Falls back to sequential score() calls on any batching error (e.g. when
        the PRM does not support batched input_ids shape, or the tokenizer does
        not support list inputs).

        Args:
            texts: List of strings to score.

        Returns:
            List of scalar outcome reward floats in the same order as texts.
        """
        if not texts:
            return []
        try:
            enc = self._tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_length,
                padding=True,
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                out = self._model(
                    enc["input_ids"],
                    enc["attention_mask"],
                    return_process_rewards=True,
                )
            outcome_rewards = out["outcome_reward"]
            # outcome_reward may be shape [B] or [B, 1] depending on the PRM head.
            # Flatten to 1-D and return one scalar per text.
            outcome_rewards = outcome_rewards.view(-1)
            if outcome_rewards.shape[0] != len(texts):
                # Shape mismatch: PRM returned unexpected batch dimension — fall through.
                raise ValueError(
                    f"outcome_reward batch size {outcome_rewards.shape[0]} != {len(texts)}"
                )
            return [float(outcome_rewards[i]) for i in range(len(texts))]
        except Exception:
            # Sequential fallback: ensures correctness even when the PRM or
            # tokenizer does not support batched inference.
            return [self.score(t) for t in texts]

    def score_text(self, text: str) -> float:
        """
        Score a single text string (no prompt/completion split).

        Args:
            text: Full text to score.

        Returns:
            Scalar reward as float.
        """
        return self.score(text, "")

    def score_branch_preference(
        self,
        steps_a: List[float],
        steps_b: List[float],
        mode: str = "prm_min",
    ) -> float:
        """Bradley-Terry preference probability P(branch_a > branch_b).

        Compares two reasoning branches using their per-step PRM scores (as
        returned by ``score_steps()``).  Aggregates each branch to a scalar
        via ``mode``, then returns ``sigmoid(agg_a - agg_b)`` — the canonical
        Bradley-Terry pairwise preference probability.

        Used by Tree-GRPO to drive preference-based process reward normalisation
        across sibling branches at the same MCTS parent node.

        Args:
            steps_a: Per-step PRM scores for branch A (from ``score_steps()``).
            steps_b: Per-step PRM scores for branch B (from ``score_steps()``).
            mode: Aggregation strategy applied to both lists before comparison.
                  ``'prm_min'`` (default) — min(steps), most conservative,
                  matches PRM-Min aggregation from the Q* literature (a proof
                  is only as strong as its weakest step).
                  ``'prm_mean'`` — arithmetic mean.
                  ``'prm_last'`` — last step score only.

        Returns:
            Scalar in [0, 1]: probability that branch A is preferred over B.
            0.5 means indifference; >0.5 means A is preferred.
        """
        if not steps_a or not steps_b:
            return 0.5

        def _aggregate(steps: List[float]) -> float:
            if mode == "prm_min":
                return min(steps)
            if mode == "prm_mean":
                return sum(steps) / len(steps)
            if mode == "prm_last":
                return steps[-1]
            raise ValueError(
                f"Unknown mode '{mode}'. Choose 'prm_min', 'prm_mean', or 'prm_last'."
            )

        agg_a = _aggregate(steps_a)
        agg_b = _aggregate(steps_b)
        return float(torch.sigmoid(torch.tensor(agg_a - agg_b, dtype=torch.float32)))


# =============================================================================
# HIDDEN-CoT SAMPLE COLLECTION
# =============================================================================

@dataclass
class HiddenCoTSample:
    """One training-ready hidden-chain-of-thought sample.

    full_sequence is the SFT/GRPO training target: the policy must learn to
    emit ``<think>reasoning</think>answer`` in the completion token stream.
    answer_only is what HiddenCoTAdapter serves at inference time.

    Fields:
        prompt:           Original user prompt.
        full_sequence:    ``prompt + <think>thinking</think>answer`` — training target.
        answer_only:      ``prompt + answer`` — serving target (thinking stripped).
        thinking:         The raw thinking text between tags, for diagnostics.
        reasoning_tokens: Approximate token count of the thinking section.
        answer_tokens:    Approximate token count of the answer section.
        thinking_score:   Optional PRM score of the thinking section (from
                          ``ChainOfThoughtGenerator`` if a PRM scorer is configured).
        tag_valid:        True when ``HiddenCoTAdapter._validate_hidden_cot()`` passed.
    """
    prompt: str
    full_sequence: str
    answer_only: str
    thinking: str
    reasoning_tokens: int
    answer_tokens: int
    thinking_score: Optional[float] = None
    tag_valid: bool = True


class HiddenCoTSampleCollector:
    """Bridge ChainOfThoughtGenerator output to training-ready HiddenCoTSample objects.

    This class is the contract-enforcement layer between the two-phase CoT
    generator in ``inference_optimizations.py`` and the training loop in
    ``rlhf.py``.  Contract is enforced at collection time via ``HiddenCoTAdapter``
    so that invalid samples are caught before they enter a training dataset.

    The primary training path for collected samples::

        collector.collect(prompts) -> List[HiddenCoTSample]
        collector.to_sft_items(samples) -> List[{'prompt': str, 'response': str}]
        SFTDataset(data=items, tokenizer=...)

    The response in each SFT item is ``full_sequence[len(prompt):]``, which is
    exactly the ``<think>...</think>answer`` suffix.  Because ``SFTDataset``
    masks the prompt tokens from the loss, the policy is trained exclusively
    to generate the hidden-CoT + answer structure — matching what
    ``HiddenCoTAdapter`` will strip at serve time.

    For RL training (GRPO / Tree-GRPO), pass the ``full_sequence`` as the
    completion and score the ``answer_only`` portion with the reward function.
    Gradient flows through all tokens including thinking tokens — the model
    learns to reason better via RL advantage signal.
    """

    def __init__(
        self,
        generator: Any,  # ChainOfThoughtGenerator from inference_optimizations
        think_start_tag: str = DEFAULT_THINK_START_TAG,
        think_end_tag: str = DEFAULT_THINK_END_TAG,
    ) -> None:
        self._generator = generator
        self._think_start_tag = think_start_tag
        self._think_end_tag = think_end_tag
        self._adapter = HiddenCoTAdapter(
            generator,
            think_start_tag=think_start_tag,
            think_end_tag=think_end_tag,
        )

    def collect(
        self,
        prompts: List[str],
        max_thinking_tokens: int = 512,
        max_answer_tokens: int = 256,
        skip_invalid: bool = True,
    ) -> List[HiddenCoTSample]:
        """Generate hidden-CoT samples for each prompt.

        Calls ``HiddenCoTAdapter.generate_with_hidden_cot()`` per prompt.
        Contract (tag presence, ordering) is validated by the adapter.

        Args:
            prompts:             List of prompt strings.
            max_thinking_tokens: Token budget for the thinking section.
            max_answer_tokens:   Token budget for the answer section.
            skip_invalid:        If True, drop samples that fail tag validation.
                                 If False, include them with ``tag_valid=False``
                                 for diagnostic purposes.

        Returns:
            List of HiddenCoTSample objects, ordered as prompts.
        """
        samples: List[HiddenCoTSample] = []
        for prompt in prompts:
            tag_valid = True
            try:
                full_sequence, answer_only = self._adapter.generate_with_hidden_cot(
                    prompt,
                    max_thinking_tokens=max_thinking_tokens,
                    max_answer_tokens=max_answer_tokens,
                )
            except (InferenceProtocolError, Exception) as exc:
                logger.warning("HiddenCoTSampleCollector: generation failed for prompt: %s", exc)
                tag_valid = False
                if skip_invalid:
                    continue
                full_sequence = prompt
                answer_only = prompt

            # Extract thinking text between tags for diagnostics.
            thinking = ""
            start_idx = full_sequence.find(self._think_start_tag)
            end_idx = full_sequence.find(self._think_end_tag)
            if start_idx >= 0 and end_idx > start_idx:
                thinking = full_sequence[start_idx + len(self._think_start_tag):end_idx]

            # Approximate token counts (character/4 is a rough estimate;
            # accurate counts require the tokenizer which we don't hold here).
            reasoning_tokens = max(1, len(thinking) // 4)
            answer_suffix = full_sequence[full_sequence.find(self._think_end_tag) + len(self._think_end_tag):] if tag_valid else answer_only
            answer_tokens = max(1, len(answer_suffix) // 4)

            # Extract thinking_score without a second generation pass.
            # ChainOfThoughtGenerator.generate() already ran inside the adapter's
            # _call_generic_generate() path.  We can't recover its return value
            # here without refactoring the adapter, so we call generate() only
            # when the generator exposes thinking_score AND the adapter used the
            # native path (meaning it called generate_with_hidden_cot, not generic
            # generate).  For the generic-generate path the adapter already ran
            # generate() once; we read thinking_score from a fresh call ONLY when
            # tag_valid is False (generation failed anyway, no wasted compute) or
            # when the generator advertises a zero-cost score attribute.
            thinking_score: Optional[float] = None
            if tag_valid and hasattr(self._generator, "generate"):
                # Avoid a second full generation pass.  If the generator caches
                # its last result or exposes a last_thinking_score attribute, use
                # that.  Otherwise leave thinking_score as None — it is optional
                # diagnostic data, not required for training.
                _last = getattr(self._generator, "_last_result", None)
                if isinstance(_last, dict):
                    ts = _last.get("thinking_score")
                    if ts is not None:
                        thinking_score = float(ts)

            sample = HiddenCoTSample(
                prompt=prompt,
                full_sequence=full_sequence,
                answer_only=answer_only,
                thinking=thinking,
                reasoning_tokens=reasoning_tokens,
                answer_tokens=answer_tokens,
                thinking_score=thinking_score,
                tag_valid=tag_valid,
            )
            if not tag_valid and skip_invalid:
                continue
            samples.append(sample)

        return samples

    def to_sft_items(
        self,
        samples: List[HiddenCoTSample],
    ) -> List[Dict[str, str]]:
        """Convert HiddenCoTSample objects to SFTDataset-compatible dicts.

        ``SFTDataset`` expects ``{'prompt': str, 'response': str}``.  The
        response is ``full_sequence[len(prompt):]`` — the
        ``<think>...</think>answer`` suffix.  Only samples where ``tag_valid``
        is True are included; invalid samples are silently skipped.

        The resulting list feeds directly into::

            SFTDataset(data=items, tokenizer=tokenizer, max_length=...)

        No schema translation needed.  The policy learns to generate the
        full ``<think>...</think>answer`` completion, and ``HiddenCoTAdapter``
        strips the thinking section at serve time.
        """
        items: List[Dict[str, str]] = []
        for s in samples:
            if not s.tag_valid:
                continue
            response = s.full_sequence[len(s.prompt):]
            if not response.strip():
                continue
            items.append({"prompt": s.prompt, "response": response})
        return items
