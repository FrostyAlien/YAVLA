"""Typed data containers for model module boundaries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, cast

import torch
from torch import Tensor


def _map_tensor_tree(value: Any, transform: Callable[[Tensor], Tensor]) -> Any:
    """Return a copy of ``value`` with ``transform`` applied to every tensor leaf."""
    if isinstance(value, Tensor):
        return transform(value)
    if isinstance(value, dict):
        return {key: _map_tensor_tree(child, transform) for key, child in value.items()}
    if isinstance(value, list):
        return [_map_tensor_tree(child, transform) for child in value]
    if isinstance(value, tuple):
        return tuple(_map_tensor_tree(child, transform) for child in value)
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{field.name: _map_tensor_tree(getattr(value, field.name), transform) for field in fields(value)}
        )
    return value


@dataclass
class ObservationBatch:
    """Batch of observations from the environment."""

    images: dict[str, Tensor]  # camera_name -> [B, C, H, W]
    proprio: Tensor  # [B, D_proprio]
    language: str | list[str] | None = None
    timestamps: Tensor | None = None  # [B]
    masks: Tensor | None = None  # [B]

    def map_tensors(self, transform: Callable[[Tensor], Tensor]) -> ObservationBatch:
        """Return a same-type copy with ``transform`` applied to every tensor leaf."""
        return cast(ObservationBatch, _map_tensor_tree(self, transform))

    def to(self, device: torch.device | str, non_blocking: bool = False) -> ObservationBatch:
        """Return a same-type copy with all tensors moved to ``device``."""
        return self.map_tensors(lambda tensor: tensor.to(device=device, non_blocking=non_blocking))


@dataclass
class TokenBatch:
    """Merged token sequence ready for backbone consumption."""

    tokens: Tensor  # [B, N, D]
    attention_mask: Tensor  # [B, N]
    token_type_ids: Tensor  # [B, N] — 0=image/bidirectional, 1=causal
    modality_ids: Tensor  # [B, N] — 0=vision, 1=language, 2=proprio, 3=readout
    readout_indices: Tensor | None = None


@dataclass
class BackboneOutput:
    """Output from the backbone module."""

    readout_states: Tensor | None  # [B, N_readout, D]
    token_states: Tensor | None  # [B, N, D]
    attention_mask: Tensor
    aux: dict[str, Tensor] = field(default_factory=dict)


@dataclass
class ActionPrediction:
    """Raw action prediction from the action head."""

    mean: Tensor  # [B, chunk_len, action_dim]
    samples: Tensor | None = None
    log_prob: Tensor | None = None
    aux: dict[str, Tensor] = field(default_factory=dict)


@dataclass
class ActionChunk:
    """Decoded action chunk ready for execution."""

    actions: Tensor  # [B, chunk_len, action_dim]
    dt_hz: float
    chunk_len: int
    action_mask: Tensor | None = None


@dataclass
class LossDict:
    """Training loss with per-component breakdown."""

    total: Tensor  # scalar
    breakdown: dict[str, Tensor] = field(default_factory=dict)


@dataclass
class TrainingBatch:
    """Complete training batch combining observations and ground-truth actions."""

    observations: ObservationBatch
    actions: Tensor  # [B, chunk_len, action_dim]
    dt_hz: float
    chunk_len: int
    action_mask: Tensor | None = None  # [B, chunk_len] — True = padded/invalid (action_is_pad polarity)
    action_dim_mask: Tensor | None = None  # [action_dim] — True = inactive/invalid dimension

    def map_tensors(self, transform: Callable[[Tensor], Tensor]) -> TrainingBatch:
        """Return a same-type copy with ``transform`` applied to every tensor leaf."""
        return cast(TrainingBatch, _map_tensor_tree(self, transform))

    def to(self, device: torch.device | str, non_blocking: bool = False) -> TrainingBatch:
        """Return a same-type copy with all tensors moved to ``device``."""
        return self.map_tensors(lambda tensor: tensor.to(device=device, non_blocking=non_blocking))


@dataclass
class ActionSpaceSpec:
    """Specification of a robot's action space."""

    names: list[str]
    units: list[str]
    # WARNING: The action decoder DOES NOT clamp unnormalized predictions to [-1, 1].
    # To maximize inference speed by avoiding GPU-to-CPU syncs or extra graph ops,
    # the Action Decoder assumes the action head was successfully trained to output
    # natively bounded values. Any clipping must be handled by the environment or dataset.
    limits: Tensor | None  # [action_dim, 2] — (min, max) per dim; None = no normalization
    frame: str = ""
    control_mode: str = ""


@dataclass
class ProprioSpec:
    """Specification of a robot's proprioceptive state."""

    names: list[str]
    units: list[str]
    limits: Tensor | None = None


@dataclass
class FreezeConfig:
    """Controls which VLM modules to freeze and LoRA configuration."""

    freeze_modules: list[str] = field(default_factory=list)
    lora_target_modules: list[str] = field(default_factory=list)  # peft leaf names, e.g. ["q_proj", "v_proj"]
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
