"""Typed data containers for model module boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class ObservationBatch:
    """Batch of observations from the environment."""

    images: dict[str, Tensor]  # camera_name -> [B, C, H, W]
    proprio: Tensor  # [B, D_proprio]
    language: str | list[str] | None = None
    timestamps: Tensor | None = None  # [B]
    masks: Tensor | None = None  # [B]


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
    action_mask: Tensor | None = None


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
