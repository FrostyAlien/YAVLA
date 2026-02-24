"""Protocols, ABC base classes, and capability negotiation for model modules."""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from torch import Tensor, nn

if TYPE_CHECKING:
    from yavla.training.config import OptimizerConfig

from yavla.models.types import (
    ActionChunk,
    ActionPrediction,
    ActionSpaceSpec,
    BackboneOutput,
    LossDict,
    ObservationBatch,
    TrainingBatch,
)


class IntegrationMode(enum.Enum):
    READOUT = "readout"
    JOINT_TOKENS = "joint_tokens"


@dataclass
class BackboneCapabilities:
    supported_modes: set[IntegrationMode]
    supports_kv_cache: bool = False


@dataclass
class ActionHeadRequirements:
    required_mode: IntegrationMode
    accepts_readout: bool = True


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class VisionEncoderProto(Protocol):
    @property
    def output_dim(self) -> int: ...
    @property
    def num_patches(self) -> int: ...
    def encode_images(self, images: dict[str, Tensor]) -> Tensor: ...


@runtime_checkable
class BackboneProto(Protocol):
    @property
    def capabilities(self) -> BackboneCapabilities: ...
    @property
    def hidden_dim(self) -> int: ...
    def embed_language(self, texts: list[str]) -> tuple[Tensor, Tensor]: ...
    def forward(self, inputs_embeds: Tensor, attention_mask: Tensor, token_type_ids: Tensor) -> BackboneOutput: ...


@runtime_checkable
class ActionHeadProto(Protocol):
    @property
    def requirements(self) -> ActionHeadRequirements: ...
    def compute_loss(self, backbone_output: BackboneOutput, batch: TrainingBatch) -> LossDict: ...
    def predict(self, backbone_output: BackboneOutput) -> ActionPrediction: ...


@runtime_checkable
class ActionDecoderProto(Protocol):
    @property
    def action_space_spec(self) -> ActionSpaceSpec: ...
    def decode(self, pred: ActionPrediction) -> ActionChunk: ...


@runtime_checkable
class ProprioEncoderProto(Protocol):
    @property
    def output_dim(self) -> int: ...
    def encode_proprio(self, proprio: Tensor) -> Tensor: ...


@runtime_checkable
class TokenMergerProto(Protocol):
    def merge(
        self,
        vision_tokens: Tensor,
        proprio_tokens: Tensor,
        language_tokens: Tensor,
        language_attn_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]: ...


# ---------------------------------------------------------------------------
# ABC base classes
# ---------------------------------------------------------------------------


class VisionEncoderBase(nn.Module, ABC):
    @property
    @abstractmethod
    def output_dim(self) -> int: ...

    @property
    @abstractmethod
    def num_patches(self) -> int: ...

    @abstractmethod
    def encode_images(self, images: dict[str, Tensor]) -> Tensor: ...


class BackboneBase(nn.Module, ABC):
    @property
    @abstractmethod
    def capabilities(self) -> BackboneCapabilities: ...

    @property
    @abstractmethod
    def hidden_dim(self) -> int: ...

    @abstractmethod
    def embed_language(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        """Tokenize and embed language texts.

        Args:
            texts: Non-empty list of language strings, one per batch element.

        Returns:
            Tuple of (embeddings ``[B, T, D]``, attention_mask ``[B, T]``).

        Raises:
            ValueError: If *texts* is empty.
        """
        ...

    @abstractmethod
    def forward(self, inputs_embeds: Tensor, attention_mask: Tensor, token_type_ids: Tensor) -> BackboneOutput: ...

    @property
    def tokenizer(self) -> Any:
        """The tokenizer, if exposed. Not part of the abstract contract."""
        raise NotImplementedError

    @property
    def base_model(self) -> Any:
        """The unwrapped model (never a PEFT wrapper).

        Used to access the original embedding layer / model config, and as the
        base argument when loading a PEFT adapter via ``PeftModel.from_pretrained``.
        """
        raise NotImplementedError

    @property
    def model(self) -> Any:
        """The active forward-pass model.

        Equals ``base_model`` when no adapter is applied; replaced with a
        ``PeftModel`` wrapper when LoRA is active. Used for adapter save/load
        and gradient-flow setup (``enable_input_require_grads``).
        """
        raise NotImplementedError

    @model.setter
    def model(self, value: Any) -> None:
        raise NotImplementedError


class ActionHeadBase(nn.Module, ABC):
    @property
    @abstractmethod
    def requirements(self) -> ActionHeadRequirements: ...

    @abstractmethod
    def compute_loss(self, backbone_output: BackboneOutput, batch: TrainingBatch) -> LossDict: ...

    @abstractmethod
    def predict(self, backbone_output: BackboneOutput) -> ActionPrediction: ...


class ActionDecoderBase(nn.Module, ABC):
    @property
    @abstractmethod
    def action_space_spec(self) -> ActionSpaceSpec: ...

    @abstractmethod
    def decode(self, pred: ActionPrediction) -> ActionChunk: ...


class ProprioEncoderBase(nn.Module, ABC):
    @property
    @abstractmethod
    def output_dim(self) -> int: ...

    @abstractmethod
    def encode_proprio(self, proprio: Tensor) -> Tensor: ...


class TokenMergerBase(nn.Module, ABC):
    @abstractmethod
    def merge(
        self,
        vision_tokens: Tensor,
        proprio_tokens: Tensor,
        language_tokens: Tensor,
        language_attn_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]: ...

class PolicyBase(nn.Module, ABC):
    """Minimal contract for all YAVLA policies.

    Concrete subclasses MUST define:
        name: str            — policy identifier (e.g. "vla", "ar_token", "flow_match")
        config_class: type   — the config dataclass this policy expects
    """

    name: str
    config_class: type

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Skip enforcement on intermediate abstract classes
        if ABC in cls.__bases__:
            return
        if not getattr(cls, "name", None):
            raise TypeError(f"Class {cls.__name__} must define 'name'")
        if not getattr(cls, "config_class", None):
            raise TypeError(f"Class {cls.__name__} must define 'config_class'")

    @abstractmethod
    def forward(self, batch: TrainingBatch) -> LossDict:
        """Compute loss for a training batch."""
        ...

    @abstractmethod
    def predict(self, obs: ObservationBatch) -> ActionChunk:
        """Predict an action chunk from observations (inference)."""
        ...

    def reset(self) -> None:
        """Clear caches (KV cache, action buffer, etc.). Default: no-op."""
        pass

    def get_optim_params(self) -> dict[str, Any]:
        """Parameter groups for the optimizer. Default: all params, single group."""
        return {"params": self.parameters()}

    def get_optimizer_preset(self) -> OptimizerConfig | None:
        """Return policy-specific optimizer defaults, or None to use config as-is."""
        return None


# ---------------------------------------------------------------------------
# Capability validation
# ---------------------------------------------------------------------------


class IncompatibleError(Exception):
    pass


def validate_integration(backbone: BackboneProto, head: ActionHeadProto) -> IntegrationMode:
    mode = head.requirements.required_mode
    if mode not in backbone.capabilities.supported_modes:
        raise IncompatibleError(
            f"Head requires {mode.value} but backbone supports "
            f"{{{', '.join(m.value for m in backbone.capabilities.supported_modes)}}}"
        )
    return mode
