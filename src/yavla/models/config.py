"""PolicyConfig dataclass tree composing all sub-configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from yavla.models.encoders.proprio import ProprioEncoderConfig
from yavla.models.encoders.vision import VisionEncoderConfig
from yavla.models.heads.mlp import MLPHeadConfig
from yavla.models.merger import TokenMergerConfig
from yavla.models.types import ActionSpaceSpec, FreezeConfig, ProprioSpec

EmbodimentMode = Literal["exact", "max_padded"]
CURRENT_POLICY_CONFIG_VERSION = "1.1"


@dataclass
class BackboneConfig:
    type: str = "paligemma"
    vlm_name: str = "google/paligemma-3b-pt-224"
    gradient_checkpointing: bool = True


@dataclass
class EmbodimentConfig:
    """Embodiment dimensions for exact-width and pretrained-VLA max-padded modes."""

    mode: EmbodimentMode = "exact"
    action_dim: int = 7
    proprio_dim: int = 7
    max_action_dim: int | None = None
    max_proprio_dim: int | None = None

    def normalized(self) -> EmbodimentConfig:
        action_dim = self.action_dim
        proprio_dim = self.proprio_dim
        max_action_dim = action_dim if self.max_action_dim is None else self.max_action_dim
        max_proprio_dim = proprio_dim if self.max_proprio_dim is None else self.max_proprio_dim

        if self.mode == "exact":
            max_action_dim = action_dim
            max_proprio_dim = proprio_dim

        normalized = EmbodimentConfig(
            mode=self.mode,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            max_action_dim=max_action_dim,
            max_proprio_dim=max_proprio_dim,
        )
        normalized.validate()
        return normalized

    def validate(self) -> None:
        if self.mode not in {"exact", "max_padded"}:
            raise ValueError(f"Unsupported embodiment mode {self.mode!r}; expected 'exact' or 'max_padded'")

        if self.action_dim is None or self.proprio_dim is None:
            raise ValueError("EmbodimentConfig requires action_dim and proprio_dim after normalization")
        if self.max_action_dim is None or self.max_proprio_dim is None:
            raise ValueError("EmbodimentConfig requires max_action_dim and max_proprio_dim after normalization")

        if self.action_dim <= 0 or self.proprio_dim <= 0:
            raise ValueError("Embodiment dimensions must be positive")
        if self.max_action_dim <= 0 or self.max_proprio_dim <= 0:
            raise ValueError("Embodiment maximum dimensions must be positive")
        if self.action_dim > self.max_action_dim:
            raise ValueError(
                "Embodiment action_dim must not exceed max_action_dim: "
                f"{self.action_dim} > {self.max_action_dim}"
            )
        if self.proprio_dim > self.max_proprio_dim:
            raise ValueError(
                "Embodiment proprio_dim must not exceed max_proprio_dim: "
                f"{self.proprio_dim} > {self.max_proprio_dim}"
            )
        if self.mode == "exact" and (
            self.action_dim != self.max_action_dim or self.proprio_dim != self.max_proprio_dim
        ):
            raise ValueError(
                "Embodiment exact mode requires active and maximum dimensions to match, got "
                f"action_dim={self.action_dim}, max_action_dim={self.max_action_dim}, "
                f"proprio_dim={self.proprio_dim}, max_proprio_dim={self.max_proprio_dim}"
            )


@dataclass
class PolicyConfig:
    vision_encoder: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    proprio_encoder: ProprioEncoderConfig = field(default_factory=ProprioEncoderConfig)
    merger: TokenMergerConfig = field(default_factory=TokenMergerConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    action_head: MLPHeadConfig = field(default_factory=MLPHeadConfig)
    embodiment: EmbodimentConfig = field(default_factory=EmbodimentConfig)
    freeze: FreezeConfig = field(default_factory=FreezeConfig)
    action_space: ActionSpaceSpec = field(default_factory=lambda: ActionSpaceSpec(names=[], units=[], limits=None))
    proprio: ProprioSpec = field(default_factory=lambda: ProprioSpec(names=[], units=[]))
    dt_hz: float = 10.0
    config_version: str = CURRENT_POLICY_CONFIG_VERSION

    def __post_init__(self) -> None:
        self.embodiment = self.embodiment.normalized()
        assert self.embodiment.action_dim is not None
        assert self.embodiment.proprio_dim is not None
        assert self.embodiment.max_action_dim is not None
        assert self.embodiment.max_proprio_dim is not None

        default_action_dim = MLPHeadConfig().action_dim
        default_proprio_dim = ProprioEncoderConfig().proprio_dim
        if self.action_head.action_dim not in {default_action_dim, self.embodiment.max_action_dim}:
            raise ValueError(
                "PolicyConfig.action_head.action_dim is a legacy width override; "
                "set policy.embodiment.max_action_dim instead"
            )
        if self.proprio_encoder.proprio_dim not in {default_proprio_dim, self.embodiment.max_proprio_dim}:
            raise ValueError(
                "PolicyConfig.proprio_encoder.proprio_dim is a legacy width override; "
                "set policy.embodiment.max_proprio_dim instead"
            )

        self.action_head.action_dim = self.embodiment.max_action_dim
        self.proprio_encoder.proprio_dim = self.embodiment.max_proprio_dim
        self._validate_specs()

    @property
    def action_dim(self) -> int:
        assert self.embodiment.action_dim is not None
        return self.embodiment.action_dim

    @property
    def proprio_dim(self) -> int:
        assert self.embodiment.proprio_dim is not None
        return self.embodiment.proprio_dim

    @property
    def max_action_dim(self) -> int:
        assert self.embodiment.max_action_dim is not None
        return self.embodiment.max_action_dim

    @property
    def max_proprio_dim(self) -> int:
        assert self.embodiment.max_proprio_dim is not None
        return self.embodiment.max_proprio_dim

    def _validate_specs(self) -> None:
        if self.action_space.limits is not None and self.action_space.limits.shape[0] != self.action_dim:
            raise ValueError(
                "ActionSpaceSpec.limits must match the active action dimension, "
                f"expected {self.action_dim}, got {self.action_space.limits.shape[0]}"
            )
        if self.action_space.names and len(self.action_space.names) != self.action_dim:
            raise ValueError(
                "ActionSpaceSpec.names must match the active action dimension, "
                f"expected {self.action_dim}, got {len(self.action_space.names)}"
            )
        if self.action_space.units and len(self.action_space.units) != self.action_dim:
            raise ValueError(
                "ActionSpaceSpec.units must match the active action dimension, "
                f"expected {self.action_dim}, got {len(self.action_space.units)}"
            )
        if self.proprio.names and len(self.proprio.names) != self.proprio_dim:
            raise ValueError(
                "ProprioSpec.names must match the active proprio dimension, "
                f"expected {self.proprio_dim}, got {len(self.proprio.names)}"
            )
        if self.proprio.units and len(self.proprio.units) != self.proprio_dim:
            raise ValueError(
                "ProprioSpec.units must match the active proprio dimension, "
                f"expected {self.proprio_dim}, got {len(self.proprio.units)}"
            )
        if self.proprio.limits is not None and self.proprio.limits.shape[0] != self.proprio_dim:
            raise ValueError(
                "ProprioSpec.limits must match the active proprio dimension, "
                f"expected {self.proprio_dim}, got {self.proprio.limits.shape[0]}"
            )
