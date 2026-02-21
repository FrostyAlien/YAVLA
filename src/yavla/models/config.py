"""PolicyConfig dataclass tree composing all sub-configs."""

from __future__ import annotations

from dataclasses import dataclass, field

from yavla.models.backbone import BackboneConfig
from yavla.models.encoders.proprio import ProprioEncoderConfig
from yavla.models.encoders.vision import VisionEncoderConfig
from yavla.models.heads.mlp import MLPHeadConfig
from yavla.models.merger import TokenMergerConfig
from yavla.models.types import ActionNormalizationConfig, ActionSpaceSpec, FreezeConfig, ProprioSpec


@dataclass
class PolicyConfig:
    vision_encoder: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    proprio_encoder: ProprioEncoderConfig = field(default_factory=ProprioEncoderConfig)
    merger: TokenMergerConfig = field(default_factory=TokenMergerConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    action_head: MLPHeadConfig = field(default_factory=MLPHeadConfig)
    freeze: FreezeConfig = field(default_factory=FreezeConfig)
    action_normalization: ActionNormalizationConfig = field(default_factory=ActionNormalizationConfig)
    action_space: ActionSpaceSpec = field(default_factory=lambda: ActionSpaceSpec(names=[], units=[], limits=None))
    proprio: ProprioSpec = field(default_factory=lambda: ProprioSpec(names=[], units=[]))
    dt_hz: float = 10.0
    config_version: str = "1.0"
