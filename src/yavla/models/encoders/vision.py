"""Vision encoder config and registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from yavla.models.protocols import VisionEncoderBase
from yavla.models.registry import Registry

vision_registry: Registry[Any, VisionEncoderBase] = Registry("vision_encoder")


@dataclass
class VisionEncoderConfig:
    type: str = "paligemma_siglip"
