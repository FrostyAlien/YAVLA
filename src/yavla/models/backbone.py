"""Backbone config and legacy registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from yavla.models.protocols import BackboneBase
from yavla.models.registry import Registry

backbone_registry: Registry[Any, BackboneBase] = Registry("backbone")


@dataclass
class BackboneConfig:
    type: str = "paligemma"
    vlm_name: str = "google/paligemma-3b-pt-224"
    gradient_checkpointing: bool = True
