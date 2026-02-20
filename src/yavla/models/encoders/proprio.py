"""Proprioceptive state encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from yavla.models.protocols import ProprioEncoderBase
from yavla.models.registry import Registry

proprio_registry: Registry[Any, ProprioEncoderBase] = Registry("proprio_encoder")


@dataclass
class ProprioEncoderConfig:
    type: str = "linear"
    proprio_dim: int = 7
    backbone_dim: int = 2048


class ProprioEncoder(ProprioEncoderBase):
    def __init__(self, config: ProprioEncoderConfig) -> None:
        super().__init__()
        self._config = config
        self.proj = nn.Linear(config.proprio_dim, config.backbone_dim)

    @property
    def output_dim(self) -> int:
        return self._config.backbone_dim

    def encode_proprio(self, proprio: Tensor) -> Tensor:
        return self.proj(proprio).unsqueeze(1)  # [B, D] -> [B, 1, D]


proprio_registry.register("linear", ProprioEncoderConfig)(ProprioEncoder)
