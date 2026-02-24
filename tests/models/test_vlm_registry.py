"""Unit tests for the VLM registry."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from yavla.models.config import BackboneConfig
from yavla.models.protocols import (
    BackboneBase,
    BackboneCapabilities,
    IntegrationMode,
    VisionEncoderBase,
)
from yavla.models.types import BackboneOutput, FreezeConfig
from yavla.models.vlm_registry import VLMRegistry


class _StubVisionEncoder(VisionEncoderBase):
    @property
    def output_dim(self) -> int:
        return 16

    @property
    def num_patches(self) -> int:
        return 4

    def encode_images(self, images: dict[str, Tensor]) -> Tensor:
        v = next(iter(images.values()))
        return torch.zeros(v.shape[0], 4, 16)


class _StubBackbone(BackboneBase):
    @property
    def capabilities(self) -> BackboneCapabilities:
        return BackboneCapabilities(supported_modes={IntegrationMode.READOUT})

    @property
    def hidden_dim(self) -> int:
        return 16

    def embed_language(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        b = len(texts)
        return torch.zeros(b, 1, 16), torch.ones(b, 1)

    def forward(self, inputs_embeds: Tensor, attention_mask: Tensor, token_type_ids: Tensor) -> BackboneOutput:
        return BackboneOutput(readout_states=torch.zeros(1, 1, 16), token_states=None, attention_mask=attention_mask)


def _build_stub(
    config: BackboneConfig, freeze: FreezeConfig, num_readout_tokens: int
) -> tuple[VisionEncoderBase, BackboneBase]:
    return _StubVisionEncoder(), _StubBackbone()


class TestVLMRegistry:
    def test_register_and_build(self) -> None:
        reg = VLMRegistry()
        reg.register("stub")(_build_stub)
        vision, backbone = reg.build(BackboneConfig(type="stub"), FreezeConfig(), 64)
        assert isinstance(vision, VisionEncoderBase)
        assert isinstance(backbone, BackboneBase)

    def test_unknown_type_raises(self) -> None:
        reg = VLMRegistry()
        with pytest.raises(KeyError, match="nonexistent"):
            reg.build(BackboneConfig(type="nonexistent"), FreezeConfig(), 64)

    def test_list(self) -> None:
        reg = VLMRegistry()
        reg.register("a")(_build_stub)
        reg.register("b")(_build_stub)
        assert sorted(reg.list()) == ["a", "b"]

    def test_duplicate_raises(self) -> None:
        reg = VLMRegistry()
        reg.register("dup")(_build_stub)
        with pytest.raises(ValueError, match="Duplicate"):
            reg.register("dup")(_build_stub)
