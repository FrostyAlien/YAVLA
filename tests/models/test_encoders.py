"""Unit tests for vision and proprio encoders."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor

from yavla.models.encoders.proprio import ProprioEncoder, ProprioEncoderConfig, proprio_registry
from yavla.models.encoders.vision import PaliGemmaVisionEncoder, VisionEncoderConfig, vision_registry
from yavla.models.protocols import ProprioEncoderProto, VisionEncoderProto


def _make_mock_paligemma(hidden_size: int = 2048, image_size: int = 224, patch_size: int = 14) -> MagicMock:
    mock = MagicMock()
    mock.config.text_config.hidden_size = hidden_size
    mock.config.vision_config.image_size = image_size
    mock.config.vision_config.patch_size = patch_size

    def fake_get_image_features(pixel_values: Tensor) -> Tensor:
        B = pixel_values.shape[0]
        num_patches = (image_size // patch_size) ** 2
        return torch.randn(B, num_patches, hidden_size)

    mock.get_image_features = fake_get_image_features
    return mock


class TestPaliGemmaVisionEncoder:
    def test_encode_single_camera(self) -> None:
        base = _make_mock_paligemma()
        enc = PaliGemmaVisionEncoder(base)
        out = enc.encode_images({"cam0": torch.randn(2, 3, 224, 224)})
        assert out.shape == (2, 256, 2048)

    def test_reject_multi_camera(self) -> None:
        base = _make_mock_paligemma()
        enc = PaliGemmaVisionEncoder(base)
        with pytest.raises(ValueError, match="single-camera"):
            enc.encode_images(
                {
                    "cam0": torch.randn(1, 3, 224, 224),
                    "cam1": torch.randn(1, 3, 224, 224),
                }
            )

    def test_output_dim(self) -> None:
        base = _make_mock_paligemma(hidden_size=1024)
        enc = PaliGemmaVisionEncoder(base)
        assert enc.output_dim == 1024

    def test_num_patches(self) -> None:
        base = _make_mock_paligemma(image_size=224, patch_size=14)
        enc = PaliGemmaVisionEncoder(base)
        assert enc.num_patches == 256

    def test_protocol_conformance(self) -> None:
        base = _make_mock_paligemma()
        enc = PaliGemmaVisionEncoder(base)
        assert isinstance(enc, VisionEncoderProto)

    def test_reject_empty_images(self) -> None:
        base = _make_mock_paligemma()
        enc = PaliGemmaVisionEncoder(base)
        with pytest.raises(ValueError, match="No camera"):
            enc.encode_images({})

    def test_registry(self) -> None:
        assert "paligemma_siglip" in vision_registry.list()


class TestProprioEncoder:
    def test_encode_shape(self) -> None:
        cfg = ProprioEncoderConfig(proprio_dim=7, backbone_dim=2048)
        enc = ProprioEncoder(cfg)
        out = enc.encode_proprio(torch.randn(2, 7))
        assert out.shape == (2, 1, 2048)

    def test_output_dim(self) -> None:
        cfg = ProprioEncoderConfig(backbone_dim=1024)
        enc = ProprioEncoder(cfg)
        assert enc.output_dim == 1024

    def test_protocol_conformance(self) -> None:
        cfg = ProprioEncoderConfig()
        enc = ProprioEncoder(cfg)
        assert isinstance(enc, ProprioEncoderProto)

    def test_registry(self) -> None:
        assert "linear" in proprio_registry.list()

    def test_default_config(self) -> None:
        cfg = ProprioEncoderConfig()
        assert cfg.type == "linear"
        assert cfg.proprio_dim == 7
        assert cfg.backbone_dim == 2048
