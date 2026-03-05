"""Unit tests for vision and proprio encoders."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor

from yavla.models.backbones.paligemma import PaliGemmaVisionEncoder
from yavla.models.encoders.proprio import ProprioEncoder, ProprioEncoderConfig, proprio_registry
from yavla.models.protocols import ProprioEncoderProto, VisionEncoderProto


def _make_mock_paligemma(hidden_size: int = 2048, image_size: int = 224, patch_size: int = 14) -> MagicMock:
    mock = MagicMock()
    mock.config.text_config.hidden_size = hidden_size
    mock.config.vision_config.image_size = image_size
    mock.config.vision_config.patch_size = patch_size

    def fake_get_image_features(pixel_values: Tensor) -> Tensor:
        B = pixel_values.shape[0]
        num_patches = (image_size // patch_size) ** 2
        signature = pixel_values.mean(dim=(1, 2, 3))  # [B]
        return signature[:, None, None].expand(B, num_patches, hidden_size).contiguous()

    mock.get_image_features = MagicMock(side_effect=fake_get_image_features)
    return mock


class TestPaliGemmaVisionEncoder:
    def test_encode_single_camera(self) -> None:
        base = _make_mock_paligemma()
        enc = PaliGemmaVisionEncoder(base)
        out = enc.encode_images({"cam0": torch.randn(2, 3, 224, 224)})
        assert out.shape == (2, 256, 2048)

    def test_encode_multi_camera_scales_tokens(self) -> None:
        base = _make_mock_paligemma()
        enc = PaliGemmaVisionEncoder(base)
        out = enc.encode_images(
            {
                "cam0": torch.randn(2, 3, 224, 224),
                "cam1": torch.randn(2, 3, 224, 224),
            }
        )
        assert out.shape == (2, 2 * 256, 2048)
        base.get_image_features.assert_called_once()
        pixel_values = base.get_image_features.call_args[0][0]
        assert pixel_values.shape == (4, 3, 224, 224)

    def test_multi_camera_ordering_is_deterministic(self) -> None:
        base = _make_mock_paligemma(hidden_size=8)
        enc = PaliGemmaVisionEncoder(base)

        cam_a = torch.full((1, 3, 224, 224), 2.0)
        cam_b = torch.full((1, 3, 224, 224), 1.0)

        out1 = enc.encode_images({"b": cam_b, "a": cam_a})
        out2 = enc.encode_images({"a": cam_a, "b": cam_b})

        assert torch.allclose(out1, out2)
        n_patch = enc.num_patches
        assert out1[0, 0, 0].item() == pytest.approx(2.0)
        assert out1[0, n_patch, 0].item() == pytest.approx(1.0)

    def test_reject_mismatched_camera_shapes(self) -> None:
        base = _make_mock_paligemma()
        enc = PaliGemmaVisionEncoder(base)
        with pytest.raises(ValueError, match="Mismatched camera tensor shapes"):
            enc.encode_images(
                {
                    "cam0": torch.randn(1, 3, 224, 224),
                    "cam1": torch.randn(2, 3, 224, 224),
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
