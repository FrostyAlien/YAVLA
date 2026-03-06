"""Tests for SigLIP image preprocessing helpers."""

from __future__ import annotations

import logging

import pytest

from yavla.training.config import TrainingConfig
from yavla.training.siglip_preprocess import (
    autowire_siglip_image_transforms,
    build_siglip_image_transform_specs,
)


def test_build_siglip_image_transform_specs_uses_resize_list_form() -> None:
    specs = build_siglip_image_transform_specs(224, 224)
    assert specs[0] == "Resize([224, 224], 3)"
    assert "Normalize" in specs[1]


@pytest.mark.parametrize(
    ("strategy", "expected_resize_spec"),
    [
        ("warp", "Resize([224, 224], 3)"),
        ("letterbox", "LetterboxPad([224, 224], 3)"),
    ],
)
def test_build_siglip_image_transform_specs_supports_resize_strategies(
    strategy: str, expected_resize_spec: str
) -> None:
    specs = build_siglip_image_transform_specs(224, 224, resize_strategy=strategy)
    assert specs[0] == expected_resize_spec
    assert "Normalize" in specs[1]


def test_autowire_siglip_transforms_defaults_to_ckpt_size() -> None:
    cfg = TrainingConfig()
    assert cfg.dataset.image_transforms is None

    autowire_siglip_image_transforms(cfg, ckpt_image_size=224)
    assert cfg.dataset.image_transforms == build_siglip_image_transform_specs(224, 224)


def test_autowire_siglip_transforms_honors_letterbox_strategy() -> None:
    cfg = TrainingConfig(vlm_image_resize_strategy="letterbox")

    autowire_siglip_image_transforms(cfg, ckpt_image_size=224)
    assert cfg.dataset.image_transforms == build_siglip_image_transform_specs(224, 224, resize_strategy="letterbox")


def test_autowire_siglip_transforms_respects_explicit_disable() -> None:
    cfg = TrainingConfig()
    cfg.dataset.image_transforms = []

    autowire_siglip_image_transforms(cfg, ckpt_image_size=224)
    assert cfg.dataset.image_transforms == []


def test_autowire_siglip_transforms_invalid_strategy_errors() -> None:
    cfg = TrainingConfig()
    cfg.vlm_image_resize_strategy = "not-a-real-strategy"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Unknown vlm_image_resize_strategy"):
        autowire_siglip_image_transforms(cfg, ckpt_image_size=224)


def test_autowire_siglip_transforms_override_warns_and_applies(caplog: pytest.LogCaptureFixture) -> None:
    cfg = TrainingConfig()
    cfg.vlm_image_height_override = 448
    cfg.vlm_image_width_override = 448

    caplog.set_level(logging.WARNING)
    autowire_siglip_image_transforms(cfg, ckpt_image_size=224)

    assert cfg.dataset.image_transforms == build_siglip_image_transform_specs(448, 448)
    assert any("Overriding VLM checkpoint image size" in rec.message for rec in caplog.records)


def test_autowire_siglip_transforms_warns_when_strategy_is_ignored(caplog: pytest.LogCaptureFixture) -> None:
    cfg = TrainingConfig(vlm_image_resize_strategy="letterbox")
    cfg.dataset.image_transforms = ["Resize([128, 128], 3)"]

    caplog.set_level(logging.WARNING)
    autowire_siglip_image_transforms(cfg, ckpt_image_size=224)

    assert cfg.dataset.image_transforms == ["Resize([128, 128], 3)"]
    assert any("strategy will not be auto-wired" in rec.message for rec in caplog.records)


def test_autowire_siglip_transforms_requires_both_overrides() -> None:
    cfg = TrainingConfig()
    cfg.vlm_image_height_override = 448

    with pytest.raises(ValueError, match="both-or-none"):
        autowire_siglip_image_transforms(cfg, ckpt_image_size=224)
