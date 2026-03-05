"""Unit tests for shared data transforms."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from yavla.data.transforms import (
    ImageTransform,
    NormalizeTransform,
    RepackTransform,
    UnnormalizeTransform,
    build_torchvision_transforms,
    compose,
)


def test_compose_accepts_plain_function() -> None:
    def add_flag(sample: dict[str, object]) -> dict[str, object]:
        output = dict(sample)
        output["flag"] = True
        return output

    transform = compose(add_flag)
    assert transform({"x": 1}) == {"x": 1, "flag": True}


def test_compose_identity_when_empty() -> None:
    transform = compose()
    sample = {"x": 1}
    assert transform(sample) == sample


def test_repack_transform_preserves_unmapped_keys() -> None:
    repack = RepackTransform({"observation.images.laptop": "image"})
    sample = {
        "observation.images.laptop": "pixels",
        "timestamp": 1.0,
    }
    transformed = repack(sample)
    assert transformed["image"] == "pixels"
    assert transformed["timestamp"] == 1.0
    assert "observation.images.laptop" not in transformed


def test_normalize_zero_std_maps_to_zero() -> None:
    stats = {"state": {"mean": [2.0, 4.0], "std": [0.0, 2.0]}}
    transform = NormalizeTransform(stats=stats, mode="z-score", keys=["state"])
    sample = {"state": torch.tensor([2.0, 8.0])}
    normalized = transform(sample)
    assert torch.allclose(normalized["state"], torch.tensor([0.0, 2.0]))


def test_unnormalize_minmax_roundtrip_and_zero_range() -> None:
    stats = {"action": {"min": np.array([5.0, 1.0]), "max": np.array([5.0, 3.0])}}
    normalize = NormalizeTransform(stats=stats, mode="min-max", keys=["action"])
    unnormalize = UnnormalizeTransform(stats=stats, mode="min-max", keys=["action"])

    sample = {"action": np.array([5.0, 2.0], dtype=np.float32)}
    normalized = normalize(sample)
    unnormalized = unnormalize(normalized)
    assert isinstance(normalized["action"], torch.Tensor)
    assert normalized["action"].dtype == torch.float32
    assert torch.allclose(normalized["action"], torch.tensor([0.0, 0.5]), atol=1e-6)
    assert torch.allclose(unnormalized["action"], torch.tensor([5.0, 2.0]), atol=1e-6)


def test_image_transform_applies_to_multiple_keys() -> None:
    transform = ImageTransform(
        transforms=[lambda tensor: tensor + 1],
        camera_keys=["image_left", "image_right"],
    )
    sample = {
        "image_left": torch.tensor([0, 1]),
        "image_right": torch.tensor([2, 3]),
        "state": torch.tensor([1.0]),
    }
    transformed = transform(sample)
    assert torch.equal(transformed["image_left"], torch.tensor([1, 2]))
    assert torch.equal(transformed["image_right"], torch.tensor([3, 4]))
    assert torch.equal(transformed["state"], sample["state"])


def test_image_transform_uint8_is_coerced_before_torchvision_normalize() -> None:
    transforms = build_torchvision_transforms(["Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))"])
    image_transform = ImageTransform(transforms=transforms, camera_keys=["image"])
    sample = {"image": torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8)}

    transformed = image_transform(sample)
    image = transformed["image"]
    assert isinstance(image, torch.Tensor)
    assert image.dtype == torch.float32


@pytest.mark.parametrize("target_hw", [(224, 224), (448, 448)])
def test_siglip_transform_recipe_shape_dtype_and_range(target_hw: tuple[int, int]) -> None:
    height, width = target_hw
    transforms = build_torchvision_transforms(
        [
            f"Resize([{height}, {width}], 3)",
            "Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))",
        ]
    )
    image_transform = ImageTransform(transforms=transforms, camera_keys=["image"])
    base_h, base_w = 32, 16
    yy = torch.linspace(0, 1, base_h).view(1, base_h, 1)
    xx = torch.linspace(0, 1, base_w).view(1, 1, base_w)
    image_u8 = ((yy * 0.7 + xx * 0.3).clamp(0, 1).repeat(3, 1, 1) * 255).round().to(torch.uint8)
    sample = {"image": image_u8}

    transformed = image_transform(sample)
    image = transformed["image"]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, height, width)
    assert image.dtype == torch.float32
    assert float(image.min()) >= -1.05
    assert float(image.max()) <= 1.05
