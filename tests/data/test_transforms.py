"""Unit tests for shared data transforms."""

from __future__ import annotations

import numpy as np
import torch

from yavla.data.transforms import (
    ImageTransform,
    NormalizeTransform,
    RepackTransform,
    UnnormalizeTransform,
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
