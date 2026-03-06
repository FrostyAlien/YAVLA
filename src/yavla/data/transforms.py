"""Composable data transforms shared across dataset backends."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import torch
import torch.nn.functional as functional
from torchvision.transforms import v2  # type: ignore[import-untyped]

Sample = dict[str, Any]


class DataTransformFn(Protocol):
    """Protocol for per-sample data transforms."""

    def __call__(self, sample: Sample) -> Sample:
        """Transform a sample and return the transformed sample."""
        ...


@dataclass(slots=True)
class _ComposedTransform:
    transforms: tuple[DataTransformFn, ...]

    def __call__(self, sample: Sample) -> Sample:
        output = sample
        for transform in self.transforms:
            output = transform(output)
        return output


def compose(*transforms: DataTransformFn) -> DataTransformFn:
    """Compose transforms in order.

    Args:
        *transforms: Transform functions.

    Returns:
        Single composed transform. If no transforms are provided, returns identity.
    """

    if not transforms:

        def _identity(sample: Sample) -> Sample:
            return sample

        return _identity
    return _ComposedTransform(tuple(transforms))


@dataclass(slots=True)
class RepackTransform:
    """Remap sample keys from source names to target names."""

    key_mapping: Mapping[str, str]

    def __call__(self, sample: Sample) -> Sample:
        output = dict(sample)
        for source_key, target_key in self.key_mapping.items():
            if source_key in output:
                output[target_key] = output.pop(source_key)
        return output


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, (list, tuple)):
        return torch.as_tensor(value)
    if isinstance(value, np.generic):
        return torch.as_tensor(value.item())
    if isinstance(value, (int, float, bool)):
        return torch.as_tensor(value)
    raise TypeError(f"Unsupported value type for normalization: {type(value)!r}")


def _extract_stat(stats_entry: Mapping[str, Any], primary: str, fallback: str | None = None) -> torch.Tensor | None:
    if primary in stats_entry:
        return _to_tensor(stats_entry[primary]).to(dtype=torch.float32)
    if fallback and fallback in stats_entry:
        return _to_tensor(stats_entry[fallback]).to(dtype=torch.float32)
    return None


@dataclass(slots=True)
class NormalizeTransform:
    """Apply feature normalization from dataset statistics."""

    stats: Mapping[str, Mapping[str, Any]]
    mode: str = "z-score"
    keys: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"z-score", "min-max"}:
            raise ValueError(f"Unsupported normalization mode: {self.mode}")

    def __call__(self, sample: Sample) -> Sample:
        output = dict(sample)
        target_keys = self.keys if self.keys is not None else [k for k in self.stats if k in sample]
        for key in target_keys:
            if key not in output or key not in self.stats:
                continue

            stats_entry = self.stats[key]
            value_tensor = _to_tensor(output[key]).to(dtype=torch.float32)

            if self.mode == "z-score":
                mean = _extract_stat(stats_entry, "mean")
                std = _extract_stat(stats_entry, "std")
                if mean is None or std is None:
                    continue
                normalized = torch.where(std == 0, torch.zeros_like(value_tensor), (value_tensor - mean) / std)
            else:
                minimum = _extract_stat(stats_entry, "min")
                maximum = _extract_stat(stats_entry, "max")
                if minimum is None or maximum is None:
                    continue
                denom = maximum - minimum
                normalized = torch.where(denom == 0, torch.zeros_like(value_tensor), (value_tensor - minimum) / denom)

            output[key] = normalized
        return output


@dataclass(slots=True)
class UnnormalizeTransform:
    """Invert normalization and return values in original scale."""

    stats: Mapping[str, Mapping[str, Any]]
    mode: str = "z-score"
    keys: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"z-score", "min-max"}:
            raise ValueError(f"Unsupported unnormalization mode: {self.mode}")

    def __call__(self, sample: Sample) -> Sample:
        output = dict(sample)
        target_keys = self.keys if self.keys is not None else [k for k in self.stats if k in sample]
        for key in target_keys:
            if key not in output or key not in self.stats:
                continue

            stats_entry = self.stats[key]
            value_tensor = _to_tensor(output[key]).to(dtype=torch.float32)

            if self.mode == "z-score":
                mean = _extract_stat(stats_entry, "mean")
                std = _extract_stat(stats_entry, "std")
                if mean is None or std is None:
                    continue
                unnormalized = torch.where(std == 0, mean, (value_tensor * std) + mean)
            else:
                minimum = _extract_stat(stats_entry, "min")
                maximum = _extract_stat(stats_entry, "max")
                if minimum is None or maximum is None:
                    continue
                denom = maximum - minimum
                unnormalized = torch.where(denom == 0, minimum, (value_tensor * denom) + minimum)

            output[key] = unnormalized
        return output


@dataclass(slots=True)
class ImageTransform:
    """Apply torchvision-style transforms to configured camera keys."""

    transforms: Sequence[Callable[[Any], Any]]
    camera_keys: Sequence[str]

    def __call__(self, sample: Sample) -> Sample:
        output = dict(sample)
        for camera_key in self.camera_keys:
            if camera_key not in output:
                continue
            value = output[camera_key]
            if isinstance(value, torch.Tensor) and value.dtype == torch.uint8:
                value = value.to(dtype=torch.float32).div(255.0)
            for transform in self.transforms:
                value = transform(value)
            output[camera_key] = value
        return output


def _coerce_hw(size: Sequence[int] | int) -> tuple[int, int]:
    if isinstance(size, int):
        return size, size
    if len(size) != 2:
        raise ValueError(f"Expected size=(H, W), got: {size!r}")
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Image size must be positive, got (H, W)=({height}, {width})")
    return height, width


def _interpolation_mode(interpolation: int) -> str:
    if interpolation == 0:
        return "nearest"
    if interpolation == 2:
        return "bilinear"
    if interpolation == 3:
        return "bicubic"
    raise ValueError(f"Unsupported interpolation={interpolation}; expected 0(nearest), 2(bilinear), or 3(bicubic)")


@dataclass(slots=True)
class LetterboxPad:
    """Aspect-ratio-preserving resize-to-fit + symmetric pad to a fixed size.

    Pads with a per-channel fill value of 0.5 in [0, 1] space so that, under
    SigLIP normalization (mean=std=0.5), padded regions become ~0.0.
    """

    size: tuple[int, int]
    interpolation: int = 3
    fill: float = 0.5

    def __init__(self, size: Sequence[int] | int, interpolation: int = 3) -> None:
        self.size = _coerce_hw(size)
        self.interpolation = int(interpolation)
        self.fill = 0.5

    def __call__(self, image: Any) -> Any:
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"LetterboxPad expects torch.Tensor, got {type(image)!r}")

        if image.ndim == 3:
            has_batch_dim = False
            image_bchw = image[None]
        elif image.ndim == 4:
            has_batch_dim = True
            image_bchw = image
        else:
            raise ValueError(f"Expected image with shape [C, H, W] or [B, C, H, W], got {tuple(image.shape)}")

        image_bchw = image_bchw.to(dtype=torch.float32)
        target_h, target_w = self.size
        in_h, in_w = int(image_bchw.shape[-2]), int(image_bchw.shape[-1])

        scale = min(target_h / in_h, target_w / in_w)
        new_h = max(1, min(target_h, int(round(in_h * scale))))
        new_w = max(1, min(target_w, int(round(in_w * scale))))

        mode = _interpolation_mode(self.interpolation)
        align_corners = False if mode in {"bilinear", "bicubic"} else None
        antialias = True if mode in {"bilinear", "bicubic"} else False
        resized = functional.interpolate(
            image_bchw,
            size=(new_h, new_w),
            mode=mode,
            align_corners=align_corners,
            antialias=antialias,
        )

        pad_h = target_h - new_h
        pad_w = target_w - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padded = functional.pad(
            resized,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=float(self.fill),
        )
        return padded if has_batch_dim else padded[0]


_TRANSFORM_EXPR = re.compile(r"^(?P<name>\w+)(?:\((?P<args>.*)\))?$")


def build_torchvision_transforms(transform_specs: Iterable[str]) -> list[Callable[[Any], Any]]:
    """Build torchvision v2 transforms from simple string specs.

    Supported forms:
    - ``Resize(224)``
    - ``CenterCrop((224, 224))``
    - ``ToImage``
    - ``LetterboxPad([224, 224], 3)``
    """

    custom_transforms: dict[str, type[Any]] = {
        "LetterboxPad": LetterboxPad,
    }

    built: list[Callable[[Any], Any]] = []
    for spec in transform_specs:
        match = _TRANSFORM_EXPR.match(spec.strip())
        if match is None:
            raise ValueError(f"Invalid transform specification: {spec}")
        name = match.group("name")
        args_raw = match.group("args")

        if name in custom_transforms:
            transform_cls = custom_transforms[name]
        elif hasattr(v2, name):
            transform_cls = cast(type[Any], getattr(v2, name))
        else:
            raise ValueError(f"Unknown transform: {name}")

        if args_raw is None or args_raw.strip() == "":
            built.append(cast(Callable[[Any], Any], transform_cls()))
            continue

        parsed = ast.literal_eval(args_raw)
        if isinstance(parsed, tuple):
            built.append(cast(Callable[[Any], Any], transform_cls(*parsed)))
        else:
            built.append(cast(Callable[[Any], Any], transform_cls(parsed)))
    return built
