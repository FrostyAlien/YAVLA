"""Composable data transforms shared across dataset backends."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import torch
from torchvision.transforms import v2  # type: ignore[import-untyped]

Sample = dict[str, Any]


class DataTransformFn(Protocol):
    """Protocol for per-sample data transforms."""

    def __call__(self, sample: Sample) -> Sample:
        """Transform a sample and return the transformed sample."""


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
            for transform in self.transforms:
                value = transform(value)
            output[camera_key] = value
        return output


_TRANSFORM_EXPR = re.compile(r"^(?P<name>\w+)(?:\((?P<args>.*)\))?$")


def build_torchvision_transforms(transform_specs: Iterable[str]) -> list[Callable[[Any], Any]]:
    """Build torchvision v2 transforms from simple string specs.

    Supported forms:
    - ``Resize(224)``
    - ``CenterCrop((224, 224))``
    - ``ToImage``
    """

    built: list[Callable[[Any], Any]] = []
    for spec in transform_specs:
        match = _TRANSFORM_EXPR.match(spec.strip())
        if match is None:
            raise ValueError(f"Invalid transform specification: {spec}")
        name = match.group("name")
        args_raw = match.group("args")

        if not hasattr(v2, name):
            raise ValueError(f"Unknown torchvision v2 transform: {name}")

        transform_cls = cast(type[Any], getattr(v2, name))
        if args_raw is None or args_raw.strip() == "":
            built.append(cast(Callable[[Any], Any], transform_cls()))
            continue

        parsed = ast.literal_eval(args_raw)
        if isinstance(parsed, tuple):
            built.append(cast(Callable[[Any], Any], transform_cls(*parsed)))
        else:
            built.append(cast(Callable[[Any], Any], transform_cls(parsed)))
    return built
