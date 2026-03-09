"""Train entry configuration and YAML loading helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, TypeVar, cast

import torch
import yaml

from yavla.data.factory import DataConfig
from yavla.models.config import BackboneConfig, EmbodimentConfig, PolicyConfig
from yavla.models.encoders.proprio import ProprioEncoderConfig
from yavla.models.encoders.vision import (
    FromBackboneVisionEncoderConfig,
    MultiTowerVisionEncoderConfig,
    SimplePatchVisionEncoderConfig,
)
from yavla.models.heads.mlp import MLPHeadConfig
from yavla.models.merger import TokenMergerConfig
from yavla.models.types import ActionSpaceSpec, FreezeConfig, ProprioSpec
from yavla.training.config import OptimizerConfig, SchedulerConfig, TrainingConfig
from yavla.visualization.config import VizConfig

T = TypeVar("T")
FieldNormalizer = Callable[[Any, str], Any]


@dataclass
class TrainConfig:
    """Top-level config composing training and policy settings."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)


def load_train_config_file(path: Path) -> TrainConfig:
    """Load a nested TrainConfig YAML file into typed defaults."""

    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    with open(path) as f:
        loaded = yaml.safe_load(f)

    raw = loaded if isinstance(loaded, dict) else {}
    if not isinstance(raw, dict) or not {"training", "policy"} & set(raw):
        raise ValueError(
            "unsupported train config format; nest fields under top-level 'training:' and 'policy:' blocks"
        )

    extra_top_level = sorted(set(raw) - {"training", "policy"})
    if extra_top_level:
        raise ValueError(
            "unsupported top-level train config fields: "
            f"{extra_top_level}; expected only 'training' and 'policy'"
        )

    return TrainConfig(
        training=_load_training_config(raw.get("training", {})),
        policy=_load_policy_config(raw.get("policy", {})),
    )


def _expect_mapping(raw: object, path: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a mapping")
    return cast(dict[str, Any], raw)


def _replace_dataclass[T](
    default: T,
    raw: object,
    path: str,
    *,
    field_normalizers: dict[str, FieldNormalizer] | None = None,
) -> T:
    mapping = _expect_mapping(raw, path)
    valid_fields = {f.name for f in fields(type(default))}
    unknown = sorted(set(mapping) - valid_fields)
    if unknown:
        raise ValueError(f"unsupported fields under {path}: {unknown}")

    normalized: dict[str, Any] = {}
    for key, value in mapping.items():
        if field_normalizers is not None and key in field_normalizers:
            normalized[key] = field_normalizers[key](value, f"{path}.{key}")
        else:
            normalized[key] = value
    return replace(default, **normalized)


def _load_float_tuple(raw: object, path: str) -> tuple[float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{path} must be a two-element list or tuple")
    return (float(raw[0]), float(raw[1]))


def _load_limits(raw: object, path: str) -> torch.Tensor | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a nested list or null")
    return torch.tensor(raw, dtype=torch.float32)


def _load_training_config(raw: object) -> TrainingConfig:
    mapping = _expect_mapping(raw, "training")
    default = TrainingConfig()
    kwargs: dict[str, Any] = {}

    nested_loaders: dict[str, Callable[[object], Any]] = {
        "dataset": _load_data_config,
        "viz": _load_viz_config,
        "optimizer": _load_optimizer_config,
        "scheduler": _load_scheduler_config,
    }
    for key, loader in nested_loaders.items():
        if key in mapping:
            kwargs[key] = loader(mapping[key])

    scalar_keys = set(mapping) - set(nested_loaders)
    for key in scalar_keys:
        kwargs[key] = mapping[key]

    return replace(default, **kwargs)


def _load_data_config(raw: object) -> DataConfig:
    mapping = _expect_mapping(raw, "training.dataset")
    if "repo_id" not in mapping:
        raise ValueError("training.dataset.repo_id is required when a dataset block is provided")
    return _replace_dataclass(DataConfig(repo_id=str(mapping["repo_id"])), mapping, "training.dataset")


def _load_viz_config(raw: object) -> VizConfig:
    return _replace_dataclass(VizConfig(), raw, "training.viz")


def _load_optimizer_config(raw: object) -> OptimizerConfig:
    return _replace_dataclass(
        OptimizerConfig(),
        raw,
        "training.optimizer",
        field_normalizers={"betas": _load_float_tuple},
    )


def _load_scheduler_config(raw: object) -> SchedulerConfig:
    return _replace_dataclass(SchedulerConfig(), raw, "training.scheduler")


def _load_policy_config(raw: object) -> PolicyConfig:
    mapping = _expect_mapping(raw, "policy")
    default = PolicyConfig()
    kwargs: dict[str, Any] = {}

    nested_loaders: dict[str, Callable[[object], Any]] = {
        "vision_encoder": lambda value: _load_vision_encoder_config(value, "policy.vision_encoder"),
        "proprio_encoder": lambda value: _replace_dataclass(
            ProprioEncoderConfig(), value, "policy.proprio_encoder"
        ),
        "merger": lambda value: _replace_dataclass(TokenMergerConfig(), value, "policy.merger"),
        "backbone": lambda value: _replace_dataclass(BackboneConfig(), value, "policy.backbone"),
        "action_head": lambda value: _replace_dataclass(MLPHeadConfig(), value, "policy.action_head"),
        "embodiment": lambda value: _replace_dataclass(EmbodimentConfig(), value, "policy.embodiment"),
        "freeze": lambda value: _replace_dataclass(FreezeConfig(), value, "policy.freeze"),
        "action_space": _load_action_space_spec,
        "proprio": _load_proprio_spec,
    }
    for key, loader in nested_loaders.items():
        if key in mapping:
            kwargs[key] = loader(mapping[key])

    scalar_keys = set(mapping) - set(nested_loaders)
    for key in scalar_keys:
        kwargs[key] = mapping[key]

    return replace(default, **kwargs)


def _load_action_space_spec(raw: object) -> ActionSpaceSpec:
    return _replace_dataclass(
        ActionSpaceSpec(names=[], units=[], limits=None),
        raw,
        "policy.action_space",
        field_normalizers={"limits": _load_limits},
    )


def _load_proprio_spec(raw: object) -> ProprioSpec:
    return _replace_dataclass(
        ProprioSpec(names=[], units=[]),
        raw,
        "policy.proprio",
        field_normalizers={"limits": _load_limits},
    )


def _load_vision_encoder_config(raw: object, path: str):
    mapping = _expect_mapping(raw, path)
    type_name = mapping.get("type", "from_backbone")

    if not isinstance(type_name, str):
        raise ValueError(f"{path}.type must be a string when provided")

    if type_name in {"from_backbone", "paligemma_siglip"}:
        return _replace_dataclass(FromBackboneVisionEncoderConfig(), mapping, path)
    if type_name == "simple_patch":
        return _replace_dataclass(SimplePatchVisionEncoderConfig(), mapping, path)
    if type_name == "multi_tower":
        multi_tower = _replace_dataclass(MultiTowerVisionEncoderConfig(), mapping, path)
        towers = mapping.get("towers", [])
        if not isinstance(towers, list):
            raise ValueError(f"{path}.towers must be a list")
        return replace(
            multi_tower,
            towers=[_load_vision_encoder_config(tower, f"{path}.towers[{idx}]") for idx, tower in enumerate(towers)],
        )

    raise ValueError(
        f"unsupported {path}.type={type_name!r}; expected one of 'from_backbone', 'simple_patch', or 'multi_tower'"
    )
