"""Train a VLA policy (CLI entry point).

Composes ``TrainingConfig`` and ``PolicyConfig`` into a single
``TrainConfig`` dataclass parsed by **tyro**.  An optional ``--config``
flag loads YAML defaults (e.g. ``configs/train.yaml``) before tyro
applies CLI overrides, giving the standard YAML-first / CLI-override
workflow.

Compatible with both single-process and distributed launches::

    # Single GPU
    python scripts/train.py --training.num-steps 50000

    # Distributed via Accelerate
    accelerate launch scripts/train.py --training.num-steps 50000

    # YAML defaults + CLI overrides
    python scripts/train.py --config configs/train.yaml --training.wandb True
"""

from __future__ import annotations

import logging
import sys
import types
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, cast, get_args, get_origin, get_type_hints

import tyro

from yavla.models.config import PolicyConfig
from yavla.models.encoders.vision import VisionEncoderConfig, get_vision_config_class
from yavla.models.policy import build_policy
from yavla.models.types import TrainingBatch
from yavla.training import Trainer, TrainingConfig, create_training_dataloader
from yavla.training.siglip_preprocess import autowire_siglip_image_transforms

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Top-level config composing training and policy settings."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)

def _resolve_config_dataclass(expected_type: Any, raw: object) -> Any:
    if expected_type is VisionEncoderConfig and isinstance(raw, dict):
        type_name = raw.get("type")
        return get_vision_config_class(type_name if isinstance(type_name, str) else None)
    return expected_type


def _coerce_config_value(expected_type: Any, raw: object) -> object:
    if expected_type is Any:
        return raw

    expected_type = _resolve_config_dataclass(expected_type, raw)
    if isinstance(expected_type, type) and is_dataclass(expected_type) and isinstance(raw, dict):
        return _build_dataclass(expected_type, raw)

    origin = get_origin(expected_type)
    if origin is None:
        return raw

    if origin is list:
        item_type = get_args(expected_type)[0] if get_args(expected_type) else Any
        if not isinstance(raw, list):
            return raw
        return [_coerce_config_value(item_type, item) for item in raw]

    if origin is dict:
        _, value_type = get_args(expected_type) if get_args(expected_type) else (Any, Any)
        if not isinstance(raw, dict):
            return raw
        return {str(key): _coerce_config_value(value_type, value) for key, value in raw.items()}

    if origin is tuple:
        item_types = get_args(expected_type)
        if not isinstance(raw, (list, tuple)):
            return raw
        if len(item_types) == 2 and item_types[1] is Ellipsis:
            return tuple(_coerce_config_value(item_types[0], item) for item in raw)
        if len(item_types) == len(raw):
            return tuple(_coerce_config_value(item_type, item) for item_type, item in zip(item_types, raw, strict=True))
        return tuple(raw)

    if origin in (types.UnionType, Union):
        union_args = get_args(expected_type)
        if raw is None and type(None) in union_args:
            return None
        for option in union_args:
            if option is type(None):
                continue
            resolved_option = _resolve_config_dataclass(option, raw)
            if isinstance(resolved_option, type) and is_dataclass(resolved_option) and isinstance(raw, dict):
                return _coerce_config_value(option, raw)
        return raw

    return raw


def _build_dataclass(cls: type[Any], raw: dict[str, object]) -> Any:
    hints = get_type_hints(cls)
    kwargs: dict[str, object] = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        kwargs[f.name] = _coerce_config_value(hints.get(f.name, f.type), raw[f.name])
    return cls(**kwargs)


def _pop_config_flag() -> TrainConfig | None:
    """Extract ``--config <path>`` before tyro sees it, return YAML defaults."""
    if "--config" not in sys.argv:
        return None
    idx = sys.argv.index("--config")
    if idx + 1 >= len(sys.argv):
        sys.exit("error: --config requires a path argument")
    path = Path(sys.argv[idx + 1])
    sys.argv = sys.argv[:idx] + sys.argv[idx + 2 :]

    if not path.exists():
        sys.exit(f"error: config file not found: {path}")

    import yaml  # type: ignore[import-untyped]

    with open(path) as f:
        loaded = yaml.safe_load(f)
    raw: dict[str, object] = loaded if isinstance(loaded, dict) else {}

    if "training" in raw or "policy" in raw:
        return cast(TrainConfig, _build_dataclass(TrainConfig, raw))

    sys.exit(
        "error: legacy flat train config format is not supported; nest fields under top-level "
        "'training:' and 'policy:'"
    )


def _peek_training_batch(dataloader: Any) -> TrainingBatch:
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        sys.exit("error: training dataloader is empty; cannot validate action/proprio dimensions")

    if not isinstance(batch, TrainingBatch):
        sys.exit(f"error: training dataloader yielded {type(batch).__name__}, expected TrainingBatch")
    return batch


def _validate_training_dimensions(cfg: TrainConfig, batch: TrainingBatch) -> None:
    configured_chunk_len = cfg.policy.action_head.chunk_len
    configured_action_dim = cfg.policy.action_dim
    configured_proprio_dim = cfg.policy.proprio_dim
    dataset_chunk_len = cfg.training.dataset.action_chunk_size

    if dataset_chunk_len is not None and dataset_chunk_len != configured_chunk_len:
        sys.exit(
            "error: training.dataset.action_chunk_size="
            f"{dataset_chunk_len} does not match policy.action_head.chunk_len={configured_chunk_len}; "
            "fix the dataset action chunk size or the policy action head chunk length"
        )

    actual_chunk_len = batch.actions.shape[1]
    if actual_chunk_len != configured_chunk_len:
        sys.exit(
            "error: first batch action chunk length mismatch: "
            f"expected policy.action_head.chunk_len={configured_chunk_len}, got {actual_chunk_len}; "
            "fix policy.action_head.chunk_len or training.dataset.action_chunk_size"
        )

    actual_action_dim = batch.actions.shape[2]
    if actual_action_dim != configured_action_dim:
        sys.exit(
            "error: first batch action dimension mismatch: "
            f"expected policy.embodiment.action_dim={configured_action_dim}, got {actual_action_dim}; "
            "fix policy.embodiment.action_dim to match the dataset embodiment"
        )

    actual_proprio_dim = batch.observations.proprio.shape[-1]
    if actual_proprio_dim != configured_proprio_dim:
        sys.exit(
            "error: first batch proprio dimension mismatch: "
            f"expected policy.embodiment.proprio_dim={configured_proprio_dim}, got {actual_proprio_dim}; "
            "fix policy.embodiment.proprio_dim to match the dataset embodiment"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    default = _pop_config_flag() or TrainConfig()
    cfg = tyro.cli(TrainConfig, default=default)
    LOGGER.info("Effective train config: %s", cfg)

    from accelerate.utils import set_seed  # type: ignore[import-untyped]

    set_seed(cfg.training.dataset.seed)

    LOGGER.info("Building policy...")
    policy = build_policy(cfg.policy)

    if cfg.policy.backbone.type == "paligemma":
        try:
            ckpt_image_size = int(policy.backbone.base_model.config.vision_config.image_size)
        except Exception as exc:  # pragma: no cover
            sys.exit(f"error: failed to derive SigLIP checkpoint image_size from loaded model config: {exc}")
        try:
            autowire_siglip_image_transforms(cfg.training, ckpt_image_size=ckpt_image_size)
        except ValueError as exc:
            sys.exit(f"error: {exc}")

    LOGGER.info("Creating dataloader...")
    dataloader = create_training_dataloader(
        cfg.training, dt_hz=cfg.policy.dt_hz, chunk_len=cfg.policy.action_head.chunk_len
    )
    _validate_training_dimensions(cfg, _peek_training_batch(dataloader))

    trainer = Trainer(policy, cfg.training, dataloader)
    LOGGER.info("Starting training for %d steps", cfg.training.num_steps)
    trainer.run()


if __name__ == "__main__":
    main()
