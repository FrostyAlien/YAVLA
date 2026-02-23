"""Training-facing data loader helpers."""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from yavla.data.factory import create_dataloader, dataclass_to_dict, set_dataloader_epoch
from yavla.models.types import ObservationBatch, TrainingBatch
from yavla.training.config import TrainingConfig

LOGGER = logging.getLogger(__name__)

_IMAGE_PREFIX = "observation.images."


class TrainingCollate:
    """Collate raw LeRobot dict samples into a typed ``TrainingBatch``.

    Converts ``list[dict[str, Any]]`` (PyTorch DataLoader convention) into a
    single ``TrainingBatch`` with nested ``ObservationBatch``, using LeRobot
    key conventions:

    - ``observation.images.<cam>`` → ``ObservationBatch.images[cam]`` ``[B, C, H, W]``
    - ``observation.state`` → ``ObservationBatch.proprio`` ``[B, D]``
    - ``task`` → ``ObservationBatch.language`` ``list[str]`` (``None`` if absent)
    - ``action`` → ``TrainingBatch.actions`` ``[B, chunk_len, action_dim]``
    - ``action_is_pad`` → ``TrainingBatch.action_mask`` ``[B, chunk_len]`` (``None`` if absent)

    Keys not matching any convention are silently ignored. Only keys present
    in *every* sample (intersection) are collated, so optional fields are
    all-or-nothing per batch.

    Args:
        dt_hz: Control frequency in Hz, passed through to ``TrainingBatch.dt_hz``.
        chunk_len: Action chunk length, passed through to ``TrainingBatch.chunk_len``.

    Raises:
        ValueError: If ``observation.state`` is missing (proprio is required).
        ValueError: If ``action`` is missing or has 2-D shape ``[B, action_dim]``
            (indicates ``action_chunk_size`` was not set in ``DataConfig``).
    """

    def __init__(self, *, dt_hz: float, chunk_len: int) -> None:
        self.dt_hz = dt_hz
        self.chunk_len = chunk_len

    def __call__(self, samples: list[dict[str, Any]]) -> TrainingBatch:
        # Intersect keys so optional fields are all-or-nothing per batch
        common_keys = set(samples[0].keys())
        for s in samples[1:]:
            common_keys &= s.keys()

        collated: dict[str, Tensor | list[str]] = {}
        for key in common_keys:
            values = [s[key] for s in samples]
            if isinstance(values[0], Tensor):
                collated[key] = torch.stack(values)
            elif isinstance(values[0], str):
                collated[key] = values

        # Validate proprio
        if "observation.state" not in collated:
            raise ValueError("observation.state is required but missing from all samples")

        # Images
        images: dict[str, Tensor] = {}
        for key, val in collated.items():
            if key.startswith(_IMAGE_PREFIX) and isinstance(val, Tensor):
                cam_name = key[len(_IMAGE_PREFIX) :]
                images[cam_name] = val

        # Proprio
        proprio = cast(Tensor, collated["observation.state"])

        # Language (optional)
        language: list[str] | None = None
        if "task" in collated:
            language = cast(list[str], collated["task"])

        # Actions (required)
        actions = collated.get("action")
        if not isinstance(actions, Tensor):
            raise ValueError("action key is required")
        if actions.ndim == 2:
            raise ValueError(
                "Actions have shape [B, action_dim] (2D) — set action_chunk_size "
                "in your DataConfig to produce chunked [B, chunk_len, action_dim] actions"
            )

        # Action mask (optional)
        action_mask: Tensor | None = None
        if "action_is_pad" in collated:
            action_mask = cast(Tensor, collated["action_is_pad"])

        return TrainingBatch(
            observations=ObservationBatch(images=images, proprio=proprio, language=language),
            actions=actions,
            dt_hz=self.dt_hz,
            chunk_len=self.chunk_len,
            action_mask=action_mask,
        )


def create_training_dataloader(
    config: TrainingConfig, *, dt_hz: float, chunk_len: int
) -> DataLoader[Any]:
    """Create a training dataloader that yields ``TrainingBatch`` instances.

    Wraps ``create_dataloader`` with a ``TrainingCollate`` collate function
    that converts raw LeRobot dict batches into typed dataclass instances.

    Args:
        config: Training configuration (dataset settings are in ``config.dataset``).
        dt_hz: Control frequency sourced from ``PolicyConfig.dt_hz``.
        chunk_len: Action chunk length sourced from the action head config.
    """

    collate = TrainingCollate(dt_hz=dt_hz, chunk_len=chunk_len)
    dataloader = create_dataloader(config.dataset, collate_fn=collate)
    backend = getattr(dataloader, "yavla_backend", "unknown")
    reason = getattr(dataloader, "yavla_backend_reason", "n/a")
    LOGGER.info(
        "Training dataloader backend=%s reason=%s config=%s",
        backend,
        reason,
        dataclass_to_dict(config.dataset),
    )
    return dataloader


def advance_data_epoch(dataloader: DataLoader[Any], epoch: int) -> None:
    """Advance sampler/dataset epoch state."""

    set_dataloader_epoch(dataloader, epoch)
