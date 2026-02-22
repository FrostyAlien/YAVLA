"""Training-facing data loader helpers."""

from __future__ import annotations

import logging
from typing import Any

from torch.utils.data import DataLoader

from yavla.data.factory import create_dataloader, dataclass_to_dict, set_dataloader_epoch
from yavla.training.config import TrainingConfig

LOGGER = logging.getLogger(__name__)


def create_training_dataloader(config: TrainingConfig) -> DataLoader[Any]:
    """Create and log training dataloader selection details."""

    dataloader = create_dataloader(config.dataset)
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
