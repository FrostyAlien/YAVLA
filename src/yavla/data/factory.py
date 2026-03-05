"""Dataset factory for selecting and wiring data backends."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from lerobot.datasets.lerobot_dataset import (  # type: ignore[import-untyped]
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from yavla.data.lazy import LazyLeRobotDataset
from yavla.data.streaming import ShardInterleavedDataset
from yavla.data.transforms import (
    DataTransformFn,
    ImageTransform,
    NormalizeTransform,
    RepackTransform,
    build_torchvision_transforms,
    compose,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DataConfig:
    """Configuration for dataset backend construction."""

    repo_id: str
    root: str | Path | None = None
    backend: Literal["default", "lazy", "streaming"] = "default"
    delta_timestamps: dict[str, list[float]] | None = None
    action_chunk_size: int | None = None

    batch_size: int = 32
    num_workers: int = 4
    persistent_workers: bool = False
    drop_last: bool = False

    parquet_cache_size: int = 32
    max_video_decoders: int = 128

    shuffle_buffer_size: int = 10_000
    num_interleaved_shards: int = 8
    streaming_parquet_batch_size: int = 256

    repack_keys: dict[str, str] | None = None
    feature_keys: list[str] | None = None
    image_transforms: list[str] | None = None
    normalize: bool = True
    normalize_mode: Literal["z-score", "min-max"] = "z-score"
    normalize_keys: list[str] | None = None

    video_backend: str = "pyav"
    seed: int = 0
    pin_memory: bool | None = None


@dataclass(slots=True)
class BackendSelection:
    """Selected backend and a human-readable reason."""

    backend: Literal["default", "lazy", "streaming"]
    reason: str


class _TransformingMapDataset(Dataset[dict[str, Any]]):
    def __init__(self, dataset: Dataset[dict[str, Any]], transform: DataTransformFn | None) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(cast(Any, self.dataset))

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        if self.transform is None:
            return sample
        return self.transform(sample)


def _is_distributed_active() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _camera_keys_from_metadata(metadata: LeRobotDatasetMetadata) -> list[str]:
    if hasattr(metadata, "camera_keys"):
        return list(metadata.camera_keys)
    return [
        key
        for key, feature in metadata.features.items()
        if feature.get("dtype") in {"video", "image"}
    ]


def plan_feature_columns(config: DataConfig, metadata: LeRobotDatasetMetadata) -> list[str]:
    """Plan feature columns needed by the configured data pipeline."""

    known_features = set(metadata.features.keys())
    planned: set[str] = set()

    if config.feature_keys is not None:
        planned.update(config.feature_keys)

    if config.repack_keys:
        planned.update(config.repack_keys.keys())

    if config.image_transforms:
        planned.update(_camera_keys_from_metadata(metadata))

    if config.normalize:
        if config.normalize_keys is None:
            planned.update(known_features)
        else:
            planned.update(config.normalize_keys)

    if config.delta_timestamps:
        planned.update(config.delta_timestamps.keys())

    if config.action_chunk_size is not None:
        planned.add("action")

    if not planned:
        planned.update(known_features)

    return sorted(key for key in planned if key in known_features)


def build_transform_pipeline(config: DataConfig, metadata: LeRobotDatasetMetadata) -> DataTransformFn | None:
    """Build a composed transform pipeline from config."""

    transforms: list[DataTransformFn] = []
    if config.repack_keys:
        transforms.append(RepackTransform(config.repack_keys))

    if config.normalize and metadata.stats is not None:
        normalize_keys = config.normalize_keys
        if normalize_keys is None:
            camera_keys = set(_camera_keys_from_metadata(metadata))
            normalize_keys = [key for key in metadata.stats if key not in camera_keys]
        transforms.append(
            NormalizeTransform(
                stats=metadata.stats,
                mode=config.normalize_mode,
                keys=normalize_keys,
            )
        )

    if config.image_transforms:
        torchvision_transforms = build_torchvision_transforms(config.image_transforms)
        transforms.append(
            ImageTransform(
                transforms=torchvision_transforms,
                camera_keys=_camera_keys_from_metadata(metadata),
            )
        )

    if not transforms:
        return None
    return compose(*transforms)


def select_backend(config: DataConfig) -> BackendSelection:
    """Validate backend constraints and return the selected backend."""

    if config.backend == "default":
        if config.action_chunk_size is not None:
            raise ValueError("default backend does not support action_chunk_size; use lazy backend")
        return BackendSelection(backend="default", reason="default backend (LeRobotDataset)")

    if config.backend == "lazy":
        return BackendSelection(backend="lazy", reason="explicit backend=lazy")

    if config.delta_timestamps is not None:
        raise ValueError("streaming backend does not support delta_timestamps; use lazy/default backend")
    if config.action_chunk_size is not None:
        raise ValueError("streaming backend does not support action_chunk_size; use lazy/default backend")
    return BackendSelection(backend="streaming", reason="explicit backend=streaming")


def create_dataloader(
    config: DataConfig,
    *,
    collate_fn: Any = None,
) -> DataLoader[Any]:
    """Create a dataloader using the selected dataset backend."""

    metadata = LeRobotDatasetMetadata(repo_id=config.repo_id, root=config.root)
    transform = build_transform_pipeline(config, metadata)
    feature_columns = plan_feature_columns(config, metadata)
    selection = select_backend(config)
    LOGGER.info("Selected data backend: %s | reason=%s", selection.backend, selection.reason)

    if selection.backend == "default":
        base_dataset = LeRobotDataset(
            repo_id=config.repo_id,
            root=config.root,
            delta_timestamps=config.delta_timestamps,
            video_backend=config.video_backend,
        )
        dataset: Any = _TransformingMapDataset(base_dataset, transform)
    elif selection.backend == "lazy":
        dataset = LazyLeRobotDataset(
            repo_id=config.repo_id,
            root=config.root,
            transforms=transform,
            feature_columns=feature_columns,
            delta_timestamps=config.delta_timestamps,
            action_chunk_size=config.action_chunk_size,
            parquet_cache_size=config.parquet_cache_size,
            video_backend=config.video_backend,
            max_video_decoders=config.max_video_decoders,
        )
    else:
        dataset = ShardInterleavedDataset(
            repo_id=config.repo_id,
            root=config.root,
            transforms=transform,
            feature_columns=feature_columns,
            shuffle_buffer_size=config.shuffle_buffer_size,
            num_interleaved_shards=config.num_interleaved_shards,
            parquet_batch_size=config.streaming_parquet_batch_size,
            seed=config.seed,
            video_backend=config.video_backend,
        )

    pin_memory = torch.cuda.is_available() if config.pin_memory is None else config.pin_memory
    sampler: DistributedSampler[Any] | None = None
    shuffle = False
    if selection.backend in {"default", "lazy"}:
        if _is_distributed_active():
            sampler = DistributedSampler(dataset, shuffle=True)
        else:
            shuffle = True

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        drop_last=config.drop_last,
        pin_memory=pin_memory,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        collate_fn=collate_fn,
    )
    setattr(dataloader, "yavla_backend", selection.backend)
    setattr(dataloader, "yavla_backend_reason", selection.reason)
    return dataloader


def set_dataloader_epoch(dataloader: DataLoader[Any], epoch: int) -> None:
    """Propagate epoch to distributed sampler and epoch-aware datasets."""

    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        return
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
        return
    wrapped = getattr(dataset, "dataset", None)
    if wrapped is not None and hasattr(wrapped, "set_epoch"):
        wrapped.set_epoch(epoch)


def dataclass_to_dict(config: DataConfig) -> dict[str, Any]:
    """Convert DataConfig to a plain dictionary for logging/debugging."""

    return dataclasses.asdict(config)
