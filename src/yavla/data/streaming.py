"""Streaming iterable dataset with shard interleaving and shuffle buffer."""

from __future__ import annotations

import collections
import multiprocessing
import random
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # type: ignore[import-untyped]
from lerobot.datasets.video_utils import decode_video_frames  # type: ignore[import-untyped]
from torch.utils.data import IterableDataset, get_worker_info

from yavla.data.metadata_utils import (
    EpisodeMediaReference,
    build_episode_media_lookup,
    build_task_lookup,
    metadata_records,
)
from yavla.data.schema import validate_sample_schema


class ShardInterleavedDataset(IterableDataset[dict[str, Any]]):
    """Iterable dataset with multi-shard interleaving and shuffle-buffer sampling."""

    # TODO(dataset-layer): add optional episode-subset filtering for parity with LazyLeRobotDataset.

    def __init__(
        self,
        repo_id: str,
        *,
        root: str | Path | None = None,
        transforms: Any = None,
        feature_columns: Sequence[str] | None = None,
        shuffle_buffer_size: int = 10_000,
        num_interleaved_shards: int = 8,
        parquet_batch_size: int = 256,
        seed: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
        delta_timestamps: Mapping[str, Sequence[float]] | None = None,
        action_chunk_size: int | None = None,
        video_backend: str = "pyav",
        tolerance_s: float = 1e-4,
    ) -> None:
        if delta_timestamps is not None:
            raise ValueError("streaming backend does not support delta_timestamps; use lazy/default backend")
        if action_chunk_size is not None:
            raise ValueError("streaming backend does not support action_chunk_size; use lazy/default backend")
        if num_interleaved_shards <= 0:
            raise ValueError("num_interleaved_shards must be positive")
        if shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be positive")

        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root is not None else None
        self.transforms = transforms
        self.feature_columns = tuple(feature_columns) if feature_columns is not None else None
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_interleaved_shards = num_interleaved_shards
        self.parquet_batch_size = parquet_batch_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.video_backend = video_backend
        self.tolerance_s = tolerance_s
        self._epoch_state = multiprocessing.Value("q", 0, lock=True)

        self.meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
        self._media_keys = self._resolve_media_keys()
        self._episode_media_lookup: dict[int, dict[str, EpisodeMediaReference]] = {}
        self._shard_paths = self._discover_shards()
        self._task_lookup = self._build_task_lookup()

    def _build_task_lookup(self) -> dict[int, str]:
        return build_task_lookup(self.meta.tasks)

    def _resolve_media_keys(self) -> tuple[str, ...]:
        keys: list[str] = []

        if hasattr(self.meta, "camera_keys"):
            keys.extend(str(key) for key in getattr(self.meta, "camera_keys"))
        if hasattr(self.meta, "video_keys"):
            keys.extend(str(key) for key in getattr(self.meta, "video_keys"))

        if not keys:
            keys.extend(
                key for key, feature in self.meta.features.items() if feature.get("dtype") in {"video", "image"}
            )

        deduped: list[str] = []
        seen: set[str] = set()
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return tuple(deduped)

    def _discover_shards(self) -> list[Path]:
        records = metadata_records(self.meta.episodes)
        self._episode_media_lookup = build_episode_media_lookup(records, self._media_keys)

        ordered_keys: collections.OrderedDict[tuple[int, int], None] = collections.OrderedDict()
        for record in records:
            key = (int(record["data/chunk_index"]), int(record["data/file_index"]))
            ordered_keys.setdefault(key, None)

        shard_paths = []
        root = Path(self.meta.root) if self.root is None else self.root
        for chunk_index, file_index in ordered_keys.keys():
            rel_path = self.meta.data_path.format(chunk_index=chunk_index, file_index=file_index)
            shard_paths.append(root / rel_path)
        return shard_paths

    def set_epoch(self, epoch: int) -> None:
        """Update epoch used for deterministic shard/buffer shuffling."""

        with self._epoch_state.get_lock():
            self._epoch_state.value = int(epoch)

    def _current_epoch(self) -> int:
        with self._epoch_state.get_lock():
            return int(self._epoch_state.value)

    def _distributed_context(self) -> tuple[int, int]:
        if self.rank is not None and self.world_size is not None:
            return self.rank, self.world_size
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def _partition_shards(self, shuffled_indices: list[int]) -> list[int]:
        rank, world_size = self._distributed_context()
        rank_indices = [index for index in shuffled_indices if index % world_size == rank]

        worker_info = get_worker_info()
        if worker_info is None:
            return rank_indices
        worker_id = int(worker_info.id)
        worker_count = int(worker_info.num_workers)
        return self._assign_worker_shards(rank_indices, worker_id=worker_id, worker_count=worker_count)

    def _assign_worker_shards(
        self,
        rank_indices: Sequence[int],
        *,
        worker_id: int,
        worker_count: int,
    ) -> list[int]:
        """Assign rank-local shards to one worker via round-robin slicing."""

        if worker_count <= 1:
            return list(rank_indices)
        return list(rank_indices[worker_id::worker_count])

    def _columns_to_read(self) -> list[str] | None:
        metadata_columns = {"episode_index", "index", "timestamp", "frame_index", "task_index"}
        if self.feature_columns is None:
            feature_columns = set(self.meta.features.keys())
        else:
            feature_columns = {key for key in self.feature_columns if key in self.meta.features}
        columns = sorted(metadata_columns.union(feature_columns))
        return columns if columns else None

    def _read_shard_rows(self, shard_path: Path, columns: Sequence[str] | None) -> Iterator[dict[str, Any]]:
        parquet_file = pq.ParquetFile(shard_path)
        for batch in parquet_file.iter_batches(batch_size=self.parquet_batch_size, columns=columns):
            for row in batch.to_pylist():
                yield dict(row)

    def _timestamp_from_context(
        self,
        *,
        sample_timestamp: Any = None,
        sample_frame_index: Any = None,
    ) -> float | None:
        if sample_timestamp is not None:
            try:
                return float(sample_timestamp)
            except (TypeError, ValueError):
                return None
        if sample_frame_index is not None and self.meta.fps:
            try:
                return float(sample_frame_index) / float(self.meta.fps)
            except (TypeError, ValueError, ZeroDivisionError):
                return None
        return None

    def _decode_video_at_timestamp(self, *, video_path: Path, timestamp: float) -> torch.Tensor:
        frames = decode_video_frames(
            video_path=video_path,
            timestamps=[float(timestamp)],
            tolerance_s=self.tolerance_s,
            backend=self.video_backend,
        )
        return frames.squeeze(0)

    def _decode_from_row_payload(
        self,
        value: Any,
        *,
        sample_timestamp: Any = None,
        sample_frame_index: Any = None,
    ) -> torch.Tensor | None:
        raw_path: Path | None = None
        timestamp_value: float | None = None

        if isinstance(value, Mapping):
            if "path" not in value:
                return None
            raw_path = Path(str(value["path"]))
            if "timestamp" in value:
                try:
                    timestamp_value = float(value["timestamp"])
                except (TypeError, ValueError):
                    timestamp_value = None
        elif isinstance(value, (str, Path)):
            raw_path = Path(str(value))
        else:
            return None

        if raw_path is None:
            return None
        if timestamp_value is None:
            timestamp_value = self._timestamp_from_context(
                sample_timestamp=sample_timestamp,
                sample_frame_index=sample_frame_index,
            )
        if timestamp_value is None:
            return None

        data_root = Path(self.meta.root) if self.root is None else self.root
        video_path = raw_path if raw_path.is_absolute() else data_root / raw_path
        return self._decode_video_at_timestamp(video_path=video_path, timestamp=timestamp_value)

    def _resolve_episode_media_path(self, media_key: str, episode_index: int) -> tuple[Path, float] | None:
        video_path_template = getattr(self.meta, "video_path", None)
        if video_path_template is None and hasattr(self.meta, "info"):
            video_path_template = self.meta.info.get("video_path")
        if not video_path_template:
            return None

        episode_media = self._episode_media_lookup.get(episode_index)
        if episode_media is None:
            return None
        reference = episode_media.get(media_key)
        if reference is None:
            return None

        rel_path = str(video_path_template).format(
            video_key=media_key,
            chunk_index=reference.chunk_index,
            file_index=reference.file_index,
        )
        data_root = Path(self.meta.root) if self.root is None else self.root
        return data_root / rel_path, reference.from_timestamp

    def _resolve_media_value(
        self,
        *,
        media_key: str,
        row_value: Any,
        episode_index: Any = None,
        sample_timestamp: Any = None,
        sample_frame_index: Any = None,
    ) -> Any:
        decoded_from_row = self._decode_from_row_payload(
            row_value,
            sample_timestamp=sample_timestamp,
            sample_frame_index=sample_frame_index,
        )
        if decoded_from_row is not None:
            return decoded_from_row

        timestamp_value = self._timestamp_from_context(
            sample_timestamp=sample_timestamp,
            sample_frame_index=sample_frame_index,
        )
        if timestamp_value is None:
            return row_value

        try:
            episode_idx = int(episode_index)
        except (TypeError, ValueError):
            return row_value

        media_path = self._resolve_episode_media_path(media_key, episode_idx)
        if media_path is None:
            return row_value
        video_path, from_timestamp = media_path
        return self._decode_video_at_timestamp(video_path=video_path, timestamp=from_timestamp + timestamp_value)

    def _prepare_sample(self, row: dict[str, Any]) -> dict[str, Any]:
        sample = dict(row)
        for media_key in self._media_keys:
            resolved_media = self._resolve_media_value(
                media_key=media_key,
                row_value=sample.get(media_key),
                episode_index=sample.get("episode_index"),
                sample_timestamp=sample.get("timestamp"),
                sample_frame_index=sample.get("frame_index"),
            )
            if media_key in sample or resolved_media is not None:
                sample[media_key] = resolved_media

        task_index = sample.get("task_index")
        if isinstance(task_index, (int, np.integer)):
            task_name = self._task_lookup.get(int(task_index))
            if task_name is not None:
                sample["task"] = task_name

        if self.transforms is not None:
            sample = self.transforms(sample)
        validate_sample_schema(sample, require_task_name=False)
        return sample

    def _next_row(
        self,
        active: list[Iterator[dict[str, Any]]],
        pending: collections.deque[int],
        *,
        columns: Sequence[str] | None,
        rr_state: list[int],
    ) -> dict[str, Any] | None:
        while True:
            if not active:
                if not pending:
                    return None
                while pending and len(active) < self.num_interleaved_shards:
                    shard_id = pending.popleft()
                    active.append(self._read_shard_rows(self._shard_paths[shard_id], columns))
                rr_state[0] = 0
                if not active:
                    return None

            active_index = rr_state[0] % len(active)
            iterator = active[active_index]
            try:
                row = next(iterator)
                rr_state[0] = (active_index + 1) % max(len(active), 1)
                return row
            except StopIteration:
                active.pop(active_index)
                if pending:
                    shard_id = pending.popleft()
                    active.insert(active_index, self._read_shard_rows(self._shard_paths[shard_id], columns))
                if not active and not pending:
                    return None
                rr_state[0] = active_index % max(len(active), 1)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        epoch = self._current_epoch()
        rng = random.Random(self.seed + epoch)
        shard_indices = list(range(len(self._shard_paths)))
        rng.shuffle(shard_indices)
        shard_indices = self._partition_shards(shard_indices)

        pending = collections.deque(shard_indices)
        active: list[Iterator[dict[str, Any]]] = []
        columns = self._columns_to_read()
        rr_state = [0]
        buffer: list[dict[str, Any]] = []

        while len(active) < self.num_interleaved_shards and pending:
            shard_id = pending.popleft()
            active.append(self._read_shard_rows(self._shard_paths[shard_id], columns))

        while len(buffer) < self.shuffle_buffer_size:
            row = self._next_row(active, pending, columns=columns, rr_state=rr_state)
            if row is None:
                break
            buffer.append(row)

        while buffer:
            random_index = rng.randrange(len(buffer))
            candidate = buffer[random_index]
            replacement = self._next_row(active, pending, columns=columns, rr_state=rr_state)
            if replacement is None:
                buffer.pop(random_index)
            else:
                buffer[random_index] = replacement
            yield self._prepare_sample(candidate)

        # Defensive tail flush for implementations that leave residual rows in the buffer.
        if buffer:
            rng.shuffle(buffer)
            for row in buffer:
                yield self._prepare_sample(row)
