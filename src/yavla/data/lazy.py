"""Map-style lazy LeRobot dataset backed by direct Parquet reads."""

from __future__ import annotations

import bisect
import collections
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # type: ignore[import-untyped]
from lerobot.datasets.video_utils import (  # type: ignore[import-untyped]
    _default_decoder_cache,
    decode_video_frames,
)
from torch.utils.data import Dataset, get_worker_info

from yavla.data.metadata_utils import build_task_lookup
from yavla.data.schema import validate_sample_schema


@dataclass(slots=True, frozen=True)
class _EpisodeWindow:
    episode_index: int
    local_start: int
    local_end: int
    global_start: int
    global_end: int
    chunk_index: int
    file_index: int


@dataclass(slots=True, frozen=True)
class _FileSegment:
    chunk_index: int
    file_index: int
    path_id: int
    file_offset_start: int
    local_start: int
    local_end: int


class LazyLeRobotDataset(Dataset[dict[str, Any]]):
    """Map-style dataset that lazily reads LeRobot Parquet shards."""

    def __init__(
        self,
        repo_id: str,
        *,
        root: str | Path | None = None,
        episodes: Sequence[int] | None = None,
        transforms: Any = None,
        feature_columns: Sequence[str] | None = None,
        delta_timestamps: Mapping[str, Sequence[float]] | None = None,
        action_chunk_size: int | None = None,
        video_backend: str = "pyav",
        parquet_cache_size: int = 32,
        max_video_decoders: int = 128,
        tolerance_s: float = 1e-4,
    ) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root is not None else None
        self.transforms = transforms
        self.feature_columns = tuple(feature_columns) if feature_columns is not None else None
        self.delta_timestamps = delta_timestamps
        self.action_chunk_size = action_chunk_size
        self.video_backend = video_backend
        self.parquet_cache_size = parquet_cache_size
        self.max_video_decoders = max_video_decoders
        self.tolerance_s = tolerance_s

        self.meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
        codebase_version = self.meta.info.get("codebase_version")
        if codebase_version != "v3.0":
            raise ValueError(f"Unsupported LeRobot codebase version: {codebase_version!r}; expected 'v3.0'")

        self._selected_episodes = set(episodes) if episodes is not None else None
        self._episode_windows: list[_EpisodeWindow] = []
        self._episode_end_boundaries: list[int] = []
        self._file_segments: list[_FileSegment] = []
        self._file_boundaries: list[int] = [0]
        self._file_paths: list[Path] = []
        self._worker_file_cache: dict[int, collections.OrderedDict[int, pq.ParquetFile]] = {}
        self._row_group_boundaries: dict[int, list[int]] = {}
        self._task_lookup = build_task_lookup(self.meta.tasks)
        self._decoder_paths: collections.OrderedDict[str, None] = collections.OrderedDict()
        self._epoch: int = 0

        self._build_indexes()
        self.delta_indices = self._build_delta_indices(delta_timestamps)

    def _iter_episode_records(self) -> list[dict[str, Any]]:
        episodes_obj = self.meta.episodes
        if hasattr(episodes_obj, "to_dict"):
            records = episodes_obj.to_dict(orient="records")
        elif isinstance(episodes_obj, list):
            records = episodes_obj
        else:
            raise TypeError(f"Unsupported episodes metadata type: {type(episodes_obj)!r}")
        return [dict(record) for record in records]

    def _build_indexes(self) -> None:
        records = sorted(self._iter_episode_records(), key=lambda record: int(record["dataset_from_index"]))
        file_first_global_index: dict[tuple[int, int], int] = {}
        file_path_ids: dict[tuple[int, int], int] = {}
        file_paths: list[Path] = []
        for record in records:
            key = (int(record["data/chunk_index"]), int(record["data/file_index"]))
            global_start = int(record["dataset_from_index"])
            if key not in file_first_global_index:
                file_first_global_index[key] = global_start
            else:
                file_first_global_index[key] = min(file_first_global_index[key], global_start)
            if key not in file_path_ids:
                file_path_ids[key] = len(file_paths)
                file_paths.append(self._resolve_file_path(*key))

        filtered_records = []
        for record in records:
            episode_index = int(record["episode_index"])
            if self._selected_episodes is not None and episode_index not in self._selected_episodes:
                continue
            filtered_records.append(record)

        local_cursor = 0
        file_segments: list[_FileSegment] = []
        for record in filtered_records:
            global_start = int(record["dataset_from_index"])
            global_end = int(record["dataset_to_index"])
            length = global_end - global_start
            if length <= 0:
                continue

            chunk_index = int(record["data/chunk_index"])
            file_index = int(record["data/file_index"])
            file_key = (chunk_index, file_index)
            path_id = file_path_ids[file_key]
            file_offset_start = global_start - file_first_global_index[file_key]
            episode_window = _EpisodeWindow(
                episode_index=int(record["episode_index"]),
                local_start=local_cursor,
                local_end=local_cursor + length,
                global_start=global_start,
                global_end=global_end,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            self._episode_windows.append(episode_window)
            self._episode_end_boundaries.append(episode_window.local_end)

            file_segments.append(
                _FileSegment(
                    chunk_index=chunk_index,
                    file_index=file_index,
                    path_id=path_id,
                    file_offset_start=file_offset_start,
                    local_start=episode_window.local_start,
                    local_end=episode_window.local_end,
                )
            )

            local_cursor += length

        self._file_segments = file_segments
        self._file_paths = file_paths
        self._file_boundaries = [0] + [segment.local_end for segment in file_segments]

    def _build_delta_indices(
        self,
        delta_timestamps: Mapping[str, Sequence[float]] | None,
    ) -> dict[str, list[int]] | None:
        if delta_timestamps is None:
            return None
        fps = float(self.meta.fps)
        return {
            key: [int(round(delta_seconds * fps)) for delta_seconds in deltas]
            for key, deltas in delta_timestamps.items()
        }

    def _resolve_file_path(self, chunk_index: int, file_index: int) -> Path:
        rel_path = str(self.meta.data_path).format(chunk_index=chunk_index, file_index=file_index)
        if self.root is not None:
            return self.root / rel_path
        return Path(str(self.meta.root)) / rel_path

    def _worker_id(self) -> int:
        worker_info = get_worker_info()
        if worker_info is None:
            return -1
        return int(worker_info.id)

    def _get_file_handle(self, path_id: int) -> pq.ParquetFile:
        worker_id = self._worker_id()
        cache = self._worker_file_cache.setdefault(worker_id, collections.OrderedDict())
        if path_id in cache:
            cache.move_to_end(path_id)
            return cache[path_id]

        file_handle = pq.ParquetFile(self._file_paths[path_id])
        cache[path_id] = file_handle
        cache.move_to_end(path_id)
        while len(cache) > self.parquet_cache_size:
            cache.popitem(last=False)
        return file_handle

    def _get_row_group_boundaries(self, path_id: int) -> list[int]:
        if path_id in self._row_group_boundaries:
            return self._row_group_boundaries[path_id]

        file_handle = self._get_file_handle(path_id)
        boundaries = [0]
        total = 0
        for row_group_id in range(file_handle.num_row_groups):
            total += file_handle.metadata.row_group(row_group_id).num_rows
            boundaries.append(total)
        self._row_group_boundaries[path_id] = boundaries
        return boundaries

    def _file_id_for_local_index(self, local_index: int) -> int:
        file_id = bisect.bisect_right(self._file_boundaries, local_index) - 1
        if file_id < 0 or file_id >= len(self._file_segments):
            raise IndexError(f"Index {local_index} out of range for file segments")
        return file_id

    def _episode_window_for_local_index(self, local_index: int) -> _EpisodeWindow:
        episode_pos = bisect.bisect_right(self._episode_end_boundaries, local_index)
        if episode_pos >= len(self._episode_windows):
            raise IndexError(f"Index {local_index} out of range for episode windows")
        return self._episode_windows[episode_pos]

    def _read_local_rows(
        self,
        file_id: int,
        local_indices_within_file: Sequence[int],
        *,
        columns: Sequence[str] | None,
    ) -> list[dict[str, Any]]:
        if not local_indices_within_file:
            return []

        segment = self._file_segments[file_id]
        file_handle = self._get_file_handle(segment.path_id)
        row_group_boundaries = self._get_row_group_boundaries(segment.path_id)

        grouped: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
        for output_order, local_in_file in enumerate(local_indices_within_file):
            row_group_id = bisect.bisect_right(row_group_boundaries, local_in_file) - 1
            row_in_group = local_in_file - row_group_boundaries[row_group_id]
            grouped[row_group_id].append((output_order, row_in_group))

        output_rows: list[dict[str, Any] | None] = [None] * len(local_indices_within_file)
        for row_group_id, entries in grouped.items():
            table = file_handle.read_row_group(row_group_id, columns=list(columns) if columns is not None else None)
            rows = table.take([entry[1] for entry in entries]).to_pylist()
            for (order, _), row in zip(entries, rows, strict=True):
                output_rows[order] = dict(row)

        return [row for row in output_rows if row is not None]

    def _default_columns(self) -> list[str]:
        metadata_columns = ["episode_index", "index", "timestamp", "frame_index", "task_index"]
        if self.feature_columns is not None:
            feature_columns = [column for column in self.feature_columns if column in self.meta.features]
        else:
            feature_columns = list(self.meta.features.keys())
        selected = metadata_columns + feature_columns
        seen: set[str] = set()
        deduped = []
        for column in selected:
            if column in seen:
                continue
            seen.add(column)
            deduped.append(column)
        return deduped

    def _local_to_file_offset(self, local_index: int, file_id: int) -> int:
        file_segment = self._file_segments[file_id]
        return file_segment.file_offset_start + (local_index - file_segment.local_start)

    def _task_name(self, task_index: int | None) -> str | None:
        if task_index is None:
            return None
        return self._task_lookup.get(int(task_index))

    def _decode_video_field(self, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        if "path" not in value or "timestamp" not in value:
            return value

        raw_path = Path(str(value["path"]))
        data_root = Path(self.meta.root) if self.root is None else self.root
        video_path = raw_path if raw_path.is_absolute() else (data_root / raw_path)
        if self.video_backend == "torchcodec":
            cache_key = str(video_path)
            self._decoder_paths[cache_key] = None
            self._decoder_paths.move_to_end(cache_key)
            if len(self._decoder_paths) > self.max_video_decoders:
                _default_decoder_cache.clear()
                self._decoder_paths.clear()

        frames = decode_video_frames(
            video_path=video_path,
            timestamps=[float(value["timestamp"])],
            tolerance_s=self.tolerance_s,
            backend=self.video_backend,
        )
        return frames.squeeze(0)

    def set_epoch(self, epoch: int) -> None:
        """Track epoch and clear shared decoder cache at boundaries."""

        if epoch != self._epoch and self.video_backend == "torchcodec":
            _default_decoder_cache.clear()
            self._decoder_paths.clear()
        self._epoch = epoch

    def clear_video_cache(self) -> None:
        """Clear torchcodec decoder cache when using torchcodec backend."""

        if self.video_backend == "torchcodec":
            _default_decoder_cache.clear()
            self._decoder_paths.clear()

    def __len__(self) -> int:
        return self._file_boundaries[-1]

    def _fetch_rows(self, local_indices: Sequence[int]) -> list[dict[str, Any]]:
        grouped_by_file: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
        for output_order, local_index in enumerate(local_indices):
            file_id = self._file_id_for_local_index(local_index)
            local_in_file = self._local_to_file_offset(local_index, file_id)
            grouped_by_file[file_id].append((output_order, local_in_file))

        ordered_rows: list[dict[str, Any] | None] = [None] * len(local_indices)
        columns = self._default_columns()
        for file_id, entries in grouped_by_file.items():
            rows = self._read_local_rows(
                file_id,
                [local_in_file for _, local_in_file in entries],
                columns=columns,
            )
            for (output_order, _), row in zip(entries, rows, strict=True):
                ordered_rows[output_order] = row
        return [row for row in ordered_rows if row is not None]

    def _augment_with_temporal_queries(self, idx: int, sample: dict[str, Any]) -> dict[str, Any]:
        if self.delta_indices is None:
            return sample

        episode_window = self._episode_window_for_local_index(idx)
        output = dict(sample)
        for key, delta_indices in self.delta_indices.items():
            query_indices = []
            pad_mask = []
            for delta_index in delta_indices:
                raw_index = idx + delta_index
                is_pad = raw_index < episode_window.local_start or raw_index >= episode_window.local_end
                clamped = min(max(raw_index, episode_window.local_start), episode_window.local_end - 1)
                query_indices.append(clamped)
                pad_mask.append(is_pad)

            queried_rows = self._fetch_rows(query_indices)
            if key in self.meta.video_keys:
                values = [self._decode_video_field(row.get(key)) for row in queried_rows]
            else:
                values = [row.get(key) for row in queried_rows]

            tensor_values = [torch.as_tensor(value) for value in values]
            output[key] = torch.stack(tensor_values)
            output[f"{key}_is_pad"] = torch.as_tensor(pad_mask, dtype=torch.bool)
        return output

    def _augment_with_action_chunk(self, idx: int, sample: dict[str, Any]) -> dict[str, Any]:
        if self.action_chunk_size is None:
            return sample

        episode_window = self._episode_window_for_local_index(idx)
        query_indices = []
        pad_mask = []
        for step in range(self.action_chunk_size):
            raw_index = idx + step
            is_pad = raw_index >= episode_window.local_end
            clamped = min(raw_index, episode_window.local_end - 1)
            query_indices.append(clamped)
            pad_mask.append(is_pad)

        queried_rows = self._fetch_rows(query_indices)
        actions = [torch.as_tensor(row["action"]) for row in queried_rows]
        output = dict(sample)
        output["action"] = torch.stack(actions)
        output["action_is_pad"] = torch.as_tensor(pad_mask, dtype=torch.bool)
        return output

    def _build_sample(self, row: dict[str, Any], idx: int) -> dict[str, Any]:
        sample = dict(row)

        for video_key in self.meta.video_keys:
            if video_key in sample:
                sample[video_key] = self._decode_video_field(sample[video_key])

        task_name = self._task_name(sample.get("task_index"))
        if task_name is not None:
            sample["task"] = task_name

        sample = self._augment_with_temporal_queries(idx, sample)
        sample = self._augment_with_action_chunk(idx, sample)
        if self.transforms is not None:
            sample = self.transforms(sample)
        validate_sample_schema(sample, require_task_name=False)
        return sample

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        row = self._fetch_rows([idx])[0]
        return self._build_sample(row, idx)

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        normalized_indices = []
        for idx in indices:
            normalized = idx + len(self) if idx < 0 else idx
            if normalized < 0 or normalized >= len(self):
                raise IndexError(idx)
            normalized_indices.append(normalized)

        rows = self._fetch_rows(normalized_indices)
        return [self._build_sample(row, idx) for row, idx in zip(rows, normalized_indices, strict=True)]
