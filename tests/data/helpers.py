"""Test helpers for synthetic LeRobot-like metadata and parquet shards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet_rows(path: Path, rows: list[dict[str, Any]], *, row_group_size: int = 2) -> None:
    """Write rows to parquet using deterministic row-group sizes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, row_group_size=row_group_size)


@dataclass(slots=True)
class FakeMetadata:
    """Minimal LeRobotDatasetMetadata-compatible object for tests."""

    root: Path
    info: dict[str, Any]
    episodes: pd.DataFrame
    tasks: pd.DataFrame
    stats: dict[str, Any]
    fps: int = 10

    @property
    def features(self) -> dict[str, Any]:
        return self.info["features"]

    @property
    def video_keys(self) -> list[str]:
        return [key for key, feature in self.features.items() if feature.get("dtype") == "video"]

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, feature in self.features.items() if feature.get("dtype") in {"video", "image"}]

    @property
    def total_frames(self) -> int:
        return int(self.info["total_frames"])

    @property
    def data_path(self) -> str:
        return str(self.info["data_path"])


def make_fake_metadata(
    root: Path,
    *,
    episode_lengths: list[int],
    file_assignments: list[tuple[int, int]] | None = None,
) -> FakeMetadata:
    """Create fake metadata with contiguous episode index ranges."""

    if file_assignments is None:
        file_assignments = [(0, 0) for _ in episode_lengths]
    if len(file_assignments) != len(episode_lengths):
        raise ValueError("file_assignments must match episode_lengths")

    cursor = 0
    episode_rows: list[dict[str, Any]] = []
    for episode_index, (length, file_assignment) in enumerate(zip(episode_lengths, file_assignments, strict=True)):
        chunk_index, file_index = file_assignment
        episode_rows.append(
            {
                "episode_index": episode_index,
                "dataset_from_index": cursor,
                "dataset_to_index": cursor + length,
                "data/chunk_index": chunk_index,
                "data/file_index": file_index,
            }
        )
        cursor += length

    features = {
        "episode_index": {"dtype": "int64", "shape": []},
        "index": {"dtype": "int64", "shape": []},
        "timestamp": {"dtype": "float32", "shape": []},
        "frame_index": {"dtype": "int64", "shape": []},
        "task_index": {"dtype": "int64", "shape": []},
        "observation.state": {"dtype": "float32", "shape": [2]},
        "action": {"dtype": "float32", "shape": [2]},
    }
    info = {
        "codebase_version": "v3.0",
        "total_frames": cursor,
        "features": features,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    }
    tasks = pd.DataFrame({"task_index": [0, 1]}, index=["pick", "place"])
    stats = {
        "observation.state": {"mean": [0.0, 0.0], "std": [1.0, 1.0], "min": [0.0, 0.0], "max": [1.0, 1.0]},
        "action": {"mean": [0.0, 0.0], "std": [1.0, 1.0], "min": [0.0, 0.0], "max": [1.0, 1.0]},
    }
    episodes = pd.DataFrame(episode_rows)
    return FakeMetadata(root=root, info=info, episodes=episodes, tasks=tasks, stats=stats, fps=10)
