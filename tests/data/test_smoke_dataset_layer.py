"""Smoke tests for dataset-layer integration points."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.data.helpers import make_fake_metadata, write_parquet_rows
from yavla.data.factory import DataConfig, create_dataloader
from yavla.data.lazy import LazyLeRobotDataset


def test_smoke_lazy_metadata_only_large_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "dataset"
    metadata = make_fake_metadata(
        root,
        episode_lengths=[1_000_000, 1_000_000],
        file_assignments=[(0, 0), (0, 1)],
    )
    metadata.info["total_frames"] = 2_000_000
    write_parquet_rows(
        root / "data/chunk-000/file-000.parquet",
        [
            {
                "episode_index": 0,
                "index": 0,
                "timestamp": 0.0,
                "frame_index": 0,
                "task_index": 0,
                "observation.state": [0.0, 0.0],
                "action": [0.0, 0.0],
            }
        ],
    )
    write_parquet_rows(
        root / "data/chunk-000/file-001.parquet",
        [
            {
                "episode_index": 1,
                "index": 1_000_000,
                "timestamp": 0.0,
                "frame_index": 0,
                "task_index": 1,
                "observation.state": [0.0, 0.0],
                "action": [0.0, 0.0],
            }
        ],
    )

    monkeypatch.setattr("yavla.data.lazy.LeRobotDatasetMetadata", lambda repo_id, root=None: metadata)
    dataset = LazyLeRobotDataset(repo_id="dummy/repo", root=root)
    assert len(dataset._file_boundaries) == 3


def test_smoke_create_dataloader_default_and_lazy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "dataset"
    metadata = make_fake_metadata(root, episode_lengths=[4], file_assignments=[(0, 0)])
    rows = [
        {
            "episode_index": 0,
            "index": index,
            "timestamp": 0.1 * index,
            "frame_index": index,
            "task_index": 0,
            "observation.state": [float(index), float(index)],
            "action": [float(index), float(index)],
        }
        for index in range(4)
    ]
    write_parquet_rows(root / "data/chunk-000/file-000.parquet", rows)

    monkeypatch.setattr("yavla.data.factory.LeRobotDatasetMetadata", lambda repo_id, root=None: metadata)
    monkeypatch.setattr(
        "yavla.data.factory.LeRobotDataset",
        lambda *args, **kwargs: LazyLeRobotDataset(repo_id="dummy"),
    )
    monkeypatch.setattr(
        "yavla.data.factory.LazyLeRobotDataset",
        lambda *args, **kwargs: LazyLeRobotDataset(repo_id="dummy"),
    )

    monkeypatch.setattr("yavla.data.lazy.LeRobotDatasetMetadata", lambda repo_id, root=None: metadata)

    default_loader = create_dataloader(DataConfig(repo_id="dummy/repo", root=root, backend="default", num_workers=0))
    lazy_loader = create_dataloader(DataConfig(repo_id="dummy/repo", root=root, backend="lazy", num_workers=0))

    assert getattr(default_loader, "yavla_backend") == "default"
    assert getattr(lazy_loader, "yavla_backend") == "lazy"
