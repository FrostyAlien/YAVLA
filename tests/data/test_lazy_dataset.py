"""Unit tests for the lazy map-style dataset backend."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from tests.data.helpers import make_fake_metadata, write_parquet_rows
from yavla.data.lazy import LazyLeRobotDataset


def _build_lazy_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[LazyLeRobotDataset, Path]:
    root = tmp_path / "dataset"
    metadata = make_fake_metadata(
        root,
        episode_lengths=[3, 2],
        file_assignments=[(0, 0), (0, 1)],
    )

    shard0 = [
        {
            "episode_index": 0,
            "index": 0,
            "timestamp": 0.0,
            "frame_index": 0,
            "task_index": 0,
            "observation.state": [1.0, 1.5],
            "action": [0.0, 0.5],
        },
        {
            "episode_index": 0,
            "index": 1,
            "timestamp": 0.1,
            "frame_index": 1,
            "task_index": 0,
            "observation.state": [2.0, 2.5],
            "action": [1.0, 1.5],
        },
        {
            "episode_index": 0,
            "index": 2,
            "timestamp": 0.2,
            "frame_index": 2,
            "task_index": 0,
            "observation.state": [3.0, 3.5],
            "action": [2.0, 2.5],
        },
    ]
    shard1 = [
        {
            "episode_index": 1,
            "index": 3,
            "timestamp": 0.0,
            "frame_index": 0,
            "task_index": 1,
            "observation.state": [4.0, 4.5],
            "action": [3.0, 3.5],
        },
        {
            "episode_index": 1,
            "index": 4,
            "timestamp": 0.1,
            "frame_index": 1,
            "task_index": 1,
            "observation.state": [5.0, 5.5],
            "action": [4.0, 4.5],
        },
    ]
    write_parquet_rows(root / "data/chunk-000/file-000.parquet", shard0, row_group_size=2)
    write_parquet_rows(root / "data/chunk-000/file-001.parquet", shard1, row_group_size=2)

    monkeypatch.setattr("yavla.data.lazy.LeRobotDatasetMetadata", lambda repo_id, root=None: metadata)
    dataset = LazyLeRobotDataset(repo_id="dummy/repo", root=root, parquet_cache_size=2, video_backend="pyav")
    return dataset, root


def test_lazy_dataset_len_and_item(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, _ = _build_lazy_fixture(tmp_path, monkeypatch)
    assert len(dataset) == 5
    sample = dataset[0]
    assert sample["episode_index"] == 0
    assert sample["index"] == 0
    assert sample["task"] == "pick"


def test_lazy_dataset_delta_timestamps_padding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, root = _build_lazy_fixture(tmp_path, monkeypatch)
    metadata = make_fake_metadata(root, episode_lengths=[3, 2], file_assignments=[(0, 0), (0, 1)])
    monkeypatch.setattr("yavla.data.lazy.LeRobotDatasetMetadata", lambda repo_id, root=None: metadata)

    dataset = LazyLeRobotDataset(
        repo_id="dummy/repo",
        root=root,
        delta_timestamps={"observation.state": [-0.1, 0.0, 0.1]},
    )
    sample = dataset[0]
    assert torch.equal(sample["observation.state_is_pad"], torch.tensor([True, False, False]))


def test_lazy_dataset_action_chunk_padding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, _ = _build_lazy_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr("yavla.data.lazy.LeRobotDatasetMetadata", lambda repo_id, root=None: dataset.meta)
    chunked = LazyLeRobotDataset(
        repo_id="dummy/repo",
        root=dataset.meta.root,
        action_chunk_size=3,
    )
    sample = chunked[4]
    assert sample["action"].shape == (3, 2)
    assert torch.equal(sample["action_is_pad"], torch.tensor([False, True, True]))


def test_lazy_dataset_getitems_batched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, _ = _build_lazy_fixture(tmp_path, monkeypatch)
    samples = dataset.__getitems__([1, 3, 4])
    assert [sample["index"] for sample in samples] == [1, 3, 4]


def test_lazy_dataset_episode_filtering_across_non_contiguous_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    metadata = make_fake_metadata(
        root,
        episode_lengths=[2, 2, 2],
        file_assignments=[(0, 0), (0, 0), (0, 0)],
    )
    rows = []
    for index in range(6):
        rows.append(
            {
                "episode_index": index // 2,
                "index": index,
                "timestamp": float(index % 2) / 10.0,
                "frame_index": index % 2,
                "task_index": 0,
                "observation.state": [float(index), float(index)],
                "action": [float(index), float(index)],
            }
        )
    write_parquet_rows(root / "data/chunk-000/file-000.parquet", rows, row_group_size=2)

    monkeypatch.setattr("yavla.data.lazy.LeRobotDatasetMetadata", lambda repo_id, root=None: metadata)
    dataset = LazyLeRobotDataset(repo_id="dummy/repo", root=root, episodes=[0, 2])
    assert [dataset[i]["index"] for i in range(len(dataset))] == [0, 1, 4, 5]


def test_lazy_dataset_decodes_video_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "dataset"
    metadata = make_fake_metadata(root, episode_lengths=[1], file_assignments=[(0, 0)])
    metadata.info["features"]["observation.images.cam"] = {"dtype": "video", "shape": []}
    rows = [
        {
            "episode_index": 0,
            "index": 0,
            "timestamp": 0.0,
            "frame_index": 0,
            "task_index": 0,
            "observation.state": [0.0, 0.0],
            "action": [0.0, 0.0],
            "observation.images.cam": {"path": "videos/cam.mp4", "timestamp": 0.0},
        }
    ]
    write_parquet_rows(root / "data/chunk-000/file-000.parquet", rows)

    calls: list[Path] = []

    def _decode_video_frames(**kwargs):
        calls.append(Path(kwargs["video_path"]))
        return torch.zeros((1, 3, 4, 4))

    monkeypatch.setattr("yavla.data.lazy.LeRobotDatasetMetadata", lambda repo_id, root=None: metadata)
    monkeypatch.setattr("yavla.data.lazy.decode_video_frames", _decode_video_frames)
    dataset = LazyLeRobotDataset(repo_id="dummy/repo", root=root)
    sample = dataset[0]

    assert "observation.images.cam" in sample
    assert isinstance(sample["observation.images.cam"], torch.Tensor)
    assert len(calls) == 1


def test_lazy_dataset_task_lookup_with_numeric_dataframe_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, root = _build_lazy_fixture(tmp_path, monkeypatch)
    metadata = dataset.meta
    metadata.tasks = pd.DataFrame({"task_index": [0, 1], "task": ["pick", "place"]})
    monkeypatch.setattr("yavla.data.lazy.LeRobotDatasetMetadata", lambda repo_id, root=None: metadata)

    ds = LazyLeRobotDataset(repo_id="dummy/repo", root=root)
    assert ds[0]["task"] == "pick"
