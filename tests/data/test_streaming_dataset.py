"""Unit tests for the streaming dataset backend."""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd
import pytest
import torch

from tests.data.helpers import make_fake_metadata, write_parquet_rows
from yavla.data.streaming import ShardInterleavedDataset


def _build_streaming_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ShardInterleavedDataset:
    root = tmp_path / "dataset"
    metadata = make_fake_metadata(
        root,
        episode_lengths=[2, 2, 2],
        file_assignments=[(0, 0), (0, 1), (0, 2)],
    )
    shard_rows = [
        [
            {
                "episode_index": episode_index,
                "index": episode_index * 2 + offset,
                "timestamp": float(offset) / 10.0,
                "frame_index": offset,
                "task_index": episode_index % 2,
                "observation.state": [float(episode_index), float(offset)],
                "action": [float(offset), float(offset + 1)],
            }
            for offset in range(2)
        ]
        for episode_index in range(3)
    ]

    for file_index, rows in enumerate(shard_rows):
        write_parquet_rows(root / f"data/chunk-000/file-{file_index:03d}.parquet", rows)

    monkeypatch.setattr("yavla.data.streaming.LeRobotDatasetMetadata", lambda repo_id, root=None: metadata)
    return ShardInterleavedDataset(
        repo_id="dummy/repo",
        root=root,
        shuffle_buffer_size=4,
        num_interleaved_shards=2,
        seed=7,
    )


def test_streaming_same_seed_same_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _build_streaming_fixture(tmp_path, monkeypatch)
    dataset.set_epoch(3)
    first = [sample["index"] for sample in itertools.islice(iter(dataset), 6)]
    dataset.set_epoch(3)
    second = [sample["index"] for sample in itertools.islice(iter(dataset), 6)]
    assert first == second


def test_streaming_rank_partitioning_no_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _build_streaming_fixture(tmp_path, monkeypatch)
    metadata = dataset.meta
    monkeypatch.setattr("yavla.data.streaming.LeRobotDatasetMetadata", lambda repo_id, root=None: metadata)

    rank0 = ShardInterleavedDataset(
        repo_id="dummy/repo",
        root=metadata.root,
        shuffle_buffer_size=4,
        num_interleaved_shards=2,
        seed=1,
        rank=0,
        world_size=2,
    )
    rank1 = ShardInterleavedDataset(
        repo_id="dummy/repo",
        root=metadata.root,
        shuffle_buffer_size=4,
        num_interleaved_shards=2,
        seed=1,
        rank=1,
        world_size=2,
    )
    rank0_indices = {sample["index"] for sample in rank0}
    rank1_indices = {sample["index"] for sample in rank1}
    assert rank0_indices.isdisjoint(rank1_indices)


def test_streaming_worker_assignment_round_robin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _build_streaming_fixture(tmp_path, monkeypatch)
    rank_local = [0, 4, 8, 12, 16]
    worker0 = dataset._assign_worker_shards(rank_local, worker_id=0, worker_count=2)
    worker1 = dataset._assign_worker_shards(rank_local, worker_id=1, worker_count=2)

    assert worker0 == [0, 8, 16]
    assert worker1 == [4, 12]
    assert set(worker0).isdisjoint(worker1)


def test_streaming_guardrails_for_unsupported_features() -> None:
    with pytest.raises(ValueError, match="delta_timestamps"):
        ShardInterleavedDataset(repo_id="dummy", delta_timestamps={"k": [0.1]})
    with pytest.raises(ValueError, match="action_chunk_size"):
        ShardInterleavedDataset(repo_id="dummy", action_chunk_size=4)


def test_streaming_epoch_changes_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _build_streaming_fixture(tmp_path, monkeypatch)
    dataset.set_epoch(0)
    epoch0 = [sample["index"] for sample in dataset]
    dataset.set_epoch(1)
    epoch1 = [sample["index"] for sample in dataset]
    assert epoch0 != epoch1


def test_streaming_decodes_video_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr("yavla.data.streaming.LeRobotDatasetMetadata", lambda repo_id, root=None: metadata)
    monkeypatch.setattr("yavla.data.streaming.decode_video_frames", _decode_video_frames)
    dataset = ShardInterleavedDataset(
        repo_id="dummy/repo",
        root=root,
        shuffle_buffer_size=1,
        num_interleaved_shards=1,
    )
    sample = next(iter(dataset))

    assert "observation.images.cam" in sample
    assert isinstance(sample["observation.images.cam"], torch.Tensor)
    assert len(calls) == 1


def test_streaming_decodes_video_key_without_embedded_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    metadata = make_fake_metadata(root, episode_lengths=[1], file_assignments=[(0, 0)])
    metadata.info["features"]["observation.images.cam"] = {"dtype": "video", "shape": []}
    rows = [
        {
            "episode_index": 0,
            "index": 0,
            "timestamp": 0.3,
            "frame_index": 3,
            "task_index": 0,
            "observation.state": [0.0, 0.0],
            "action": [0.0, 0.0],
            "observation.images.cam": {"path": "videos/cam.mp4"},
        }
    ]
    write_parquet_rows(root / "data/chunk-000/file-000.parquet", rows)

    calls: list[float] = []

    def _decode_video_frames(**kwargs):
        calls.extend(kwargs["timestamps"])
        return torch.zeros((1, 3, 4, 4))

    monkeypatch.setattr("yavla.data.streaming.LeRobotDatasetMetadata", lambda repo_id, root=None: metadata)
    monkeypatch.setattr("yavla.data.streaming.decode_video_frames", _decode_video_frames)
    dataset = ShardInterleavedDataset(
        repo_id="dummy/repo",
        root=root,
        shuffle_buffer_size=1,
        num_interleaved_shards=1,
    )
    sample = next(iter(dataset))

    assert isinstance(sample["observation.images.cam"], torch.Tensor)
    assert calls == [0.3]


def test_streaming_decodes_video_from_episode_metadata_when_row_has_no_media_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    metadata = make_fake_metadata(root, episode_lengths=[1], file_assignments=[(0, 0)])
    metadata.info["features"]["observation.images.cam"] = {"dtype": "video", "shape": []}
    metadata.info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    metadata.episodes = metadata.episodes.copy()
    metadata.episodes["videos/observation.images.cam/chunk_index"] = [0]
    metadata.episodes["videos/observation.images.cam/file_index"] = [3]
    metadata.episodes["videos/observation.images.cam/from_timestamp"] = [1.0]

    rows = [
        {
            "episode_index": 0,
            "index": 0,
            "timestamp": 0.3,
            "frame_index": 3,
            "task_index": 0,
            "observation.state": [0.0, 0.0],
            "action": [0.0, 0.0],
        }
    ]
    write_parquet_rows(root / "data/chunk-000/file-000.parquet", rows)

    calls: list[tuple[Path, list[float]]] = []

    def _decode_video_frames(**kwargs):
        calls.append((Path(kwargs["video_path"]), list(kwargs["timestamps"])))
        return torch.zeros((1, 3, 4, 4))

    monkeypatch.setattr("yavla.data.streaming.LeRobotDatasetMetadata", lambda repo_id, root=None: metadata)
    monkeypatch.setattr("yavla.data.streaming.decode_video_frames", _decode_video_frames)
    dataset = ShardInterleavedDataset(
        repo_id="dummy/repo",
        root=root,
        shuffle_buffer_size=1,
        num_interleaved_shards=1,
    )
    sample = next(iter(dataset))

    assert isinstance(sample["observation.images.cam"], torch.Tensor)
    assert len(calls) == 1
    assert calls[0][0] == root / "videos/observation.images.cam/chunk-000/file-003.mp4"
    assert calls[0][1] == pytest.approx([1.3])


def test_streaming_task_lookup_with_numeric_dataframe_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _build_streaming_fixture(tmp_path, monkeypatch)
    metadata = dataset.meta
    metadata.tasks = pd.DataFrame({"task_index": [0, 1], "task": ["pick", "place"]})
    monkeypatch.setattr("yavla.data.streaming.LeRobotDatasetMetadata", lambda repo_id, root=None: metadata)

    ds = ShardInterleavedDataset(
        repo_id="dummy/repo",
        root=metadata.root,
        shuffle_buffer_size=4,
        num_interleaved_shards=2,
    )
    assert next(iter(ds))["task"] in {"pick", "place"}
