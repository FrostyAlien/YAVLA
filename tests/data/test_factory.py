"""Unit tests for dataset factory selection and wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from torch.utils.data import Dataset, IterableDataset

from tests.data.helpers import make_fake_metadata, write_parquet_rows
from yavla.data.factory import DataConfig, create_dataloader, plan_feature_columns, select_backend


class _DummyMapDataset(Dataset[dict[str, Any]]):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "episode_index": 0,
            "index": index,
            "timestamp": 0.1 * index,
            "frame_index": index,
            "task_index": 0,
            "action": [0.0, 1.0],
            "observation.state": [0.0, 1.0],
        }


class _DummyLazyDataset(_DummyMapDataset):
    pass


class _DummyStreamingDataset(IterableDataset[dict[str, Any]]):
    def __iter__(self):
        for index in range(4):
            yield {
                "episode_index": 0,
                "index": index,
                "timestamp": 0.1 * index,
                "frame_index": index,
                "task_index": 0,
                "action": [0.0, 1.0],
                "observation.state": [0.0, 1.0],
            }


def _patch_factory_dependencies(monkeypatch: pytest.MonkeyPatch, metadata: Any) -> None:
    monkeypatch.setattr("yavla.data.factory.LeRobotDatasetMetadata", lambda repo_id, root=None: metadata)
    monkeypatch.setattr("yavla.data.factory.LeRobotDataset", lambda *args, **kwargs: _DummyMapDataset())
    monkeypatch.setattr("yavla.data.factory.LazyLeRobotDataset", lambda *args, **kwargs: _DummyLazyDataset())
    monkeypatch.setattr("yavla.data.factory.ShardInterleavedDataset", lambda *args, **kwargs: _DummyStreamingDataset())


def _metadata_with_local_data(tmp_path: Path) -> Any:
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
    return metadata


def test_select_backend_defaults_to_default() -> None:
    config = DataConfig(repo_id="dummy/repo")
    selection = select_backend(config)
    assert selection.backend == "default"


def test_select_backend_explicit_lazy() -> None:
    config = DataConfig(repo_id="dummy/repo", backend="lazy")
    selection = select_backend(config)
    assert selection.backend == "lazy"


def test_select_backend_explicit_streaming() -> None:
    config = DataConfig(repo_id="dummy/repo", backend="streaming")
    selection = select_backend(config)
    assert selection.backend == "streaming"


def test_select_backend_default_rejects_action_chunk_size() -> None:
    with pytest.raises(ValueError, match="default backend does not support action_chunk_size"):
        select_backend(DataConfig(repo_id="dummy/repo", action_chunk_size=2))


def test_create_dataloader_explicit_modes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    metadata = _metadata_with_local_data(tmp_path)
    _patch_factory_dependencies(monkeypatch, metadata)

    config_default = DataConfig(repo_id="dummy/repo", root=metadata.root, backend="default", num_workers=0)
    config_lazy = DataConfig(repo_id="dummy/repo", root=metadata.root, backend="lazy", num_workers=0)
    config_streaming = DataConfig(repo_id="dummy/repo", root=metadata.root, backend="streaming", num_workers=0)

    loader_default = create_dataloader(config_default)
    loader_lazy = create_dataloader(config_lazy)
    loader_streaming = create_dataloader(config_streaming)

    assert getattr(loader_default, "yavla_backend") == "default"
    assert getattr(loader_lazy, "yavla_backend") == "lazy"
    assert getattr(loader_streaming, "yavla_backend") == "streaming"


def test_create_dataloader_streaming_rejects_temporal_features(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    metadata = _metadata_with_local_data(tmp_path)
    _patch_factory_dependencies(monkeypatch, metadata)

    with pytest.raises(ValueError, match="delta_timestamps"):
        create_dataloader(
            DataConfig(
                repo_id="dummy/repo",
                root=metadata.root,
                backend="streaming",
                delta_timestamps={"observation.state": [-0.1, 0.0]},
                num_workers=0,
            )
        )


def test_create_dataloader_streaming_rejects_action_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    metadata = _metadata_with_local_data(tmp_path)
    _patch_factory_dependencies(monkeypatch, metadata)

    with pytest.raises(ValueError, match="action_chunk_size"):
        create_dataloader(
            DataConfig(
                repo_id="dummy/repo",
                root=metadata.root,
                backend="streaming",
                action_chunk_size=2,
                num_workers=0,
            )
        )


def test_create_dataloader_default_rejects_action_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    metadata = _metadata_with_local_data(tmp_path)
    _patch_factory_dependencies(monkeypatch, metadata)

    with pytest.raises(ValueError, match="default backend does not support action_chunk_size"):
        create_dataloader(
            DataConfig(
                repo_id="dummy/repo",
                root=metadata.root,
                backend="default",
                action_chunk_size=2,
                num_workers=0,
            )
        )


def test_dataloader_config_passthrough(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    metadata = _metadata_with_local_data(tmp_path)
    _patch_factory_dependencies(monkeypatch, metadata)

    config = DataConfig(
        repo_id="dummy/repo",
        root=metadata.root,
        backend="lazy",
        batch_size=64,
        num_workers=0,
    )
    dataloader = create_dataloader(config)
    assert dataloader.batch_size == 64


def test_plan_feature_columns_respects_config(tmp_path: Path) -> None:
    metadata = _metadata_with_local_data(tmp_path)
    config = DataConfig(
        repo_id="dummy/repo",
        root=metadata.root,
        normalize=False,
        repack_keys={"observation.state": "state"},
        delta_timestamps={"observation.state": [-0.1, 0.0]},
        action_chunk_size=3,
    )
    columns = plan_feature_columns(config, metadata)
    assert columns == ["action", "observation.state"]
