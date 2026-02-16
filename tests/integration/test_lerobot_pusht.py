"""Integration tests against real ``lerobot/pusht`` data."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from lerobot.datasets.lerobot_dataset import (  # type: ignore[import-untyped]
    LeRobotDataset,
    LeRobotDatasetMetadata,
)

from yavla.data.factory import DataConfig, create_dataloader
from yavla.data.lazy import LazyLeRobotDataset
from yavla.data.schema import REQUIRED_METADATA_KEYS, validate_sample_schema
from yavla.data.streaming import ShardInterleavedDataset
from yavla.data.transforms import NormalizeTransform

REPO_ID = "lerobot/pusht"


def _meta(root: Path) -> LeRobotDatasetMetadata:
    return LeRobotDatasetMetadata(repo_id=REPO_ID, root=root)


def _non_video_feature_columns(metadata: LeRobotDatasetMetadata) -> list[str]:
    return [key for key, feature in metadata.features.items() if feature.get("dtype") not in {"video", "image"}]


def _sample_indices(length: int) -> list[int]:
    if length == 0:
        return []
    if length == 1:
        return [0]
    middle = length // 2
    return sorted({0, middle, length - 1})


def _camera_keys(metadata: LeRobotDatasetMetadata) -> list[str]:
    if hasattr(metadata, "camera_keys"):
        camera_keys = list(metadata.camera_keys)
        if camera_keys:
            return camera_keys

    return [key for key, feature in metadata.features.items() if feature.get("dtype") in {"video", "image"}]


def _iter_index_sequence(dataset: ShardInterleavedDataset, *, limit: int) -> list[int]:
    return [int(sample["index"]) for sample in itertools.islice(iter(dataset), limit)]


def _is_raw_media_payload(value: Any) -> bool:
    if isinstance(value, (str, Path)):
        return True
    if isinstance(value, Mapping):
        return "path" in value or "bytes" in value
    return False


def _is_decoded_media_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.ndim == 3 and value.dtype in {torch.float32, torch.uint8}


def _decoded_media_tensors(sample: Mapping[str, Any], camera_keys: Sequence[str]) -> dict[str, torch.Tensor]:
    decoded: dict[str, torch.Tensor] = {}
    for key in camera_keys:
        if key not in sample:
            continue
        value = sample[key]
        if _is_decoded_media_tensor(value):
            decoded[key] = value
    return decoded


def _baseline_has_decoded_media_tensor(root: Path, camera_keys: Sequence[str], *, limit: int = 256) -> bool:
    baseline = LeRobotDataset(repo_id=REPO_ID, root=root, video_backend="pyav")
    search_limit = min(len(baseline), limit)
    for idx in range(search_limit):
        sample = baseline[idx]
        if _decoded_media_tensors(sample, camera_keys):
            return True
    return False


def test_lazy_loads_full_dataset(pusht_root: Path) -> None:
    metadata = _meta(pusht_root)
    dataset = LazyLeRobotDataset(
        repo_id=REPO_ID,
        root=pusht_root,
        feature_columns=_non_video_feature_columns(metadata),
    )

    assert len(dataset) == int(metadata.total_frames)
    sample = dataset[0]
    validate_sample_schema(sample, require_task_name=False)


def test_lazy_episode_filtering(pusht_root: Path) -> None:
    metadata = _meta(pusht_root)
    full_dataset = LazyLeRobotDataset(
        repo_id=REPO_ID,
        root=pusht_root,
        feature_columns=_non_video_feature_columns(metadata),
    )
    filtered_dataset = LazyLeRobotDataset(
        repo_id=REPO_ID,
        root=pusht_root,
        episodes=[0],
        feature_columns=_non_video_feature_columns(metadata),
    )

    assert len(filtered_dataset) < len(full_dataset)
    for idx in _sample_indices(len(filtered_dataset)):
        assert int(filtered_dataset[idx]["episode_index"]) == 0


def test_lazy_delta_timestamps(pusht_root: Path) -> None:
    metadata = _meta(pusht_root)
    dataset = LazyLeRobotDataset(
        repo_id=REPO_ID,
        root=pusht_root,
        episodes=[0],
        feature_columns=_non_video_feature_columns(metadata),
        delta_timestamps={"observation.state": [-0.1, 0.0, 0.1]},
    )

    sample = dataset[0]
    state = torch.as_tensor(sample["observation.state"])
    state_is_pad = torch.as_tensor(sample["observation.state_is_pad"])

    assert state.ndim == 2
    assert state.shape[0] == 3
    assert state_is_pad.dtype == torch.bool
    assert tuple(state_is_pad.shape) == (3,)


def test_lazy_action_chunk(pusht_root: Path) -> None:
    metadata = _meta(pusht_root)
    dataset = LazyLeRobotDataset(
        repo_id=REPO_ID,
        root=pusht_root,
        episodes=[0],
        feature_columns=_non_video_feature_columns(metadata),
        action_chunk_size=4,
    )

    sample = dataset[len(dataset) - 1]
    action = torch.as_tensor(sample["action"])
    action_is_pad = torch.as_tensor(sample["action_is_pad"])

    assert action.ndim == 2
    assert action.shape[0] == 4
    assert action_is_pad.dtype == torch.bool
    assert tuple(action_is_pad.shape) == (4,)
    assert torch.any(action_is_pad).item() is True


def test_lazy_getitems_batched(pusht_root: Path) -> None:
    metadata = _meta(pusht_root)
    dataset = LazyLeRobotDataset(
        repo_id=REPO_ID,
        root=pusht_root,
        feature_columns=_non_video_feature_columns(metadata),
    )

    if len(dataset) < 3:
        pytest.skip(f"{REPO_ID} contains fewer than 3 frames")

    samples = dataset.__getitems__([0, 1, 2])
    assert len(samples) == 3
    assert [int(sample["index"]) for sample in samples] == [0, 1, 2]
    for sample in samples:
        validate_sample_schema(sample, require_task_name=False)


def test_lazy_decodes_video(pusht_root: Path) -> None:
    metadata = _meta(pusht_root)
    camera_keys = _camera_keys(metadata)
    if not camera_keys:
        pytest.skip(f"{REPO_ID} metadata did not expose camera/video keys")
    if not _baseline_has_decoded_media_tensor(pusht_root, camera_keys):
        pytest.skip("LeRobotDataset did not expose decoded 3-D media tensors in first 256 samples")

    dataset = LazyLeRobotDataset(repo_id=REPO_ID, root=pusht_root, episodes=[0], video_backend="pyav")

    search_limit = min(len(dataset), 256)
    saw_raw_payload = False
    for idx in range(search_limit):
        sample = dataset[idx]
        if _decoded_media_tensors(sample, camera_keys):
            return

        for key in camera_keys:
            value = sample.get(key)
            if _is_raw_media_payload(value):
                saw_raw_payload = True

    if saw_raw_payload:
        pytest.fail("lazy backend returned only raw media payloads in first 256 samples")
    pytest.fail("expected at least one decoded 3-D media tensor in lazy samples")


def test_streaming_yields_valid_samples(pusht_root: Path) -> None:
    metadata = _meta(pusht_root)
    dataset = ShardInterleavedDataset(
        repo_id=REPO_ID,
        root=pusht_root,
        feature_columns=_non_video_feature_columns(metadata),
        shuffle_buffer_size=256,
        num_interleaved_shards=4,
        seed=7,
    )

    samples = list(itertools.islice(iter(dataset), 8))
    assert samples
    for sample in samples:
        validate_sample_schema(sample, require_task_name=False)
        assert isinstance(sample.get("task"), str)
        assert sample["task"].strip() != ""


def test_streaming_epoch_reproducibility(pusht_root: Path) -> None:
    metadata = _meta(pusht_root)
    dataset = ShardInterleavedDataset(
        repo_id=REPO_ID,
        root=pusht_root,
        feature_columns=_non_video_feature_columns(metadata),
        shuffle_buffer_size=256,
        num_interleaved_shards=4,
        seed=11,
    )

    dataset.set_epoch(3)
    first = _iter_index_sequence(dataset, limit=20)
    dataset.set_epoch(3)
    second = _iter_index_sequence(dataset, limit=20)
    assert first == second


def test_streaming_decodes_video(pusht_root: Path) -> None:
    metadata = _meta(pusht_root)
    camera_keys = _camera_keys(metadata)
    if not camera_keys:
        pytest.skip(f"{REPO_ID} metadata did not expose camera/video keys")
    if not _baseline_has_decoded_media_tensor(pusht_root, camera_keys):
        pytest.skip("LeRobotDataset did not expose decoded 3-D media tensors in first 256 samples")

    dataset = ShardInterleavedDataset(
        repo_id=REPO_ID,
        root=pusht_root,
        shuffle_buffer_size=64,
        num_interleaved_shards=2,
        seed=0,
        video_backend="pyav",
    )

    saw_raw_payload = False
    for sample in itertools.islice(iter(dataset), 256):
        if _decoded_media_tensors(sample, camera_keys):
            return
        for key in camera_keys:
            value = sample.get(key)
            if _is_raw_media_payload(value):
                saw_raw_payload = True

    if saw_raw_payload:
        pytest.fail("streaming backend returned only raw media payloads in first 256 samples")
    pytest.fail("expected to observe at least one decoded 3-D media tensor in streaming samples")


def test_default_backend_loads(pusht_root: Path) -> None:
    dataset = LeRobotDataset(repo_id=REPO_ID, root=pusht_root, video_backend="pyav")

    assert len(dataset) > 0
    sample = dataset[0]
    validate_sample_schema(sample, require_task_name=False)


def test_default_backend_decodes_video(pusht_root: Path) -> None:
    metadata = _meta(pusht_root)
    camera_keys = _camera_keys(metadata)
    if not camera_keys:
        pytest.skip(f"{REPO_ID} metadata did not expose camera/video keys")

    dataset = LeRobotDataset(repo_id=REPO_ID, root=pusht_root, video_backend="pyav")
    search_limit = min(len(dataset), 256)
    saw_raw_payload = False
    for idx in range(search_limit):
        sample = dataset[idx]
        if _decoded_media_tensors(sample, camera_keys):
            return
        for key in camera_keys:
            value = sample.get(key)
            if _is_raw_media_payload(value):
                saw_raw_payload = True

    if saw_raw_payload:
        pytest.skip("LeRobotDataset exposed raw media payloads but no decoded 3-D tensors in first 256 samples")
    pytest.fail("expected at least one decoded 3-D media tensor in default backend samples")


def test_factory_creates_lazy_dataloader(pusht_root: Path) -> None:
    dataloader = create_dataloader(
        DataConfig(
            repo_id=REPO_ID,
            root=pusht_root,
            backend="lazy",
            batch_size=2,
            num_workers=0,
            normalize=False,
            feature_keys=["observation.state", "action"],
        )
    )

    batch = next(iter(dataloader))
    assert isinstance(batch, dict)
    for key in REQUIRED_METADATA_KEYS.union({"task_index"}):
        assert key in batch
    assert torch.as_tensor(batch["index"]).shape[0] == 2


def test_factory_creates_default_dataloader(pusht_root: Path) -> None:
    metadata = _meta(pusht_root)
    camera_keys = _camera_keys(metadata)
    if not camera_keys:
        pytest.skip(f"{REPO_ID} metadata did not expose camera/video keys")
    if not _baseline_has_decoded_media_tensor(pusht_root, camera_keys):
        pytest.skip("LeRobotDataset did not expose decoded 3-D media tensors in first 256 samples")

    dataloader = create_dataloader(
        DataConfig(
            repo_id=REPO_ID,
            root=pusht_root,
            backend="default",
            batch_size=2,
            num_workers=0,
            normalize=False,
        )
    )

    assert getattr(dataloader, "yavla_backend") == "default"
    batch = next(iter(dataloader))
    assert isinstance(batch, dict)
    for key in REQUIRED_METADATA_KEYS.union({"task_index"}):
        assert key in batch
    assert torch.as_tensor(batch["index"]).shape[0] == 2

    decoded: list[str] = []
    for key in camera_keys:
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 4 and value.dtype in {torch.float32, torch.uint8}:
            decoded.append(key)
    assert decoded, "expected at least one decoded 4-D batched media tensor in default dataloader batch"


def test_normalize_transform_on_real_data(pusht_root: Path) -> None:
    metadata = _meta(pusht_root)
    feature_columns = _non_video_feature_columns(metadata)
    normalize = NormalizeTransform(stats=metadata.stats, mode="z-score", keys=["observation.state"])

    raw_dataset = LazyLeRobotDataset(
        repo_id=REPO_ID,
        root=pusht_root,
        episodes=[0],
        feature_columns=feature_columns,
    )
    normalized_dataset = LazyLeRobotDataset(
        repo_id=REPO_ID,
        root=pusht_root,
        episodes=[0],
        feature_columns=feature_columns,
        transforms=normalize,
    )

    raw_sample = raw_dataset[0]
    normalized_sample = normalized_dataset[0]

    raw_state = torch.as_tensor(raw_sample["observation.state"], dtype=torch.float32)
    normalized_state = torch.as_tensor(normalized_sample["observation.state"], dtype=torch.float32)
    assert raw_state.shape == normalized_state.shape

    stats_entry = metadata.stats.get("observation.state")
    if stats_entry is None or "std" not in stats_entry:
        pytest.skip("observation.state stats unavailable")

    std = torch.as_tensor(stats_entry["std"], dtype=torch.float32)
    if torch.all(std == 0):
        pytest.skip("observation.state std is zero everywhere")

    assert not torch.allclose(raw_state, normalized_state)
