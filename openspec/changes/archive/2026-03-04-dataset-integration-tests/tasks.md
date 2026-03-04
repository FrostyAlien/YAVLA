## 1. Project Configuration

- [x] 1.1 Register `integration` marker in `pyproject.toml` under `[tool.pytest.ini_options]` with `markers = ["integration: marks tests that require external data (deselect with '-m not integration')"]`
- [x] 1.2 Update `addopts` in `pyproject.toml` to include `-m "not integration"` so the default `pytest` run excludes integration tests

## 2. Test Infrastructure

- [x] 2.1 Create `tests/integration/__init__.py`
- [x] 2.2 Create `tests/integration/conftest.py` with a session-scoped `pusht_root` fixture that uses `LeRobotDataset(repo_id="lerobot/pusht", root=~/.cache/yavla-test-data/)` for first-time download, returns the resolved cached dataset root `Path` on subsequent runs, and calls `pytest.skip` with a clear reason when first-time download fails
- [x] 2.3 Add a `pytest_collection_modifyitems` hook in `conftest.py` to auto-apply `pytest.mark.integration` to all tests under `tests/integration/`

## 3. Lazy Backend Tests

- [x] 3.1 Write `test_lazy_loads_full_dataset` — construct `LazyLeRobotDataset` with cached root and non-video `feature_columns`, assert `len(dataset)` matches metadata `total_frames`, assert `dataset[0]` passes `validate_sample_schema`
- [x] 3.2 Write `test_lazy_episode_filtering` — construct with `episodes=[0]` and non-video `feature_columns`, assert `len(dataset)` < full length, assert sampled indices from the filtered dataset have `episode_index == 0`
- [x] 3.3 Write `test_lazy_delta_timestamps` — construct with `delta_timestamps={"observation.state": [-0.1, 0.0, 0.1]}`, `episodes=[0]`, and non-video `feature_columns`; assert `observation.state` is 2-D with first dim 3, assert `observation.state_is_pad` is bool tensor of shape `(3,)`
- [x] 3.4 Write `test_lazy_action_chunk` — construct with `action_chunk_size=4`, `episodes=[0]`, and non-video `feature_columns`; access last frame, assert `action` shape `(4, action_dim)`, assert `action_is_pad` has at least one `True`
- [x] 3.5 Write `test_lazy_getitems_batched` — call `dataset.__getitems__([0, 1, 2])` on a non-video configuration, assert 3 dicts returned, each passing `validate_sample_schema`
- [x] 3.6 Write `test_lazy_decodes_video` — construct with cached root, access a sample with a video key, assert value is a 3-D `torch.Tensor` (C, H, W)

## 4. Streaming Backend Tests

- [x] 4.1 Write `test_streaming_yields_valid_samples` — construct `ShardInterleavedDataset` with cached root and non-video `feature_columns`, iterate for a limited count, assert each sample passes `validate_sample_schema` and has a non-empty `task` string
- [x] 4.2 Write `test_streaming_epoch_reproducibility` — iterate with same seed/epoch twice on a non-video configuration, assert `index` sequences match
- [x] 4.3 Write `test_streaming_decodes_video` — construct with cached root, iterate to get a sample with a video key, assert value is a 3-D `torch.Tensor`

## 5. Default Backend & Factory Tests

- [x] 5.1 Write `test_default_backend_loads` — construct `LeRobotDataset` with cached root, assert `len(dataset) > 0`, assert `dataset[0]` passes `validate_sample_schema`
- [x] 5.2 Write `test_factory_creates_dataloader` — call `create_dataloader` with `DataConfig(repo_id="lerobot/pusht", root=cached_root, batch_size=2, backend="lazy", num_workers=0, normalize=False, feature_keys=["observation.state", "action"])`; assert at least one batch is yielded, required metadata keys exist in the collated batch dict, and batch dimension is 2

## 6. Transform Pipeline Test

- [x] 6.1 Write `test_normalize_transform_on_real_data` — construct raw and normalized `LazyLeRobotDataset` instances (same `episodes=[0]`, non-video `feature_columns`) using `NormalizeTransform` with dataset stats; assert both samples load successfully with matching shapes and normalized `observation.state` differs from raw when `std` contains non-zero entries

## 7. Verification

- [x] 7.1 Run `pytest -m integration` and verify integration suite is selectable (tests pass when dataset is available; otherwise skip with clear download-failure reason)
- [x] 7.2 Run `pytest` (default) and verify no integration tests execute
