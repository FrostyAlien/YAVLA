## ADDED Requirements

### Requirement: Integration test marker registration
The project SHALL register a `integration` pytest marker in `pyproject.toml` so that integration tests are excluded from the default `pytest` invocation and only run when explicitly selected via `-m integration`.

#### Scenario: Default pytest run excludes integration tests
- **WHEN** a developer runs `pytest` without any marker flags
- **THEN** no tests marked with `@pytest.mark.integration` SHALL execute

#### Scenario: Explicit marker selection runs integration tests
- **WHEN** a developer runs `pytest -m integration`
- **THEN** all tests marked with `@pytest.mark.integration` SHALL execute

### Requirement: Integration tests are consistently marked
All tests under `tests/integration/` SHALL be marked as `integration` through centralized test-collection logic so marker behavior does not rely on per-test decorators.

#### Scenario: Collection-time marker application
- **WHEN** pytest collects tests under `tests/integration/`
- **THEN** those tests SHALL be marked with `integration` before marker filtering is applied

### Requirement: Session-scoped dataset download fixture
The test infrastructure SHALL provide a session-scoped pytest fixture that downloads `lerobot/pusht` from HuggingFace Hub and caches it under `~/.cache/yavla-test-data/`. Subsequent test runs MUST reuse the cached copy without re-downloading.

#### Scenario: First run downloads and caches the dataset
- **WHEN** the integration test suite runs for the first time and no local cache exists
- **THEN** the fixture SHALL download `lerobot/pusht` from HuggingFace Hub under `~/.cache/yavla-test-data/` and return the resolved dataset root path for test construction

#### Scenario: Subsequent runs use cached data
- **WHEN** the integration test suite runs and a usable cached dataset root already exists under `~/.cache/yavla-test-data/`
- **THEN** the fixture SHALL use the cached data without making network requests to HuggingFace Hub

#### Scenario: First-run download failure skips integration tests
- **WHEN** the fixture cannot download `lerobot/pusht` because network access or Hub availability is unavailable
- **THEN** integration tests SHALL be skipped with a clear reason message

### Requirement: LazyLeRobotDataset loads real data end-to-end
The integration tests SHALL verify that `LazyLeRobotDataset` can load `lerobot/pusht` without monkeypatching, producing samples that pass `validate_sample_schema`.

#### Scenario: Lazy backend loads full dataset
- **WHEN** `LazyLeRobotDataset` is constructed with `repo_id="lerobot/pusht"` pointing at the cached root
- **THEN** `len(dataset)` SHALL equal the total frame count reported in the dataset metadata and `dataset[0]` SHALL return a dict that passes `validate_sample_schema`

#### Scenario: Lazy backend with episode filtering
- **WHEN** `LazyLeRobotDataset` is constructed with `episodes=[0]`
- **THEN** `len(dataset)` SHALL be less than the full dataset length and every sample's `episode_index` SHALL equal `0`

### Requirement: LazyLeRobotDataset supports delta_timestamps with real data
The integration tests SHALL verify that temporal context queries work against real Parquet data, producing correctly shaped and padded tensors.

#### Scenario: Delta timestamps produce stacked tensors with pad masks
- **WHEN** `LazyLeRobotDataset` is constructed with `delta_timestamps={"observation.state": [-0.1, 0.0, 0.1]}` and `episodes=[0]`
- **THEN** `sample["observation.state"]` SHALL be a 2-D tensor with first dimension equal to 3 (the number of delta offsets) and `sample["observation.state_is_pad"]` SHALL be a boolean tensor of shape `(3,)`

### Requirement: LazyLeRobotDataset supports action_chunk_size with real data
The integration tests SHALL verify that action chunking works against real Parquet data, producing correctly shaped and padded action tensors.

#### Scenario: Action chunk produces stacked actions with pad mask
- **WHEN** `LazyLeRobotDataset` is constructed with `action_chunk_size=4` and `episodes=[0]`, and the last frame of the episode is accessed
- **THEN** `sample["action"]` SHALL be a 2-D tensor with first dimension equal to 4 and `sample["action_is_pad"]` SHALL be a boolean tensor of shape `(4,)` with at least one `True` entry for the padded positions

### Requirement: ShardInterleavedDataset loads real data end-to-end
The integration tests SHALL verify that `ShardInterleavedDataset` can iterate over `lerobot/pusht` without monkeypatching, producing valid samples.

#### Scenario: Streaming backend yields valid samples
- **WHEN** `ShardInterleavedDataset` is constructed with `repo_id="lerobot/pusht"` pointing at the cached root and iterated for a limited number of samples
- **THEN** each yielded sample SHALL pass `validate_sample_schema` and contain a `task` field with a non-empty string value

#### Scenario: Streaming backend epoch reproducibility
- **WHEN** `ShardInterleavedDataset` iterates with the same seed and epoch twice
- **THEN** both iterations SHALL produce samples in the same order (verified by comparing `index` values)

### Requirement: Default LeRobotDataset backend loads real data
The integration tests SHALL verify that the default `LeRobotDataset` wrapper (from lerobot) can load `lerobot/pusht`, producing samples that pass schema validation.

#### Scenario: Default backend loads and returns valid samples
- **WHEN** `LeRobotDataset` is constructed with `repo_id="lerobot/pusht"` pointing at the cached root
- **THEN** `len(dataset)` SHALL be greater than zero and `dataset[0]` SHALL pass `validate_sample_schema`

### Requirement: Factory path creates a working DataLoader
The integration tests SHALL verify that `create_dataloader` with a `DataConfig` pointing at real data produces a functional `DataLoader` that yields valid batches.

#### Scenario: Factory produces iterable DataLoader
- **WHEN** `create_dataloader` is called with a `DataConfig` using `repo_id="lerobot/pusht"`, the cached root, `backend="lazy"`, `batch_size=2`, and `num_workers=0`
- **THEN** the returned `DataLoader` SHALL yield at least one collated batch dict containing required metadata keys (`episode_index`, `index`, `timestamp`, `frame_index`, `task_index`) with leading batch dimension `2`

### Requirement: Transform pipeline runs on real data
The integration tests SHALL verify that the transform pipeline (normalize, image transforms) executes without error on real dataset samples.

#### Scenario: Transforms applied to lazy dataset samples
- **WHEN** raw and normalized `LazyLeRobotDataset` instances are constructed for the same sample index using `NormalizeTransform` with dataset stats and `episodes=[0]`
- **THEN** both samples SHALL load successfully with matching `observation.state` shape, and normalized `observation.state` SHALL differ from raw values when the corresponding `std` has at least one non-zero element

### Requirement: Video frame decoding with real data
The integration tests SHALL verify that video keys in `lerobot/pusht` are decoded into image tensors by both the lazy and streaming backends.

#### Scenario: Lazy backend decodes video frames
- **WHEN** `LazyLeRobotDataset` is constructed with the cached `lerobot/pusht` root and a sample containing a video key is accessed
- **THEN** the video key's value SHALL be a `torch.Tensor` with 3 dimensions (C, H, W) and dtype `torch.float32` or `torch.uint8`

#### Scenario: Streaming backend decodes video frames
- **WHEN** `ShardInterleavedDataset` is constructed with the cached `lerobot/pusht` root and iterated to yield a sample containing a video key
- **THEN** the video key's value SHALL be a `torch.Tensor` with 3 dimensions (C, H, W) and dtype `torch.float32` or `torch.uint8`

### Requirement: Batched access via __getitems__
The integration tests SHALL verify that `LazyLeRobotDataset.__getitems__` works correctly with real data for DataLoader batch collation.

#### Scenario: Batched index access returns correct samples
- **WHEN** `dataset.__getitems__([0, 1, 2])` is called on a `LazyLeRobotDataset` loaded with real data
- **THEN** the result SHALL be a list of 3 dicts, each passing `validate_sample_schema`, with `index` values matching the requested indices
