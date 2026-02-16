## ADDED Requirements

### Requirement: create_dataloader factory function
`create_dataloader()` SHALL accept a `DataConfig` dataclass and return a fully configured `torch.utils.data.DataLoader` with the appropriate dataset backend and transforms wired up.

#### Scenario: Explicit backend selection
- **WHEN** Something like `DataConfig(repo_id="lerobot/aloha_sim", backend="lazy")` is passed to `create_dataloader()`
- **THEN** it SHALL return a `DataLoader` wrapping a `LazyLeRobotDataset` instance

#### Scenario: Default backend selection
- **WHEN** Something like `DataConfig(repo_id="lerobot/aloha_sim", backend="default")` is passed
- **THEN** it SHALL return a `DataLoader` wrapping a standard `LeRobotDataset` instance

#### Scenario: Streaming backend selection
- **WHEN** Something like `DataConfig(repo_id="lerobot/aloha_sim", backend="streaming")` is passed
- **THEN** it SHALL return a `DataLoader` wrapping a `ShardInterleavedDataset` instance

### Requirement: Auto backend selection
When `backend="auto"`, the factory SHALL select the backend based on dataset characteristics and configured features.

#### Scenario: Small dataset selects default backend
- **WHEN** `backend="auto"` and the estimated uncompressed tabular data size (computed from non-video entries in `info["features"]` schema: dtypes × shapes × `total_frames`, with video features treated as path/metadata) is below the configured threshold (default 50GB)
- **THEN** the factory SHALL select the `default` (LeRobotDataset) backend

#### Scenario: Large dataset selects lazy backend
- **WHEN** `backend="auto"` and the estimated uncompressed data size exceeds the threshold
- **THEN** the factory SHALL select the `lazy` backend

#### Scenario: Data not locally available selects streaming backend
- **WHEN** `backend="auto"` and the dataset is not cached locally (Parquet files not present on disk) and neither `delta_timestamps` nor `action_chunk_size` is configured
- **THEN** the factory SHALL select the `streaming` backend

#### Scenario: v1 simplification excludes streaming in distributed auto mode
- **WHEN** `backend="auto"`, distributed training is active, and selecting `streaming` would otherwise be valid
- **THEN** the factory SHALL select `lazy` or `default` instead, and SHALL log `SC-001` as the selection reason

#### Scenario: delta_timestamps or action_chunk_size excludes streaming
- **WHEN** `backend="auto"` and `delta_timestamps` or `action_chunk_size` is configured
- **THEN** the factory SHALL NOT select the `streaming` backend, even if data is not locally available (fall back to `lazy` which will trigger download)

#### Scenario: Backend selection logging
- **WHEN** `backend="auto"` and a backend is selected
- **THEN** the factory SHALL log the chosen backend and the reason for the selection

### Requirement: Transform pipeline wiring
The factory SHALL compose and attach the transform pipeline to the dataset based on `DataConfig` settings.

#### Scenario: Normalization enabled
- **WHEN** `DataConfig.normalize=True`
- **THEN** the factory SHALL load stats from `LeRobotDatasetMetadata.stats` and include a `NormalizeTransform` in the pipeline

#### Scenario: Normalization disabled
- **WHEN** `DataConfig.normalize=False`
- **THEN** no `NormalizeTransform` SHALL be included in the pipeline

#### Scenario: Custom repack mapping
- **WHEN** `DataConfig.repack_keys` specifies a key mapping
- **THEN** a `RepackTransform` with that mapping SHALL be prepended to the pipeline

#### Scenario: Image transforms configured
- **WHEN** `DataConfig.image_transforms` specifies transform names
- **THEN** an `ImageTransform` with the corresponding torchvision v2 transforms SHALL be included in the pipeline, applied to all camera keys from dataset metadata

### Requirement: DataConfig dataclass with tyro compatibility
`DataConfig` SHALL be a Python dataclass with typed fields and default values, compatible with tyro for CLI and YAML configuration.

#### Scenario: CLI override
- **WHEN** a training script uses `tyro.cli(DataConfig)` and the user passes `--backend lazy --batch-size 64`
- **THEN** the resulting `DataConfig` SHALL have `backend="lazy"` and `batch_size=64` with all other fields at defaults

#### Scenario: YAML configuration
- **WHEN** a YAML config file specifies `repo_id: lerobot/droid` and `backend: lazy`
- **THEN** `DataConfig` SHALL be constructable from the parsed YAML dict

#### Scenario: Default values
- **WHEN** `DataConfig` is constructed with only `repo_id`
- **THEN** `backend` SHALL default to `"auto"`, `batch_size` to `32`, `num_workers` to `4`, `normalize` to `True`

### Requirement: DistributedSampler integration
The factory SHALL configure the appropriate sampler for distributed training when a distributed process group is active.

#### Scenario: Map-style dataset with DDP
- **WHEN** `torch.distributed.is_initialized()` is `True` and the backend is `default` or `lazy`
- **THEN** the factory SHALL wrap the dataset with `DistributedSampler` and pass it to the `DataLoader`

#### Scenario: Streaming dataset with DDP
- **WHEN** `torch.distributed.is_initialized()` is `True` and the backend is `streaming`
- **THEN** the factory SHALL configure `ShardInterleavedDataset` with the current rank and world size for shard partitioning (no separate `DistributedSampler` needed)

### Requirement: DataLoader configuration passthrough
The factory SHALL pass through relevant `DataConfig` fields to the `DataLoader` constructor.

#### Scenario: Worker and batch configuration
- **WHEN** `DataConfig(batch_size=64, num_workers=8)` is provided
- **THEN** the returned `DataLoader` SHALL have `batch_size=64` and `num_workers=8`

#### Scenario: Pin memory for GPU training
- **WHEN** CUDA is available
- **THEN** the `DataLoader` SHALL be configured with `pin_memory=True`
