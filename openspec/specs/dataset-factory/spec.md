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

#### Scenario: Explicit streaming backend rejects temporal features
- **WHEN** `backend="streaming"` is explicitly selected and `delta_timestamps` or `action_chunk_size` is configured
- **THEN** `create_dataloader()` SHALL raise a `ValueError` indicating temporal features require the `lazy` or `default` backend

### Requirement: Backend selection and guardrails
The factory SHALL use `DataConfig.backend` directly and validate unsupported feature/backend combinations.

#### Scenario: Default backend is used when backend is omitted
- **WHEN** `DataConfig` is constructed with only `repo_id` (and no explicit backend)
- **THEN** the factory SHALL use the `default` (LeRobotDataset) backend

#### Scenario: Explicit lazy backend selection
- **WHEN** `backend="lazy"` is configured
- **THEN** the factory SHALL use `LazyLeRobotDataset`

#### Scenario: Explicit streaming backend selection
- **WHEN** `backend="streaming"` is configured
- **THEN** the factory SHALL use `ShardInterleavedDataset`

#### Scenario: Default backend rejects action chunking
- **WHEN** `backend="default"` and `action_chunk_size` is configured
- **THEN** `create_dataloader()` SHALL raise a `ValueError` indicating action chunking requires the `lazy` backend

#### Scenario: Streaming backend rejects temporal features
- **WHEN** `backend="streaming"` and `delta_timestamps` or `action_chunk_size` is configured
- **THEN** `create_dataloader()` SHALL raise a `ValueError` indicating temporal features require the `lazy` or `default` backend

#### Scenario: Backend selection logging
- **WHEN** a backend is selected
- **THEN** the factory SHALL log the chosen backend and the reason for the selection

### Requirement: Transform pipeline wiring
The factory SHALL compose and attach the transform pipeline to the dataset based on `DataConfig` settings.

#### Scenario: Normalization enabled with default keys excludes camera keys
- **WHEN** `DataConfig.normalize=True`, dataset stats are available via `LeRobotDatasetMetadata.stats`, and `DataConfig.normalize_keys is None`
- **THEN** the factory SHALL include a `NormalizeTransform` in the pipeline configured with an explicit key list derived from stats keys that EXCLUDES all camera keys from dataset metadata

#### Scenario: Normalization enabled with explicit keys
- **WHEN** `DataConfig.normalize=True`, dataset stats are available via `LeRobotDatasetMetadata.stats`, and `DataConfig.normalize_keys` is explicitly provided
- **THEN** the factory SHALL include a `NormalizeTransform(keys=DataConfig.normalize_keys)` in the pipeline (even if that list includes camera/image keys)

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
- **THEN** `backend` SHALL default to `"default"`, `batch_size` to `32`, `num_workers` to `4`, `normalize` to `True`

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
