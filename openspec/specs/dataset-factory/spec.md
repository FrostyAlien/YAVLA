# dataset-factory Specification

## Purpose
Define expected behavior of `create_dataloader()` and dataset-factory guardrails: backend selection, transform wiring, and `DataLoader` configuration for LeRobot-format datasets.

## Requirements

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

When `backend="default"` and `action_chunk_size` is configured, the factory SHALL support action chunking and MUST assemble chunked actions through LeRobot temporal queries (via `delta_timestamps` on the `action` key).

`action_chunk_size` is a convenience alias for contiguous forward action deltas (including the current frame at step 0). For custom/non-contiguous action deltas, users MAY configure `delta_timestamps["action"]` directly and leave `action_chunk_size` unset.

If `action_chunk_size` is set and `delta_timestamps` is also set with an explicit `"action"` entry, the factory SHALL raise a `ValueError` indicating the configuration is ambiguous (choose exactly one way to specify action chunking).

#### Scenario: Default backend is used when backend is omitted
- **WHEN** `DataConfig` is constructed with only `repo_id` (and no explicit backend)
- **THEN** the factory SHALL use the `default` (LeRobotDataset) backend

#### Scenario: Explicit lazy backend selection
- **WHEN** `backend="lazy"` is configured
- **THEN** the factory SHALL use `LazyLeRobotDataset`

#### Scenario: Explicit streaming backend selection
- **WHEN** `backend="streaming"` is configured
- **THEN** the factory SHALL use `ShardInterleavedDataset`

#### Scenario: Default backend supports action chunking via LeRobot temporal queries
- **WHEN** `backend="default"` and `action_chunk_size=4` is configured and `delta_timestamps` is unset (or does not contain an `"action"` entry)
- **THEN** `create_dataloader()` SHALL succeed (no `ValueError`), and default-backend samples SHALL include `action` as a stacked tensor with first dimension `4` and `action_is_pad` as a boolean tensor of shape `(4,)`

#### Scenario: Default backend rejects conflicting action chunk configuration
- **WHEN** `backend="default"`, `action_chunk_size` is configured, and `delta_timestamps` is configured with an explicit `"action"` entry
- **THEN** `create_dataloader()` SHALL raise a `ValueError` explaining that `"action"` chunking cannot be specified by both `action_chunk_size` and `delta_timestamps["action"]`

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
