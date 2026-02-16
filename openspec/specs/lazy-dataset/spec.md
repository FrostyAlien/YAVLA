## ADDED Requirements

### Requirement: Lazy initialization with metadata-only loading
`LazyLeRobotDataset` SHALL load only `LeRobotDatasetMetadata` and build a shard index at initialization time. It SHALL NOT load HuggingFace `datasets` Arrow tables or memory-map Parquet files during `__init__`.

#### Scenario: Initialization memory footprint on large dataset
- **WHEN** `LazyLeRobotDataset` is initialized with a repo_id pointing to a dataset with 10 million frames across 1000 Parquet files
- **THEN** peak memory usage during `__init__` SHALL be dominated by metadata loading, not frame-level indexing (shard index is O(#files), approximately 8KB for 1000 files)

#### Scenario: Initialization with episode filtering
- **WHEN** `LazyLeRobotDataset` is initialized with an `episodes` parameter containing a subset of episode indices
- **THEN** the shard index SHALL only contain entries for frames belonging to the specified episodes, and `__len__` SHALL return the total frame count of those episodes only

### Requirement: Shard index maps global frame indices to Parquet locations
The shard index SHALL store file boundary offsets (O(#files) memory) and a `file_id → (chunk_index, file_index)` lookup table. Global frame index resolution SHALL use right-biased binary-search semantics (`searchsorted(..., side="right") - 1`), not per-row storage.

#### Scenario: Index correctness for multi-shard dataset
- **WHEN** a dataset has frames spread across 3 Parquet shards with 1000, 500, and 800 frames respectively
- **THEN** `file_boundaries` SHALL be `[0, 1000, 1500, 2300]`, and right-biased search over `file_boundaries` for index 1000 SHALL resolve to file 1 with `local_idx = 0`

#### Scenario: Memory efficiency
- **WHEN** a dataset has 1000 Parquet files
- **THEN** the shard index SHALL use O(#files) memory (approximately 8KB for 1000 files), not O(#frames)

#### Scenario: Version validation
- **WHEN** `LazyLeRobotDataset` is initialized with a dataset whose `codebase_version` is not `"v3.0"`
- **THEN** it SHALL raise a `ValueError` with a message indicating the unsupported version

### Requirement: On-demand Parquet row reading in __getitem__
`LazyLeRobotDataset.__getitem__(idx)` SHALL read the required row(s) from the Parquet file identified by the shard index using row-group-aware reads via `pyarrow.parquet.ParquetFile`, with column pruning, and return a dictionary matching the schema of `LeRobotDataset.__getitem__`.

#### Scenario: Single frame retrieval without delta_timestamps
- **WHEN** `__getitem__(idx)` is called without `delta_timestamps` configured
- **THEN** it SHALL resolve the file and local index via right-biased search over `file_boundaries` (`... side="right") - 1`), read the containing row group via `read_row_group(rg_id, columns=needed_columns)`, extract the row via `table.take([local_idx_within_rg])`, and return a dict containing all feature columns for that frame, plus `episode_index`, `index`, `timestamp`, `frame_index`, and `task` (string)

#### Scenario: Column pruning
- **WHEN** reading from a Parquet file
- **THEN** only the columns required by the configured features and transforms SHALL be read, not the entire row

#### Scenario: Parquet file handle caching
- **WHEN** consecutive `__getitem__` calls access rows from the same Parquet shard
- **THEN** the `ParquetFile` handle SHALL be reused from an LRU cache rather than reopened

#### Scenario: Batched access via __getitems__
- **WHEN** PyTorch requests multiple indices at once via `__getitems__([idx1, idx2, ...])`
- **THEN** indices SHALL be grouped by file and row group to amortize decompression cost across multiple samples

### Requirement: delta_timestamps support
When `delta_timestamps` is configured, `__getitem__` SHALL compute query indices by converting timestamp deltas to frame offsets, clamping to episode boundaries, and returning temporal context frames with padding masks.

#### Scenario: Temporal context within episode bounds
- **WHEN** `delta_timestamps={"observation.state": [-0.1, 0.0, 0.1]}` at 10 fps, and `__getitem__` is called for frame 50 (mid-episode)
- **THEN** the returned dict SHALL contain `observation.state` with data from frames [49, 50, 51] stacked along a new temporal dimension

#### Scenario: Temporal context at episode boundary with padding
- **WHEN** `delta_timestamps={"observation.state": [-0.1, 0.0, 0.1]}` at 10 fps, and `__getitem__` is called for the last frame of an episode
- **THEN** the forward-looking frame SHALL be clamped to the last frame, and `observation.state_is_pad` SHALL be `[False, False, True]`

#### Scenario: Batch read optimization for delta_timestamps
- **WHEN** multiple query indices fall within the same Parquet file and row group
- **THEN** they SHALL be read in a single `read_row_group()` + `table.take(indices)` call rather than separate reads

### Requirement: Video frame decoding
When the dataset contains video keys, `__getitem__` SHALL decode video frames using LeRobot's `decode_video_frames()` function.

#### Scenario: Video frame retrieval
- **WHEN** `__getitem__` is called for a frame in a dataset with video keys
- **THEN** video frames SHALL be decoded via `decode_video_frames()` with the configured `video_backend` and included in the returned dict

#### Scenario: Default video backend
- **WHEN** no `video_backend` is explicitly configured
- **THEN** the default backend SHALL be `"pyav"` (to avoid unbounded torchcodec decoder cache growth)

### Requirement: Action chunking assembles future action frames
When `action_chunk_size` is configured, `__getitem__` SHALL assemble `chunk_size` consecutive action frames starting from the current frame, clamping at episode boundaries and providing padding masks. This is a dataset-level concern (requires multi-frame access), not a per-sample transform.

#### Scenario: Action chunking mid-episode
- **WHEN** `action_chunk_size=4` and `__getitem__` is called for frame 50 in an episode of length 100
- **THEN** the returned dict SHALL contain `action` as a tensor of shape `(4, action_dim)` with actions from frames [50, 51, 52, 53]

#### Scenario: Action chunking at episode end with padding
- **WHEN** `action_chunk_size=4` and `__getitem__` is called for frame 98 in an episode of length 100
- **THEN** `action` SHALL be `(4, action_dim)` with frames [98, 99, 99, 99] (last frame repeated), and `action_is_pad` SHALL be `[False, False, True, True]`

#### Scenario: Action chunking uses batch read optimization
- **WHEN** the chunk frames fall within the same Parquet shard
- **THEN** they SHALL be read in a single `read_row_group()` + `table.take()` call

### Requirement: Map-style Dataset compatibility
`LazyLeRobotDataset` SHALL implement `torch.utils.data.Dataset` (map-style) with `__len__` and `__getitem__`, enabling use with `RandomSampler`, `DistributedSampler`, and multi-worker `DataLoader`.

#### Scenario: RandomSampler integration
- **WHEN** `LazyLeRobotDataset` is wrapped in a `DataLoader` with `shuffle=True`
- **THEN** `RandomSampler` SHALL produce uniformly random indices over the full dataset, and each `__getitem__` call SHALL return the correct frame

#### Scenario: DistributedSampler integration
- **WHEN** `LazyLeRobotDataset` is used with `DistributedSampler` across 4 GPUs
- **THEN** each rank SHALL receive a disjoint partition of frame indices, and all frames SHALL be covered across ranks

#### Scenario: Multi-worker DataLoader
- **WHEN** `DataLoader` is configured with `num_workers=4`
- **THEN** each worker process SHALL maintain its own LRU cache of `ParquetFile` handles, and total memory usage SHALL NOT grow proportionally to dataset size
