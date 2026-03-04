# streaming-dataset Specification

## Purpose
Define the `ShardInterleavedDataset` streaming backend for scalable iteration over LeRobot v3 datasets via
interleaved shard readers, shuffle-buffer-based randomness, rank/worker-aware shard partitioning, batched Parquet
reads, deterministic epoch seeding, and video decoding, while explicitly rejecting random-access temporal features.
## Requirements
### Requirement: Multi-shard interleaved iteration
`ShardInterleavedDataset` SHALL implement `torch.utils.data.IterableDataset` that discovers all Parquet shards for a LeRobot v3.0 dataset, shuffles shard order per epoch, opens K shards simultaneously, and round-robin reads batches from each.

#### Scenario: Shard order shuffled per epoch
- **WHEN** a new epoch begins (i.e., `__iter__` is called again)
- **THEN** the shard order SHALL be re-shuffled using a seed derived from the epoch number, producing a different iteration order than the previous epoch

#### Scenario: Interleaved reading from K shards
- **WHEN** `num_interleaved_shards=4` and the dataset has 10 shards
- **THEN** the iterator SHALL maintain 4 active shard readers at any time, reading batches in round-robin order, and replace exhausted readers with the next shard in the shuffled order

### Requirement: Shuffle buffer for near-uniform randomness
The iterator SHALL maintain a shuffle buffer that accumulates rows from interleaved shards and yields random samples from the buffer.

#### Scenario: Buffer-based random yielding
- **WHEN** the shuffle buffer reaches `shuffle_buffer_size` capacity
- **THEN** each `__next__` call SHALL pop a uniformly random element from the buffer and replace it with the next row from the interleaved readers

#### Scenario: Epoch tail flush
- **WHEN** all shard readers are exhausted and the buffer still contains rows
- **THEN** the remaining buffer SHALL be shuffled and yielded in the shuffled order

#### Scenario: Near-uniform randomness with sufficient buffer and shards
- **WHEN** the dataset has many shards and `shuffle_buffer_size` is large relative to shard size
- **THEN** the output distribution SHALL approximate a uniform distribution over the full dataset, with quality improving as shard count and buffer size increase. Perfect uniform shuffling requires buffer ≥ dataset size.

### Requirement: Distributed rank-aware shard partitioning
In multi-GPU training, each rank SHALL process a disjoint subset of shards to avoid duplicate samples across ranks.

#### Scenario: Shard partitioning across 4 GPUs
- **WHEN** the dataset has 12 shards and training runs on 4 GPUs
- **THEN** each rank SHALL process exactly 3 shards (shard assignment: `shard_indices where shard_index % world_size == rank`)

#### Scenario: Uneven shard distribution
- **WHEN** the dataset has 10 shards and training runs on 4 GPUs
- **THEN** ranks 0 and 1 SHALL process 3 shards each, and ranks 2 and 3 SHALL process 2 shards each, with no shard assigned to multiple ranks

### Requirement: DataLoader worker-aware shard partitioning
When used with multi-worker `DataLoader`, each worker within a rank SHALL process a disjoint subset of that rank's shards.

#### Scenario: Worker partitioning
- **WHEN** rank 0 has 8 shards assigned and `DataLoader` uses `num_workers=4`
- **THEN** each worker SHALL process 2 shards, determined by `worker_info.id` within the rank's shard subset

### Requirement: Parquet row reading via PyArrow
Each shard reader SHALL read rows from Parquet files using `pyarrow.parquet.ParquetFile.iter_batches()` with column pruning, converting batches per-column for optimal performance.

#### Scenario: Column pruning
- **WHEN** the transform pipeline only requires `observation.state` and `action` columns
- **THEN** `iter_batches()` SHALL specify only those columns plus required metadata columns (`episode_index`, `index`, `timestamp`, `frame_index`, `task_index`)

#### Scenario: Batch size for shard reading
- **WHEN** a shard reader iterates over a Parquet file
- **THEN** it SHALL read in batches (default 256 rows) rather than row-by-row, to amortize I/O overhead

#### Scenario: Primitive numeric columns use numpy conversion
- **WHEN** a batch contains primitive numeric columns (int64, float64)
- **THEN** each column SHALL be converted via `column.to_numpy(zero_copy_only=False)` and per-row values SHALL be numpy scalars obtained by indexing

#### Scenario: List/string columns use per-column to_pylist
- **WHEN** a batch contains list-type or string columns
- **THEN** each such column SHALL be converted via `column.to_pylist()` at the column level

### Requirement: Video frame decoding in streaming mode
When the dataset contains video keys, the iterator SHALL decode video frames using LeRobot's `decode_video_frames()` for each yielded sample.

#### Scenario: Video decoding per sample
- **WHEN** a sample is yielded from the shuffle buffer for a dataset with video keys
- **THEN** video frames SHALL be decoded from the corresponding MP4 file using `decode_video_frames()` with the configured backend

### Requirement: delta_timestamps and action chunking not supported
The streaming backend SHALL NOT support `delta_timestamps` or `action_chunk_size` because both require random access to nearby frames, which is incompatible with sequential shard reading.

#### Scenario: delta_timestamps rejected
- **WHEN** `ShardInterleavedDataset` is constructed with `delta_timestamps` configured
- **THEN** it SHALL raise a `ValueError` indicating that delta_timestamps requires the `lazy` or `default` backend

#### Scenario: action_chunk_size rejected
- **WHEN** `ShardInterleavedDataset` is constructed with `action_chunk_size` configured
- **THEN** it SHALL raise a `ValueError` indicating that action chunking requires the `lazy` or `default` backend

### Requirement: Epoch seeding for reproducibility
The shard shuffle and buffer operations SHALL be seeded deterministically for reproducible training.

#### Scenario: Same seed produces same order
- **WHEN** `ShardInterleavedDataset` is iterated twice with the same epoch seed
- **THEN** the shard order and shuffle buffer random choices SHALL produce the identical sample sequence

#### Scenario: set_epoch interface
- **WHEN** `set_epoch(epoch)` is called before iteration
- **THEN** the seed for shard shuffling SHALL incorporate the epoch number, producing a different shard order per epoch

#### Scenario: set_epoch with persistent workers
- **WHEN** `DataLoader(persistent_workers=True)` is used and `set_epoch(epoch)` is called on the main-process dataset
- **THEN** worker dataset replicas SHALL read the updated epoch from shared worker-visible state before their next `__iter__`, so shard order changes across epochs without recreating workers

### Requirement: Episode metadata container compatibility for shard discovery
`ShardInterleavedDataset` SHALL normalize `meta.episodes` records from supported container types without relying on container-specific `to_dict(orient=...)` behavior.

#### Scenario: HF Dataset metadata records are supported
- **WHEN** `meta.episodes` is a HuggingFace `datasets.Dataset`
- **THEN** shard discovery SHALL build shard paths and episode media references without calling `Dataset.to_dict(orient="records")`

#### Scenario: Pandas and list-backed metadata records are supported
- **WHEN** `meta.episodes` is a pandas DataFrame or list-like record collection
- **THEN** shard discovery SHALL normalize those records and discover the correct shard path set

### Requirement: Dual-path media decode source resolution in streaming backend
For media keys, `ShardInterleavedDataset` SHALL decode from row payloads when present and SHALL fall back to canonical LeRobot v3 episode metadata when row payloads are absent.

#### Scenario: Row payload media path uses timestamp fallback order
- **WHEN** a streaming row contains a media payload path but omits an embedded payload timestamp
- **THEN** streaming decode SHALL derive timestamp in this order: row payload timestamp, sample timestamp, sample `frame_index / fps`

#### Scenario: Canonical episode metadata media path is used when row payload is absent
- **WHEN** a streaming row does not include a media key payload and episode metadata includes `videos/{media_key}/chunk_index`, `videos/{media_key}/file_index`, and `videos/{media_key}/from_timestamp`
- **THEN** streaming decode SHALL resolve media path via `video_path` template and decode at `from_timestamp + sample_timestamp`

