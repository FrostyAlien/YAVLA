## 1. Data Layer Foundations

- [x] 1.1 Create `src/yavla/data/` module skeleton (`lazy.py`, `streaming.py`, `transforms.py`, `factory.py`, `__init__.py`)
- [x] 1.2 Define shared sample/schema contract for dataset outputs (required metadata keys and task fields)
- [x] 1.3 Add typed `DataConfig` dataclass with tyro/YAML-compatible defaults and backend options

## 2. Shared Transform Pipeline

- [x] 2.1 Implement `DataTransformFn` protocol and `compose(*transforms)` with identity behavior for empty composition
- [x] 2.2 Implement `RepackTransform` key remapping while preserving unmapped keys
- [x] 2.3 Implement `NormalizeTransform` (z-score and min-max) with zero-std/zero-range safe handling
- [x] 2.4 Implement `UnnormalizeTransform` round-trip inversion for both normalization modes
- [x] 2.5 Implement `ImageTransform` wrapper that applies torchvision v2 transforms to configured camera keys

## 3. Lazy Dataset Backend

- [x] 3.1 Implement `LazyLeRobotDataset` initialization with metadata-only loading, v3.0 validation, and optional episode filtering
- [x] 3.2 Build O(#files) shard index (`file_boundaries` + file lookup) using episode metadata
- [x] 3.3 Implement Parquet column planning and per-worker LRU cache for `ParquetFile` handles
- [x] 3.4 Implement `__getitem__` row-group-aware reads with column pruning and LeRobot-compatible output fields
- [x] 3.5 Implement `__getitems__` grouped-by-file/row-group batched reads to amortize decompression
- [x] 3.6 Implement `delta_timestamps` query index computation, episode-boundary clamping, and pad masks
- [x] 3.7 Implement `action_chunk_size` future-action assembly with boundary clamping and pad masks
- [x] 3.8 Implement video decoding integration with default `video_backend="pyav"` and bounded torchcodec cache management hooks

## 4. Streaming Dataset Backend

- [x] 4.1 Implement `ShardInterleavedDataset` iterable skeleton, shard discovery, and explicit `set_epoch(epoch)` API
- [x] 4.2 Implement deterministic shard shuffling and RNG seeding that changes by epoch
- [x] 4.3 Implement rank-aware and worker-aware shard partitioning without shard duplication
- [x] 4.4 Implement K-way shard interleaving with `iter_batches()` readers and configurable batch size
- [x] 4.5 Implement shuffle buffer random pop/replace behavior and end-of-epoch tail flush shuffle
- [x] 4.6 Implement video decoding for yielded samples and enforce Parquet column-pruning path
- [x] 4.7 Enforce unsupported feature guards (`delta_timestamps`, `action_chunk_size`) with clear `ValueError` messages
- [x] 4.8 Implement persistent-worker epoch propagation mechanism for `set_epoch()` correctness

## 5. Dataloader Factory and Selection Policy

- [x] 5.1 Implement `create_dataloader()` explicit backend wiring (`default`, `lazy`, `streaming`)
- [x] 5.2 Implement `auto` backend selection with size estimation and local-data availability checks
- [x] 5.3 Enforce simplification constraint `SC-001` (distributed + `auto` SHALL NOT choose `streaming`) with structured log reason
- [x] 5.4 Enforce `streaming` exclusion when `delta_timestamps` or `action_chunk_size` is configured
- [x] 5.5 Implement distributed sampler behavior (`DistributedSampler` for map-style, rank/world-size injection for streaming)
- [x] 5.6 Wire transform pipeline construction and DataLoader passthrough options (`batch_size`, `num_workers`, `pin_memory`, etc.)

## 6. Training Integration and Config Wiring

- [x] 6.1 Replace direct `LeRobotDataset` construction in training entrypoints with `create_dataloader()`
- [x] 6.2 Add epoch handoff calls for both sampler-driven and streaming backends (`set_epoch` paths)
- [x] 6.3 Add dataset-layer YAML config fields (backend, cache sizes, interleave/shuffle knobs, `video_backend`)
- [x] 6.4 Add backend-selection logging in training startup output, including explicit `SC-001` reason when applied

## 7. Validation and Documentation

- [x] 7.1 Add unit tests for transform protocol/composition and normalization edge cases
- [x] 7.2 Add unit tests for lazy shard index resolution, row-group read grouping, and temporal/action padding semantics
- [x] 7.3 Add unit tests for streaming determinism, partitioning correctness, and unsupported-feature guardrails
- [x] 7.4 Add factory selection tests covering explicit modes, `auto` decision matrix, and `SC-001` distributed override
- [x] 7.5 Add smoke tests for small local datasets (default/lazy) plus metadata-only large-dataset readiness checks
- [x] 7.6 Update dataset-layer docs with backend choice guidance, streaming limitations, and `SC-001` lookup reference
