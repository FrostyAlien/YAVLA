## Context

YAVLA trains Vision-Language-Action models on robotics datasets stored in LeRobot v3.0 format (Parquet + MP4, hosted on HuggingFace Hub). The standard `LeRobotDataset` materializes Parquet via LeRobot helpers (internally using HuggingFace `Dataset.from_parquet`), which produces uncompressed Arrow on disk, then memory-maps it. This works for small datasets but causes OOM on large ones (e.g., DROID ~400GB). The OOM has multiple contributing factors:

1. Large uncompressed Arrow caches on disk (2-5x the Parquet size) that get memory-mapped.
2. Per-worker Python heap overhead and object duplication across `fork()`-ed DataLoader workers.
3. LeRobot's video decoder cache (`_default_decoder_cache`) grows unbounded — no eviction policy, keeps file handles and decoded frames in memory.
4. The combination of these factors causes OOM before the first training step on large datasets with multiple workers/GPUs.

The exact memory growth mechanism needs profiling (RSS vs PSS vs page cache) to confirm, but the practical result is clear: `LeRobotDataset` cannot scale to datasets like DROID.

The alternative `StreamingLeRobotDataset` loses random-access indexing, and naive shuffle buffers produce poor randomness (correlated batches hurt VLA convergence).

LeRobot v3.0 organizes data as chunked Parquet shards (`data/chunk-{NNN}/file-{NNN}.parquet`) with episode metadata tracking `dataset_from_index` / `dataset_to_index` per episode, and chunk/file indices for both data and video files. Video frames are decoded on-the-fly from MP4 files via torchcodec or pyav.

## Goals / Non-Goals

**Goals:**
- Load large LeRobot v3.0 datasets without OOM, regardless of dataset size
- Preserve uniform random sampling quality (equivalent to `RandomSampler` over the full dataset)
- Support `delta_timestamps` for temporal context windows
- Support multi-GPU training via `DistributedSampler`
- Share a single transforms pipeline across all dataset backends
- Keep `LeRobotDataset` as the default for datasets that fit in memory (no regression)
- Config-driven backend selection via tyro/YAML

**Non-Goals:**
- Writing data / dataset creation (read-only consumption for training)
- RLDS/TFRecord ingestion (rely on community converters like any4lerobot)
- Custom video decoding — reuse LeRobot's `decode_video_frames()` as-is
- Distributed dataset sharding across nodes at the storage level (rely on each node having local cache or Hub access)
- Online data augmentation beyond image transforms (handled by model-specific transforms)

## Decisions

### D1: LazyLeRobotDataset reads Parquet directly via PyArrow, bypassing HuggingFace `datasets`

**Choice:** Build a map-style `torch.utils.data.Dataset` that loads only `LeRobotDatasetMetadata` + a shard index at init time, then reads individual Parquet rows on demand in `__getitem__` using `pyarrow.parquet.ParquetFile`.

**Why not alternatives:**
- *HuggingFace `datasets` with `keep_in_memory=False`*: Still converts to Arrow on disk and memory-maps it. The large Arrow cache + per-worker overhead is the root cause of OOM — this doesn't fix it.
- *Patching `LeRobotDataset` to use streaming internally*: Would require deep changes to LeRobot's `__getitem__` which indexes into `self.hf_dataset[idx]`. Fragile across LeRobot version updates.
- *Using `datasets.load_dataset(streaming=True)`*: Loses `__getitem__` / map-style access entirely.

**How the shard index works:**
- At init, read `LeRobotDatasetMetadata.episodes` to get `dataset_from_index`, `dataset_to_index`, `data/chunk_index`, `data/file_index` per episode.
- Build a sorted array of file boundary offsets: `file_boundaries[i]` = cumulative frame count up to file `i`. This is O(#files) memory, not O(#frames). For a dataset with 1000 Parquet files, this is ~8KB.
- Maintain a `file_id → (chunk_index, file_index)` lookup table to format paths via `info["data_path"]` template.
- `__getitem__(idx)` uses right-biased binary search semantics (`searchsorted(..., side="right") - 1`, e.g. `np.searchsorted` or `bisect_right`) to find the file, computes `local_idx = idx - file_boundaries[file_id]`, then reads from the Parquet file.

**Row-group-aware reading:**
- Parquet files are organized into row groups (the smallest decompression unit). Reading a single row still requires decompressing the entire row group it belongs to.
- `__getitem__` reads via `ParquetFile.read_row_group(rg_id, columns=needed_columns)` then `table.take([local_idx_within_rg])`, with column pruning to minimize decompression cost.
- LRU cache of `ParquetFile` handles (configurable size, default 32) avoids repeated file opens. The OS page cache handles hot row group data.
- For batched access (PyTorch `__getitems__` protocol), group indices by file and row group to amortize decompression across multiple samples.

### D2: ShardInterleavedDataset uses multi-shard interleaving for near-uniform streaming

**Choice:** PyTorch `IterableDataset` that shuffles shard order per epoch, opens K shards simultaneously, round-robin reads batches from each, and applies a shuffle buffer on top.

**Why this approach:**
- Technique inspired by TensorFlow's tf.data pipeline (Derek Murray, Stack Overflow, Oct 2017): shuffle filenames, interleave across files, then apply a shuffle buffer. Shuffling quality improves with more shards, more interleaving, and larger buffers. Perfect uniform shuffling requires buffer ≥ dataset size, but in practice shard interleaving + moderate buffer is sufficient for training convergence.
- No full dataset materialization — reads Parquet shards sequentially via `ParquetFile.iter_batches()`.
- Suitable for Hub-only scenarios where data isn't fully downloaded.

**Limitations:** Does not support `delta_timestamps` or action chunking (both require random access to nearby frames). The factory SHALL force `lazy` or `default` backend when these features are configured.

**Distributed sharding:** Each worker/rank takes a disjoint subset of shards (shard index mod world_size == rank), similar to how `DistributedSampler` partitions indices.

### D3: Composable transforms pipeline as a protocol, not a base class

**Choice:** Define a `DataTransformFn` protocol (`__call__(sample: dict[str, Any]) -> dict[str, Any]`) and a `compose()` function that chains transforms. Each dataset backend applies transforms in `__getitem__` / `__iter__`.

**Why a protocol over inheritance:**
- Transforms are orthogonal to dataset backends — any callable works.
- Easy to test transforms in isolation.
- Matches openpi's proven pattern (`RepackTransform → DataTransform → Normalize → ModelTransform`).
- Users can inject custom transforms without subclassing.

**Standard transform stages:**
1. **Repack**: Remap keys between dataset format and model format (e.g., `observation.images.laptop` → `image`).
2. **Normalize**: Z-score or min-max normalization using dataset stats from `LeRobotDatasetMetadata.stats`.
3. **Image transforms**: Torchvision v2 augmentations (resize, crop, color jitter). Applied to camera keys.
4. **Model transforms**: Out of scope for this dataset-layer spec; model-specific preprocessing (e.g., tokenization for language instructions) is applied in the model/input pipeline.

**Not transforms — dataset-level concerns:**
- **Action chunking** and **frame stacking** require access to multiple frames (future actions, neighboring observations). These are handled inside `__getitem__` / `__iter__` alongside `delta_timestamps`, not in the transform pipeline. The dataset assembles the temporal window; transforms operate on the assembled sample.

### D4: Factory function as the single entry point, configured via dataclass + tyro

**Choice:** A `create_dataloader()` function that takes a `DataConfig` dataclass, selects the backend, wires up transforms, and returns a `torch.utils.data.DataLoader`.

```python
@dataclass
class DataConfig:
    repo_id: str
    backend: Literal["auto", "default", "lazy", "streaming"] = "auto"
    delta_timestamps: dict[str, list[float]] | None = None
    action_chunk_size: int | None = None
    batch_size: int = 32
    num_workers: int = 4
    # Lazy backend options
    parquet_cache_size: int = 32  # LRU cache for open ParquetFile handles
    # Streaming backend options
    shuffle_buffer_size: int = 10_000
    num_interleaved_shards: int = 8
    # Transforms
    repack_keys: dict[str, str] | None = None
    image_transforms: list[str] | None = None  # names of torchvision transforms
    normalize: bool = True
```

**`auto` backend selection:** Load `LeRobotDatasetMetadata` and estimate the uncompressed tabular data size from non-video entries in `info["features"]` schema (dtypes × shapes × `total_frames`), treating video features as path/metadata references rather than dense frame tensors. If estimated size > threshold (configurable, default 50GB), use `lazy`. If data isn't locally available, use `streaming`. Otherwise, use `default`. If `delta_timestamps` or `action_chunk_size` is configured, `streaming` is excluded from auto-selection (force `lazy` or `default`). During the v1 simplification scope (`SC-001`), `streaming` is also excluded from `auto` when distributed training is active.

### D5: Reuse LeRobot's video decoding and metadata loading with bounded torchcodec cache management

**Choice:** `LazyLeRobotDataset` and `ShardInterleavedDataset` both use `LeRobotDatasetMetadata` for metadata and `decode_video_frames()` for MP4 decoding. No custom video pipeline.

**Why:** Video decoding is already well-handled by LeRobot (torchcodec/pyav backends, timestamp-based seeking, tolerance validation). The OOM problem is in the Parquet/Arrow layer, not video. Reusing LeRobot's video utils means we stay compatible with future LeRobot improvements and avoid duplicating complex codec logic.

**Caveat:** LeRobot's torchcodec path uses a global `_default_decoder_cache` that grows unbounded — no eviction policy, keeps file handles and decoded state in memory. On large datasets with many video files, this causes continuous memory growth. Expose `video_backend` as a first-class config knob. When using torchcodec, wrap LeRobot decoding with bounded per-worker decoder-cache management (or equivalent monkey-patch) and clear caches at epoch boundaries. pyav does not have this caching issue but is slower.

### D6: delta_timestamps handled by replicating LeRobot's query logic

**Choice:** `LazyLeRobotDataset.__getitem__` replicates the `_get_query_indices` → `_query_hf_dataset` → `_get_query_timestamps` → `_query_videos` flow from `LeRobotDataset`, but reads from Parquet instead of `hf_dataset`.

**Why:** `delta_timestamps` is critical for VLA models (temporal context for action prediction). The logic is well-defined: convert timestamp deltas to frame index offsets, clamp to episode boundaries, generate padding masks. Reimplementing against PyArrow is straightforward since we have the shard index.

**Optimization:** When `delta_timestamps` requests N frames, group query indices by file and row group, then batch-read via `read_row_group()` + `table.take(indices)` to amortize decompression. Action chunking (assembling `chunk_size` future action frames) uses the same mechanism — it's a dataset-level concern alongside `delta_timestamps`, not a per-sample transform.

### D7: Simplification constraint log for v1 rollout

**Choice:** Prioritize correctness and operational simplicity in the first rollout by restricting production `auto` behavior under DDP.

**Constraint ID:** `SC-001`

**Constraint:** While distributed training is active, `backend="auto"` SHALL NOT select `streaming`; it SHALL select `lazy` or `default` and emit a log message containing `SC-001`.

**Why this is the top simplification:** It removes the highest-risk ambiguity (uneven per-rank sample counts and epoch-step mismatches in iterable streaming) while preserving the core goal of stable, uniform-sampling map-style training.

## Risks / Trade-offs

**[Per-row Parquet reads are slower than memory-mapped Arrow]** → Acceptable trade-off: we trade throughput for memory safety. Row-group-aware reads with column pruning minimize decompression cost. The OS page cache keeps hot row groups in memory. Multi-worker DataLoader prefetching hides I/O latency. Support `__getitems__` (batched access) to group reads by file/row group. If throughput becomes a bottleneck, add row-group-level LRU caching of decompressed data.

**[Shard index assumes stable Parquet file layout]** → LeRobot v3.0's chunked format is versioned (`codebase_version: "v3.0"`) and the `data_path` template is in metadata. We validate the version at init and fail fast if it changes. The index is rebuilt each time from episode metadata, not persisted.

**[Video decoder cache grows unbounded]** → LeRobot's torchcodec `_default_decoder_cache` has no eviction policy. Cap cache size per worker and clear at epoch boundaries. Expose `video_backend` as a config knob (torchcodec for speed, pyav for safety). Monitor open file descriptors — K-way interleaving + per-worker caches can hit `ulimit -n`.

**[ShardInterleavedDataset randomness is approximate]** → Perfect uniform shuffling requires buffer ≥ dataset size. Shard interleaving + moderate buffer is "good enough" for training convergence but not mathematically uniform. Quality improves with more shards and larger buffers. Document this constraint and recommend `LazyLeRobotDataset` when local storage is available.

**[Streaming backend does not support delta_timestamps or action chunking]** → Both require random access to nearby frames. The factory enforces this constraint: if `delta_timestamps` or `action_chunk_size` is configured, `streaming` is excluded from backend selection.

**[Tight coupling to LeRobot v3.0 internals]** → We depend on `LeRobotDatasetMetadata`, `decode_video_frames()`, and the Parquet shard naming convention. LeRobot is pinned from git in our dependencies, so we control the version. If LeRobot v4 changes the format, we update our shard index builder — the rest of the architecture (transforms, factory, DataLoader) is format-agnostic.

**[No integration tests against real large datasets in CI]** → Unit tests will use small synthetic datasets. Real-scale validation happens during training runs with wandb logging. Add a smoke test that loads DROID metadata without downloading data.

**[File descriptor limits]** → K-way shard interleaving + per-worker ParquetFile LRU caches can exhaust `ulimit -n`. Add FD budget awareness: cap total open files per worker, document recommended ulimit settings.
