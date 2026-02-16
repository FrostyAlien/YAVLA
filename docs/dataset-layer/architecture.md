# Architecture

This page describes the dataset-layer architecture: the three backends, how `create_dataloader()` selects and wires them, and where the key code lives.

## Overview

All data loading flows through a single entry point:

```python
from yavla.data.factory import create_dataloader, DataConfig

config = DataConfig(repo_id="lerobot/aloha_sim", backend="auto")
dataloader = create_dataloader(config)
```

`create_dataloader()` does four things:

1. Loads `LeRobotDatasetMetadata` for the given `repo_id`.
2. Builds a transform pipeline from `DataConfig` settings (normalization, image augmentation, key repacking).
3. Selects a backend (`default`, `lazy`, or `streaming`) based on config, dataset size, data locality, and distributed context.
4. Returns a configured `torch.utils.data.DataLoader` with the appropriate dataset, sampler, and transforms attached.

## Backends

### `default`

Wraps the upstream `LeRobotDataset` (HuggingFace Arrow-backed). Loads the full dataset into memory-mapped Arrow tables at init. Supports `delta_timestamps` and all standard LeRobot features.

Best for small-to-medium datasets where the full Arrow table fits comfortably in memory.

Implementation: `lerobot.datasets.lerobot_dataset.LeRobotDataset` (upstream), wrapped by `_TransformingMapDataset` in `src/yavla/data/factory.py`.

### `lazy`

Map-style dataset (`torch.utils.data.Dataset`) that loads only metadata at init and reads Parquet rows on demand via PyArrow. Uses a shard index with O(#files) memory — roughly 8KB for 1000 Parquet files — instead of loading full Arrow tables.

Key capabilities:
- On-demand Parquet row reading with row-group-aware batching and column pruning.
- `delta_timestamps` support (temporal context frames with clamping and padding masks).
- `action_chunk_size` support (future action frame assembly with padding).
- LRU cache for `ParquetFile` handles (configurable via `parquet_cache_size`).
- Batched `__getitems__` that groups indices by file and row group to amortize I/O.
- Video frame decoding via `decode_video_frames()` with configurable backend.

Best for large datasets where loading the full Arrow table is too expensive, or when you need `delta_timestamps` / `action_chunk_size`.

Implementation: `LazyLeRobotDataset` in `src/yavla/data/lazy.py`.

### `streaming`

Iterable dataset (`torch.utils.data.IterableDataset`) that discovers Parquet shards, shuffles shard order per epoch, opens K shards simultaneously, and round-robin reads batches from each into a shuffle buffer.

Key capabilities:
- Multi-shard interleaved iteration with configurable `num_interleaved_shards`.
- Shuffle buffer for approximate randomness (configurable `shuffle_buffer_size`).
- Rank-aware shard partitioning for distributed training (modulo assignment).
- Worker-aware shard partitioning within each rank (round-robin slicing).
- Deterministic seeding via `set_epoch()` for reproducible shard order per epoch.
- Column pruning via PyArrow `iter_batches()`.

Does **not** support `delta_timestamps` or `action_chunk_size` (raises `ValueError`).

Best for remote-only datasets where local Parquet files are not available.

Implementation: `ShardInterleavedDataset` in `src/yavla/data/streaming.py`.

## Data Flow

```
DataConfig
    │
    ▼
create_dataloader()
    ├── LeRobotDatasetMetadata(repo_id)
    ├── build_transform_pipeline(config, metadata)
    │       └── RepackTransform → NormalizeTransform → ImageTransform
    ├── plan_feature_columns(config, metadata)
    ├── select_backend(config, metadata)
    │       └── returns BackendSelection(backend, reason)
    │
    ├── backend == "default"  → LeRobotDataset + _TransformingMapDataset
    ├── backend == "lazy"     → LazyLeRobotDataset (transforms built-in)
    └── backend == "streaming"→ ShardInterleavedDataset (transforms built-in)
            │
            ▼
      DataLoader(dataset, sampler, batch_size, num_workers, ...)
```

For map-style backends (`default`, `lazy`):
- Distributed training → `DistributedSampler` with `shuffle=True`.
- Single-process → `shuffle=True` on the DataLoader.

For `streaming`:
- Shard partitioning is handled internally by the dataset (rank + worker aware).
- No external sampler needed.

## Implementation Map

| Component | File | Role |
|-----------|------|------|
| `DataConfig` | `src/yavla/data/factory.py` | Configuration dataclass |
| `create_dataloader()` | `src/yavla/data/factory.py` | Factory entry point |
| `select_backend()` | `src/yavla/data/factory.py` | Auto-selection logic |
| `build_transform_pipeline()` | `src/yavla/data/factory.py` | Transform composition |
| `set_dataloader_epoch()` | `src/yavla/data/factory.py` | Epoch propagation helper |
| `LazyLeRobotDataset` | `src/yavla/data/lazy.py` | Lazy map-style backend |
| `ShardInterleavedDataset` | `src/yavla/data/streaming.py` | Streaming iterable backend |
| `DataTransformFn` | `src/yavla/data/transforms.py` | Transform protocol + built-ins |
| `validate_sample_schema()` | `src/yavla/data/schema.py` | Output schema validation |

## Normative References

- [`openspec/specs/dataset-factory/spec.md`](../../openspec/specs/dataset-factory/spec.md)
- [`openspec/specs/lazy-dataset/spec.md`](../../openspec/specs/lazy-dataset/spec.md)
- [`openspec/specs/streaming-dataset/spec.md`](../../openspec/specs/streaming-dataset/spec.md)
- [`openspec/specs/data-transforms/spec.md`](../../openspec/specs/data-transforms/spec.md)
