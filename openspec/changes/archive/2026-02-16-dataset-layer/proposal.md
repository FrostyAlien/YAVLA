## Why

`LeRobotDataset` converts Parquet to uncompressed Arrow on disk, then memory-maps it. With multi-worker DataLoaders on large datasets (e.g., DROID at ~400GB), the combination of large Arrow caches, per-worker Python heap overhead, and video decoder caches causes OOM before the first training step. The only alternative — `StreamingLeRobotDataset` — loses random-access indexing, and naive shuffle buffers produce poor randomness (nearby samples in file appear nearby in output), which hurts VLA training convergence. YAVLA needs a dataset layer that scales to large robotics datasets without OOM while preserving high-quality shuffling.

## What Changes

- Add a `LazyLeRobotDataset` that reads Parquet rows on demand via PyArrow, loading only metadata + a shard index into memory. Provides full map-style `Dataset` semantics (uniform random sampling via `RandomSampler`/`DistributedSampler`) without materializing Arrow tables.
- Add a `ShardInterleavedDataset` (PyTorch `IterableDataset`) that shuffles shard order per epoch, interleaves K shards simultaneously, and applies a shuffle buffer on top. Near-uniform randomness for streaming-only / Hub-only scenarios.
- Wrap the standard `LeRobotDataset` as the default path for small-to-medium datasets that fit in memory.
- Introduce a shared composable transforms pipeline (inspired by openpi's `DataTransformFn` pattern) used by all three dataset backends: repack transforms → normalize → image transforms → model transforms.
- Provide a unified factory that selects the appropriate backend based on dataset size and user config.

## Capabilities

### New Capabilities

- `lazy-dataset`: Map-style dataset that reads LeRobot v3.0 Parquet shards on demand via PyArrow without loading full Arrow tables into memory. Supports `RandomSampler`, `DistributedSampler`, `delta_timestamps`, and multi-worker `DataLoader`.
- `streaming-dataset`: `IterableDataset` with multi-shard interleaving and shuffle buffer for near-uniform randomness over large or Hub-only datasets. Supports `DistributedSampler`-style sharding across ranks. Does not support `delta_timestamps` or action chunking (use `lazy-dataset` for those).
- `data-transforms`: Composable transform pipeline shared across all dataset backends. Covers key remapping, normalization/unnormalization, image augmentation, and model-specific preprocessing. Note: action chunking and temporal windowing are dataset-level concerns (require multi-frame access), not per-sample transforms.
- `dataset-factory`: Unified entry point that selects the appropriate dataset backend (LeRobot default, lazy, or streaming) based on config, and wires up transforms.

### Modified Capabilities

_(none — no existing specs)_

## Impact

- **New code**: `src/yavla/data/` — new modules for `LazyLeRobotDataset`, `ShardInterleavedDataset`, transforms, and factory.
- **Dependencies**: PyArrow (already a transitive dep of `datasets`), LeRobot (already pinned from git). No new external dependencies.
- **APIs**: Public `create_dataloader()` factory function and transform protocol used by training loop.
- **Training loop**: Will consume dataloaders from the factory instead of directly instantiating `LeRobotDataset`.
- **Config**: New dataset config section in YAML training configs (backend selection, shuffle buffer size, number of interleaved shards, etc.).
