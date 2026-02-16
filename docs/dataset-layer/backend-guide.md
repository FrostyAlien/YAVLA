# Backend Decision Guide

This page explains how to choose a dataset backend and how auto-selection works.

## Quick Decision Flow

```
Is backend explicitly set in DataConfig?
  ├── Yes → use that backend
  └── No (backend="auto")
        │
        ├── Does local data probe succeed (metadata root exists + first shard path exists)?
        │     ├── Yes → estimated size > threshold (50GB)?
        │     │           ├── Yes → lazy
        │     │           └── No  → default
        │     └── No  → are delta_timestamps or action_chunk_size configured?
        │                 ├── Yes → lazy (streaming excluded; triggers download)
        │                 └── No  → streaming
        │
        └── Is distributed training active AND backend == streaming?
              └── Yes → SC-001: fall back to lazy or default
```

## Explicit Backend Selection

Set `backend` in `DataConfig` to bypass auto-selection:

| Value | Dataset class | When to use |
|-------|--------------|-------------|
| `"default"` | `_TransformingMapDataset(LeRobotDataset)` | Small datasets that fit in Arrow memory-mapped tables |
| `"lazy"` | `LazyLeRobotDataset` | Large datasets, or when you need `delta_timestamps` / `action_chunk_size` |
| `"streaming"` | `ShardInterleavedDataset` | Remote-only data without temporal features |

Setting `backend="streaming"` with `delta_timestamps` or `action_chunk_size` configured raises a `ValueError`.

## Auto-Selection Logic

When `backend="auto"` (the default), the factory inspects:

1. **Data locality**: are Parquet shards present on disk?
2. **Estimated size**: uncompressed tabular size computed from metadata schema (dtypes x shapes x total_frames, excluding video features).
3. **Feature requirements**: `delta_timestamps` and `action_chunk_size` exclude streaming.
4. **Distributed context**: `SC-001` constraint.

Locality check note: auto-selection uses a fast local-data probe (dataset root + first shard existence), not an exhaustive validation that every shard is present.

### Size threshold

The threshold defaults to 50GB (`auto_size_threshold_gb` in `DataConfig`). The estimate covers non-video tabular features only — video frames are decoded separately and don't factor into the size calculation.

### SC-001: Distributed Auto-Mode Constraint

When distributed training is active (`torch.distributed.is_initialized()`) and auto-selection would choose `streaming`, the factory overrides to `lazy` or `default` instead. The selection reason is logged with the identifier `SC-001`.

Rationale: streaming shard partitioning in distributed mode was deferred from v1 validation scope. The constraint prevents untested distributed-streaming combinations from being silently selected.

If you need streaming in distributed mode, set `backend="streaming"` explicitly — the constraint only applies to `auto`.

### Streaming Temporal-Feature Guardrails

The streaming backend (`ShardInterleavedDataset`) reads shards sequentially and cannot perform random access to nearby frames. Both `delta_timestamps` and `action_chunk_size` require multi-frame lookups, so:

- Auto-selection excludes streaming when either is configured.
- Explicit `backend="streaming"` with either configured raises `ValueError`.

Use `lazy` or `default` for temporal features.

## Right-Biased Frame-Index Lookup (Lazy Backend)

The lazy backend resolves a global frame index to a Parquet file using right-biased binary search:

```python
file_id = bisect.bisect_right(file_boundaries, local_index) - 1
```

For a dataset with shards of 1000, 500, and 800 frames:
- `file_boundaries = [0, 1000, 1500, 2300]`
- Index 1000 → `bisect_right([0, 1000, 1500, 2300], 1000) - 1 = 2 - 1 = 1` → file 1, local offset 0

This is an O(log n) lookup over O(#files) memory, not per-row storage.

## Backend Selection Logging

Every call to `create_dataloader()` logs the selected backend and reason:

```
INFO Selected data backend: default | reason=local data available + estimated size <= threshold (53687091200 bytes vs 53687091200 bytes)
```

For `SC-001` overrides:

```
INFO Selected data backend: lazy | reason=SC-001: distributed auto-mode excludes streaming, falling back to lazy
```

## Normative References

- [`openspec/specs/dataset-factory/spec.md`](../../openspec/specs/dataset-factory/spec.md) — auto-selection requirements, `SC-001`
- [`openspec/specs/streaming-dataset/spec.md`](../../openspec/specs/streaming-dataset/spec.md) — temporal feature rejection
- [`openspec/specs/lazy-dataset/spec.md`](../../openspec/specs/lazy-dataset/spec.md) — shard index and right-biased lookup
