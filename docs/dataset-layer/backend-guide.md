# Backend Decision Guide

This page explains how to choose a dataset backend and what each backend supports.

## Quick Decision Flow

```
Do you need action_chunk_size?
  ├── Yes → default or lazy
  └── No
        │
        ├── Do you need shard-iterable loading with shuffle buffer?
        │     ├── Yes → streaming
        │     └── No  → default
        │
        └── Are delta_timestamps configured?
              ├── Yes → default or lazy
              └── No  → any backend
```

## Backend Selection

Set `backend` in `DataConfig`:

| Value | Dataset class | When to use |
|-------|--------------|-------------|
| `"default"` | `_TransformingMapDataset(LeRobotDataset)` | Standard path; default choice |
| `"lazy"` | `LazyLeRobotDataset` | Large local datasets or when you want lazy Parquet reads |
| `"streaming"` | `ShardInterleavedDataset` | Iterable shard-based loading with shuffle buffer |

`DataConfig.backend` defaults to `"default"`.

## Feature Compatibility

| Feature | default | lazy | streaming |
|---------|---------|------|-----------|
| `delta_timestamps` | Supported | Supported | Not supported (`ValueError`) |
| `action_chunk_size` | Supported | Supported | Not supported (`ValueError`) |
| Random-access map dataset | Yes | Yes | No (iterable) |
| Shuffle buffer interleaving | No | No | Yes |

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
INFO Selected data backend: default | reason=default backend (LeRobotDataset)
```

## Normative References

- [`openspec/specs/dataset-factory/spec.md`](../../openspec/specs/dataset-factory/spec.md) — backend selection and guardrails
- [`openspec/specs/streaming-dataset/spec.md`](../../openspec/specs/streaming-dataset/spec.md) — temporal feature rejection
- [`openspec/specs/lazy-dataset/spec.md`](../../openspec/specs/lazy-dataset/spec.md) — shard index and right-biased lookup
