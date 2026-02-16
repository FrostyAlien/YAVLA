# Must-Know Caveats

This page documents correctness-affecting constraints and operational pitfalls that are easy to miss when using the dataset layer.

## Streaming Does Not Support Temporal Features

The streaming backend (`ShardInterleavedDataset`) reads shards sequentially and cannot perform random access to nearby frames. Constructing it with `delta_timestamps` or `action_chunk_size` raises `ValueError`:

```
ValueError: streaming backend does not support delta_timestamps; use lazy/default backend
ValueError: streaming backend does not support action_chunk_size; use lazy/default backend
```

Use `lazy` or `default` backend for any workload that requires temporal context or action chunking.

## Streaming Shuffle Is Approximate

The streaming shuffle buffer provides near-uniform randomness, not perfect uniform shuffling. Quality depends on two factors:

- **Buffer size**: larger `shuffle_buffer_size` → better approximation. Perfect uniformity requires buffer >= dataset size.
- **Shard count and interleaving**: more shards with higher `num_interleaved_shards` → more diverse samples entering the buffer.

For small datasets, the approximation may noticeably differ from true random shuffling. If shuffle quality is critical, use `lazy` or `default` backend with `RandomSampler` (which provides true uniform random access).

At epoch end, remaining buffer contents are shuffled and yielded in shuffled order (tail flush).

## Video Decoder Cache Behavior

### Default backend: `pyav`

The default `video_backend` is `"pyav"`. This is stateless — no decoder cache accumulates across samples.

### `torchcodec` backend

When using `video_backend="torchcodec"`, the lazy backend tracks decoder paths in an LRU-style ordered dict. When the number of unique video paths exceeds `max_video_decoders` (default 128), the entire `_default_decoder_cache` from LeRobot is cleared.

At epoch boundaries, `set_epoch()` also clears the decoder cache. You can manually clear it with `dataset.clear_video_cache()`.

The streaming backend does not implement dataset-level decoder-cache management. Decoder caching behavior depends on the configured video backend and LeRobot decoder internals.

### Recommendation

Use `pyav` (the default) unless you have a specific reason to use `torchcodec`. The torchcodec cache can grow unbounded if not managed, and the lazy backend's eviction strategy is a full cache flush rather than per-entry eviction.

## Media Source Resolution

Both custom backends (`LazyLeRobotDataset` and `ShardInterleavedDataset`) resolve media keys from two sources:

- **Row payload path**: If a sample row includes a media value with a path (for example `{"path": ...}`), that path is decoded directly.
- **Canonical LeRobot v3 episode metadata path**: If a row does not include the media key, the backend falls back to episode metadata fields:
  - `videos/{media_key}/chunk_index`
  - `videos/{media_key}/file_index`
  - `videos/{media_key}/from_timestamp`
  with `video_path` from dataset metadata to build the MP4 path.

Timestamp selection for row payload decoding follows this order:

1. payload timestamp (`value["timestamp"]`)
2. sample timestamp (`sample["timestamp"]`)
3. `sample["frame_index"] / fps`

When decoding via canonical episode metadata, the query timestamp is shifted by episode offset:

- `decode_timestamp = from_timestamp + sample_timestamp`

### Difference vs default LeRobot backend

- **Parity**: Both implementations apply episode-level `from_timestamp` shifting when resolving frames from canonical metadata.
- **Intentional extension in YAVLA backends**: Lazy/streaming fallback handles missing row media payloads by resolving directly from canonical episode metadata, so real datasets where data parquet omits media columns still decode correctly.

### Memory behavior

This media source fallback does not change initialization memory behavior:

- no eager frame-level parquet load is introduced during dataset `__init__`
- frame rows and video frames are loaded on sample access

## Parquet File Handle Caching (Lazy Backend)

The lazy backend maintains an LRU cache of `ParquetFile` handles per DataLoader worker. Each worker has its own cache (keyed by `worker_info.id`), so total open file handles scale as `num_workers × parquet_cache_size`.

Default: 32 cached handles per worker. Adjust `parquet_cache_size` if you see excessive file open/close overhead or if you're hitting OS file descriptor limits.

## Distributed Training: Sampler and Shard Partitioning

### Map-style backends (`default`, `lazy`)

The factory attaches a `DistributedSampler` when `torch.distributed.is_initialized()` is true. You must call `set_dataloader_epoch(dataloader, epoch)` each epoch to reshuffle the sampler.

### Streaming backend

`ShardInterleavedDataset` handles rank partitioning internally — shards are assigned by `shard_index % world_size == rank`. No external `DistributedSampler` is used.

Worker partitioning within a rank uses round-robin slicing: worker `i` gets every `num_workers`-th shard from the rank's subset.

Uneven shard distribution is handled naturally: if 10 shards are split across 4 ranks, ranks 0-1 get 3 shards each and ranks 2-3 get 2 shards each.

## Epoch Propagation with Persistent Workers

When using `persistent_workers=True` with the streaming backend, epoch updates are propagated via `multiprocessing.Value` (shared memory). Workers read the updated epoch at the start of each `__iter__` call without needing to be recreated.

For map-style backends, `DistributedSampler.set_epoch()` is called on the main process and takes effect on the next iteration.

Always call `set_dataloader_epoch(dataloader, epoch)` before iterating — skipping this means the same shard/sample order repeats every epoch.

## LeRobot Version Requirement

The lazy backend requires LeRobot dataset format `v3.0`. Initializing with a different `codebase_version` raises:

```
ValueError: Unsupported LeRobot codebase version: 'v2.0'; expected 'v3.0'
```

The streaming and default backends inherit version requirements from upstream LeRobot.

## Normative References

- [`openspec/specs/streaming-dataset/spec.md`](../../openspec/specs/streaming-dataset/spec.md) — shuffle buffer, shard partitioning, temporal feature rejection, epoch seeding
- [`openspec/specs/lazy-dataset/spec.md`](../../openspec/specs/lazy-dataset/spec.md) — Parquet caching, video decoding, version validation
- [`openspec/specs/dataset-factory/spec.md`](../../openspec/specs/dataset-factory/spec.md) — backend guardrails and distributed sampler wiring
