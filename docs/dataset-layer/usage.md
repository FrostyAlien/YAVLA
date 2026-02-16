# Usage and Configuration

This page covers `DataConfig` fields, YAML/CLI configuration, common recipes, and epoch handoff.

## DataConfig Reference

`DataConfig` is a Python dataclass in `src/yavla/data/factory.py`. All fields have defaults except `repo_id`.

### Core Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `repo_id` | `str` | *(required)* | HuggingFace dataset repository ID |
| `root` | `str \| Path \| None` | `None` | Local data root override |
| `backend` | `"auto" \| "default" \| "lazy" \| "streaming"` | `"auto"` | Backend selection mode |
| `delta_timestamps` | `dict[str, list[float]] \| None` | `None` | Temporal context offsets per feature key |
| `action_chunk_size` | `int \| None` | `None` | Number of future action frames to assemble |

### DataLoader Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | `int` | `32` | Batch size |
| `num_workers` | `int` | `4` | DataLoader worker count |
| `persistent_workers` | `bool` | `False` | Keep workers alive between epochs |
| `drop_last` | `bool` | `False` | Drop incomplete final batch |
| `pin_memory` | `bool \| None` | `None` | Pin memory (auto-detects CUDA if `None`) |
| `seed` | `int` | `0` | Random seed for streaming shard shuffle |

### Backend-Specific Fields

| Field | Type | Default | Used by |
|-------|------|---------|---------|
| `parquet_cache_size` | `int` | `32` | `lazy` — LRU cache slots for ParquetFile handles |
| `max_video_decoders` | `int` | `128` | `lazy` — max torchcodec decoders before cache flush |
| `auto_size_threshold_gb` | `float` | `50.0` | `auto` — size threshold for default vs lazy |
| `shuffle_buffer_size` | `int` | `10_000` | `streaming` — shuffle buffer capacity |
| `num_interleaved_shards` | `int` | `8` | `streaming` — concurrent shard readers |
| `streaming_parquet_batch_size` | `int` | `256` | `streaming` — rows per PyArrow batch |

### Transform Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `normalize` | `bool` | `True` | Enable statistical normalization |
| `normalize_mode` | `"z-score" \| "min-max"` | `"z-score"` | Normalization method |
| `normalize_keys` | `list[str] \| None` | `None` | Keys to normalize (`None` = all features) |
| `repack_keys` | `dict[str, str] \| None` | `None` | Key remapping (dataset key → model key) |
| `feature_keys` | `list[str] \| None` | `None` | Explicit feature column selection |
| `image_transforms` | `list[str] \| None` | `None` | Torchvision v2 transform names for camera keys |
| `video_backend` | `str` | `"pyav"` | Video decoding backend |

## YAML Configuration

`DataConfig` is compatible with YAML-based config loading. Example from `configs/train.yaml`:

```yaml
dataset:
  repo_id: "lerobot/aloha_sim"
  backend: "auto"
  batch_size: 32
  num_workers: 4
  persistent_workers: true
  parquet_cache_size: 32
  shuffle_buffer_size: 10000
  num_interleaved_shards: 8
  video_backend: "pyav"
  normalize: true
  normalize_mode: "z-score"
  image_transforms: []
```

`DataConfig` is also compatible with `tyro` for CLI overrides:

```bash
python train.py --dataset.backend lazy --dataset.batch-size 64
```

## Common Recipes

### Auto backend (most common)

```python
config = DataConfig(repo_id="lerobot/aloha_sim")
dataloader = create_dataloader(config)
```

The factory picks the best backend based on data locality and size.

### Lazy backend with temporal features

```python
config = DataConfig(
    repo_id="lerobot/aloha_sim",
    backend="lazy",
    delta_timestamps={
        "observation.state": [-0.1, 0.0, 0.1],
    },
    action_chunk_size=4,
)
dataloader = create_dataloader(config)
```

Each sample will include temporal context frames for `observation.state` and a chunk of 4 future action frames, both with padding masks at episode boundaries.

### Streaming backend for remote data

```python
config = DataConfig(
    repo_id="lerobot/droid",
    backend="streaming",
    shuffle_buffer_size=50_000,
    num_interleaved_shards=16,
)
dataloader = create_dataloader(config)
```

No local data needed. Shards are read directly via PyArrow. Increase `shuffle_buffer_size` and `num_interleaved_shards` for better randomness on large datasets.

### Custom key repacking

```python
config = DataConfig(
    repo_id="lerobot/aloha_sim",
    repack_keys={
        "observation.images.laptop": "image",
        "observation.state": "state",
    },
)
dataloader = create_dataloader(config)
# Samples will have "image" and "state" keys instead of the dataset-native names.
```

## Epoch Handoff

At each epoch boundary, call `set_dataloader_epoch()` to propagate the epoch number:

```python
from yavla.data.factory import create_dataloader, set_dataloader_epoch, DataConfig

config = DataConfig(repo_id="lerobot/aloha_sim", persistent_workers=True)
dataloader = create_dataloader(config)

for epoch in range(num_epochs):
    set_dataloader_epoch(dataloader, epoch)
    for batch in dataloader:
        # training step
        ...
```

This function handles:
- `DistributedSampler.set_epoch(epoch)` — ensures different shuffle order per epoch in DDP.
- `ShardInterleavedDataset.set_epoch(epoch)` — re-seeds shard shuffle for the new epoch.
- `LazyLeRobotDataset.set_epoch(epoch)` — clears torchcodec decoder cache at epoch boundaries (when using torchcodec backend).

With `persistent_workers=True`, the streaming dataset uses `multiprocessing.Value` to propagate the epoch to worker processes without recreating them.

## Normative References

- [`openspec/specs/dataset-factory/spec.md`](../../openspec/specs/dataset-factory/spec.md) — `DataConfig`, factory behavior, transform wiring
- [`openspec/specs/data-transforms/spec.md`](../../openspec/specs/data-transforms/spec.md) — transform protocol and built-in transforms
