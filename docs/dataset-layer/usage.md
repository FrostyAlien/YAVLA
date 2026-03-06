# Usage and Configuration

This page covers `DataConfig` fields, YAML/CLI configuration, common recipes, and epoch handoff.

## DataConfig Reference

`DataConfig` is a Python dataclass in `src/yavla/data/factory.py`. All fields have defaults except `repo_id`.

### Core Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `repo_id` | `str` | *(required)* | HuggingFace dataset repository ID |
| `root` | `str \| Path \| None` | `None` | Local data root override |
| `backend` | `"default" \| "lazy" \| "streaming"` | `"default"` | Dataset backend |
| `delta_timestamps` | `dict[str, list[float]] \| None` | `None` | Temporal context offsets per feature key |
| `action_chunk_size` | `int \| None` | `None` | Number of contiguous future action frames to assemble |

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
| `shuffle_buffer_size` | `int` | `10_000` | `streaming` — shuffle buffer capacity |
| `num_interleaved_shards` | `int` | `8` | `streaming` — concurrent shard readers |
| `streaming_parquet_batch_size` | `int` | `256` | `streaming` — rows per PyArrow batch |

### Transform Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `normalize` | `bool` | `True` | Enable statistical normalization |
| `normalize_mode` | `"z-score" \| "min-max"` | `"z-score"` | Normalization method |
| `normalize_keys` | `list[str] \| None` | `None` | Keys to normalize (`None` = all stat-backed keys excluding camera keys) |
| `repack_keys` | `dict[str, str] \| None` | `None` | Key remapping (dataset key → model key) |
| `feature_keys` | `list[str] \| None` | `None` | Explicit feature column selection |
| `image_transforms` | `list[str] \| None` | `None` | Torchvision v2 transform names for camera keys |
| `video_backend` | `str` | `"pyav"` | Video decoding backend |

## YAML Configuration

`DataConfig` is compatible with YAML-based config loading. Example from `configs/train.yaml`:

```yaml
dataset:
  repo_id: "lerobot/aloha_sim"
  backend: "default"
  batch_size: 32
  num_workers: 4
  persistent_workers: true
  parquet_cache_size: 32
  shuffle_buffer_size: 10000
  num_interleaved_shards: 8
  video_backend: "pyav"
  normalize: true
  normalize_mode: "z-score"
```

`DataConfig` is also compatible with `tyro` for CLI overrides:

```bash
python train.py --dataset.backend lazy --dataset.batch-size 64
```

## Notes: Camera Normalization + SigLIP/PaliGemma Preprocessing

### Dataset-stat normalization excludes camera keys by default

When `normalize=true` and `normalize_keys` is omitted / `null`, YAVLA normalizes all keys that have dataset stats
**except** camera keys (image/video). To include camera keys, explicitly list them in `normalize_keys`:

```yaml
dataset:
  normalize: true
  normalize_keys:
    - observation.state
    - action
    - observation.images.laptop  # opt-in: normalize images using dataset stats
```

### Canonical SigLIP/PaliGemma recipe (dataset layer)

SigLIP/PaliGemma expects dataset-layer preprocessed `pixel_values` (no model-internal processor). The canonical recipe is:

- One resize step:
  - `Resize([H, W], 3)` *(warp; distorts aspect ratio)*
  - `LetterboxPad([H, W], 3)` *(letterbox; resize-to-fit + symmetric pad)*
- `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`

Use the list form for resize (`[H, W]`) to avoid tuple-parsing pitfalls.

When training SigLIP/PaliGemma backbones, `scripts/train.py` auto-wires this recipe when `image_transforms` is
omitted / `null`. The resize step is selected by `vlm_image_resize_strategy`:

- `warp` → `Resize([H, W], 3)` (default; warps aspect ratio)
- `letterbox` → `LetterboxPad([H, W], 3)` (resize-to-fit + symmetric pad)

Set `image_transforms: []` to explicitly disable preprocessing, or provide a non-empty list to take full control.
If you provide `image_transforms`, auto-wiring is disabled (and `vlm_image_resize_strategy` is ignored).

Padding fill behavior: the letterbox strategy fills padded pixels with value **0.5** in `[0, 1]` space so that after
SigLIP normalization (`mean=std=0.5`) those regions become approximately **0.0**.

To override the default checkpoint-derived resize target, set both `vlm_image_height_override` and
`vlm_image_width_override` in the training config. If the override differs from the checkpoint size, training
logs a warning (you are responsible for verifying VLM compatibility).

## Notes: Action Chunking

- `action_chunk_size=K` is a convenience alias for contiguous forward action deltas (includes the current frame at step 0).
- For custom/non-contiguous action deltas, set `delta_timestamps["action"]` directly and leave `action_chunk_size` unset.
- Do not set both `action_chunk_size` and `delta_timestamps["action"]` (ambiguous; raises `ValueError`).

## Common Recipes

### Default backend (most common)

```python
config = DataConfig(repo_id="lerobot/aloha_sim")
dataloader = create_dataloader(config)
```

This uses the upstream `LeRobotDataset` backend.

### Default backend with action chunking

```python
config = DataConfig(
    repo_id="lerobot/aloha_sim",
    backend="default",
    action_chunk_size=4,
)
dataloader = create_dataloader(config)
```

This delegates to upstream LeRobot `delta_timestamps["action"]` and yields `action_is_pad` masks at episode boundaries.

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

### Streaming backend for shard iteration

```python
config = DataConfig(
    repo_id="lerobot/droid",
    backend="streaming",
    shuffle_buffer_size=50_000,
    num_interleaved_shards=16,
)
dataloader = create_dataloader(config)
```

Use this when you want iterable shard-based loading from local parquet shards. Increase `shuffle_buffer_size` and `num_interleaved_shards` for better randomness on large datasets.

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
