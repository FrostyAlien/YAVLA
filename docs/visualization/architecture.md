# Architecture

This page describes the visualization package structure, data flow, and how the FiftyOne loader converts LeRobot episodes into browsable samples.

## Package Structure

```
src/yavla/visualization/
├── __init__.py            # Public API: VizConfig, load_lerobot_to_fiftyone, add_embeddings_to_dataset
├── config.py              # VizConfig dataclass (all features disabled by default)
└── fiftyone_loader.py     # LeRobot → FiftyOne conversion + embedding visualization
```

Planned (deferred until model/training/eval exist):

```
├── attention.py           # Attention rollout / Grad-CAM overlays
├── snapshot.py            # Periodic training snapshot capture
└── rerun_logger.py        # Rerun eval-time 3D logging
```

## Design Principles

1. **Config-driven, default-off.** Every feature is gated by `VizConfig` fields that default to disabled. Zero overhead when visualization is not used.
2. **Optional dependencies.** FiftyOne, scikit-learn, and Rerun are declared as `[viz]` extras with lazy imports. Core training never imports them.
3. **Offline-first.** FiftyOne browsing is a standalone offline activity — not wired into the training loop.
4. **Subsample for scale.** The dataset is ~300GB. The loader takes every Nth frame per episode to keep FiftyOne datasets manageable.

## Data Flow: LeRobot → FiftyOne

```
LeRobotDataset(repo_id)
        │
        ▼
  Episode iteration
  (meta.episodes["dataset_from_index"] / ["dataset_to_index"])
        │
        ▼
  Subsample: every Nth frame (subsample_rate)
        │
        ├── Extract camera image → save as JPEG (idempotent, skip if exists + dims match)
        ├── Extract metadata: episode_index, frame_index, timestamp, task, action
        │
        ▼
  fo.Sample(filepath=jpeg_path, **metadata)
        │
        ▼
  Bulk insert (batches of 2000) → fo.Dataset
```

### Typed Fields

The FiftyOne dataset is created with explicit field types for filtering and aggregation:

| Field | FiftyOne Type | Source |
|-------|--------------|--------|
| `episode_index` | `IntField` | Episode loop counter |
| `frame_index` | `IntField` | `frame["index"]` |
| `timestamp` | `FloatField` | `frame["timestamp"]` |
| `task` | `StringField` | Reverse lookup from `meta.tasks` |
| `action` | `ListField(FloatField)` | `frame["action"].tolist()` |
| `camera_key` | `StringField` | First key from `meta.camera_keys` |

### JPEG Export

Images are saved as `ep{NNNNNN}_frame{NNNNNNNN}.jpg` under `{output_dir}/{dataset_name}/images/`. The save is idempotent: if a file exists with matching dimensions, it is skipped.

## Embedding Visualization

`add_embeddings_to_dataset()` takes precomputed embeddings and runs FiftyOne Brain's dimensionality reduction:

```
embeddings (N, D)
      │
      ├── torch → numpy conversion (if needed)
      ├── Count validation (N == len(dataset))
      ├── Optional: load from cache if shape matches
      ├── Optional: PCA pre-reduction (D → pca_dims)
      ├── Save to cache
      │
      ▼
fob.compute_visualization(dataset, embeddings, method="umap")
```

This is intended for use after a model produces embeddings — not currently wired to any model code.

## Integration Point

`VizConfig` is nested inside `TrainingConfig` at `src/yavla/training/data.py`:

```python
@dataclass(slots=True)
class TrainingConfig:
    ...
    viz: VizConfig = field(default_factory=VizConfig)
```

This allows future training-loop integration (snapshots, Rerun) without changing the config surface.

## Implementation Map

| Component | File | Role |
|-----------|------|------|
| `VizConfig` | `src/yavla/visualization/config.py` | Central config dataclass |
| `load_lerobot_to_fiftyone()` | `src/yavla/visualization/fiftyone_loader.py` | Episode → FiftyOne conversion |
| `add_embeddings_to_dataset()` | `src/yavla/visualization/fiftyone_loader.py` | Embedding viz via FiftyOne Brain |
| `browse_dataset.py` | `scripts/browse_dataset.py` | Standalone CLI for FiftyOne browsing |
