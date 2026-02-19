# Usage and Configuration

This page covers `VizConfig` fields, the `browse_dataset.py` CLI, Python API usage, and common recipes.

## VizConfig Reference

`VizConfig` is a Python dataclass in `src/yavla/visualization/config.py`. All fields default to disabled/sensible values.

### FiftyOne Fields (implemented)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fiftyone_subsample_rate` | `int` | `10` | Take every Nth frame per episode |
| `fiftyone_umap_pca_dims` | `int` | `50` | PCA pre-reduction dimensions before UMAP |

### Snapshot Fields (deferred)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `snapshot_enabled` | `bool` | `False` | Enable periodic training snapshots |
| `snapshot_interval_steps` | `int` | `10_000` | Steps between snapshots |
| `snapshot_num_samples` | `int` | `4` | Samples per snapshot |
| `snapshot_methods` | `list[str]` | `["attention_rollout", "grad_cam"]` | Visualization methods |
| `snapshot_layers` | `list[str]` | `["last"]` | Target layers |
| `snapshot_seed` | `int` | `42` | Fixed seed for reproducible sample selection |

### Rerun Fields (deferred)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rerun_enabled` | `bool` | `False` | Enable Rerun eval logging |
| `rerun_log_images` | `bool` | `True` | Log camera images |
| `rerun_log_actions` | `bool` | `True` | Log action trajectories |
| `rerun_log_attention` | `bool` | `False` | Log attention maps |
| `rerun_output_dir` | `str` | `"./rerun_logs"` | Output directory for `.rrd` files |

## Browse Script CLI

The standalone script `scripts/browse_dataset.py` loads LeRobot datasets into FiftyOne and launches the web app.

```bash
pixi run -e dev python scripts/browse_dataset.py <repo_ids> [options]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `repo_ids` | Yes | Comma-separated HuggingFace repo IDs |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--subsample-rate N` | `10` | Take every Nth frame per episode |
| `--output-dir PATH` | `/tmp/yavla_fiftyone` | Directory for exported JPEG frames |
| `--port N` | `5151` | FiftyOne app port |
| `--persistent` | off | Persist dataset in FiftyOne's MongoDB |
| `--no-app` | off | Load dataset without launching the web app |

### Examples

Browse a single dataset:

```bash
pixi run -e dev python scripts/browse_dataset.py lerobot/pusht
```

Browse multiple datasets with higher sampling density:

```bash
pixi run -e dev python scripts/browse_dataset.py lerobot/pusht,lerobot/aloha_sim --subsample-rate 5
```

Load without launching the app (useful for persistent datasets):

```bash
pixi run -e dev python scripts/browse_dataset.py lerobot/pusht --persistent --no-app
```

## Python API

### Load a dataset

```python
from yavla.visualization import load_lerobot_to_fiftyone

dataset = load_lerobot_to_fiftyone(
    "lerobot/pusht",
    subsample_rate=10,
    output_dir="/tmp/yavla_fiftyone",
)
print(f"Loaded {len(dataset)} samples")
```

### Launch the FiftyOne app

```python
import fiftyone as fo

session = fo.launch_app(dataset, port=5151)
session.wait()
```

### Add embeddings for visualization

```python
import numpy as np
from yavla.visualization import add_embeddings_to_dataset

# embeddings: (N, D) array where N == len(dataset)
embeddings = np.random.randn(len(dataset), 512)

add_embeddings_to_dataset(
    dataset,
    embeddings,
    brain_key="my_embeddings",
    method="umap",
    pca_dims=50,
    cache_dir="/tmp/yavla_fiftyone/cache",
)
```

After running this, the FiftyOne app will show an "Embeddings" panel where you can explore the 2D projection.

## Function Signatures

### `load_lerobot_to_fiftyone()`

```python
def load_lerobot_to_fiftyone(
    repo_id: str,
    *,
    root: str | Path | None = None,
    subsample_rate: int = 10,
    output_dir: str | Path = "/tmp/yavla_fiftyone",
    dataset_name: str | None = None,
    persistent: bool = False,
) -> fo.Dataset
```

### `add_embeddings_to_dataset()`

```python
def add_embeddings_to_dataset(
    dataset: fo.Dataset,
    embeddings: np.ndarray | Any,
    brain_key: str = "default_vis",
    method: str = "umap",
    pca_dims: int | None = 50,
    cache_dir: str | Path | None = None,
) -> None
```
