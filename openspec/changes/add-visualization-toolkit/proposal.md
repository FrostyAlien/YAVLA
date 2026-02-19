## Why

The dataset layer is working but we have no way to visually inspect what the model sees, debug decoded images, understand attention patterns, or analyze predicted actions vs ground truth. For a VLA pipeline (SigLIP → PaliGemma → Flow Matching Action Expert), visual debugging at every stage is critical — from data quality to feature extraction to action prediction. Without it, training failures are black boxes.

## What Changes

- Add `src/yavla/visualization/` subpackage with config-driven visualization utilities
- Add **FiftyOne dataset browser** integration: load subsampled LeRobot frames into FiftyOne for browsing, filtering by episode/task, and embedding visualization (UMAP via FiftyOne Brain)
- Add **training-time attention snapshots**: configurable interval (e.g. every 10K steps), fixed sample set chosen once at training start, generates attention rollout + Grad-CAM heatmaps from SigLIP encoder, logged to W&B as images
- Add **Captum integration** for deep post-hoc attribution analysis (Integrated Gradients, LayerActivation extraction for feeding embeddings to FiftyOne Brain)
- Add **Rerun integration for eval only**: log camera frames, predicted vs GT actions per dimension, and optional attention overlays to Rerun's synchronized timeline viewer during evaluation rollouts
- Add new optional dependencies (`[project.optional-dependencies] viz`): `rerun-sdk>=0.27`, `pytorch-grad-cam>=1.5`, `captum>=0.7`, `umap-learn>=0.5`, `scikit-learn>=1.3` (minimum bounds only — pixi lockfile pins exact versions)
- Flow matching denoising visualization is explicitly **deferred** to a future change

## Capabilities

### New Capabilities
- `visualization-config`: Central `VizConfig` dataclass controlling all visualization toggles — snapshot intervals, methods, Rerun flags, FiftyOne subsampling params
- `attention-visualization`: Attention rollout and Grad-CAM heatmap generation for SigLIP vision encoder, plus Captum-based deep attribution (Integrated Gradients, layer activation extraction). Supports both training-time snapshots (W&B) and interactive post-hoc analysis
- `training-snapshot`: Interval-based callback that generates attention visualizations on a fixed sample set during training and logs to W&B. Configurable step interval, sample count, methods, and target layers
- `fiftyone-dataset-loader`: Utility to load subsampled LeRobot dataset frames into FiftyOne with episode metadata, action vectors, and optional precomputed embeddings (via Captum LayerActivation or model forward pass) for UMAP/t-SNE scatter visualization
- `rerun-eval-logger`: Eval-only Rerun logging — camera frames, per-dimension predicted vs GT action scalars, and optional attention map overlays on a synchronized timeline

### Modified Capabilities
<!-- No existing spec requirements change. The visualization layer is additive and does not modify dataset loading, transforms, or factory behavior. -->

## Impact

- **New subpackage**: `src/yavla/visualization/` — 5 modules: `config.py`, `attention.py`, `snapshot.py`, `fiftyone_loader.py`, `rerun_logger.py`
- **Dependencies**: `rerun-sdk`, `pytorch-grad-cam`, `captum`, `umap-learn`, `scikit-learn` added as optional extras `[viz]` in `pyproject.toml`
- **Training loop**: Optional snapshot callback — requires adding `viz: VizConfig` field to `TrainingConfig` and two call sites (`init_snapshot_state` + `maybe_generate_snapshot`). No change if disabled.
- **Eval scripts**: Optional Rerun logging — requires wrapping eval episodes with `RerunEvalLogger` context manager. No change if disabled.
- **Existing code**: Minimal modifications — `TrainingConfig` and `EvalConfig` each gain one `VizConfig` field. No changes to `src/yavla/data/` or `src/yavla/models/`.
- **Dataset scale**: FiftyOne loader handles 300GB+ datasets via subsampling (every Nth frame)
