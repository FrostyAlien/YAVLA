## 1. Project Setup

- [x] 1.1 Add optional `[viz]` dependency group to `pyproject.toml` with `rerun-sdk`, `pytorch-grad-cam`, `captum`, `umap-learn`, `scikit-learn` (minimum version bounds only — lockfile pins exact versions)
- [x] 1.2 Create `src/yavla/visualization/` package directory with empty `__init__.py`

## 2. VizConfig

- [x] 2.1 Create `src/yavla/visualization/config.py` with `VizConfig` dataclass (`@dataclass(slots=True)`) — all snapshot, rerun, and fiftyone fields with disabled-by-default values per spec
- [x] 2.2 Export `VizConfig` from `__init__.py` with `__all__`

## 3. Attention Visualization

- [ ] 3.1 Create `src/yavla/visualization/attention.py` with lazy-import helpers (`_import_grad_cam`, `_import_captum_layer_activation`, `_import_captum_ig`) that raise `ImportError` with install instructions
- [ ] 3.2 Implement `generate_attention_heatmap` — Grad-CAM path using pytorch-grad-cam with ViT `reshape_transform` (strip CLS, reshape patches to spatial grid, bilinear upsample), internal context manager for hook cleanup. Raise `ValueError` for unsupported method strings.
- [ ] 3.3 Implement `generate_attention_heatmap` — attention rollout path (layer-by-layer multiply with identity residual, CLS-to-patch row, reshape + upsample). Accepts `model_forward_kwargs` for HF `output_attentions=True`. No backward pass required.
- [ ] 3.4 Implement `extract_layer_embeddings` using Captum `LayerActivation` with hook cleanup context manager and invariant (hook count before == after)
- [ ] 3.5 Implement `compute_integrated_gradients` using Captum `IntegratedGradients`, returning `(N, C, H, W)` attribution tensor with hook cleanup
- [ ] 3.6 Export all three public functions from `__init__.py`

## 4. Training Snapshots

- [ ] 4.1 Create `src/yavla/visualization/snapshot.py` with `SnapshotState` dataclass (`@dataclass(slots=True)`) holding `config`, `fixed_indices`, `snapshot_dataloader`, `last_snapshot_step`
- [ ] 4.2 Implement `init_snapshot_state` — returns `None` if disabled or rank != 0, selects indices via `torch.Generator` + `randperm`, builds deterministic DataLoader (no stochastic augmentations, `num_workers=0`). Clamp to dataset length if fewer samples than requested.
- [ ] 4.3 Implement `maybe_generate_snapshot` accepting `SnapshotState | None` — no-op if None, interval check, gradient isolation (save grads → eval → enable_grad + autocast off → generate heatmaps → zero_grad → restore grads → train), W&B logging with `"viz/{method}/{sample_idx}"` keys. If `wandb.run is None`, skip logging with `logger.warning`.
- [ ] 4.4 Export `SnapshotState`, `init_snapshot_state`, `maybe_generate_snapshot` from `__init__.py`

## 5. FiftyOne Dataset Loader

- [x] 5.1 Create `src/yavla/visualization/fiftyone_loader.py` with `load_lerobot_to_fiftyone` — per-episode uniform subsampling (every Nth frame), JPEG saving to `{output_dir}/{dataset_name}/images/ep{idx:06d}_frame{idx:08d}.jpg` (quality 95, skip if file exists with matching dimensions), `persistent` param for ephemeral vs MongoDB-backed datasets, FiftyOne dataset with typed fields (`episode_index`, `frame_index`, `timestamp`, `task`, `action`, `camera_key`), bulk `add_samples`
- [x] 5.2 Implement `add_embeddings_to_dataset` — numpy conversion, `ValueError` if embedding count != sample count, optional PCA pre-reduction via sklearn, `fob.compute_visualization` call, cache embeddings to `{output_dir}/{dataset_name}/embeddings_{brain_key}.npy` (load from cache if exists with matching shape)
- [x] 5.3 Export both functions from `__init__.py`

## 6. Rerun Eval Logger

- [ ] 6.1 Create `src/yavla/visualization/rerun_logger.py` with `RerunEvalLogger` context manager — lazy rerun import, `__enter__` initializes recording (app ID "yavla-eval"), `__exit__` flushes/closes (exception-safe)
- [ ] 6.2 Implement `log_step` — `rr.set_time_sequence("step", step)`, config-driven logging: images under `vision/{camera_key}` (multi-camera support), per-dimension action scalars under `action/pred/dim_{i}` and `action/gt/dim_{i}` (logged as-is, denormalized), attention under `vision/attention`
- [ ] 6.3 Implement headless detection (`DISPLAY` env var) — always save to `{rerun_output_dir}/{episode_id}.rrd`, optionally `rr.spawn()` when display available
- [ ] 6.4 Export `RerunEvalLogger` from `__init__.py`

## 7. Integration Points

- [x] 7.1 Add `viz: VizConfig = field(default_factory=VizConfig)` to `TrainingConfig` in `src/yavla/training/data.py`
- [x] 7.2 Verify `pyproject.toml` is valid and `pixi install` succeeds with the new optional group

> **Deferred** (D2 points #2-4 — blocked on code that doesn't exist yet):
> - EvalConfig gains `viz: VizConfig` field — when EvalConfig is created
> - Training loop calls `init_snapshot_state()` + `maybe_generate_snapshot()` — when training loop is built
> - Eval loop wraps episodes with `RerunEvalLogger` — when eval scripts are built
