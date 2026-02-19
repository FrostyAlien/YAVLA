## Context

YAVLA is a VLA model (SigLIP → PaliGemma → Flow Matching Action Expert) with a working dataset layer but zero visualization tooling. The training loop and eval scripts don't exist yet — only data loading helpers (`TrainingConfig` composing `DataConfig`). All configs use `@dataclass(slots=True)`. The project uses wandb for experiment tracking and already has `fiftyone>=1.2` in deps.

The visualization subpackage is self-contained — all new code lives in `src/yavla/visualization/`. However, integration is **not purely additive**: composing `VizConfig` into `TrainingConfig`/`EvalConfig` requires adding a field to those dataclasses, and the training loop must call `maybe_generate_snapshot()` at each step. These are minimal, well-defined insertion points documented in D2 and D4. The subpackage works today for dataset inspection (FiftyOne) and is ready to plug into training/eval when those are built.

## Goals / Non-Goals

**Goals:**
- Config-driven visualization with all features toggleable (zero overhead when disabled)
- FiftyOne integration for dataset-level browsing, filtering, and embedding scatter on 300GB+ datasets via subsampling
- Attention heatmap generation (attention rollout + Grad-CAM) for SigLIP encoder, usable both as training snapshots (W&B) and interactive post-hoc analysis
- Captum integration for deep attribution (Integrated Gradients, layer activation extraction for embeddings)
- Rerun integration for eval-only episode replay with synchronized camera frames and per-dimension action scalars
- Match existing codebase conventions: `@dataclass(slots=True)`, `__all__` exports, Google docstrings, type hints

**Non-Goals:**
- Flow matching denoising visualization (deferred)
- PaliGemma cross-attention visualization (requires model to be built first)
- Real-time Rerun logging during training (eval only)
- Replacing wandb as the training metrics sink
- Visualization of 3D robot meshes or URDF models

## Decisions

### D1: Module structure — flat subpackage, not nested

**Decision**: `src/yavla/visualization/` with 5 flat modules + `__init__.py`.

```
src/yavla/visualization/
├── __init__.py            # public API, __all__ exports
├── config.py              # VizConfig dataclass
├── attention.py           # attention rollout, Grad-CAM, Captum wrappers
├── snapshot.py            # training-time snapshot callback
├── fiftyone_loader.py     # LeRobot → FiftyOne dataset builder
└── rerun_logger.py        # eval-only Rerun logging
```

**Why not nested** (e.g. `visualization/attention/gradcam.py`): The scope is 5 modules with clear boundaries. Nesting adds import complexity for no benefit at this scale. If attention.py grows beyond ~300 lines, split then.

**Alternative considered**: Putting viz utilities in `src/yavla/utils/`. Rejected — visualization is a distinct concern with its own dependencies, not a grab-bag utility.

### D2: Config composition — VizConfig as a peer of DataConfig, not nested inside TrainingConfig

**Decision**: `VizConfig` is a standalone dataclass that gets composed into future `TrainingConfig` and `EvalConfig` the same way `DataConfig` is today.

```python
@dataclass(slots=True)
class VizConfig:
    # Training snapshots
    snapshot_enabled: bool = False
    snapshot_interval_steps: int = 10_000
    snapshot_num_samples: int = 4
    snapshot_methods: list[str] = field(default_factory=lambda: ["attention_rollout", "grad_cam"])
    snapshot_layers: list[str] = field(default_factory=lambda: ["last"])
    snapshot_seed: int = 42

    # Rerun (eval only)
    rerun_enabled: bool = False
    rerun_log_images: bool = True
    rerun_log_actions: bool = True
    rerun_log_attention: bool = False
    rerun_output_dir: str = "./rerun_logs"

    # FiftyOne (offline)
    fiftyone_subsample_rate: int = 10
    fiftyone_umap_pca_dims: int = 50
```

**Why standalone**: The viz config is used by training, eval, AND offline scripts. Nesting it inside TrainingConfig would make it inaccessible to eval/FiftyOne scripts. Keeping it standalone follows the same pattern as DataConfig — composed where needed.

**Why all defaults are "off"**: Zero overhead when visualization isn't needed. Training and eval work identically with or without viz configured.

**Integration points** (the only modifications to existing code when training/eval are built):
1. `TrainingConfig` gains `viz: VizConfig = field(default_factory=VizConfig)` — one line
2. `EvalConfig` gains the same field — one line
3. Training loop calls `init_snapshot_state()` before the loop and `maybe_generate_snapshot()` inside the loop — two call sites
4. Eval loop wraps episodes with `RerunEvalLogger` context manager — one `with` block

These are the **complete** set of touch points. No other existing code changes.

### D3: Attention visualization — pytorch-grad-cam for heatmaps, Captum for attribution and embedding extraction

**Decision**: Use both libraries with a unified wrapper in `attention.py`.

pytorch-grad-cam handles:
- Attention rollout (raw self-attention from ViT)
- GradCAM with ViT `reshape_transform` (patches → spatial grid)
- Fast — one backward pass per image, suitable for training snapshots

Captum handles:
- `LayerActivation` — extract intermediate patch embeddings for FiftyOne Brain UMAP
- `IntegratedGradients` — rigorous pixel-level attribution for post-hoc debugging
- PyTorch official — guaranteed forward compatibility

**Why both**: They serve different purposes. pytorch-grad-cam produces spatial heatmaps fast (training snapshots). Captum produces rigorous attributions and extracts embeddings (post-hoc analysis, FiftyOne). No hook conflicts when used sequentially (not simultaneously).

**ViT-specific Grad-CAM details** (CRITICAL — SigLIP has no conv layers):
- **Target layer**: The final `nn.LayerNorm` before the projection head in SigLIP's vision encoder (typically `model.vision_model.encoder.layers[-1].layer_norm1`). This is where patch-level features are richest.
- **Token-to-spatial reshape**: ViT outputs `(batch, num_patches+1, hidden_dim)`. Strip the CLS token (index 0), reshape remaining `num_patches` tokens to `(batch, H_patches, W_patches, hidden_dim)` where `H_patches = W_patches = sqrt(num_patches)`. pytorch-grad-cam's `reshape_transform` parameter handles this.
- **CLS token handling**: Excluded from spatial heatmaps. For attention rollout, CLS-to-patch attention weights are used as the final attribution map.
- **Attention rollout algorithm**: Multiply attention matrices layer-by-layer: `rollout = A_1 @ A_2 @ ... @ A_L` where each `A_i` is the attention matrix averaged across heads with identity residual added (`A_i = 0.5 * attention_i + 0.5 * I`). Final attribution = CLS row of the rollout matrix, reshaped to `(H_patches, W_patches)` and upsampled to image resolution via bilinear interpolation.

**HuggingFace model integration**: SigLIP/PaliGemma via `transformers` requires `output_attentions=True` for attention rollout. The wrapper function accepts a `model_forward_kwargs: dict` parameter to pass this through. For Grad-CAM, only the target layer reference is needed — no special HF flags required.

**Hook cleanup contract** (CRITICAL): All hook-based functions (`generate_attention_heatmap`, `extract_layer_embeddings`) use an internal context manager that:
1. Registers hooks on `__enter__`
2. Removes all hooks on `__exit__` (including on exception)
3. Invariant: `len(module._forward_hooks) + len(module._backward_hooks)` is identical before and after the call
4. No reference cycles — hooks do not capture the model itself, only the target layer

**Integrated Gradients output**: `compute_integrated_gradients` returns a `Tensor` of shape `(N, C, H, W)` — per-pixel attribution. The caller decides how to surface it: save as numpy for offline analysis, log as `wandb.Image` with colormap overlay, or store in FiftyOne as a heatmap field.

**Key constraint**: Grad-CAM requires gradients — it cannot run inside `torch.no_grad()`. See D4 for the concrete gradient isolation mechanism.

**Interface**:
```python
def generate_attention_heatmap(
    model: nn.Module,
    images: Tensor,           # (N, C, H, W)
    target_layer: nn.Module,
    method: str = "grad_cam", # "grad_cam" | "attention_rollout"
    model_forward_kwargs: dict[str, Any] | None = None,
) -> Tensor:                  # (N, H_img, W_img) float32 heatmaps in [0, 1]
    ...

def extract_layer_embeddings(
    model: nn.Module,
    images: Tensor,
    target_layer: nn.Module,
) -> Tensor:                  # (N, num_tokens, hidden_dim)
    ...

def compute_integrated_gradients(
    model: nn.Module,
    images: Tensor,
    target_layer: nn.Module,
    n_steps: int = 50,
) -> Tensor:                  # (N, C, H, W) per-pixel attribution
    ...
```

### D4: Training snapshots — fixed sample set, interval-based, W&B logging

**Decision**: At training start, randomly select `snapshot_num_samples` dataset **indices** (not decoded tensors) using `snapshot_seed`. Store these indices. Every `snapshot_interval_steps` steps:

1. Save current model gradient state: `grad_states = {p: p.requires_grad for p in model.parameters()}`
2. Call `model.eval()` and `torch.set_grad_enabled(True)` (Grad-CAM needs gradients)
3. Disable AMP autocast: `with torch.autocast(device_type, enabled=False):`
4. Load fixed samples from dataset **with augmentations disabled** (use a dedicated `DataLoader` with `seed=snapshot_seed` and deterministic transforms only — no random crop/flip/color jitter)
5. Register attention hooks on target layer(s)
6. Run forward + backward on the fixed samples (separate from training batch)
7. Generate heatmaps via configured methods
8. Remove all hooks
9. Call `model.zero_grad(set_to_none=True)` to discard snapshot gradients
10. Restore model to `model.train()` and prior gradient state
11. Log to W&B

**Gradient isolation mechanism** (CRITICAL): The snapshot runs in a fully isolated context:
```python
# Pseudocode for the isolation boundary
saved_training = model.training
saved_grads = {p: p.grad for p in model.parameters() if p.grad is not None}
model.eval()
try:
    with torch.enable_grad(), torch.autocast(device_type, enabled=False):
        # ... generate heatmaps ...
finally:
    model.zero_grad(set_to_none=True)
    # Restore saved gradients
    for p, g in saved_grads.items():
        p.grad = g
    model.train(saved_training)
```
This ensures: (a) no snapshot gradients leak into the optimizer step, (b) AMP scaling is not corrupted, (c) any accumulated training gradients from gradient accumulation are preserved.

**DDP/FSDP behavior** (CRITICAL): Snapshots run on **rank 0 only**. The mechanism:
- `init_snapshot_state` returns `None` on non-zero ranks (checked via `torch.distributed.get_rank() if torch.distributed.is_initialized() else 0`)
- `maybe_generate_snapshot` is a no-op when `state is None`
- No `all_gather` or broadcast needed — snapshot samples are loaded independently on rank 0
- Other ranks do NOT call the snapshot function, so no synchronization barrier is introduced
- W&B logging is inherently rank-0 only (standard practice)

**Fixed sample determinism** (IMPORTANT): "Fixed samples" means fixed **dataset indices** selected once. To ensure visual determinism across snapshots:
- Indices are selected via `torch.Generator(device='cpu').manual_seed(snapshot_seed)` + `torch.randperm`
- Samples are loaded through a **snapshot-specific DataLoader** with `num_workers=0`, no shuffle, and a transform pipeline that excludes stochastic augmentations (no random crop, flip, color jitter). Only deterministic transforms (resize, normalize) are applied.
- The same indices + deterministic transforms guarantee identical input images at every snapshot interval

**W&B logging contract** (IMPORTANT):
- **Key naming**: `"viz/{method}/{sample_idx}"` (e.g., `"viz/grad_cam/0"`, `"viz/attention_rollout/3"`)
- **Image format**: `wandb.Image(heatmap_overlay, caption=f"step={step} sample={idx} method={method}")`
- **Layout**: One image per sample per method — no grids. W&B's panel grouping handles layout.
- **Metadata**: Each image logged with `step=step` (W&B's x-axis), caption includes sample index, method, and epoch
- **W&B disabled/offline**: If `wandb.run is None`, skip logging and emit `logger.warning("W&B not initialized, skipping snapshot logging")`. Heatmaps are still generated (useful for debugging the snapshot pipeline itself).

**Why fixed samples**: Seeing the same images' attention evolve over training reveals learning dynamics. Random samples per snapshot make comparison impossible.

**Why not a PyTorch callback/hook system**: No Trainer class exists yet. The snapshot logic is a simple function `maybe_generate_snapshot(model, step, config, fixed_samples)` that the future training loop calls. When a Trainer is built, it wraps this call.

**Interface**:
```python
@dataclass(slots=True)
class SnapshotState:
    """Holds fixed sample indices and tracks last snapshot step."""
    config: VizConfig
    fixed_indices: list[int]          # dataset indices, not decoded tensors
    snapshot_dataloader: DataLoader   # deterministic, no stochastic augmentations
    last_snapshot_step: int = 0

def init_snapshot_state(
    config: VizConfig,
    dataset: Dataset,
    rank: int = 0,
) -> SnapshotState | None:
    """Select fixed samples at training start. Returns None if disabled or rank != 0."""
    ...

def maybe_generate_snapshot(
    state: SnapshotState | None,
    model: nn.Module,
    step: int,
    wandb_run: Any | None = None,
) -> None:
    """Generate and log snapshots if interval reached. No-op if state is None."""
    ...
```

### D5: FiftyOne loader — subsampled, offline, script-driven

**Decision**: A standalone utility that reads a LeRobot dataset, subsamples every Nth frame, saves decoded images to a temp directory, and creates a FiftyOne dataset with metadata fields.

**Why subsampled**: 300GB dataset = millions of frames. FiftyOne handles ~500K samples. At subsample_rate=10, a 1M-frame dataset becomes 100K samples — well within limits.

**Subsampling strategy** (IMPORTANT): Uniform temporal subsampling **per episode**. For each episode, take every Nth frame (0, N, 2N, ...). This preserves temporal coverage across all episodes and tasks, unlike global random sampling which could over-represent long episodes. The `subsample_rate` parameter controls N.

**JPEG saving details** (IMPORTANT):
- **Output directory**: `{output_dir}/{dataset_name}/images/` (default `output_dir=/tmp/yavla_fiftyone`)
- **File naming**: `ep{episode_index:06d}_frame{frame_index:08d}.jpg` — deterministic, no collisions
- **Idempotency**: If the file already exists with matching dimensions, skip re-encoding. This makes re-runs fast.
- **Color space**: LeRobot stores RGB. JPEG is written as RGB via PIL. No BGR conversion needed.
- **Compression quality**: 95 (high quality, ~3x smaller than PNG, negligible visual loss for debugging)
- **Scale concern**: At subsample_rate=10 on a 1M-frame dataset, 100K JPEGs ≈ 5-10GB on disk. Acceptable for debugging.

**Why save to disk**: FiftyOne requires filepaths, not tensors. Images are saved as JPEG to a configurable output directory.

**Why offline**: This is a data curation tool, not a training component. Run it as a script before training to inspect data quality.

**Embedding pipeline** (IMPORTANT): The full flow for UMAP visualization:
1. Load subsampled dataset into FiftyOne (creates samples with filepaths)
2. Extract embeddings: iterate samples in batches (batch_size=64), load images, run through model's vision encoder to the target layer via `extract_layer_embeddings()`, collect as numpy array of shape `(N_samples, hidden_dim)` — use **pooled** output (mean over patch tokens), not per-patch
3. PCA pre-reduction: if `pca_dims` is set, reduce `(N, hidden_dim)` → `(N, pca_dims)` via `sklearn.decomposition.PCA`. This is critical for UMAP performance on >10K samples.
4. Call `fob.compute_visualization(dataset, embeddings=np_array, method="umap", brain_key=brain_key)`
5. Embeddings are stored as a FiftyOne Brain run (not as sample fields) — FiftyOne manages storage internally
6. Caching: embeddings can be saved to `{output_dir}/{dataset_name}/embeddings_{brain_key}.npy` for reuse without re-computing

**Interface**:
```python
def load_lerobot_to_fiftyone(
    repo_id: str,
    *,
    root: str | Path | None = None,
    subsample_rate: int = 10,
    output_dir: str | Path = "/tmp/yavla_fiftyone",
    dataset_name: str | None = None,
    persistent: bool = False,
) -> fo.Dataset:
    ...

def add_embeddings_to_dataset(
    dataset: fo.Dataset,
    embeddings: np.ndarray,
    brain_key: str = "default_vis",
    method: str = "umap",
    pca_dims: int | None = 50,
) -> None:
    ...
```

### D6: Rerun logger — eval only, per-episode, headless-compatible

**Decision**: Rerun logging wraps eval rollouts. Each eval episode is a Rerun recording with a time sequence indexed by environment step number.

**Timeline time base** (IMPORTANT): The timeline is keyed by **environment step index** (0, 1, 2, ..., T) — not wall time, not dataset timestamp. This is the natural unit for action prediction: step N corresponds to observation N and action N. Multiple cameras and action dimensions all share the same step-indexed timeline via `rr.set_time_sequence("step", step)`. Wall-clock time is not logged (irrelevant for eval replay).

**Why eval only**: Rerun adds I/O overhead (image serialization, IPC to viewer). During training, wandb handles metrics. Rerun's value is in synchronized multi-modal replay — only meaningful for eval episodes.

**Headless support and .rrd output** (IMPORTANT):
- **Detection**: Headless if `os.environ.get("DISPLAY")` is None/empty on Linux, or always when `rerun_output_dir` is set (which it is by default)
- **Output path**: `{rerun_output_dir}/{episode_id}.rrd` (default `rerun_output_dir="./rerun_logs"`)
- **File naming**: `{episode_id}.rrd` — episode_id is caller-provided, must be filesystem-safe
- **Size expectation**: ~1-5MB per episode (100 steps × 224×224 JPEG + action scalars). 1000 eval episodes ≈ 1-5GB.
- **Behavior**: In headless mode, `rr.save(path)` is called instead of `rr.spawn()`. In local mode, both `rr.spawn()` AND `rr.save(path)` are called (stream to viewer + persist for later).
- **No streaming + save split**: Always save .rrd. Viewer is optional bonus when display is available.

**Interface**:
```python
class RerunEvalLogger:
    """Context manager for logging one eval episode to Rerun."""

    def __init__(self, config: VizConfig, episode_id: str) -> None: ...
    def __enter__(self) -> RerunEvalLogger: ...
    def __exit__(self, *exc: object) -> None: ...

    def log_step(
        self,
        step: int,
        image: Tensor | np.ndarray | None = None,
        pred_action: Tensor | np.ndarray | None = None,
        gt_action: Tensor | np.ndarray | None = None,
        attention_map: Tensor | np.ndarray | None = None,
    ) -> None:
        ...
```

### D7: Dependencies — optional extras with lazy imports

**Decision**: Add visualization dependencies as an optional extras group `[project.optional-dependencies] viz = [...]` in pyproject.toml. All visualization modules use **lazy imports** — the `rerun`, `fiftyone`, `captum`, `pytorch_grad_cam` packages are imported inside functions, not at module top level.

```toml
[project.optional-dependencies]
viz = [
    "rerun-sdk>=0.27",
    "pytorch-grad-cam>=1.5",
    "captum>=0.7",
    "umap-learn>=0.5",
    "scikit-learn>=1.3",  # PCA for embedding pre-reduction
]
```

**Version pinning strategy**: Minimum bounds only in `pyproject.toml`. The pixi lockfile pins exact versions for reproducibility. Known-good tested versions: `rerun-sdk==0.27.0`, `pytorch-grad-cam==1.5.4`, `captum==0.7.0`, `umap-learn==0.5.7`, `scikit-learn==1.5.2` (PyTorch 2.5+, Python 3.11+).

**Why optional (revised from original "main deps" decision)**: `fiftyone` bundles MongoDB and has platform constraints. `rerun-sdk` requires Rust binaries. These are heavy and not needed by users who only train/eval without visualization. Lazy imports + optional extras means:
- `pip install yavla` works without viz deps
- `pip install yavla[viz]` adds visualization
- Importing `yavla.visualization.config` always works (pure Python, no heavy deps)
- Importing `yavla.visualization.attention` and calling functions raises `ImportError` with a clear message if deps are missing

**Note**: `fiftyone>=1.2` is already in main deps (pre-existing). It stays in main deps since it was there before this change. New deps go to optional.

**Failure mode**: When a viz function is called without the optional dep installed, it raises:
```python
raise ImportError(
    f"{package} is required for this feature. "
    "Install with: pip install yavla[viz]"
)
```

### D8: "Disabled" semantics — config flag + missing dependency

**Decision**: A visualization feature is "disabled" when EITHER:
1. **Config flag is off** (e.g., `snapshot_enabled=False`, `rerun_enabled=False`) — the code path is never entered, no imports attempted
2. **Dependency is missing** — the function is called but the required package isn't installed → `ImportError` with install instructions

These are two distinct failure modes with different behaviors:

| Condition | Behavior |
|-----------|----------|
| Config off, dep installed | No-op. Zero overhead. No imports. |
| Config off, dep missing | No-op. Zero overhead. No imports. No error. |
| Config on, dep installed | Feature runs normally. |
| Config on, dep missing | `ImportError` with `pip install yavla[viz]` message. |

**Lazy import pattern**: Every module that depends on optional packages uses:
```python
def _import_grad_cam():
    try:
        from pytorch_grad_cam import GradCAM
        return GradCAM
    except ImportError:
        raise ImportError(
            "pytorch-grad-cam is required for attention heatmaps. "
            "Install with: pip install yavla[viz]"
        ) from None
```

This ensures `from yavla.visualization import VizConfig` never triggers heavy imports.

## Risks / Trade-offs

**[Grad-CAM gradient isolation]** → Snapshot runs in isolated context: `model.eval()` + `torch.enable_grad()` + `autocast(False)` + `zero_grad(set_to_none=True)` after. Training gradients saved/restored around the snapshot. See D4 pseudocode for exact mechanism.

**[DDP/FSDP snapshot divergence]** → Snapshots run on rank 0 only. `init_snapshot_state` returns `None` on other ranks. No synchronization barriers introduced. See D4.

**[Hook leaks from pytorch-grad-cam / Captum]** → All hook-based functions use internal context managers with guaranteed cleanup. Testable invariant: hook count before == hook count after. See D3.

**[FiftyOne MongoDB overhead]** → FiftyOne runs a local MongoDB instance. Mitigation: Use `persistent=False` for ephemeral debug sessions. For CI, skip FiftyOne entirely (config-driven). FiftyOne is already in main deps (pre-existing).

**[Rerun on headless servers]** → Always save to `.rrd` files. Viewer spawned only when DISPLAY is available. See D6.

**[300GB dataset + FiftyOne]** → Uniform per-episode subsampling (every Nth frame). At rate=10, ~100K samples. Embedding UMAP uses PCA pre-reduction to 50 dims. See D5.

**[Optional dependency breakage]** → Upper-bounded version constraints + known-good pinned set. Lazy imports mean missing deps only error when features are actually used, not on import. See D7/D8.

**[Hook conflicts between pytorch-grad-cam and Captum]** → Never run simultaneously. Snapshot uses one method at a time, cleans up before the next. Post-hoc analysis is sequential. Context manager enforces this.

**[Model not built yet]** → All functions accept `nn.Module` + target layer reference — model-agnostic. HF-specific kwargs passed via `model_forward_kwargs` dict. No coupling to a specific model class.

**[CI testability]** → Core logic (config, heatmap math, subsampling) is unit-testable with mocks. Integration tests requiring FiftyOne/Rerun/GPU are marked `@pytest.mark.integration` and skipped in minimal CI. Each spec notes its testability boundary.
