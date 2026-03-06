# Training Guide

This page covers launching training, configuration, optimizer/scheduler tuning, checkpointing, distributed training, and common recipes.

## Quick Start

```bash
# Install
GIT_LFS_SKIP_SMUDGE=1 pixi install -e dev

# Train with default config
pixi run train --config configs/train.yaml

# Verify output
ls outputs/train/checkpoint-*
```

## Launch Methods

### Single GPU

```bash
# Via pixi
pixi run train --config configs/train.yaml

# Direct invocation
python scripts/train.py --config configs/train.yaml
```

### Multi-GPU (Accelerate)

```bash
accelerate launch scripts/train.py --config configs/train.yaml
```

Accelerate handles DDP, mixed precision, and gradient accumulation automatically.

## Configuration

The CLI entry point (`scripts/train.py`) defines a top-level `TrainConfig`:

```python
@dataclass
class TrainConfig:
    training: TrainingConfig   # training loop, optimizer, scheduler, data
    policy: PolicyConfig       # model architecture (vision encoder, backbone, action head, etc.)
```

### YAML-first + CLI-override

Load YAML defaults with `--config`, then override individual fields via CLI flags:

```bash
python scripts/train.py --config configs/train.yaml --training.num-steps 50000 --training.wandb True
```

tyro naming conventions:
- Dots (`.`) for nesting: `--training.optimizer.lr 3e-4`
- Hyphens (`-`) for underscores: `--training.num-steps` maps to `num_steps`
- Booleans: `--training.wandb True` / `--training.wandb False`

## Training Config Reference

`TrainingConfig` is defined in `src/yavla/training/config.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset` | `DataConfig` | `DataConfig(repo_id="lerobot/aloha_sim")` | Dataset and dataloader settings (see [Dataset Usage](dataset-layer/usage.md)) |
| `viz` | `VizConfig` | `VizConfig()` | Visualization settings (snapshots, rerun) |
| `optimizer` | `OptimizerConfig` | see below | Optimizer hyperparameters |
| `scheduler` | `SchedulerConfig` | see below | LR scheduler hyperparameters |
| `precision` | `str` | `"bf16"` | Mixed precision mode (`"no"`, `"fp16"`, `"bf16"`) |
| `num_steps` | `int` | `100_000` | Total optimizer steps |
| `log_freq` | `int` | `100` | Log metrics every N optimizer steps |
| `save_freq` | `int` | `5000` | Save checkpoint every N optimizer steps |
| `output_dir` | `str` | `"outputs/train"` | Checkpoint and output directory |
| `resume` | `bool` | `False` | Resume from latest checkpoint in `output_dir` |
| `gradient_checkpointing` | `bool` | `True` | Enable activation checkpointing to save VRAM |
| `use_policy_preset` | `bool` | `True` | Merge policy-provided optimizer presets |
| `vlm_image_resize_strategy` | `"warp" \| "letterbox"` | `"warp"` | SigLIP image resize strategy used when auto-wiring preprocessing (`dataset.image_transforms=null`) |
| `vlm_image_height_override` | `int \| None` | `None` | Override SigLIP target height (must set with `vlm_image_width_override`) |
| `vlm_image_width_override` | `int \| None` | `None` | Override SigLIP target width (must set with `vlm_image_height_override`) |
| `wandb` | `bool` | `False` | Enable Weights & Biases logging |
| `gradient_accumulation_steps` | `int` | `1` | Micro-batches per optimizer step |

## Optimizer & Scheduler

### OptimizerConfig

Defined in `src/yavla/training/config.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | `"AdamW"` | Optimizer name (only AdamW supported) |
| `lr` | `float` | `1e-4` | Base learning rate |
| `weight_decay` | `float` | `0.01` | Weight decay |
| `betas` | `tuple[float, float]` | `(0.9, 0.999)` | Adam beta coefficients |
| `eps` | `float` | `1e-8` | Adam epsilon |
| `grad_clip_norm` | `float` | `1.0` | Max gradient norm for clipping |
| `backbone_lr_scale` | `float` | `0.1` | Backbone LR multiplier (discriminative LR) |

### SchedulerConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | `"cosine"` | Scheduler name |
| `warmup_steps` | `int` | `1000` | Linear warmup steps |
| `min_lr_ratio` | `float` | `0.1` | Minimum LR as fraction of `lr * backbone_lr_scale` |

### Discriminative Learning Rate

The optimizer splits parameters into two groups:
- **Action head + encoders**: train at full `lr`
- **Backbone (pretrained VLM)**: train at `lr * backbone_lr_scale`

This preserves pretrained features while allowing the task-specific head to adapt quickly. The default 0.1× scale is standard for VLA fine-tuning.

### Warmup + Cosine Decay

The scheduler uses `SequentialLR([LinearLR, CosineAnnealingLR])`:
1. Linear warmup from 1% of target LR over `warmup_steps`
2. Cosine decay to `lr * backbone_lr_scale * min_lr_ratio` over remaining steps

### Policy Presets

When `use_policy_preset=True`, the policy's `get_optimizer_preset()` can override optimizer defaults. Preset fields that differ from `OptimizerConfig()` defaults take priority; all others fall through to the user config.

## Data

YAVLA supports three dataset backends:

| Backend | Best for | Key constraint |
|---------|----------|----------------|
| `default` | Standard LeRobot datasets | Supports `action_chunk_size` (delegates to LeRobot `delta_timestamps["action"]`) |
| `lazy` | Large datasets, temporal features | Supports `delta_timestamps` + `action_chunk_size` |
| `streaming` | Shard-based iteration | No `delta_timestamps` or `action_chunk_size` |

To enable chunked action targets (required by `TrainingCollate`), set `dataset.action_chunk_size` on either `default` or `lazy` backends. Prefer `lazy` for large datasets.

The transform pipeline applies in order: repack → normalize → image transforms.

For full `DataConfig` reference and recipes, see [Dataset Layer Usage](dataset-layer/usage.md).

### SigLIP preprocessing precedence

When training SigLIP-based backbones (e.g., PaliGemma), `scripts/train.py` auto-wires a canonical image preprocessing
recipe into `training.dataset.image_transforms` **only** when `dataset.image_transforms` is omitted / `null`.

Precedence:
- If `training.dataset.image_transforms` is provided (including `[]`), it is used as-is and `vlm_image_resize_strategy`
  is ignored.
- If `training.dataset.image_transforms` is `null`, the wired resize step is selected by `vlm_image_resize_strategy`
  (`warp` or `letterbox`) and followed by SigLIP normalization.

### Multi-camera images

Many real LeRobot datasets include multiple camera views per timestep (e.g., `cam_high`, `cam_left_wrist`, `cam_right_wrist`).
YAVLA supports this via `ObservationBatch.images: dict[str, Tensor]` where each camera tensor has shape `[B, C, H, W]`.

- **Canonical camera order:** cameras are concatenated in ascending lexicographic key order (`sorted(images.keys())`). This is
  deterministic and does not depend on dict insertion order.
- **Token shape:** vision tokens are returned as a single tensor `[B, N_img, D]` with `N_img = K * N_patch_per_camera`
  where `K` is the number of camera keys present.
- **Camera identity (v1):** no explicit camera embeddings; camera identity is implicit via its position in the concatenated token stream.
- **Validation:** all cameras must have the same `[B, C, H, W]` shape; empty `images` is rejected.
- **Compute/VRAM scaling:** `N_img` grows linearly with `K`, so attention cost and VRAM/runtime typically increase with the
  number of cameras.

## Distributed Training

### Multi-GPU with Accelerate

```bash
accelerate launch --num_processes 4 scripts/train.py --config configs/train.yaml
```

The `Trainer` creates an `Accelerator` internally and calls `accelerator.prepare()` on the model, optimizer, dataloader, and scheduler. `DistributedSampler` is added automatically when DDP is active.

### Gradient Accumulation

Simulate larger batch sizes without more VRAM:

```bash
python scripts/train.py --config configs/train.yaml --training.gradient-accumulation-steps 4
```

With `batch_size=32` and `gradient_accumulation_steps=4`, the effective batch size is 128. Logging and checkpointing only fire on real optimizer steps (gated on `accelerator.sync_gradients`).

### Mixed Precision

Controlled by `precision`:

```bash
# BFloat16 (default, recommended for Ampere+)
--training.precision bf16

# Float16
--training.precision fp16

# Disable
--training.precision no
```

## Checkpointing & Resume

### Checkpoint Structure

Checkpoints are saved via `accelerator.save_state()` to `{output_dir}/checkpoint-{step}/`. Each contains the full training state: model weights, optimizer state, scheduler state, and RNG states.

### Save Frequency

Checkpoints save every `save_freq` optimizer steps. A final checkpoint is always saved at the end of training if the last step wasn't already a save boundary.

### Resuming

```bash
python scripts/train.py --config configs/train.yaml --training.resume True
```

On resume, the trainer:
1. Scans `output_dir` for the highest-step `checkpoint-*` directory
2. Restores full training state via `accelerator.load_state()`
3. Fast-forwards the dataloader by `start_step * gradient_accumulation_steps` micro-batches via `skip_first_batches` so no data is replayed

## Logging

### Console

Every `log_freq` steps, the trainer prints:

```
step 100/100000  loss=0.4321  lr=1.00e-04  grad_norm=0.85
```

### WandB

Enable with `--training.wandb True`. Metrics logged per optimizer step:

| Metric | Description |
|--------|-------------|
| `train/loss` | Total loss |
| `train/lr` | Current learning rate |
| `train/grad_norm` | Gradient norm (before clipping) |
| `train/{component}` | Per-component loss breakdown from `LossDict.breakdown` |

The project name is `"yavla"`.

## Freezing & LoRA

`FreezeConfig` is defined in `src/yavla/models/types.py` and nested under `PolicyConfig.freeze`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `freeze_modules` | `list[str]` | `[]` | Module name prefixes to freeze |
| `lora_target_modules` | `list[str]` | `[]` | peft leaf module names for LoRA (e.g. `["q_proj", "v_proj"]`) |
| `lora_r` | `int` | `8` | LoRA rank |
| `lora_alpha` | `int` | `16` | LoRA alpha scaling |
| `lora_dropout` | `float` | `0.0` | LoRA dropout |

### Examples

```bash
# Freeze the backbone entirely
--policy.freeze.freeze-modules '["backbone"]'

# LoRA on attention projections
--policy.freeze.lora-target-modules '["q_proj", "v_proj"]' --policy.freeze.lora-r 16
```

## Common Recipes

### Short test run

```bash
pixi run train --config configs/train.yaml --training.num-steps 100 --training.log-freq 10
```

### Enable WandB logging

```bash
pixi run train --config configs/train.yaml --training.wandb True
```

### Lower learning rate with longer warmup

```bash
pixi run train --config configs/train.yaml \
  --training.optimizer.lr 5e-5 \
  --training.scheduler.warmup-steps 2000
```

### Larger effective batch size via accumulation

```bash
pixi run train --config configs/train.yaml \
  --training.gradient-accumulation-steps 8 \
  --training.dataset.batch-size 16
```

### Resume from checkpoint

```bash
pixi run train --config configs/train.yaml --training.resume True
```

### Multi-GPU training

```bash
accelerate launch --num_processes 4 scripts/train.py \
  --config configs/train.yaml --training.precision bf16
```

### Freeze backbone + LoRA fine-tune

```bash
pixi run train --config configs/train.yaml \
  --policy.freeze.freeze-modules '["backbone"]' \
  --policy.freeze.lora-target-modules '["q_proj", "v_proj"]' \
  --policy.freeze.lora-r 16
```

### Lazy backend with action chunking

```bash
pixi run train --config configs/train.yaml \
  --training.dataset.backend lazy \
  --training.dataset.action-chunk-size 4
```

### SigLIP/PaliGemma image preprocessing

SigLIP/PaliGemma expects dataset-layer preprocessed `pixel_values` (no model-internal processor). When training
SigLIP/PaliGemma backbones:

- `dataset.image_transforms`: omitted / `null` → auto-wire canonical SigLIP transforms based on the loaded checkpoint
  `vision_config.image_size`
- `dataset.image_transforms: []` → explicitly disable preprocessing
- non-empty `dataset.image_transforms` → use your transforms as-is

Canonical recipe uses list form for resize: `Resize([H, W], 3)` (avoid `Resize((H, W))`).

To override the checkpoint-derived resize target, set both `vlm_image_height_override` and
`vlm_image_width_override`. If the override differs from the checkpoint size, training logs a warning (you are
responsible for verifying VLM compatibility).

## Full Config Reference

Complete annotated YAML showing all fields with defaults:

```yaml
# Dataset and dataloader
dataset:
  repo_id: "lerobot/aloha_sim"       # HuggingFace dataset repo ID (required)
  backend: "default"                   # "default" | "lazy" | "streaming"
  batch_size: 32
  num_workers: 4
  persistent_workers: true
  normalize: true
  normalize_mode: "z-score"            # "z-score" | "min-max"
  image_transforms: null               # null/omitted = auto-wire for SigLIP; [] = disable
  video_backend: "pyav"

# Optimizer
optimizer:
  name: "AdamW"
  lr: 1e-4
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1e-8
  grad_clip_norm: 1.0
  backbone_lr_scale: 0.1              # backbone trains at lr * 0.1

# LR scheduler
scheduler:
  name: "cosine"
  warmup_steps: 1000
  min_lr_ratio: 0.1

# Training loop
precision: "bf16"                      # "no" | "fp16" | "bf16"
num_steps: 100000
log_freq: 100
save_freq: 5000
output_dir: "outputs/train"
resume: false
gradient_checkpointing: true
use_policy_preset: true
vlm_image_resize_strategy: "warp"      # "warp" | "letterbox"
vlm_image_height_override: null        # set both height+width to override VLM resize target
vlm_image_width_override: null
wandb: false
gradient_accumulation_steps: 1
```
