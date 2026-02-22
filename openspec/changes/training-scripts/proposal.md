## Why

YAVLA has a complete model layer (Phase 1 MVP, 157 tests passing) and dataset layer, but no training loop. The `yavla-model-architecture` design explicitly scopes out training infrastructure as a separate concern. Without a training script, the policy cannot be trained — `VLAPolicy.compute_loss()` and `TrainingBatch` exist but nothing calls them. This change delivers the canonical training pipeline: config-driven optimizer/scheduler, a `train_step()` function, a `Trainer` class built on HuggingFace Accelerate, and a `scripts/train.py` entry point.

Reference implementations studied:
- **LeRobot** (`huggingface/lerobot`, SHA `5f152322`): Accelerate-first training loop; `make_optimizer_and_scheduler` factory; `use_policy_training_preset`; `accelerator.backward()`, `accelerator.clip_grad_norm_()`, `accelerator.save_state()`/`load_state()` for checkpoints; WandB logging
- **π0 / openpi** (`Physical-Intelligence/openpi`, SHA `981483dc`): `TrainConfig` dataclass, AdamW + cosine-with-warmup, EMA (0.99 decay), bfloat16, gradient checkpointing, separate LR scale for backbone vs action expert
- **OpenVLA** (`openvla/openvla`): FSDP, gradient checkpointing enabled by default in `VLAConfig`, LoRA fine-tuning path, per-component freeze options

## What Changes

- Expand `src/yavla/training/config.py`: add `OptimizerConfig`, `SchedulerConfig`, expand `TrainingConfig` with optimizer, scheduler, precision (maps to Accelerate `mixed_precision`), num_steps, log_freq, save_freq, output_dir, resume, gradient_checkpointing, gradient_accumulation_steps
- Add `src/yavla/training/optim.py`: `make_optimizer_and_scheduler()` factory using built-in PyTorch `SequentialLR([LinearLR, CosineAnnealingLR])` for linear warmup + cosine decay; per-param-group LR support (backbone vs head)
- Add `src/yavla/training/trainer.py`: `train_step()` function and `Trainer` class built on HuggingFace Accelerate — AMP, gradient clipping, checkpointing, and distributed training all delegated to Accelerate APIs (no manual GradScaler/autocast/safetensors code)
- Add `scripts/train.py`: tyro CLI entry point composing `TrainingConfig` + `PolicyConfig`; launchable via `python` or `accelerate launch`
- Add `get_optimizer_preset() -> OptimizerConfig | None` to `PolicyBase` protocol
- Update `configs/train.yaml` with full training config example

## Capabilities

### New Capabilities

- `training-config`: `OptimizerConfig` + `SchedulerConfig` + expanded `TrainingConfig` dataclasses; YAML + tyro CLI compatible; `use_policy_preset` flag lets policy override optimizer defaults
- `training-optim`: `make_optimizer_and_scheduler()` factory; linear warmup + cosine decay via built-in PyTorch `SequentialLR`; per-param-group LR (backbone at `backbone_lr_scale * lr`, head at full `lr`)
- `training-loop`: `train_step()` + `Trainer` built on Accelerate; mixed precision via `Accelerator(mixed_precision=...)`, gradient clipping via `accelerator.clip_grad_norm_()`, gradient checkpointing via `model.gradient_checkpointing_enable(use_reentrant=False)`
- `training-checkpoint`: full state save/resume via `accelerator.save_state()` / `load_state()` (model in safetensors, optimizer/scheduler/RNG automatically handled)
- `training-logging`: WandB via Accelerate tracker integration (`accelerator.log()`); console logging via `accelerator.print()`
- `training-entry`: `scripts/train.py` tyro CLI entry point; `pixi run train`; compatible with `accelerate launch` for multi-GPU

### Modified Capabilities

- `model-protocols`: add `get_optimizer_preset() -> OptimizerConfig | None` to `PolicyBase`

## Impact

- **New code**: `src/yavla/training/optim.py`, `src/yavla/training/trainer.py`, `scripts/train.py`
- **Modified**: `src/yavla/training/config.py`, `src/yavla/models/protocols.py`, `configs/train.yaml`
- **Dependencies**: No new external dependencies — uses torch, accelerate, wandb, tyro (all already in pyproject.toml). Accelerate replaces ~200 lines of manual AMP/checkpoint/clipping code.
- **Contract**: `train_step(policy, batch, accelerator, optimizer, config) -> tuple[LossDict, float]` — clean boundary, testable in isolation
- **Subtask of**: `yavla-model-architecture` (training infrastructure explicitly scoped out of that change's design.md)
