## Context

YAVLA's model layer is complete (Phase 1 MVP, 157 tests passing). `VLAPolicy.compute_loss()` accepts a `TrainingBatch` and returns a `LossDict`, but nothing calls it. `src/yavla/training/` exists with a minimal `TrainingConfig` (dataset + viz only) and no optimizer, scheduler, or training loop. `scripts/train.py` does not exist. `configs/train.yaml` only has dataset config.

HuggingFace Accelerate (`>=1.5`, already in `pyproject.toml`) provides automatic mixed precision, gradient clipping, full-state checkpointing (safetensors by default since v1.12), distributed training, and gradient accumulation — all through a thin wrapper around PyTorch primitives. Building on Accelerate avoids reimplementing these features and gives us multi-GPU/FSDP support for free.

Reference implementations:
- **LeRobot** (`huggingface/lerobot`, SHA `5f152322`): Accelerate-first training loop; `make_optimizer_and_scheduler` factory; `use_policy_training_preset`; `accelerator.backward()`, `accelerator.clip_grad_norm_()`, `accelerator.save_state()`/`load_state()` for checkpoints; WandB logging. [train.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_train.py)
- **π0 / openpi** (`Physical-Intelligence/openpi`, SHA `981483dc`): `TrainConfig` dataclass; AdamW + cosine-with-warmup; EMA (0.99 decay); bfloat16; gradient checkpointing; separate LR scale for backbone vs action expert. [config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)
- **OpenVLA** (`openvla/openvla`): `enable_gradient_checkpointing=True` in `VLAConfig` by default; FSDP for distributed; LoRA fine-tuning path; per-component freeze options. [vla.py](https://github.com/openvla/openvla/blob/main/prismatic/conf/vla.py)

## Goals / Non-Goals

**Goals:**
- `OptimizerConfig` + `SchedulerConfig` dataclasses, YAML + tyro CLI compatible
- `make_optimizer_and_scheduler()` factory with linear warmup + cosine decay (built-in PyTorch `SequentialLR`)
- `train_step()` as the canonical unit: forward → loss → backward → clip → step
- `Trainer` class built on HuggingFace Accelerate: step loop, logging, checkpoint save/resume, mixed precision, gradient clipping — all delegated to Accelerate APIs
- `scripts/train.py` tyro CLI entry point, launchable via `accelerate launch`
- Gradient checkpointing via `model.gradient_checkpointing_enable(use_reentrant=False)`
- WandB logging (loss, LR, grad norm) via Accelerate tracker integration
- `get_optimizer_preset()` on `PolicyBase` for policy-specific optimizer defaults

**Non-Goals:**
- EMA — Phase 6 of this change
- Evaluation loop — Phase 6 of this change
- RL / online training — separate change
- Custom CUDA kernels or FlashAttention

## Decisions

### D1: Accelerate as the training foundation — not an add-on

**Choice:** `Trainer` creates an `Accelerator(mixed_precision=config.precision)` and uses it for all training operations: `accelerator.prepare()` wraps model/optimizer/dataloader/scheduler; `accelerator.backward(loss)` replaces manual `loss.backward()`; `accelerator.clip_grad_norm_()` replaces manual unscale + clip; `accelerator.save_state()`/`load_state()` replaces manual checkpoint code.

**Why Accelerate from day one:** Accelerate is already a dependency (`>=1.5` in pyproject.toml). It handles AMP (GradScaler internally), gradient clipping (with automatic unscaling for mixed precision), full-state checkpointing (model in safetensors, optimizer/scheduler/RNG via torch.save), and distributed training — all through a thin API. Building on it avoids ~200 lines of manual AMP/checkpoint/clipping code and gives us multi-GPU/FSDP for free. LeRobot uses this exact pattern.

**Why not manual AMP/GradScaler:** `Accelerator(mixed_precision="bf16")` + `accelerator.backward()` handles scaling internally. Manual `GradScaler` + `autocast` + `scaler.unscale_()` is error-prone boilerplate that Accelerate eliminates.

### D2: `train_step()` is the canonical unit — not a method on `Trainer`

**Choice:** `train_step(policy, batch, accelerator, optimizer, config) -> tuple[LossDict, float]` is a standalone function in `trainer.py`, not a method on `Trainer`. `Trainer` calls it in a loop.

**Why standalone:** A standalone function is directly unit-testable without constructing a full `Trainer`. It also makes the contract explicit — the function signature documents exactly what a training step needs. This matches how LeRobot's `update_policy` is a standalone function called from the training loop.

**Signature change from original:** `scaler: GradScaler` is replaced by `accelerator: Accelerator` — the accelerator handles scaling internally.

### D3: `use_policy_preset` flag — policy overrides optimizer defaults

**Choice:** `TrainingConfig` has a `use_policy_preset: bool = True` flag. When True, `make_optimizer_and_scheduler()` calls `policy.get_optimizer_preset()` and merges the result over the config defaults. `get_optimizer_preset() -> OptimizerConfig | None` is added to `PolicyBase`.

**Why:** Different action heads have different optimal optimizer settings. An MLP head trains well at `lr=1e-3`; a flow matching head with a frozen backbone needs `lr=1e-4` for the backbone and `lr=5e-4` for the head. Encoding this in the policy (not the config file) means users get good defaults without tuning. LeRobot's `use_policy_training_preset` demonstrates this pattern works in practice.

**Why `None` return:** A policy that doesn't care about optimizer settings returns `None`, and the config defaults are used as-is. No forced override.

### D4: Linear warmup + cosine decay via built-in `SequentialLR`

**Choice:** `SchedulerConfig` supports `name="cosine"` (default) which builds `SequentialLR([LinearLR(start_factor=0.01, total_iters=warmup_steps), CosineAnnealingLR(T_max=remaining_steps, eta_min=lr*min_lr_ratio)], milestones=[warmup_steps])`. Scheduler is stepped **per batch** (not per epoch). All three classes are built-in `torch.optim.lr_scheduler` — no custom code needed.

**Why `SequentialLR` over `LambdaLR`:** `SequentialLR` is the idiomatic PyTorch way to chain schedulers since 1.13. It avoids manual lambda math and is compatible with `state_dict` save/resume. Per-batch stepping gives fine-grained control during warmup.

**Why warmup:** Adam-family optimizers have poor gradient estimates in the first steps — warmup prevents early divergence. π0 uses 10k warmup steps; LeRobot uses policy-specific warmup. Default: `warmup_steps=1000` (configurable).

### D5: Per-param-group LR — backbone at reduced scale

**Choice:** `OptimizerConfig` has `backbone_lr_scale: float = 0.1`. `make_optimizer_and_scheduler()` splits `policy.parameters()` into two groups: backbone params (identified via `policy.backbone.parameters()`) at `lr * backbone_lr_scale`, all other params at full `lr`.

**Why:** Pretrained VLM backbones (PaliGemma 3B) are sensitive to large LR updates — they can catastrophically forget pretrained representations. π0 and OpenVLA both use lower LR for the backbone. A scale factor (not a separate config field) keeps the config simple while enabling the pattern.

### D6: Checkpoints via `accelerator.save_state()` / `load_state()`

**Choice:** `Trainer.save_checkpoint(step)` calls `accelerator.save_state(output_dir/checkpoint-{step})`. Resume calls `accelerator.load_state()` on the latest checkpoint directory.

Accelerate's `save_state()` saves:
- Model weights in safetensors format (default since v1.12, `safe_serialization=True`)
- Optimizer state dict
- Scheduler state dict (via `accelerator.register_for_checkpointing(scheduler)`)
- RNG states for reproducibility
- GradScaler state (if using fp16 AMP)

**Why delegate to Accelerate:** This eliminates ~50 lines of manual `safetensors.torch.save_file` + `torch.save` + directory scanning + state restoration code. It also automatically handles FSDP sharded checkpoints when we scale to multi-GPU. LeRobot uses this exact pattern.

**Why not manual two-file approach:** The original design proposed `model.safetensors` + `training_state.pt`. Accelerate's `save_state()` achieves the same result (safetensors for model, pickle for optimizer/scheduler) but handles edge cases (distributed, sharding, RNG sync) that manual code would miss.

### D7: WandB logging gated by config flag, grad norm logged before clipping

**Choice:** `TrainingConfig.wandb: bool = False` (off by default). When enabled, `Trainer` initializes Accelerate with `log_with="wandb"` and uses `accelerator.log(metrics, step=step)` every `log_freq` steps. Logged: all `LossDict` scalars, `train/lr`, `train/grad_norm` (computed before clipping). Falls back to `accelerator.print()` for console logging when WandB is off.

**Why Accelerate tracker:** `accelerator.log()` automatically handles main-process-only logging in distributed settings. No manual `if accelerator.is_main_process:` guards needed for WandB calls.

**Why grad norm before clipping:** The pre-clip grad norm is the diagnostic signal — it shows whether training is stable. Post-clip norm is always ≤ `grad_clip_norm` and carries no information. π0 and LeRobot both log pre-clip norm.

**Why off by default:** WandB requires an API key and project setup. Off-by-default avoids blocking users who haven't configured it.
