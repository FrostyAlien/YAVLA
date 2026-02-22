## Phase 1: Training Config

- [x] 1.1 Implement `OptimizerConfig` dataclass in `src/yavla/training/config.py` (`name`, `lr`, `weight_decay`, `betas`, `eps`, `grad_clip_norm`, `backbone_lr_scale`)
- [x] 1.2 Implement `SchedulerConfig` dataclass (`name`, `warmup_steps`, `min_lr_ratio`)
- [x] 1.3 Expand `TrainingConfig` with `optimizer`, `scheduler`, `precision` (maps to Accelerate `mixed_precision`), `num_steps`, `log_freq`, `save_freq`, `output_dir`, `resume`, `gradient_checkpointing`, `use_policy_preset`, `wandb`, `gradient_accumulation_steps`
- [x] 1.4 Add `get_optimizer_preset() -> OptimizerConfig | None` to `PolicyBase` in `protocols.py` (default returns `None`)
- [x] 1.5 Update `configs/train.yaml` with full training config example

## Phase 2: Optimizer & Scheduler Factory

- [x] 2.1 Implement `make_optimizer_and_scheduler(policy, config, num_training_steps)` in `src/yavla/training/optim.py`
- [x] 2.2 Implement two param groups: backbone at `lr * backbone_lr_scale`, rest at full `lr`
- [x] 2.3 Implement `SequentialLR([LinearLR, CosineAnnealingLR], milestones=[warmup_steps])` for warmup + cosine decay (built-in PyTorch schedulers, no custom code)
- [x] 2.4 Implement `use_policy_preset` merge: call `policy.get_optimizer_preset()` and override config defaults when non-None

## Phase 3: Training Loop (Accelerate-first)

- [x] 3.1 Implement `train_step(policy, batch, accelerator, optimizer, config) -> tuple[LossDict, float]` in `src/yavla/training/trainer.py` — uses `accelerator.backward()` and `accelerator.clip_grad_norm_()` (no manual GradScaler/autocast)
- [x] 3.2 Implement `Trainer.__init__`: create `Accelerator(mixed_precision=config.precision)`, call `make_optimizer_and_scheduler`, call `accelerator.prepare(policy, optimizer, dataloader, scheduler)`, register scheduler for checkpointing
- [x] 3.3 Implement `Trainer.run()` step loop with scheduler step and periodic checkpoint save via `accelerator.save_state()`
- [x] 3.4 Wire gradient checkpointing: call `policy.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})` when `config.gradient_checkpointing=True`

## Phase 4: Checkpoint Save & Resume (via Accelerate)

- [x] 4.1 Implement `Trainer.save_checkpoint(step)`: call `accelerator.save_state(f"{output_dir}/checkpoint-{step}")`
- [x] 4.2 Implement `Trainer._load_latest_checkpoint()`: scan for highest-step `checkpoint-*` dir, call `accelerator.load_state(path)`, return step
- [x] 4.3 Wire resume into `Trainer.run()`: call `_load_latest_checkpoint()` when `config.resume=True`

## Phase 5: Entry Point & Logging

- [ ] 5.1 Implement `scripts/train.py` with tyro CLI composing `TrainingConfig` + `PolicyConfig`; launchable via `python` or `accelerate launch`
- [ ] 5.2 Wire `pixi run train` to `scripts/train.py` in `pixi.toml`
- [ ] 5.3 Implement logging in `Trainer.run()`: use `accelerator.log()` for WandB when `config.wandb=True`, `accelerator.print()` for console; log `LossDict` scalars + `train/lr` + `train/grad_norm` every `log_freq` steps

## Phase 6: Advanced (post-MVP)

- [ ] 6.1 Implement EMA for policy weights (`ema_decay` in `TrainingConfig`, separate EMA state in checkpoint)
- [ ] 6.2 Add evaluation loop: periodic held-out loss on validation split, log `eval/loss`
