## ADDED Requirements

### Requirement: train_step function
`train_step(policy, batch: TrainingBatch, accelerator: Accelerator, optimizer, config: TrainingConfig) -> tuple[LossDict, float]` SHALL be implemented in `src/yavla/training/trainer.py`. It SHALL:
1. Forward pass: call `policy.forward(batch)` → `LossDict`
2. Backward: call `accelerator.backward(loss_dict.total)`
3. Compute grad norm via `accelerator.clip_grad_norm_(policy.parameters(), config.optimizer.grad_clip_norm)` which returns the pre-clip total norm
4. Call `optimizer.step()` and `optimizer.zero_grad(set_to_none=True)`
5. Return `(loss_dict, grad_norm)`

Note: AMP (autocast + GradScaler) is handled automatically by `Accelerator(mixed_precision=...)`. No manual `autocast` or `GradScaler` code needed.

#### Scenario: Returns LossDict and grad norm
- **WHEN** `train_step(policy, batch, accelerator, optimizer, config)` is called
- **THEN** it SHALL return a `LossDict` with `total` scalar and a `float` grad norm

#### Scenario: Gradient clipping applied
- **WHEN** gradients exceed `grad_clip_norm`
- **THEN** the global grad norm after clipping SHALL be ≤ `grad_clip_norm`

### Requirement: Trainer class
`Trainer` SHALL be a class in `src/yavla/training/trainer.py` with `__init__(policy, config: TrainingConfig, train_dataloader)`. It SHALL:
1. Create `Accelerator(mixed_precision=config.precision, log_with="wandb" if config.wandb else None)`
2. Call `make_optimizer_and_scheduler(policy, config, num_training_steps)`
3. Call `accelerator.prepare(policy, optimizer, train_dataloader, scheduler)` to wrap all objects
4. Register scheduler for checkpointing via `accelerator.register_for_checkpointing(scheduler)`

`Trainer.run() -> None` SHALL:
1. If `config.resume` is True, load the latest checkpoint via `accelerator.load_state()`
2. Loop for `config.num_steps` steps, calling `train_step` each iteration
3. Call `scheduler.step()` after each `train_step`
4. Call `accelerator.save_state()` every `config.save_freq` steps
5. Log metrics via `accelerator.log()` every `config.log_freq` steps

#### Scenario: Step counter advances
- **WHEN** `Trainer.run()` is called
- **THEN** it SHALL execute exactly `config.num_steps` optimizer steps

#### Scenario: Gradient checkpointing enabled
- **WHEN** `config.gradient_checkpointing=True`
- **THEN** `Trainer` SHALL call `policy.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})` before training
