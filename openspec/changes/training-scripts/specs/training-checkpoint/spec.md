## ADDED Requirements

### Requirement: Checkpoint save
`Trainer.save_checkpoint(step: int) -> None` SHALL call `accelerator.save_state(f"{config.output_dir}/checkpoint-{step}")`. Accelerate's `save_state()` automatically writes:
- Model weights in safetensors format (`model.safetensors`)
- Optimizer state dict
- Scheduler state dict (registered via `accelerator.register_for_checkpointing`)
- RNG states for reproducibility

#### Scenario: Files written
- **WHEN** `save_checkpoint(1000)` is called
- **THEN** `checkpoint-1000/` SHALL exist under `output_dir` with model and training state files

### Requirement: Checkpoint resume
`Trainer._load_latest_checkpoint() -> int` SHALL:
1. Scan `output_dir` for `checkpoint-*` directories, pick the highest step
2. Call `accelerator.load_state(checkpoint_path)` to restore all state
3. Return the restored step number

#### Scenario: Resume restores step
- **WHEN** a checkpoint at step 5000 exists and `config.resume=True`
- **THEN** `Trainer.run()` SHALL start from step 5000, not step 0

#### Scenario: No checkpoint — start from 0
- **WHEN** `output_dir` has no checkpoints and `config.resume=True`
- **THEN** training SHALL start from step 0 without error
