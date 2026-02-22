## ADDED Requirements

### Requirement: scripts/train.py CLI entry point
`scripts/train.py` SHALL use `tyro.cli` to parse a combined `TrainScript` dataclass composing `TrainingConfig` and `PolicyConfig`. It SHALL construct a `VLAPolicy` via `build_policy(policy_config)`, create a dataloader via `create_training_dataloader(training_config)`, construct a `Trainer`, and call `trainer.run()`.

The script SHALL be launchable via both:
- `python scripts/train.py` (single GPU)
- `accelerate launch scripts/train.py` (multi-GPU via Accelerate config)

#### Scenario: YAML + CLI override
- **WHEN** invoked with `--config configs/train.yaml --training.lr 5e-5`
- **THEN** the training SHALL use `lr=5e-5` with all other values from the YAML

### Requirement: pixi run train entry point
`pixi.toml` SHALL have a `train` task wired to `python scripts/train.py`.

#### Scenario: pixi run train
- **WHEN** `pixi run train` is executed
- **THEN** it SHALL invoke `scripts/train.py` without error (with default config)
