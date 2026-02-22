## ADDED Requirements

### Requirement: WandB logging via Accelerate tracker
When `config.wandb=True`, `Trainer` SHALL initialize `Accelerator(log_with="wandb")` and call `accelerator.init_trackers(project_name)`. Every `config.log_freq` steps, it SHALL call `accelerator.log(metrics, step=step)` with: all `LossDict` scalar fields prefixed `train/`, `train/lr` (current LR from scheduler), `train/grad_norm` (pre-clip, from `train_step`).

#### Scenario: Metrics logged at log_freq
- **WHEN** `config.wandb=True` and `config.log_freq=100`
- **THEN** `accelerator.log` SHALL be called at steps 100, 200, 300, ...

#### Scenario: WandB disabled by default
- **WHEN** `config.wandb=False` (default)
- **THEN** `wandb` SHALL NOT be imported or initialized

#### Scenario: Console logging always active
- **WHEN** any training run executes
- **THEN** loss and LR SHALL be printed to console every `log_freq` steps via `accelerator.print()`
