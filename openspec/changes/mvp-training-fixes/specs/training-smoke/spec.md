## ADDED Requirements

### Requirement: Repository provides a low-resource smoke training config for aloha_sim
The repository SHALL provide `configs/train_smoke.yaml` as a low-resource first-run configuration for `lerobot/aloha_sim`.

The smoke config SHALL include explicit policy embodiment dimensions and bounded runtime defaults suitable for a short validation run.

#### Scenario: Smoke config pins aloha_sim embodiment dimensions
- **WHEN** `configs/train_smoke.yaml` is loaded
- **THEN** it SHALL set `policy.action_head.action_dim: 14`, `policy.proprio_encoder.proprio_dim: 14`, and a dataset action chunk size matching `policy.action_head.chunk_len`

#### Scenario: Smoke config limits runtime and loader cost
- **WHEN** `configs/train_smoke.yaml` is loaded
- **THEN** it SHALL set a short run budget including `num_steps: 10`, `log_freq: 1`, `save_freq: 10`, `batch_size: 2`, `num_workers: 0`, and `drop_last: true`

#### Scenario: Smoke config uses adapter-style fine-tuning defaults
- **WHEN** `configs/train_smoke.yaml` is loaded
- **THEN** it SHALL freeze `vision_tower` and `multi_modal_projector` and SHALL target LoRA adapters at least on `q_proj` and `v_proj`

### Requirement: Default train.yaml is an explicit nested TrainConfig example
`configs/train.yaml` SHALL be a nested `TrainConfig` example with top-level `training:` and `policy:` blocks.

It SHALL include explicit `aloha_sim` embodiment dimensions so the default example does not rely on mismatched policy defaults.

#### Scenario: Default config includes explicit policy section
- **WHEN** `configs/train.yaml` is loaded as YAML
- **THEN** it SHALL contain both top-level `training:` and `policy:` mappings

#### Scenario: Default config declares aloha_sim dimensions
- **WHEN** `configs/train.yaml` is loaded as YAML
- **THEN** `policy.action_head.action_dim` and `policy.proprio_encoder.proprio_dim` SHALL both equal `14`
