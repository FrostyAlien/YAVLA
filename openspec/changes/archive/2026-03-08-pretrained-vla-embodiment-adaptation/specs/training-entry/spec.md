## MODIFIED Requirements

### Requirement: scripts/train.py loads full TrainConfig defaults from YAML
`scripts/train.py` SHALL treat `--config` as defaults for the full `TrainConfig`, not only `TrainingConfig`.

If the YAML contains top-level `training:` or `policy:` keys, the script SHALL parse it as a nested `TrainConfig`.
Legacy flat train-config YAML is unsupported and SHALL be rejected with a clear error instead of being coerced into `TrainingConfig`.

#### Scenario: Nested YAML populates policy defaults
- **WHEN** `scripts/train.py` is invoked with `--config <path>` and the YAML contains both `training:` and `policy:` blocks
- **THEN** the effective tyro defaults SHALL include values from both blocks before CLI overrides are applied

#### Scenario: Legacy flat YAML is rejected
- **WHEN** `scripts/train.py` is invoked with `--config <path>` and the YAML contains only flat `TrainingConfig` fields
- **THEN** the script SHALL fail with an error stating that the legacy flat format is unsupported and that the config must use top-level `training:` and `policy:` blocks

#### Scenario: CLI override wins over YAML default
- **WHEN** a config file sets `policy.embodiment.action_dim: 14` and the CLI passes `--policy.embodiment.action-dim 16`
- **THEN** the effective config SHALL use `16`
