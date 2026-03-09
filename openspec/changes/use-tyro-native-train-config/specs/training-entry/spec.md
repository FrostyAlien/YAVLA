## MODIFIED Requirements

### Requirement: scripts/train.py loads full TrainConfig defaults from YAML
`scripts/train.py` SHALL treat `--config` as defaults for the full `TrainConfig`, not only `TrainingConfig`.

The training config file SHALL use the nested `TrainConfig` schema rooted at top-level `training:` and `policy:` blocks.
The file-loading path SHALL construct a typed `TrainConfig` default object before tyro applies CLI overrides.
Unsupported config shapes SHALL be rejected with a clear error instead of being coerced through loader-specific fallback behavior.

#### Scenario: Nested YAML populates typed training and policy defaults
- **WHEN** `scripts/train.py` is invoked with `--config <path>` and the YAML contains both `training:` and `policy:` blocks in the supported nested schema
- **THEN** the effective tyro defaults SHALL include typed values from both blocks before CLI overrides are applied

#### Scenario: Unsupported top-level config shape is rejected
- **WHEN** `scripts/train.py` is invoked with `--config <path>` and the YAML does not conform to the supported nested `TrainConfig` schema
- **THEN** the script SHALL fail with a clear error instead of coercing the file into a partial config object

#### Scenario: CLI override wins over YAML default
- **WHEN** a config file sets `policy.embodiment.action_dim: 14` and the CLI passes `--policy.embodiment.action-dim 16`
- **THEN** the effective config SHALL use `16`

## ADDED Requirements

### Requirement: train-facing polymorphic config fields use explicit typed variants
The training entry capability SHALL expose train-facing polymorphic config fields through explicit supported variants in the config schema instead of relying on loader-specific string dispatch against placeholder base config types.

Each supported variant SHALL have a concrete config shape with stable YAML and CLI names.
Variant-specific fields SHALL only be accepted when the corresponding variant is selected.

#### Scenario: Default polymorphic variant loads without subtype reconstruction rules
- **WHEN** a train config omits an optional polymorphic field that has a documented default variant
- **THEN** `scripts/train.py` SHALL construct the effective config using that explicit default variant without invoking generic subtype reconstruction logic

#### Scenario: Alternate polymorphic variant loads from YAML defaults
- **WHEN** a train config selects a supported non-default variant for a train-facing polymorphic field and provides that variant's fields in YAML
- **THEN** the effective config SHALL contain the corresponding concrete variant with those typed field values before CLI overrides are applied

#### Scenario: Variant-specific overrides apply through tyro
- **WHEN** a train config selects a supported polymorphic variant and the CLI overrides one of that variant's fields
- **THEN** the effective config SHALL retain the selected variant and apply the override to that variant-specific field
