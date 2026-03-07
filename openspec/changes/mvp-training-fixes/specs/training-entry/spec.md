## ADDED Requirements

### Requirement: scripts/train.py loads full TrainConfig defaults from YAML
`scripts/train.py` SHALL treat `--config` as defaults for the full `TrainConfig`, not only `TrainingConfig`.

If the YAML contains top-level `training:` or `policy:` keys, the script SHALL parse it as a nested `TrainConfig`.
If the YAML omits both keys, the script SHALL treat it as a legacy flat `TrainingConfig` document, preserve backward compatibility for training fields, and leave policy fields at their typed defaults until overridden by CLI flags.

#### Scenario: Nested YAML populates policy defaults
- **WHEN** `scripts/train.py` is invoked with `--config <path>` and the YAML contains both `training:` and `policy:` blocks
- **THEN** the effective tyro defaults SHALL include values from both blocks before CLI overrides are applied

#### Scenario: Legacy flat YAML remains supported
- **WHEN** `scripts/train.py` is invoked with `--config <path>` and the YAML contains only flat `TrainingConfig` fields
- **THEN** the effective config SHALL use those training defaults and SHALL preserve the typed default `PolicyConfig`

#### Scenario: CLI override wins over YAML default
- **WHEN** a config file sets `policy.action_head.action_dim: 14` and the CLI passes `--policy.action-head.action-dim 16`
- **THEN** the effective config SHALL use `16`

### Requirement: scripts/train.py seeds reproducible training before model and data construction
`scripts/train.py` SHALL call `accelerate.utils.set_seed()` before building the policy or dataloader.

For this MVP fix, the seed source SHALL be `training.dataset.seed` so the existing data seed also controls global training reproducibility.

#### Scenario: Seed is applied before policy construction
- **WHEN** `training.dataset.seed` is set to a concrete integer in the effective config
- **THEN** `set_seed(seed)` SHALL run before `build_policy(...)` and before dataloader creation

