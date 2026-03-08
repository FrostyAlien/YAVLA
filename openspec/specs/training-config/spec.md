# training-config Specification

## Purpose
TBD - created by archiving change mvp-training-fixes. Update Purpose after archive.

## Requirements
### Requirement: Training startup validates policy and data dimensions before optimization
Before training begins, the system SHALL validate that the configured model embodiment matches the training data.

At minimum, startup validation SHALL check:
- `training.dataset.action_chunk_size` against `policy.action_head.chunk_len` when `action_chunk_size` is configured
- the first batch chunk length against `policy.action_head.chunk_len`
- the first batch action dimension against `policy.action_head.action_dim`
- the first batch proprio dimension against `policy.proprio_encoder.proprio_dim`

If any validation fails, the training entry point SHALL terminate before `Trainer.run()` starts and SHALL surface a clear fix-oriented message.

#### Scenario: Action chunk length mismatch exits before training
- **WHEN** `training.dataset.action_chunk_size` is set and it differs from `policy.action_head.chunk_len`
- **THEN** the training entry point SHALL exit before the first optimization step with a message showing both values

#### Scenario: Action dimension mismatch exits before training
- **WHEN** the first batch has `actions.shape[-1]` different from `policy.action_head.action_dim`
- **THEN** the training entry point SHALL exit before the first optimization step with a message showing expected and actual action dimension

#### Scenario: First-batch chunk length mismatch exits before training
- **WHEN** the first batch has `actions.shape[1]` different from `policy.action_head.chunk_len`
- **THEN** the training entry point SHALL exit before the first optimization step with a message showing expected and actual chunk length

#### Scenario: Proprio dimension mismatch exits before training
- **WHEN** the first batch has `observations.proprio.shape[-1]` different from `policy.proprio_encoder.proprio_dim`
- **THEN** the training entry point SHALL exit before the first optimization step with a message showing expected and actual proprio dimension
