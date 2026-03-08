# mvp-action-head Specification

## Purpose
TBD - created by archiving change mvp-training-fixes. Update Purpose after archive.

## Requirements
### Requirement: MLP regression loss masks padded action timesteps
`MLPRegressionHead.compute_loss()` SHALL treat `TrainingBatch.action_mask` with `True = padded/invalid` polarity and SHALL exclude padded timesteps from the L1 reduction.

If `action_mask` is `None` or contains no padded positions, the loss SHALL remain equivalent to the unmasked L1 loss over the full action chunk.

#### Scenario: Padded timesteps do not contribute to loss
- **WHEN** `TrainingBatch.action_mask` marks a suffix of the target chunk as padded
- **THEN** `compute_loss()` SHALL reduce only over unmasked elements and SHALL ignore the padded suffix

#### Scenario: Missing mask preserves prior behavior
- **WHEN** `TrainingBatch.action_mask` is `None`
- **THEN** `compute_loss()` SHALL compute the same scalar L1 loss as an unmasked reduction over the full target tensor

#### Scenario: Fully padded chunk stays finite
- **WHEN** every timestep in the compared target chunk is masked as padded
- **THEN** `compute_loss()` SHALL return a finite zero-valued loss rather than `NaN` or `Inf`

### Requirement: MLP regression loss rejects action-shape mismatch
`MLPRegressionHead.compute_loss()` SHALL fail fast if the target action tensor does not exactly match the configured `chunk_len` and `action_dim`.

It SHALL NOT silently truncate the temporal axis or the action-dimension axis.

#### Scenario: Action dimension mismatch raises an error
- **WHEN** `batch.actions.shape[-1]` differs from `MLPHeadConfig.action_dim`
- **THEN** `compute_loss()` SHALL raise an error describing the expected and actual action dimension

#### Scenario: Chunk length mismatch raises an error
- **WHEN** `batch.actions.shape[1]` differs from `MLPHeadConfig.chunk_len`
- **THEN** `compute_loss()` SHALL raise an error describing the expected and actual chunk length
