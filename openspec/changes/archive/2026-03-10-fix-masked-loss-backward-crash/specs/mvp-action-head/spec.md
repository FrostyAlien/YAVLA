## MODIFIED Requirements

### Requirement: MLP regression loss masks padded action timesteps
`MLPRegressionHead.compute_loss()` SHALL treat `TrainingBatch.action_mask` with `True = padded/invalid` polarity and SHALL exclude padded timesteps from the L1 reduction.

If `action_mask` is `None` or contains no padded positions, the loss SHALL remain equivalent to the unmasked L1 loss over the full action chunk.
If every compared timestep is masked as padded, the returned zero-valued loss SHALL remain connected to autograd and SHALL be safe to pass to `backward()`.

#### Scenario: Padded timesteps do not contribute to loss
- **WHEN** `TrainingBatch.action_mask` marks a suffix of the target chunk as padded
- **THEN** `compute_loss()` SHALL reduce only over unmasked elements and SHALL ignore the padded suffix

#### Scenario: Missing mask preserves prior behavior
- **WHEN** `TrainingBatch.action_mask` is `None`
- **THEN** `compute_loss()` SHALL compute the same scalar L1 loss as an unmasked reduction over the full target tensor

#### Scenario: Fully padded chunk remains backward-safe
- **WHEN** every timestep in the compared target chunk is masked as padded
- **THEN** `compute_loss()` SHALL return a finite zero-valued loss that remains safe to pass to `backward()`
