## ADDED Requirements

### Requirement: Typed training batches support device movement
`ObservationBatch` and `TrainingBatch` SHALL expose a public batch-transport operation that returns the same typed container with all contained tensor fields moved to a requested device.

The operation SHALL recurse through nested tensor leaves, including image tensors, proprio tensors, action tensors, and optional mask tensors, while preserving non-tensor metadata.

#### Scenario: Move nested tensors and preserve metadata
- **WHEN** a `TrainingBatch` containing `observations.images`, `observations.proprio`, `actions`, optional masks, language strings, `dt_hz`, and `chunk_len` is moved to device `D`
- **THEN** the returned value SHALL still be a `TrainingBatch` containing an `ObservationBatch`, every tensor leaf SHALL be on device `D`, and the language strings, `dt_hz`, and `chunk_len` values SHALL remain unchanged

### Requirement: Typed training batch transport preserves optional structure
The typed batch transport capability SHALL preserve optional fields and structural identity across transport operations.

Optional tensor fields that are absent SHALL remain absent, and keyed collections such as multi-camera image dictionaries SHALL preserve their keys.

#### Scenario: Optional fields and camera keys are preserved
- **WHEN** a `TrainingBatch` is moved and its `action_mask`, `action_dim_mask`, `observations.timestamps`, or `observations.masks` fields are `None`, and `observations.images` contains multiple camera keys
- **THEN** each absent optional field SHALL remain `None` in the returned typed batch and the set of camera keys in `observations.images` SHALL remain unchanged
