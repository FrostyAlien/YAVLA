## ADDED Requirements

### Requirement: Training collate function converts dict samples to TrainingBatch
`training_collate_fn` SHALL accept a `list[dict[str, Any]]` of dataset samples and return a single `TrainingBatch` instance with a nested `ObservationBatch`.

#### Scenario: Standard LeRobot dataset with images, proprio, language, and actions
- **WHEN** samples contain keys `observation.state`, `observation.images.<cam_name>`, `action`, and `task`
- **THEN** the collate function SHALL return a `TrainingBatch` where:
  - `observations.images` is a `dict[str, Tensor]` keyed by camera name with shape `[B, C, H, W]`
  - `observations.proprio` is a `Tensor` with shape `[B, D_proprio]`
  - `observations.language` is a `list[str]` of length `B`
  - `actions` is a `Tensor` with shape `[B, chunk_len, action_dim]`

#### Scenario: Multiple cameras
- **WHEN** samples contain `observation.images.cam_left` and `observation.images.cam_right`
- **THEN** `observations.images` SHALL contain both keys: `{"cam_left": [B,C,H,W], "cam_right": [B,C,H,W]}`

#### Scenario: No language key present
- **WHEN** samples do not contain a `task` key
- **THEN** `observations.language` SHALL be `None`

#### Scenario: Action padding mask present
- **WHEN** samples contain an `action_is_pad` key
- **THEN** `TrainingBatch.action_mask` SHALL be a boolean `Tensor` with shape `[B, chunk_len]`

#### Scenario: No action padding mask
- **WHEN** samples do not contain an `action_is_pad` key
- **THEN** `TrainingBatch.action_mask` SHALL be `None`

#### Scenario: Action padding mask polarity
- **WHEN** `action_is_pad` is present
- **THEN** `TrainingBatch.action_mask` SHALL preserve the same polarity: `True` = padded/invalid (matching the `action_is_pad` semantics)

#### Scenario: 2D actions raise ValueError
- **WHEN** stacked actions have shape `[B, action_dim]` (2D, no chunk dimension)
- **THEN** the collate function SHALL raise `ValueError` with a message directing the user to set `action_chunk_size`

#### Scenario: Missing proprio raises ValueError
- **WHEN** no sample contains an `observation.state` key
- **THEN** the collate function SHALL raise `ValueError` indicating that proprio data is required

#### Scenario: Extra/unknown keys silently ignored
- **WHEN** samples contain keys not matching any known convention (e.g. `timestamp`, `episode_index`)
- **THEN** the collate function SHALL silently ignore those keys without error

### Requirement: Image key detection uses prefix convention
The collate function SHALL detect image keys by matching the prefix `observation.images.` and extract the camera name as the suffix after the prefix.

#### Scenario: Key matching
- **WHEN** a batched dict contains key `observation.images.top`
- **THEN** it SHALL be placed in `observations.images["top"]`

#### Scenario: Non-image observation keys are not treated as images
- **WHEN** a batched dict contains `observation.state` or `observation.velocity`
- **THEN** these keys SHALL NOT appear in `observations.images`

### Requirement: dt_hz and chunk_len are constructor parameters
The collate function SHALL receive `dt_hz` and `chunk_len` as configuration at construction time, not from sample data.

#### Scenario: Values passed through to TrainingBatch
- **WHEN** the collate is constructed with `dt_hz=10.0` and `chunk_len=5`
- **THEN** every returned `TrainingBatch` SHALL have `dt_hz=10.0` and `chunk_len=5`

### Requirement: create_training_dataloader wires the collate function
`create_training_dataloader(config, *, dt_hz: float, chunk_len: int)` SHALL pass the training collate function as the `collate_fn` argument to `create_dataloader()`.

#### Scenario: DataLoader yields TrainingBatch
- **WHEN** `create_training_dataloader(config, dt_hz=10.0, chunk_len=5)` is called
- **THEN** iterating the returned `DataLoader` SHALL yield `TrainingBatch` instances, not raw dicts

#### Scenario: Collate receives dt_hz and chunk_len from explicit parameters
- **WHEN** `create_training_dataloader()` constructs the collate function
- **THEN** it SHALL pass `dt_hz` and `chunk_len` from its own keyword arguments to the `TrainingCollate` constructor
