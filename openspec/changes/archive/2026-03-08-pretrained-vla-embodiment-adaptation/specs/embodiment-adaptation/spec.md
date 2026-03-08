## ADDED Requirements

### Requirement: Pretrained-VLA embodiment mode is explicit
YAVLA SHALL expose an explicit pretrained-VLA embodiment adaptation mode that distinguishes active embodiment dimensions from model maximum dimensions. In this mode, policy configuration SHALL record `action_dim`, `proprio_dim`, `max_action_dim`, and `max_proprio_dim`, and the system MUST reject configurations where an active dimension exceeds its model maximum.

#### Scenario: Smaller embodiment uses a wider pretrained policy
- **WHEN** a policy is configured with `action_dim=14`, `proprio_dim=14`, `max_action_dim=32`, and `max_proprio_dim=32`
- **THEN** YAVLA SHALL build the model at width 32 while treating only the first 14 action and proprio dimensions as active for that embodiment

#### Scenario: Invalid active dimension is rejected
- **WHEN** a policy is configured with `action_dim=18` and `max_action_dim=14`
- **THEN** YAVLA SHALL raise a configuration validation error before training or checkpoint loading begins

### Requirement: Training pads smaller embodiments and masks inactive dimensions
In pretrained-VLA embodiment adaptation mode, the dataset layer SHALL remain embodiment-exact, while the model and training pipeline SHALL adapt smaller embodiments to the configured model maximum dimensions. Proprio inputs SHALL be zero-padded to `max_proprio_dim` before the proprio encoder, and action loss computation SHALL combine timestep validity with active action-dimension masks so inactive dimensions do not contribute to loss. If the combined mask contains no valid elements, loss computation SHALL return zero loss and MUST NOT produce `NaN`.

#### Scenario: Smaller proprio batch is padded inside the model path
- **WHEN** a dataset batch provides proprio tensors with shape `[B, 14]` for a policy configured with `max_proprio_dim=32`
- **THEN** YAVLA SHALL preserve the dataset tensor as 14-wide at the data boundary and zero-pad it to 32-wide only inside the pretrained-VLA adaptation path

#### Scenario: Inactive action dimensions are excluded from loss
- **WHEN** a training batch has `chunk_len=8`, only the first 14 of 32 action dimensions are active, and the batch also carries timestep padding near an episode boundary
- **THEN** loss computation SHALL ignore both the padded timesteps and the inactive action dimensions

#### Scenario: Fully masked batch does not produce NaN
- **WHEN** the combined timestep-and-dimension mask contains zero valid action elements
- **THEN** YAVLA SHALL return a finite zero loss instead of dividing by zero or emitting `NaN`

### Requirement: Inference returns the active embodiment action slice
In pretrained-VLA embodiment adaptation mode, action heads MAY predict the model maximum action width internally, but `VLAPolicy.predict()` and downstream decoding SHALL expose only the active embodiment action dimensions for the selected embodiment.

#### Scenario: Prediction is sliced to the active embodiment width
- **WHEN** a 32-wide pretrained policy is loaded with an embodiment configured for `action_dim=14`
- **THEN** inference SHALL return `ActionChunk.actions` with shape `[B, chunk_len, 14]`

### Requirement: Checkpoints record max-width and active-embodiment metadata
Checkpoint save and load flows SHALL distinguish model maximum dimensions from active embodiment dimensions. `save_pretrained()` SHALL persist both max-width metadata and active embodiment metadata, and `from_pretrained()` SHALL validate them explicitly. `strict=True` SHALL require exact max-width compatibility, while `strict=False` SHALL permit rebinding to another embodiment only when the target configuration provides explicit embodiment metadata and its active dimensions do not exceed the checkpoint's max dimensions.

#### Scenario: Legacy checkpoint metadata is rejected
- **WHEN** `from_pretrained()` loads a checkpoint whose `config.json` lacks an `embodiment` block or whose `embodiment.json` lacks max-width fields
- **THEN** YAVLA SHALL raise a validation error stating that legacy embodiment-less checkpoints are unsupported

#### Scenario: Strict load rejects max-width mismatch
- **WHEN** `from_pretrained(strict=True)` loads a checkpoint with `max_action_dim=32` into a config with `max_action_dim=24`
- **THEN** YAVLA SHALL raise a validation error before weights are applied

#### Scenario: Non-strict load allows a smaller embodiment
- **WHEN** `from_pretrained(strict=False)` loads a checkpoint with `max_action_dim=32` into a config with `action_dim=14` and `max_action_dim=32`
- **THEN** YAVLA SHALL load the shared pretrained weights and expose predictions using the 14-dimensional active embodiment
