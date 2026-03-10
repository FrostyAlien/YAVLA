## MODIFIED Requirements

### Requirement: Training pads smaller embodiments and masks inactive dimensions
In pretrained-VLA embodiment adaptation mode, the dataset layer SHALL remain embodiment-exact, while the model and training pipeline SHALL adapt smaller embodiments to the configured model maximum dimensions. Proprio inputs SHALL be zero-padded to `max_proprio_dim` before the proprio encoder, and action loss computation SHALL combine timestep validity with active action-dimension masks so inactive dimensions do not contribute to loss. If the combined mask contains no valid elements, loss computation SHALL return a finite zero-valued loss that remains safe to pass to `backward()` and MUST NOT produce `NaN`.

#### Scenario: Smaller proprio batch is padded inside the model path
- **WHEN** a dataset batch provides proprio tensors with shape `[B, 14]` for a policy configured with `max_proprio_dim=32`
- **THEN** YAVLA SHALL preserve the dataset tensor as 14-wide at the data boundary and zero-pad it to 32-wide only inside the pretrained-VLA adaptation path

#### Scenario: Inactive action dimensions are excluded from loss
- **WHEN** a training batch has `chunk_len=8`, only the first 14 of 32 action dimensions are active, and the batch also carries timestep padding near an episode boundary
- **THEN** loss computation SHALL ignore both the padded timesteps and the inactive action dimensions

#### Scenario: Fully masked batch remains backward-safe
- **WHEN** the combined timestep-and-dimension mask contains zero valid action elements
- **THEN** YAVLA SHALL return a finite zero-valued loss that remains safe to pass to `backward()` instead of dividing by zero or emitting `NaN`
