# pretrained-vla-documentation Specification

## Purpose
Define the documentation requirements for exact-dimension versus pretrained-VLA training flows, including configuration examples, checkpoint-loading guidance, and unsupported legacy formats.

## Requirements
### Requirement: Training docs distinguish exact-dimension and pretrained-VLA modes
Repository documentation SHALL explain the difference between exact-dimension MVP training and pretrained-VLA embodiment adaptation. The training guide SHALL state when users MUST keep exact dataset/model dimensions, when they MAY use model maximum dimensions larger than the active embodiment, and what extra padding and masking behavior pretrained-VLA mode introduces.

#### Scenario: Reader chooses the correct training mode
- **WHEN** a user consults the training guide before a first training run
- **THEN** the docs SHALL tell them to use exact-dimension configuration for MVP smoke training unless they are intentionally training or loading a pretrained multi-embodiment VLA

#### Scenario: Reader understands pretrained-VLA dimension rules
- **WHEN** a user reads the pretrained-VLA training section
- **THEN** the docs SHALL explain that model max dimensions can exceed active embodiment dimensions only in the explicit embodiment adaptation mode

### Requirement: Documentation includes pretrained-VLA configuration and checkpoint examples
Repository documentation SHALL provide concrete examples for configuring pretrained-VLA embodiment adaptation and for loading pretrained checkpoints with strict and non-strict embodiment validation. The docs SHALL also state that legacy flat training YAML and embodiment-less checkpoints are unsupported.

#### Scenario: Pretrained-VLA config example is available
- **WHEN** a user needs to configure a pretrained policy with `max_action_dim` larger than `action_dim`
- **THEN** the docs SHALL include a YAML example showing the relevant policy fields and explain how active embodiment dimensions map onto the wider model

#### Scenario: Checkpoint loading example is available
- **WHEN** a user needs to load a pretrained checkpoint for another embodiment
- **THEN** the docs SHALL include an example of strict load behavior and an example of non-strict rebinding to a smaller compatible embodiment

#### Scenario: Reader sees unsupported legacy formats
- **WHEN** a user consults the docs with an older flat train config or an embodiment-less checkpoint
- **THEN** the docs SHALL state that these legacy formats are unsupported and that the current nested train config plus embodiment-aware checkpoint format is required
