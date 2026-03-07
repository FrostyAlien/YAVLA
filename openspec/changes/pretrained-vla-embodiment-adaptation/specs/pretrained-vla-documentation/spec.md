## ADDED Requirements

### Requirement: Training docs distinguish exact-dimension and pretrained-VLA modes
Repository documentation SHALL explain the difference between exact-dimension MVP training and pretrained-VLA embodiment adaptation. The training guide SHALL state when users MUST keep exact dataset/model dimensions, when they MAY use model maximum dimensions larger than the active embodiment, and what extra padding and masking behavior pretrained-VLA mode introduces.

#### Scenario: Reader chooses the correct training mode
- **WHEN** a user consults the training guide before a first training run
- **THEN** the docs SHALL tell them to use exact-dimension configuration for MVP smoke training unless they are intentionally training or loading a pretrained multi-embodiment VLA

#### Scenario: Reader understands pretrained-VLA dimension rules
- **WHEN** a user reads the pretrained-VLA training section
- **THEN** the docs SHALL explain that model max dimensions can exceed active embodiment dimensions only in the explicit embodiment adaptation mode

### Requirement: Documentation includes pretrained-VLA configuration and checkpoint examples
Repository documentation SHALL provide concrete examples for configuring pretrained-VLA embodiment adaptation and for loading pretrained checkpoints with strict and non-strict embodiment validation.

#### Scenario: Pretrained-VLA config example is available
- **WHEN** a user needs to configure a pretrained policy with `max_action_dim` larger than `action_dim`
- **THEN** the docs SHALL include a YAML example showing the relevant policy fields and explain how active embodiment dimensions map onto the wider model

#### Scenario: Checkpoint loading example is available
- **WHEN** a user needs to load a pretrained checkpoint for another embodiment
- **THEN** the docs SHALL include an example of strict load behavior and an example of non-strict rebinding to a smaller compatible embodiment
