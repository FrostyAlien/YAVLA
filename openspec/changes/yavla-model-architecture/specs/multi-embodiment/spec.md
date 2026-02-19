## ADDED Requirements

### Requirement: ActionSpaceSpec and ProprioSpec
`ActionSpaceSpec` SHALL define `names: list[str]`, `units: list[str]`, `limits: Tensor [action_dim, 2]`, `frame: str`, and `control_mode: str`. `ProprioSpec` SHALL define `names`, `units`, and `limits`. Both SHALL be serializable to JSON.

#### Scenario: Per-robot action space
- **WHEN** `ActionSpaceSpec` is constructed for a 7-DOF arm with gripper
- **THEN** `names` SHALL have 7 entries and `limits` shape SHALL be `[7, 2]`

### Requirement: ObservationAdapter per robot
`ObservationAdapter` SHALL normalize raw sensor data into canonical `ObservationBatch` using robot-specific camera configs, proprio mappings, and image preprocessing.

#### Scenario: Multi-camera robot
- **WHEN** a robot has wrist + overhead cameras with different resolutions
- **THEN** the adapter SHALL resize both to the configured resolution and populate `images` dict with camera-name keys

### Requirement: Per-robot action decoder
Each robot type SHALL have its own `ActionDecoder` instance configured with its `ActionSpaceSpec` for correct unnormalization.

#### Scenario: Cross-embodiment checkpoint
- **WHEN** a checkpoint trained on robot A is loaded for robot B with `strict=False`
- **THEN** shared backbone weights SHALL load; robot-specific adapter/decoder weights SHALL be re-initialized

### Requirement: Embodiment registry
An `embodiment_registry` SHALL map robot names to `(ActionSpaceSpec, ProprioSpec, ObservationAdapterConfig)` tuples, enabling config-driven robot selection.

#### Scenario: Register and select robot
- **WHEN** `embodiment_registry.get("franka_panda")` is called
- **THEN** it SHALL return the Franka-specific specs and adapter config
