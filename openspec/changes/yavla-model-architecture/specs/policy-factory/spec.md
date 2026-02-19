## ADDED Requirements

### Requirement: build_policy factory
`build_policy(config: PolicyConfig) → VLAPolicy` SHALL build each module from its sub-config via the appropriate registry, run `validate_integration(backbone, head)`, and compose into `VLAPolicy`.

#### Scenario: Successful build
- **WHEN** `build_policy(PolicyConfig())` is called with compatible configs
- **THEN** it SHALL return a fully constructed `VLAPolicy`

#### Scenario: Integration validation failure
- **WHEN** head requires `JOINT_TOKENS` but backbone only supports `READOUT`
- **THEN** `IncompatibleError` SHALL be raised before constructing the policy

#### Scenario: PEFT auto-application
- **WHEN** `config.backbone.peft` is not `None`
- **THEN** `build_policy` SHALL call `backbone.apply_peft(config.backbone.peft)` after construction

### Requirement: PolicyConfig dataclass tree
`PolicyConfig` SHALL compose all sub-configs (`VisionEncoderConfig`, `TokenMergerConfig`, `BackboneConfig`, action head config, `ActionDecoderConfig`, `PEFTConfig | None`) with `config_version: str` and tyro CLI compatibility.

#### Scenario: Nested tyro override
- **WHEN** `tyro.cli(PolicyConfig, args=["--backbone.peft.rank", "32"])` is called
- **THEN** `config.backbone.peft.rank` SHALL be `32`
