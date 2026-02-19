## ADDED Requirements

### Requirement: IntegrationMode and capability negotiation
`IntegrationMode` SHALL be an `Enum` with `READOUT` and `JOINT_TOKENS`. `BackboneCapabilities` SHALL declare `supported_modes: set[IntegrationMode]` and `supports_kv_cache: bool`. `ActionHeadRequirements` SHALL declare `required_mode: IntegrationMode` and `accepts_readout: bool`. `validate_integration(backbone, head)` SHALL raise `IncompatibleError` if incompatible.

#### Scenario: Compatible readout mode
- **WHEN** backbone supports `{READOUT}` and head requires `READOUT`
- **THEN** `validate_integration` SHALL return `IntegrationMode.READOUT`

#### Scenario: Incompatible joint-token mode
- **WHEN** backbone supports only `{READOUT}` and head requires `JOINT_TOKENS`
- **THEN** `validate_integration` SHALL raise `IncompatibleError`

#### Scenario: Dual-expert supports both modes
- **WHEN** `DualExpertBackbone` declares `supported_modes={READOUT, JOINT_TOKENS}`
- **THEN** `validate_integration` SHALL succeed for heads requiring either mode

### Requirement: Protocol interfaces (full set)
`VisionEncoderProto`, `BackboneProto`, `ActionHeadProto`, `ActionDecoderProto`, `TokenMergerProto`, `ProprioEncoderProto`, and `ObservationAdapterProto` SHALL be `@runtime_checkable Protocol` classes.

#### Scenario: TokenMergerProto conformance
- **WHEN** `isinstance(merger, TokenMergerProto)` is called on a conforming class
- **THEN** it SHALL return `True`

#### Scenario: ObservationAdapterProto conformance
- **WHEN** a class implements `adapt(raw_obs) → ObservationBatch`
- **THEN** `isinstance(adapter, ObservationAdapterProto)` SHALL return `True`

### Requirement: ABC base classes (full set)
`VisionEncoderBase`, `BackboneBase`, `ActionHeadBase`, `ActionDecoderBase`, `TokenMergerBase`, `ProprioEncoderBase`, and `ObservationAdapterBase` SHALL extend `nn.Module` and `ABC`.

#### Scenario: Incomplete subclass
- **WHEN** a subclass of `ActionHeadBase` omits `compute_loss`
- **THEN** instantiation SHALL raise `TypeError`

### Requirement: PEFT wrapper protocol
`PEFTWrappable` Protocol SHALL declare `apply_peft(config: PEFTConfig) → None` and `merge_peft() → None`. Modules that support PEFT adaptation SHALL implement this protocol.

#### Scenario: Apply LoRA to backbone
- **WHEN** `backbone.apply_peft(LoRAConfig(rank=16))` is called
- **THEN** LoRA adapters SHALL be injected into attention+MLP projections

#### Scenario: Merge for deployment
- **WHEN** `backbone.merge_peft()` is called after training
- **THEN** LoRA weights SHALL be merged into base weights and adapter layers removed
