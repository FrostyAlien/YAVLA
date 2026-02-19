## ADDED Requirements

### Requirement: VisionEncoderProto protocol
`VisionEncoderProto` SHALL be a `@runtime_checkable Protocol` requiring `output_dim: int` property, `num_patches: int` property, and `encode_images(images: dict[str, Tensor]) → Tensor` method returning image embeddings `[B, num_patches, output_dim]`. The encoder holds a reference to the backbone's PaliGemma instance internally — callers do NOT pass the model.

#### Scenario: Protocol conformance check
- **WHEN** `isinstance(encoder, VisionEncoderProto)` is called on a class implementing all required methods
- **THEN** it SHALL return `True`

### Requirement: BackboneProto protocol
`BackboneProto` SHALL be a `@runtime_checkable Protocol` requiring `capabilities: BackboneCapabilities` property, `hidden_dim: int` property, `tokenizer` property (for language tokenization in policy wiring), and `forward(inputs_embeds, attention_mask, token_type_ids) → BackboneOutput` method.

#### Scenario: Protocol conformance check
- **WHEN** `isinstance(backbone, BackboneProto)` is called on a conforming class
- **THEN** it SHALL return `True`

### Requirement: ActionHeadProto protocol
`ActionHeadProto` SHALL be a `@runtime_checkable Protocol` requiring `requirements: ActionHeadRequirements` property, `compute_loss(backbone_output: BackboneOutput, batch: TrainingBatch) → LossDict` method, and `predict(backbone_output: BackboneOutput) → ActionPrediction` method.

#### Scenario: Protocol conformance check
- **WHEN** `isinstance(head, ActionHeadProto)` is called on a conforming class
- **THEN** it SHALL return `True`

### Requirement: ActionDecoderProto protocol
`ActionDecoderProto` SHALL be a `@runtime_checkable Protocol` requiring `action_space_spec: ActionSpaceSpec` property and `decode(pred: ActionPrediction) → ActionChunk` method.

#### Scenario: Protocol conformance check
- **WHEN** `isinstance(decoder, ActionDecoderProto)` is called on a conforming class
- **THEN** it SHALL return `True`

### Requirement: IntegrationMode and capability negotiation
`IntegrationMode` SHALL be an `Enum` with values `READOUT` and `JOINT_TOKENS`. `BackboneCapabilities` SHALL declare `supported_modes: set[IntegrationMode]` and `supports_kv_cache: bool`. `ActionHeadRequirements` SHALL declare `required_mode: IntegrationMode` and `accepts_readout: bool`. A `validate_integration(backbone, head)` function SHALL raise `IncompatibleError` if the head's required mode is not in the backbone's supported modes.

#### Scenario: Compatible readout mode
- **WHEN** backbone supports `{READOUT}` and head requires `READOUT`
- **THEN** `validate_integration` SHALL return `IntegrationMode.READOUT` without error

#### Scenario: Incompatible joint-token mode
- **WHEN** backbone supports only `{READOUT}` and head requires `JOINT_TOKENS`
- **THEN** `validate_integration` SHALL raise `IncompatibleError`

### Requirement: ProprioEncoderProto protocol
`ProprioEncoderProto` SHALL be a `@runtime_checkable Protocol` requiring `output_dim: int` property and `encode_proprio(proprio: Tensor) → Tensor` method that maps `[B, proprio_dim]` to `[B, 1, D]`.

<!-- Ref: Oracle review — proprio projection must be a separate module, not inline in merger.
     Ref: π0 action token projection pattern: https://github.com/Physical-Intelligence/openpi/blob/981483dc/src/openpi/models/pi0.py#L108-L130 -->

#### Scenario: Protocol conformance check
- **WHEN** `isinstance(encoder, ProprioEncoderProto)` is called on a class implementing `output_dim` and `encode_proprio`
- **THEN** it SHALL return `True`

### Requirement: TokenMergerProto protocol
`TokenMergerProto` SHALL be a `@runtime_checkable Protocol` requiring `merge(vision_tokens, proprio_tokens, language_tokens, language_attn_mask) → tuple[Tensor, Tensor, Tensor]` method returning `(inputs_embeds, attention_mask, token_type_ids)`. The merger owns readout token creation internally — callers do NOT pass readout tokens. The merger does NOT produce `position_ids` — the backbone lets PaliGemma compute them internally. The `language_attn_mask` parameter propagates tokenizer padding masks into the merged `attention_mask`.

<!-- Ref: PaliGemma token_type_ids: https://github.com/huggingface/transformers/blob/556312cd/src/transformers/models/paligemma/modeling_paligemma.py#L134-L138 -->

#### Scenario: Protocol conformance check
- **WHEN** `isinstance(merger, TokenMergerProto)` is called on a conforming class
- **THEN** it SHALL return `True`

### Requirement: ABC base classes
`VisionEncoderBase(nn.Module, ABC)`, `BackboneBase(nn.Module, ABC)`, `ActionHeadBase(nn.Module, ABC)`, `ActionDecoderBase(nn.Module, ABC)`, and `ProprioEncoderBase(nn.Module, ABC)` SHALL provide abstract method stubs matching their respective Protocols.

#### Scenario: Incomplete subclass error
- **WHEN** a subclass of `ActionHeadBase` does not implement `compute_loss`
- **THEN** instantiation SHALL raise `TypeError`
