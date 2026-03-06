## ADDED Requirements

### Requirement: Vision encoder selection is explicit and config-driven
`PolicyConfig.vision_encoder.type` MUST explicitly control how the policy’s vision encoder is chosen.

The system MUST support a canonical selection mode:

- `type="from_backbone"`: the vision encoder is sourced from the configured VLM backbone builder (the `(VisionEncoderBase, BackboneBase)` pair returned by `vlm_registry.build(...)`).

#### Scenario: Default vision encoder comes from the backbone
- **WHEN** `PolicyConfig.vision_encoder.type == "from_backbone"` and `build_policy(config)` is called
- **THEN** the policy’s `vision_encoder` is the instance returned by the selected VLM backbone builder

### Requirement: Non-backbone vision encoders are built via the vision registry
For any `PolicyConfig.vision_encoder.type` other than `"from_backbone"`, `build_policy()` MUST build the vision encoder using `vision_registry` and MUST use it in the constructed policy.

#### Scenario: Registry-built vision encoder is selected by type
- **WHEN** `PolicyConfig.vision_encoder.type == "<registered-vision-type>"` and `build_policy(config)` is called
- **THEN** the policy’s `vision_encoder` is built via `vision_registry` using the provided vision encoder config

#### Scenario: Unknown vision encoder type is rejected
- **WHEN** `PolicyConfig.vision_encoder.type == "nonexistent"` and `build_policy(config)` is called
- **THEN** the build fails with a `KeyError` that lists the available vision encoder types

### Requirement: Vision tokens are compatible with the backbone embedding space
The effective vision encoder used by the policy MUST produce vision tokens with hidden dimension equal to the backbone embedding dimension:

- `vision_tokens.shape[-1] == backbone.hidden_dim`

If a registry-built vision encoder produces tokens with a different hidden dimension, the system MUST apply a projection so the final vision token stream matches `backbone.hidden_dim`.

#### Scenario: Registry encoder output is projected to backbone.hidden_dim
- **WHEN** a registry-built vision encoder produces tokens shaped `[B, N_img, D_vision]` with `D_vision != backbone.hidden_dim`
- **THEN** the policy uses a projection such that the final vision token stream consumed by the merger/backbone has shape `[B, N_img, backbone.hidden_dim]`

### Requirement: Legacy vision encoder config values remain loadable
For backward compatibility, legacy `PolicyConfig.vision_encoder.type` values that historically indicated “use the VLM’s built-in vision tower” (e.g., the current default `paligemma_siglip`) MUST behave equivalently to `type="from_backbone"`.

#### Scenario: paligemma_siglip behaves like from_backbone
- **WHEN** `PolicyConfig.vision_encoder.type == "paligemma_siglip"` and `build_policy(config)` is called
- **THEN** the policy’s `vision_encoder` is sourced from the VLM backbone builder as if `type="from_backbone"`
