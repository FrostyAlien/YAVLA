## ADDED Requirements

<!--
Refs (prior art / alignment targets):
- LeRobot uses config-driven composition and registry patterns for modular policy components:
  https://github.com/huggingface/lerobot
- PEFT/LoRA is the standard HF adapter workflow (native adapter save/load):
  https://huggingface.co/docs/transformers/en/peft
-->

### Requirement: `build_policy()` is registry-driven and config-based
`build_policy(config: PolicyConfig)` SHALL construct policy modules via the registries for each module family rather
than hardcoding concrete classes. At minimum, these module families SHALL be registry-constructed:

- vision encoder
- proprio encoder
- token merger
- backbone
- action head
- action decoder

Each sub-config MUST include a `type` field selecting the registered implementation.

#### Scenario: Selecting modules by `type`
- **WHEN** `PolicyConfig` specifies `vision_encoder.type="paligemma_siglip"` and `merger.type="concat"` (etc.)
- **THEN** `build_policy` SHALL build the corresponding registered classes via their registries

#### Scenario: Unknown module type is a build-time error
- **WHEN** any sub-config has `type="<unknown>"`
- **THEN** `build_policy` SHALL raise a clear error listing available types for that registry

### Requirement: Factory enforces backbone↔head compatibility at build time
`build_policy` SHALL run capability/requirement negotiation (via `validate_integration(backbone, head)`) before
returning a policy instance.

#### Scenario: Integration validation fails early
- **WHEN** a head requires an integration mode the backbone does not support
- **THEN** `build_policy` SHALL raise an `IncompatibleError` before returning a partially-constructed policy

### Requirement: Exactly one VLM base model instance is loaded and shared
For VLM-backed policies, the factory SHALL load the underlying HuggingFace base model exactly once and SHALL ensure
that:

- The backbone’s forward path uses the active model (base or PEFT-wrapped).
- Vision encoding and language embedding lookup use a reference to the unwrapped base model when required by the HF
  API surface.

This avoids silent “two copies of the backbone” bugs and ensures checkpointing behavior is well-defined.

#### Scenario: Vision and language paths share the base model reference
- **WHEN** a policy is built
- **THEN** the vision encoder and language embedding path SHALL reference the same underlying base model instance that the backbone was initialized from

### Requirement: Factory propagates computed dimensions through the module graph
The factory SHALL propagate derived dimensions (e.g., `backbone.hidden_dim`) into downstream modules, rather than
relying on duplicated constants in configs.

#### Scenario: Proprio encoder uses backbone hidden dim
- **WHEN** the factory constructs the proprio encoder for a given backbone
- **THEN** the proprio encoder output dimension SHALL match `backbone.hidden_dim`

