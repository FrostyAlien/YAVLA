## ADDED Requirements

### Requirement: BackboneBase exposes embed_language method
`BackboneBase` SHALL provide an abstract method `embed_language(texts: list[str]) -> tuple[Tensor, Tensor]` that tokenizes and embeds a batch of text strings, returning `(embeddings [B, T, D], attention_mask [B, T])`. The embeddings MUST have the same hidden dimension as `backbone.hidden_dim`. The attention mask MUST be 1 for real tokens and 0 for padding.

#### Scenario: Single text input
- **WHEN** `embed_language(["pick up the red block"])` is called
- **THEN** returns embeddings with shape `[1, T, D]` where `D == backbone.hidden_dim` and attention_mask with shape `[1, T]` with all ones (no padding)

#### Scenario: Batched text with padding
- **WHEN** `embed_language(["short", "a longer sentence here"])` is called
- **THEN** returns embeddings with shape `[2, T_max, D]` and attention_mask `[2, T_max]` where shorter sequences have 0s in the mask for padded positions

#### Scenario: Empty string input
- **WHEN** `embed_language([""])` is called
- **THEN** returns valid embeddings and mask (backbone-specific tokenization of empty string, e.g., BOS/EOS tokens)

### Requirement: BackboneBase tokenizer property is non-abstract
`BackboneBase.tokenizer` SHALL be a concrete property that raises `NotImplementedError` by default. Subclasses MAY override it. The pipeline (`VLAPolicy`) MUST NOT call `backbone.tokenizer` directly — `embed_language()` is the interface for text processing.

#### Scenario: Default tokenizer access
- **WHEN** `backbone.tokenizer` is accessed on a subclass that does not override it
- **THEN** raises `NotImplementedError`

#### Scenario: PaliGemma tokenizer access
- **WHEN** `backbone.tokenizer` is accessed on `PaliGemmaBackbone`
- **THEN** returns the PaliGemma tokenizer instance (for serialization use)

### Requirement: VLM registry builds vision encoder and backbone pair
A `vlm_registry` SHALL map `backbone.type` string to a builder function with signature `(config: BackboneConfig) -> tuple[VisionEncoderBase, BackboneBase]`. `build_policy()` SHALL call `vlm_registry.build(config.backbone)` to obtain both components.

#### Scenario: Build PaliGemma VLM
- **WHEN** `vlm_registry.build(BackboneConfig(type="paligemma"))` is called
- **THEN** returns a `(PaliGemmaVisionEncoder, PaliGemmaBackbone)` pair with matching hidden dimensions

#### Scenario: Unknown VLM type
- **WHEN** `vlm_registry.build(BackboneConfig(type="nonexistent"))` is called
- **THEN** raises `KeyError` listing available VLM types

#### Scenario: New VLM registration
- **WHEN** a new VLM builder is registered via `vlm_registry.register("new_vlm")`
- **THEN** `build_policy()` can construct a policy with `type="new_vlm"` without any code changes to `build_policy()` or `VLAPolicy`

### Requirement: VLAPolicy.encode_observations uses backbone abstraction
`VLAPolicy.encode_observations()` SHALL call `self.backbone.embed_language(texts)` for language encoding instead of accessing `backbone.tokenizer` or `backbone.base_model.get_input_embeddings()` directly.

#### Scenario: Forward pass with language
- **WHEN** `policy.forward(batch)` is called with `batch.observations.language = ["pick up block"]`
- **THEN** `encode_observations()` calls `backbone.embed_language(["pick up block"])` and uses the returned embeddings and mask

#### Scenario: Forward pass without language
- **WHEN** `policy.forward(batch)` is called with `batch.observations.language = None`
- **THEN** `encode_observations()` calls `backbone.embed_language([""])` (empty string fallback, same as current behavior)

### Requirement: PaliGemma code isolated in dedicated module
All PaliGemma-specific code SHALL reside in `src/yavla/models/backbones/paligemma.py`, including `PaliGemmaBackbone`, `PaliGemmaVisionEncoder`, and the VLM builder function. The builder SHALL handle HF model loading, LoRA wrapping, freezing, and gradient checkpointing.

#### Scenario: PaliGemma builder produces working policy
- **WHEN** `build_policy(PolicyConfig(backbone=BackboneConfig(type="paligemma")))` is called
- **THEN** the resulting policy produces identical outputs to the pre-refactor `build_policy()` for the same config and inputs

#### Scenario: LoRA configuration
- **WHEN** `PolicyConfig.freeze.lora_target_modules` is set
- **THEN** the PaliGemma builder applies LoRA via PEFT and the backbone's `model` property returns the `PeftModel` wrapper

### Requirement: Serialization compatibility preserved
`save_pretrained()` and `from_pretrained()` SHALL produce and load checkpoints compatible with the pre-refactor format. State dict key prefixes MUST remain `backbone.`, `vision_encoder.`, etc. `from_pretrained()` SHALL read `config.backbone.type` to dispatch to the correct VLM builder.

#### Scenario: Save and reload PaliGemma policy
- **WHEN** a PaliGemma policy is saved with `save_pretrained(path)` and reloaded with `from_pretrained(path)`
- **THEN** the reloaded policy produces identical outputs for the same inputs

#### Scenario: Load pre-refactor checkpoint
- **WHEN** `from_pretrained()` loads a checkpoint saved before this refactor (no `backbone.type` in config)
- **THEN** defaults to `type="paligemma"` and loads successfully
