## ADDED Requirements

### Requirement: PolicyConfig dataclass tree
`PolicyConfig` SHALL be a `@dataclass` composing `VisionEncoderConfig`, `ProprioEncoderConfig`, `TokenMergerConfig`, `BackboneConfig`, `MLPHeadConfig`, `FreezeConfig`, `ActionSpaceSpec`, `ProprioSpec`, and `dt_hz: float = 10.0` (control frequency for inference), with a `config_version: str` field. It SHALL be compatible with tyro CLI. `FreezeConfig` controls which VLM modules to freeze and which to apply LoRA to via the `peft` library.

#### Scenario: Default construction
- **WHEN** `PolicyConfig()` is constructed with defaults
- **THEN** all sub-configs SHALL have their default values and `config_version` SHALL be `"1.0"`

#### Scenario: Tyro CLI override
- **WHEN** `tyro.cli(PolicyConfig, args=["--merger.num-readout-tokens", "128"])` is called
- **THEN** `config.merger.num_readout_tokens` SHALL be `128` with all other fields at defaults

### Requirement: VLAPolicy nn.Module
`VLAPolicy(nn.Module)` SHALL compose vision encoder, proprio encoder, token merger, backbone, action head, and action decoder into a single module. It SHALL expose `forward(batch: TrainingBatch) → LossDict` for training and `predict(obs: ObservationBatch) → ActionChunk` for inference.

<!-- Ref: LeRobot PreTrainedPolicy pattern: https://github.com/huggingface/lerobot/blob/5f152322/src/lerobot/policies/pretrained.py#L45
     Ref: OpenVLA-OFT forward pass: https://github.com/moojink/openvla-oft/blob/e4287e94/prismatic/extern/hf/modeling_prismatic.py#L571-L638 -->

#### Scenario: Training forward pass
- **WHEN** `policy.forward(training_batch)` is called
- **THEN** it SHALL:
  1. `vision_encoder.encode_images(observations.images)` → image_embeds `[B, 256, D]`
  2. `proprio_encoder.encode_proprio(observations.proprio)` → proprio_embeds `[B, 1, D]`
3. Tokenize language via `backbone.tokenizer(language)` → `input_ids`, `language_attn_mask`; embed via `backbone.base_model.get_input_embeddings()(input_ids)` → lang_embeds `[B, L, D]` (uses unwrapped base model, NOT PeftModel wrapper)
4. `merger.merge(image_embeds, proprio_embeds, lang_embeds, language_attn_mask)` → `(inputs_embeds, attention_mask, token_type_ids)` — merger appends readout tokens internally
5. `backbone.forward(inputs_embeds, attention_mask, token_type_ids)` → `BackboneOutput` — NO dummy labels; hybrid mask is driven by `token_type_ids` and works for both `train()` and `eval()` in MVP (no KV cache)
  6. `action_head.compute_loss(backbone_output, batch)` → `LossDict`

#### Scenario: Inference predict
- **WHEN** `policy.predict(observation_batch)` is called in `torch.no_grad()` context
- **THEN** it SHALL follow steps 1-5 above, then `action_head.predict(backbone_output)` → `action_decoder.decode(prediction)` → `ActionChunk`

#### Scenario: Module access
- **WHEN** `policy.vision_encoder`, `policy.proprio_encoder`, `policy.backbone`, `policy.action_head` are accessed
- **THEN** each SHALL return the corresponding `nn.Module` submodule

### Requirement: build_policy factory
`build_policy(config: PolicyConfig) → VLAPolicy` SHALL:
1. Load `PaliGemmaForConditionalGeneration` via `transformers` → `base_model`
2. Store `base_model` reference (for vision encoder's `get_image_features()` and policy's `get_input_embeddings()` — these MUST use the unwrapped model, not the PeftModel wrapper)
3. Apply freeze via `param.requires_grad_(False)` for modules matching `FreezeConfig.freeze_modules` prefixes on `base_model`
4. If `FreezeConfig.lora_target_modules` is non-empty: apply LoRA via `peft_model = peft.get_peft_model(base_model, peft.LoraConfig(target_modules=..., r=..., lora_alpha=..., lora_dropout=...))` — note: peft freezes ALL base params in the wrapper, only LoRA adapters are trainable; non-VLM modules (action head, proprio encoder, merger) are separate nn.Modules and remain trainable
5. If peft applied: call `peft_model.enable_input_require_grads()` (required for gradient flow through frozen base to LoRA adapters under gradient checkpointing)
6. If `config.backbone.gradient_checkpointing` is `True`: set `base_model.config.use_cache = False` (required for gradient checkpointing; MVP does not use KV cache)
7. If `config.backbone.gradient_checkpointing` is `True`: enable gradient checkpointing via `base_model.gradient_checkpointing_enable()`
8. Pass `num_readout_tokens` from `TokenMergerConfig` to the backbone constructor
9. Construct vision encoder with `base_model` reference (unwrapped)
10. Run `validate_integration(backbone, head)`
11. Compose all modules into `VLAPolicy`

<!-- Ref: peft library for LoRA: https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraConfig
     Ref: OpenVLA-OFT freeze + LoRA pattern: https://github.com/moojink/openvla-oft/blob/e4287e94/prismatic/training/train.py#L89-L120
     Ref: PaliGemma model sharing — a SINGLE PaliGemmaForConditionalGeneration instance is loaded by build_policy.
     The backbone OWNS this instance. The vision encoder receives a reference to the unwrapped base model to call get_image_features().
     The policy calls backbone.base_model.get_input_embeddings() for language embedding.
     When LoRA is applied, forward() goes through the PeftModel wrapper; vision/embedding access uses the unwrapped base_model. -->

#### Scenario: Successful build with defaults
- **WHEN** `build_policy(PolicyConfig())` is called
- **THEN** it SHALL return a `VLAPolicy` with SigLIP encoder, proprio encoder, concat merger, VLM backbone, and MLP action head

#### Scenario: Integration validation failure
- **WHEN** `build_policy` is called with a head requiring `JOINT_TOKENS` and a backbone supporting only `READOUT`
- **THEN** it SHALL raise `IncompatibleError` before constructing the policy

### Requirement: Action decoder (MVP)
`SimpleActionDecoder(ActionDecoderBase)` SHALL take `ActionPrediction` from the head, unnormalize actions using `ActionSpaceSpec.limits`, and return an `ActionChunk`. No temporal ensembling.

#### Scenario: Unnormalize actions
- **WHEN** `decode(prediction)` is called with normalized actions in `[-1, 1]`
- **THEN** `ActionChunk.actions` SHALL be scaled to the range defined by `ActionSpaceSpec.limits`

#### Scenario: Training target normalization contract
- **WHEN** `ActionSpaceSpec.limits` is provided and `compute_loss` is used during training
- **THEN** `TrainingBatch.actions` SHALL be normalized to `[-1, 1]` to match head prediction space; decoder unnormalization applies to inference outputs only

#### Scenario: Pass-through when no normalization
- **WHEN** `ActionSpaceSpec.limits` is `None`
- **THEN** actions SHALL pass through unchanged

### Requirement: save_pretrained / from_pretrained
`VLAPolicy.save_pretrained(path)` SHALL write `config.json`, `model.safetensors`, `action_stats.json`, and `embodiment.json`. `VLAPolicy.from_pretrained(path)` SHALL load and reconstruct the full policy.

#### Scenario: Round-trip serialization
- **WHEN** `policy.save_pretrained(tmp)` then `loaded = VLAPolicy.from_pretrained(tmp)`
- **THEN** `loaded(batch)` SHALL produce identical output to `policy(batch)` (within float tolerance)

#### Scenario: Embodiment mismatch
- **WHEN** `from_pretrained(path)` loads a checkpoint trained with `action_dim=7` but current config has `action_dim=6`
- **THEN** it SHALL raise `ValueError` unless `strict=False` is passed
