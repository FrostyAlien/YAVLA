## 1. Data Containers and Types

- [x] 1.1 Create `src/yavla/models/` package with `__init__.py`, `types.py`
- [x] 1.2 Implement `ObservationBatch`, `TokenBatch` (with `token_type_ids`), `BackboneOutput`, `ActionPrediction`, `ActionChunk`, `LossDict`, `TrainingBatch` dataclasses in `types.py`
  <!-- Ref: PaliGemma token_type_ids: https://github.com/huggingface/transformers/blob/556312cd/src/transformers/models/paligemma/modeling_paligemma.py#L134-L138 -->
- [x] 1.3 Implement `ActionSpaceSpec` and `ProprioSpec` dataclasses in `types.py`
- [x] 1.4 Implement `FreezeConfig` dataclass in `types.py` with `freeze_modules: list[str]`, `lora_target_modules: list[str]` (peft leaf module names e.g. `["q_proj", "v_proj"]`, NOT full paths), `lora_r: int = 8`, `lora_alpha: int = 16`, `lora_dropout: float = 0.0`
  <!-- Ref: peft LoraConfig: https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraConfig
       Ref: OpenVLA-OFT freeze+LoRA: https://github.com/moojink/openvla-oft/blob/e4287e94/prismatic/training/train.py#L89-L120 -->

## 2. Registry

- [x] 2.1 Implement `Registry[ConfigT, ModuleT]` generic class in `src/yavla/models/registry.py` with `register()` decorator, `build()`, `list()`, `get_default_config()`
  <!-- Ref: LeRobot ChoiceRegistry pattern: https://github.com/huggingface/lerobot/blob/5f152322/src/lerobot/policies/act/configuration_act.py#L23 -->

## 3. Protocols and Base Classes

- [x] 3.1 Implement `IntegrationMode` enum, `BackboneCapabilities`, `ActionHeadRequirements` in `src/yavla/models/protocols.py`
- [x] 3.2 Implement runtime-checkable Protocols with unified signatures:
  - `VisionEncoderProto`: `encode_images(images: dict[str, Tensor]) → Tensor` (encoder holds PaliGemma ref internally)
  - `BackboneProto`: `tokenizer` property + `forward(inputs_embeds, attention_mask, token_type_ids) → BackboneOutput`
  - `ActionHeadProto`: `compute_loss(backbone_output: BackboneOutput, batch: TrainingBatch) → LossDict`, `predict(backbone_output: BackboneOutput) → ActionPrediction`
  - `ActionDecoderProto`: `decode(pred: ActionPrediction) → ActionChunk`
  - `ProprioEncoderProto`: `encode_proprio(proprio: Tensor) → Tensor`
  - `TokenMergerProto`: `merge(vision_tokens, proprio_tokens, language_tokens, language_attn_mask) → tuple[Tensor, Tensor, Tensor]`
- [x] 3.3 Implement ABC base classes: `VisionEncoderBase`, `BackboneBase`, `ActionHeadBase`, `ActionDecoderBase`, `ProprioEncoderBase`
  <!-- Ref: LeRobot PreTrainedPolicy ABC: https://github.com/huggingface/lerobot/blob/5f152322/src/lerobot/policies/pretrained.py#L45 -->
- [x] 3.4 Implement `validate_integration(backbone, head)` function with `IncompatibleError`

## 4. Vision Encoder

- [x] 4.1 Implement `VisionEncoderConfig` dataclass in `src/yavla/models/encoders/vision.py`
- [x] 4.2 Implement `PaliGemmaVisionEncoder(VisionEncoderBase)` — holds reference to backbone's unwrapped `base_model` (NOT PeftModel wrapper, NOT its own copy), calls `base_model.get_image_features(pixel_values)` to get projected+scaled image tokens `[B, num_patches, D]` (returns tensor directly, NOT a dict). Do NOT rescale. NOT frozen by default — freeze controlled by `FreezeConfig`. MVP supports single-camera only; raise `ValueError` if more than one camera key is provided.
  <!-- Ref: PaliGemma get_image_features returns projected+scaled tensor directly: https://github.com/huggingface/transformers/blob/556312cd/src/transformers/models/paligemma/modeling_paligemma.py#L92-L100
       Ref: π0 reuses PaliGemma's SigLIP: https://github.com/Physical-Intelligence/openpi/blob/981483dc/src/openpi/models/pi0.py#L108-L130 -->
- [x] 4.3 Register `PaliGemmaVisionEncoder` with `vision_registry`

## 4b. Proprio Encoder

- [x] 4b.1 Implement `ProprioEncoderConfig` dataclass in `src/yavla/models/encoders/proprio.py`
- [x] 4b.2 Implement `ProprioEncoder(ProprioEncoderBase)` — single `nn.Linear(proprio_dim, backbone_dim)` projecting `[B, D_proprio]` → `[B, 1, D_backbone]`
  <!-- Ref: Oracle review — proprio projection must be explicit module, not inline in merger.
       Ref: π0 action token linear projection pattern: https://github.com/Physical-Intelligence/openpi/blob/981483dc/src/openpi/models/pi0.py#L108-L130 -->
- [x] 4b.3 Register `ProprioEncoder` with `proprio_registry`

## 5. Token Merger

- [x] 5.1 Implement `TokenMergerConfig` dataclass in `src/yavla/models/merger.py`
- [x] 5.2 Implement `ConcatMerger` — `merge(image_embeds, proprio_embeds, language_embeds, language_attn_mask)` concatenates `[image | proprio | language | readout]` tokens, builds `inputs_embeds`, `attention_mask`, `token_type_ids` (0=image, 1=rest). Readout tokens = zeros + learned positional embedding `N(0, 0.02)` at END of sequence. Does NOT produce `position_ids`.
  <!-- Ref: PaliGemma token_type_ids: https://github.com/huggingface/transformers/blob/556312cd/src/transformers/models/paligemma/modeling_paligemma.py#L134-L138
       Ref: Octo readout init (zeros + pos_embed): https://github.com/octo-models/octo/blob/241fb351/octo/model/octo_module.py#L315-L333
       Ref: OpenVLA-OFT inputs_embeds bypass: https://github.com/moojink/openvla-oft/blob/e4287e94/prismatic/extern/hf/modeling_prismatic.py#L571-L638 -->
- [x] 5.3 Register `ConcatMerger` with `merger_registry`

## 6. Backbone

- [x] 6.1 Implement `BackboneConfig` dataclass in `src/yavla/models/backbone.py` — NO `FreezeConfig` field (freeze is at PolicyConfig level only)
- [x] 6.2 Implement `VLMBackbone(BackboneBase)` — wrap `PaliGemmaForConditionalGeneration` (use `transformers.AutoModelForVision2Seq.from_pretrained()`). Expose `base_model` property for unwrapped access (vision encoder + language embeddings use this). When LoRA applied, store PeftModel wrapper for `forward()`. Forward with `input_ids=None, inputs_embeds=..., pixel_values=None, token_type_ids=..., attention_mask=..., output_hidden_states=True`. Do NOT pass dummy `labels` — hybrid mask is driven by `token_type_ids` + `create_causal_mask_mapping`; `is_training` only controls a validation check, actual mask driven by `is_first_iteration` (True when `past_key_values is None`, always true for MVP). Extract `readout_states = outputs.hidden_states[-1][:, -N_readout:, :]`.
  <!-- CORRECTED: Causal mask driven by create_causal_mask_mapping(..., is_training=self.training) + token_type_ids, NOT by labels.
       Ref: PaliGemma forward: https://github.com/huggingface/transformers/blob/556312cd/src/transformers/models/paligemma/modeling_paligemma.py#L304-L387
       Ref: OpenVLA-OFT inputs_embeds injection: https://github.com/moojink/openvla-oft/blob/e4287e94/prismatic/extern/hf/modeling_prismatic.py#L571-L638
       Ref: Octo readout extraction: https://github.com/octo-models/octo/blob/241fb351/octo/model/components/action_heads.py#L157-L165 -->
- [x] 6.3 Register `VLMBackbone` with `backbone_registry`

## 7. Action Head

- [x] 7.1 Implement `MLPHeadConfig` dataclass in `src/yavla/models/heads/mlp.py`
- [x] 7.2 Implement `MLPResNet` module (LayerNorm → Linear → ReLU → residual blocks → output)
- [x] 7.3 Implement `MLPRegressionHead(ActionHeadBase)` — mean-pool `readout_states [B, N_readout, D] → [B, D]`, pass through MLPResNet, predict action chunks, L1 loss
  <!-- Ref: OpenVLA-OFT MLPResNet action head: https://github.com/moojink/openvla-oft/blob/e4287e94/prismatic/models/action_heads.py
       Ref: Octo readout mean-pooling: https://github.com/octo-models/octo/blob/241fb351/octo/model/components/action_heads.py#L157-L165 -->
- [x] 7.4 Register `MLPRegressionHead` with `head_registry`

## 8. Action Decoder and Policy

- [x] 8.1 Implement `SimpleActionDecoder(ActionDecoderBase)` in `src/yavla/models/decoder.py` — `decode(prediction: ActionPrediction) → ActionChunk`, unnormalize via `ActionSpaceSpec.limits`
  <!-- Ref: LeRobot normalization: https://github.com/huggingface/lerobot/blob/5f152322/src/lerobot/processor/normalize_processor.py#L400 -->
- [x] 8.2 Implement `PolicyConfig` dataclass tree in `src/yavla/models/config.py` — includes `FreezeConfig`, `ProprioEncoderConfig`, all sub-configs, `config_version: str`
- [x] 8.3 Implement `VLAPolicy(nn.Module)` in `src/yavla/models/policy.py` — compose all modules including `ProprioEncoder`. Language tokenization: `backbone.tokenizer(language)` → `input_ids` + `language_attn_mask`, then `backbone.base_model.get_input_embeddings()(input_ids)` → `language_embeds` (uses unwrapped base model). Forward: steps 1-6 per mvp-policy spec. Predict: steps 1-5 + `action_head.predict()` + `decoder.decode()`. `dt_hz` sourced from `TrainingBatch.dt_hz` in training, from `PolicyConfig.dt_hz` in inference.
  <!-- Ref: LeRobot PreTrainedPolicy: https://github.com/huggingface/lerobot/blob/5f152322/src/lerobot/policies/pretrained.py#L45
       Ref: OpenVLA-OFT forward: https://github.com/moojink/openvla-oft/blob/e4287e94/prismatic/extern/hf/modeling_prismatic.py#L571-L638 -->
- [x] 8.4 Implement `build_policy(PolicyConfig) → VLAPolicy` factory:
  (1) load PaliGemma via `transformers` → `base_model`
  (2) store `base_model` reference (for vision encoder + language embeddings — MUST use unwrapped model)
  (3) freeze modules per `FreezeConfig.freeze_modules` via `requires_grad_(False)` on `base_model`
  (4) if `FreezeConfig.lora_target_modules` non-empty: `peft_model = peft.get_peft_model(base_model, peft.LoraConfig(target_modules=..., r=..., lora_alpha=..., lora_dropout=...))` — peft freezes ALL base params, only adapters trainable
  (5) if peft applied: `peft_model.enable_input_require_grads()` (required for gradient flow through frozen base to LoRA adapters)
  (6) if `config.backbone.gradient_checkpointing` is True: `base_model.config.use_cache = False` (required for gradient checkpointing; MVP has no KV cache)
  (7) if `config.backbone.gradient_checkpointing` is True: `base_model.gradient_checkpointing_enable()`
  (8) `validate_integration(backbone, head)`
  (9) compose `VLAPolicy`
  <!-- Ref: peft.get_peft_model + LoraConfig: https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraConfig
       Ref: OpenVLA-OFT freeze+LoRA: https://github.com/moojink/openvla-oft/blob/e4287e94/prismatic/training/train.py#L89-L120 -->
- [x] 8.5 Implement `save_pretrained` / `from_pretrained` — safetensors + config.json + action_stats.json + embodiment.json. For LoRA models, use `peft` adapter save/load (`model.save_pretrained` / `PeftModel.from_pretrained`)

## 9. Wiring and Exports

- [x] 9.1 Wire all registry imports in `src/yavla/models/__init__.py` so registrations execute on import
- [x] 9.2 Export public API: `build_policy`, `VLAPolicy`, `PolicyBase`, `PolicyConfig`, all Protocol types, all data containers

## 10. Code Review Fixes

- [x] 10.1 Fix LoRA-aware `save_pretrained` — save adapter separately via `peft.save_pretrained()`, add `checkpoint_meta.json`, save non-VLM weights as `non_vlm_weights.safetensors`
- [x] 10.2 Fix LoRA-aware `from_pretrained` — read `checkpoint_meta.json`, load adapter-only or full state dict; backward-compatible with pre-metadata checkpoints
- [x] 10.3 Add `PolicyBase(nn.Module, ABC)` to `protocols.py` — minimal contract (`forward`, `predict`, `reset`, `get_optim_params`) with `__init_subclass__` enforcement of `name` and `config_class`
- [x] 10.4 Refactor `VLAPolicy(PolicyBase)` — split pipeline into 5 overridable step methods: `encode_observations`, `merge_tokens`, `run_backbone`, `compute_loss`, `decode_prediction`
- [x] 10.5 Fix spec conflict: correct `yavla-model-architecture/design.md` hybrid mask guidance (remove incorrect `labels` advice)

## 11. Test Coverage

- [x] 11.1 Add `tests/models/test_policy_serialization.py` — round-trip tests for `_tensor_to_list`, `_dict_to_config`, checkpoint file outputs, metadata, `_has_lora`
- [x] 11.2 Add `PolicyBase` enforcement tests — `test_missing_name_raises`, `test_missing_config_class_raises`
- [x] 11.3 Add `VLAPolicy` hierarchy tests — `test_is_policy_base`, `test_has_overridable_steps`, `test_reset_is_noop`, `test_name_and_config_class`

