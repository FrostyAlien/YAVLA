## ADDED Requirements

<!-- Ref: PaliGemma forward with inputs_embeds: https://github.com/huggingface/transformers/blob/556312cd/src/transformers/models/paligemma/modeling_paligemma.py#L304-L387
     Ref: Readout extraction at end of sequence (Octo pattern): https://github.com/octo-models/octo/blob/241fb351/octo/model/components/action_heads.py#L157-L165 -->

### Requirement: PaliGemma backbone in readout mode
`VLMBackbone(BackboneBase)` SHALL wrap `PaliGemmaForConditionalGeneration` via HuggingFace `transformers`. The backbone OWNS the single PaliGemma instance — vision encoder and language embedding path receive references to the unwrapped base model (via `backbone.base_model`), never their own copies. When LoRA is applied, the backbone stores the PeftModel wrapper for `forward()` and exposes `base_model` property for direct access to the unwrapped `PaliGemmaForConditionalGeneration`. It SHALL also expose `tokenizer` for policy language tokenization. It receives `inputs_embeds`, `attention_mask`, and `token_type_ids` from the merger. It forwards through PaliGemma with `input_ids=None, pixel_values=None` (bypassing built-in vision pipeline) and extracts `readout_states` from the final `N_readout` positions of the last hidden layer.

#### Scenario: Forward pass in readout mode
- **WHEN** `forward(inputs_embeds, attention_mask, token_type_ids)` is called
- **THEN** it SHALL call PaliGemma with `output_hidden_states=True`, extract the last layer's hidden states via `outputs.hidden_states[-1]`, and return `BackboneOutput` with `readout_states = last_hidden[:, -N_readout:, :]` shape `[B, N_readout, hidden_dim]`

#### Scenario: Bypasses PaliGemma vision pipeline
- **WHEN** the backbone forwards
- **THEN** it SHALL pass `input_ids=None, pixel_values=None, inputs_embeds=inputs_embeds, output_hidden_states=True` to PaliGemma

#### Scenario: Hybrid causal mask via token_type_ids (NOT labels)
- **WHEN** the backbone forwards through PaliGemma
- **THEN** it SHALL pass `token_type_ids` to PaliGemma. The hybrid mask (bidirectional for `token_type_ids==0` image prefix, causal for `token_type_ids==1`) is constructed by `create_causal_mask_mapping(..., is_training=self.training)`. In the HF source (commit `556312cd`), `is_training` only controls a validation check — the actual mask is driven by `is_first_iteration` (True when `past_key_values is None`, which is always the case for MVP since we don't use KV cache). The hybrid mask works correctly in both `train()` and `eval()` modes. Do NOT pass dummy `labels` — labels only control LM loss computation, not masking.

<!-- CORRECTED: In HF PaliGemma (commit 556312cd), causal mask is driven by
     create_causal_mask_mapping(..., is_training=self.training) + token_type_ids.
     labels do NOT gate masking — they only control whether LM loss is computed.
     Ref: https://github.com/huggingface/transformers/blob/556312cd/src/transformers/models/paligemma/modeling_paligemma.py#L304-L387 -->

#### Scenario: Capabilities declaration
- **WHEN** `backbone.capabilities` is accessed
- **THEN** `supported_modes` SHALL contain `IntegrationMode.READOUT` and `supports_kv_cache` SHALL be `False` for MVP

#### Scenario: Gradient checkpointing
- **WHEN** `gradient_checkpointing=True`
- **THEN** `build_policy` SHALL set `base_model.config.use_cache = False` (required — gradient checkpointing is incompatible with KV cache), call `base_model.gradient_checkpointing_enable()`, and if peft is applied, call `peft_model.enable_input_require_grads()` (required for gradients to flow through frozen base weights to LoRA adapters)

### Requirement: BackboneConfig
`BackboneConfig` SHALL be a `@dataclass` with `type: str = "vlm"`, `vlm_name: str = "google/paligemma-3b-pt-224"`, and `gradient_checkpointing: bool = True`. `num_readout_tokens` is NOT stored here — it is passed at construction time from `TokenMergerConfig` by the `build_policy` factory to avoid duplication. Freeze/LoRA is controlled by `FreezeConfig` at the `PolicyConfig` level, NOT per-backbone. `BackboneConfig` does NOT contain a `FreezeConfig` field.

#### Scenario: Default config
- **WHEN** `BackboneConfig()` is constructed
- **THEN** `type` SHALL be `"vlm"` and `gradient_checkpointing` SHALL be `True`
