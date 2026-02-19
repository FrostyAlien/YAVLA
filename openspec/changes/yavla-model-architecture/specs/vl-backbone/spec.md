## ADDED Requirements

### Requirement: Standard VLM backbone (readout mode)
`VLMBackbone(BackboneBase)` SHALL wrap a HuggingFace VLM (default PaliGemma), receive `inputs_embeds` (not raw images/text), and extract `readout_states` from the last `N_readout` positions of the final hidden layer. The backbone OWNS the single PaliGemma instance — vision encoder and language embedding path receive references to it. Supports gradient checkpointing and PEFT. Forward call uses `input_ids=None, inputs_embeds=..., pixel_values=None, token_type_ids=..., labels=dummy(-100), output_hidden_states=True` to bypass PaliGemma's internal vision pipeline and enforce the hybrid causal mask. The dummy `labels` (all `-100`) are required because HF PaliGemma only activates `token_type_ids`-based causal masking when `labels is not None`. The merger does NOT pass `position_ids` — PaliGemma computes them internally.

<!-- Ref: PaliGemma forward with inputs_embeds: https://github.com/huggingface/transformers/blob/556312cd/src/transformers/models/paligemma/modeling_paligemma.py#L351-L387
     Ref: OpenVLA-OFT inputs_embeds injection: https://github.com/moojink/openvla-oft/blob/e4287e94/prismatic/extern/hf/modeling_prismatic.py#L571-L638 -->

#### Scenario: Readout mode forward
- **WHEN** `forward(token_batch)` is called with readout tokens present
- **THEN** `BackboneOutput.readout_states` SHALL have shape `[B, N_readout, hidden_dim]`

#### Scenario: PEFT application
- **WHEN** `backbone.apply_peft(LoRAConfig(rank=16, target_modules=["q_proj", "v_proj"]))` is called
- **THEN** LoRA adapters SHALL be injected and only adapter params SHALL have `requires_grad=True`

### Requirement: Dual-expert backbone (joint-token mode)
`DualExpertBackbone(BackboneBase)` SHALL contain a frozen VLM expert and a trainable action expert (~300M params) with cross-attention layers for bidirectional information flow. Supports `JOINT_TOKENS` mode.

#### Scenario: Joint-token forward
- **WHEN** `forward(token_batch)` is called with action tokens interleaved
- **THEN** `BackboneOutput.token_states` SHALL contain full sequence hidden states including action token positions

#### Scenario: Cross-attention flow
- **WHEN** the dual-expert processes a batch
- **THEN** the action expert SHALL attend to VLM expert hidden states via cross-attention at each layer

#### Scenario: Capabilities declaration
- **WHEN** `backbone.capabilities` is accessed
- **THEN** `supported_modes` SHALL contain both `READOUT` and `JOINT_TOKENS`

### Requirement: BackboneConfig
`BackboneConfig` SHALL have `type: str`, `vlm_name: str`, `num_readout_tokens: int = 64`, `gradient_checkpointing: bool = True`, and `peft: PEFTConfig | None`.

#### Scenario: Dual-expert config
- **WHEN** `BackboneConfig(type="dual_expert", action_expert_dim=1024, action_expert_layers=12)` is constructed
- **THEN** it SHALL configure a dual-expert backbone with the specified action expert dimensions
