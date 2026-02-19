## ADDED Requirements

<!-- Ref: PaliGemma token_type_ids: https://github.com/huggingface/transformers/blob/556312cd/src/transformers/models/paligemma/modeling_paligemma.py#L134-L138
     Ref: Octo readout at end: https://github.com/octo-models/octo/blob/241fb351/octo/model/octo_module.py#L248-L262
     Ref: OpenVLA-OFT inputs_embeds: https://github.com/moojink/openvla-oft/blob/e4287e94/prismatic/extern/hf/modeling_prismatic.py#L571-L638 -->

### Requirement: Concat token merger builds inputs_embeds
`ConcatMerger` SHALL concatenate pre-embedded tokens into `inputs_embeds: Tensor [B, S, D]`, build `token_type_ids: Tensor [B, S]` (0=image/bidirectional prefix, 1=causal), and `attention_mask: Tensor [B, S]`. Readout tokens MUST be at the END of the sequence. The merger does NOT produce `position_ids` — PaliGemma computes them internally.

#### Scenario: Merge all modalities
- **WHEN** `merge(image_embeds, proprio_embeds, language_embeds, language_attn_mask)` is called with image `[B, 256, D]`, proprio `[B, 1, D]`, language `[B, 20, D]`, language_attn_mask `[B, 20]`
- **THEN** `inputs_embeds` shape SHALL be `[B, 256+1+20+N_readout, D]` with ordering `[image | proprio | language | readout]`

#### Scenario: Language padding mask propagation
- **WHEN** `language_attn_mask` contains `0` at padded positions
- **THEN** the merged `attention_mask` SHALL propagate those `0` values at the corresponding language token positions, so padded language tokens are masked out during attention

#### Scenario: token_type_ids assignment
- **WHEN** the merger builds `token_type_ids`
- **THEN** image positions SHALL be `0` (bidirectional prefix), all others (proprio, language, readout) SHALL be `1` (causal)

#### Scenario: Readout token injection
- **WHEN** `num_readout_tokens=64`
- **THEN** 64 learned readout tokens (zeros + positional embedding `N(0, 0.02)`) SHALL be appended at the END

### Requirement: TokenMergerConfig
`TokenMergerConfig` SHALL be a `@dataclass` with `type: str = "concat"` and `num_readout_tokens: int = 64`.

#### Scenario: Default config
- **WHEN** `TokenMergerConfig()` is constructed
- **THEN** `type` SHALL be `"concat"` and `num_readout_tokens` SHALL be `64`
