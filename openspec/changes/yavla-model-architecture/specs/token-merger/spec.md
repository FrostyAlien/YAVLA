## ADDED Requirements

### Requirement: Concat token merger
`ConcatMerger` SHALL concatenate vision, proprio, language, readout, and context tokens into `inputs_embeds` with correct `attention_mask`, `token_type_ids` (0=image/bidirectional prefix, 1=rest/causal), and `modality_ids`. The merger does NOT produce `position_ids` — PaliGemma computes them internally. The merger SHALL accept `language_attn_mask` from the tokenizer to propagate padding masks into the merged `attention_mask`. Readout tokens MUST be at the END of the sequence for causal LM isolation.

<!-- Ref: PaliGemma token_type_ids: https://github.com/huggingface/transformers/blob/556312cd/src/transformers/models/paligemma/modeling_paligemma.py#L134-L138
     Ref: Octo readout isolation via causal attention: https://github.com/octo-models/octo/blob/241fb351/octo/model/octo_module.py#L257-L262 -->

#### Scenario: Merge with context tokens
- **WHEN** context tokens (robot embedding, task ID) are provided
- **THEN** they SHALL be appended with `modality_ids=4` before readout tokens

### Requirement: Perceiver resampler merger
`PerceiverMerger` SHALL use cross-attention with N learned queries (default 64) attending to vision tokens, reducing sequence length to a fixed `token_budget` regardless of input resolution or camera count.

#### Scenario: Token reduction
- **WHEN** vision tokens have shape `[B, 729, D]` (384x384 / 14^2) and `token_budget=64`
- **THEN** output vision tokens SHALL have shape `[B, 64, D]`

#### Scenario: Multi-camera reduction
- **WHEN** 3 cameras produce `[B, 2187, D]` total vision tokens
- **THEN** output SHALL still be `[B, 64, D]` vision tokens

#### Scenario: Token budget warning
- **WHEN** `ConcatMerger` is used with >256 vision tokens
- **THEN** a warning SHALL be logged recommending Perceiver resampler

### Requirement: TokenMergerConfig
`TokenMergerConfig` SHALL have `type: str = "concat"`, `num_readout_tokens: int = 64`, and `token_budget: int | None = None` (None = no reduction).

#### Scenario: Perceiver config
- **WHEN** `TokenMergerConfig(type="perceiver", token_budget=64)` is constructed
- **THEN** `type` SHALL be `"perceiver"` and `token_budget` SHALL be `64`
