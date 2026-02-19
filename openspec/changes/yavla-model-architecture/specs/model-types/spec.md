## ADDED Requirements

### Requirement: ObservationBatch data container
`ObservationBatch` SHALL be a `@dataclass` holding images (`dict[str, Tensor]` keyed by camera name, shape `[B, C, H, W]`), proprio (`Tensor [B, D]`), language (`str | list[str] | None`), timestamps (`Tensor [B]`), masks (`Tensor | None [B]`), and `camera_intrinsics: dict[str, Tensor] | None`.

#### Scenario: Multi-camera with intrinsics
- **WHEN** `ObservationBatch` is constructed with 3 cameras and `camera_intrinsics` for each
- **THEN** all fields SHALL be accessible and intrinsics keys SHALL match image keys

### Requirement: TokenBatch data container
`TokenBatch` SHALL hold `tokens: Tensor [B, N, D]`, `attn_mask: Tensor [B, N]`, `token_type_ids: Tensor [B, N]` (0=image/bidirectional prefix, 1=text+proprio+readout/causal), `modality_ids: Tensor [B, N]` (0=vision, 1=language, 2=proprio, 3=readout, 4=context), and `readout_indices: Tensor | None`. `position_ids` is NOT included — PaliGemma computes them internally.

<!-- Ref: PaliGemma token_type_ids: https://github.com/huggingface/transformers/blob/556312cd/src/transformers/models/paligemma/modeling_paligemma.py#L134-L138 -->

#### Scenario: Context token modality
- **WHEN** context tokens (robot embedding, task ID) are included
- **THEN** `modality_ids` SHALL assign value `4` to context token positions

### Requirement: BackboneOutput data container
`BackboneOutput` SHALL hold `readout_states: Tensor | None`, `token_states: Tensor | None`, `attn_mask: Tensor`, `kv_cache: tuple | None`, and `aux: dict[str, Tensor]`.

#### Scenario: Joint-token mode output
- **WHEN** backbone operates in joint-token mode
- **THEN** `token_states` SHALL be populated with full sequence hidden states and `readout_states` MAY be `None`

### Requirement: ActionPrediction data container
`ActionPrediction` SHALL hold `mean: Tensor [B, chunk_len, action_dim]`, `samples: Tensor | None [B, N_samples, chunk_len, action_dim]`, `log_prob: Tensor | None`, and `aux: dict[str, Tensor]`.

#### Scenario: Flow matching prediction with samples
- **WHEN** flow matching head produces a prediction with `num_samples=5`
- **THEN** `samples` SHALL have shape `[B, 5, chunk_len, action_dim]` and `mean` SHALL be the sample mean

### Requirement: SamplingConfig
`SamplingConfig` SHALL be a `@dataclass` with `num_samples: int = 1`, `temperature: float = 1.0`, `num_denoise_steps: int = 10`, and `guidance_scale: float | None = None`.

#### Scenario: Default sampling
- **WHEN** `SamplingConfig()` is constructed
- **THEN** `num_denoise_steps` SHALL be `10` and `num_samples` SHALL be `1`
