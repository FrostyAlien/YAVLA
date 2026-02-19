## ADDED Requirements

### Requirement: ObservationBatch data container
`ObservationBatch` SHALL be a `@dataclass` holding images (`dict[str, Tensor]` keyed by camera name, shape `[B, C, H, W]`), proprio (`Tensor [B, D]`), language (`str | list[str] | None`), timestamps (`Tensor [B]`), and masks (`Tensor | None [B]`).

#### Scenario: Construction with all fields
- **WHEN** `ObservationBatch(images={"cam0": img}, proprio=prop, language="pick up the cup", timestamps=ts, masks=None)` is constructed
- **THEN** all fields SHALL be accessible with correct types and no validation error

#### Scenario: Multi-camera images
- **WHEN** `images` contains keys `"cam0"` and `"cam1"` each with shape `[B, 3, 224, 224]`
- **THEN** `ObservationBatch` SHALL store both without modification

### Requirement: TokenBatch data container
`TokenBatch` SHALL be a `@dataclass` holding `tokens: Tensor [B, N, D]`, `attn_mask: Tensor [B, N]`, `token_type_ids: Tensor [B, N]` (0=image/bidirectional prefix, 1=text+proprio+readout/causal), `modality_ids: Tensor [B, N]` (0=vision, 1=language, 2=proprio, 3=readout), and optional `readout_indices: Tensor | None`. `position_ids` is NOT included — PaliGemma computes them internally.

<!-- Ref: PaliGemma token_type_ids controls bidirectional vs causal attention:
     https://github.com/huggingface/transformers/blob/556312cd/src/transformers/models/paligemma/modeling_paligemma.py#L134-L138 -->

#### Scenario: Modality tracking
- **WHEN** a `TokenBatch` is constructed with vision, language, proprio, and readout tokens
- **THEN** `modality_ids` SHALL correctly identify each token's modality

#### Scenario: PaliGemma-compatible token_type_ids
- **WHEN** a `TokenBatch` is constructed with 256 vision tokens and 85 non-vision tokens
- **THEN** `token_type_ids[:, :256]` SHALL all be `0` and `token_type_ids[:, 256:]` SHALL all be `1`

### Requirement: BackboneOutput data container
`BackboneOutput` SHALL be a `@dataclass` with `readout_states: Tensor | None [B, N_readout, D]`, `token_states: Tensor | None [B, N, D]`, `attn_mask: Tensor`, and `aux: dict[str, Tensor]`.

#### Scenario: Readout mode output
- **WHEN** backbone operates in readout mode
- **THEN** `readout_states` SHALL be populated and `token_states` MAY be `None`

### Requirement: ActionPrediction data container
`ActionPrediction` SHALL be a `@dataclass` with `mean: Tensor [B, chunk_len, action_dim]`, `samples: Tensor | None`, `log_prob: Tensor | None`, and `aux: dict[str, Tensor]`.

#### Scenario: MLP head prediction
- **WHEN** MLP action head produces a prediction
- **THEN** `mean` SHALL contain the predicted actions and `samples` SHALL be `None`

### Requirement: ActionChunk data container
`ActionChunk` SHALL be a `@dataclass` with `actions: Tensor [B, chunk_len, action_dim]`, `dt_hz: float` (sourced from `TrainingBatch.dt_hz` during training, from `PolicyConfig.dt_hz` during inference), `chunk_len: int`, and `action_mask: Tensor | None`.

#### Scenario: Chunk with valid mask
- **WHEN** an `ActionChunk` is constructed with `chunk_len=5` and `action_mask` of shape `[B, 5]`
- **THEN** `actions.shape[1]` SHALL equal `chunk_len`

### Requirement: LossDict data container
`LossDict` SHALL be a `@dataclass` with `total: Tensor` (scalar) and `breakdown: dict[str, Tensor]` for per-component logging.

#### Scenario: Loss backward compatibility
- **WHEN** `loss_dict.total.backward()` is called
- **THEN** gradients SHALL flow through the computation graph

### Requirement: TrainingBatch data container
`TrainingBatch` SHALL be a `@dataclass` with `observations: ObservationBatch`, `actions: Tensor [B, chunk_len, action_dim]`, `action_mask: Tensor | None`, `dt_hz: float`, and `chunk_len: int`.

#### Scenario: Training batch from dataloader
- **WHEN** a `TrainingBatch` is constructed from dataset output
- **THEN** `observations` SHALL be a valid `ObservationBatch` and `actions` shape SHALL match `[B, chunk_len, action_dim]`

### Requirement: ActionSpaceSpec and ProprioSpec
`ActionSpaceSpec` SHALL be a `@dataclass` with `names: list[str]`, `units: list[str]`, `limits: Tensor | None [action_dim, 2]` (None = no normalization), `frame: str`, and `control_mode: str`. `ProprioSpec` SHALL have `names`, `units`, and `limits: Tensor | None`.

#### Scenario: Normalization bounds
- **WHEN** `ActionSpaceSpec.limits` is accessed
- **THEN** it SHALL return a `[action_dim, 2]` tensor with `(min, max)` per dimension
