## ADDED Requirements

<!-- Ref: OpenVLA-OFT MLPResNet action head: https://github.com/moojink/openvla-oft/blob/e4287e94/prismatic/models/action_heads.py
     Ref: Octo readout mean-pooling: https://github.com/octo-models/octo/blob/241fb351/octo/model/components/action_heads.py#L157-L165 -->

### Requirement: MLP regression action head
`MLPRegressionHead(ActionHeadBase)` SHALL take `readout_states` from `BackboneOutput`, mean-pool over the readout token dimension `[B, N_readout, D] → [B, D]`, pass through a 2-block MLPResNet, and output `ActionPrediction` with continuous actions.

#### Scenario: Predict action chunk
- **WHEN** `predict(backbone_output)` is called with `readout_states` shape `[B, 64, D]`
- **THEN** it SHALL mean-pool to `[B, D]`, pass through MLPResNet, and return `ActionPrediction` with `mean` shape `[B, chunk_len, action_dim]`

#### Scenario: Compute L1 loss
- **WHEN** `compute_loss(backbone_output: BackboneOutput, batch: TrainingBatch)` is called with ground-truth actions
- **THEN** it SHALL return `LossDict` with `total` equal to `F.l1_loss(predicted, ground_truth)` and `breakdown` containing `{"l1": <value>}`

#### Scenario: Requirements declaration
- **WHEN** `head.requirements` is accessed
- **THEN** `required_mode` SHALL be `IntegrationMode.READOUT` and `accepts_readout` SHALL be `True`

### Requirement: MLPResNet architecture
The MLPResNet SHALL consist of `LayerNorm → Linear → ReLU → [N× (LayerNorm → Linear → ReLU + residual)] → LayerNorm → Linear`, with configurable `num_blocks`, `hidden_dim`, and `output_dim`.

#### Scenario: Two-block default
- **WHEN** `MLPResNet(num_blocks=2, input_dim=4096, hidden_dim=1024, output_dim=7)` is constructed
- **THEN** it SHALL have 2 residual blocks and output shape `[B, 7]` for input `[B, 4096]`

### Requirement: ActionHeadConfig
`MLPHeadConfig` SHALL be a `@dataclass` with `type: str = "mlp"`, `hidden_dim: int = 1024`, `num_blocks: int = 2`, `chunk_len: int = 5`, and `action_dim: int = 7`.

#### Scenario: Default config
- **WHEN** `MLPHeadConfig()` is constructed
- **THEN** `type` SHALL be `"mlp"` and `chunk_len` SHALL be `5`
