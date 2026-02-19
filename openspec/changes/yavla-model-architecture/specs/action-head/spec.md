## ADDED Requirements

### Requirement: MLP regression action head
`MLPRegressionHead(ActionHeadBase)` SHALL predict continuous action chunks via MLPResNet with L1 loss. Requires `IntegrationMode.READOUT`.

#### Scenario: Predict action chunk
- **WHEN** `predict(backbone_output, sampling_cfg)` is called
- **THEN** `ActionPrediction.mean` SHALL have shape `[B, chunk_len, action_dim]`

### Requirement: Flow matching action head (standalone)
`FlowMatchingHead(ActionHeadBase)` SHALL implement rectified flow with linear interpolation `x_t = (1-t)*x_0 + t*x_1` and velocity prediction. Operates in readout mode with cross-attention conditioning on `readout_states`.

#### Scenario: Training loss
- **WHEN** `compute_loss(backbone_output, batch)` is called
- **THEN** it SHALL sample random `t ~ U(0,1)`, interpolate noisy actions, predict velocity, and return `LossDict` with MSE between predicted and true velocity

#### Scenario: Multi-step denoising inference
- **WHEN** `predict(backbone_output, SamplingConfig(num_denoise_steps=10))` is called
- **THEN** it SHALL iteratively denoise from `x_0 ~ N(0,1)` over 10 steps, returning `ActionPrediction` with `mean` and optionally `samples`

### Requirement: Flow matching action head (pi0-style joint-token)
`Pi0FlowHead(ActionHeadBase)` SHALL require `IntegrationMode.JOINT_TOKENS` and denoise action tokens in-place within the backbone's forward pass. The backbone runs multiple times (once per denoising step).

#### Scenario: Joint-token denoising
- **WHEN** `compute_loss` is called
- **THEN** noisy action tokens SHALL be injected into the token sequence, processed by the dual-expert backbone, and the velocity prediction extracted from action token positions

### Requirement: Diffusion action head (DDPM/DDIM)
`DiffusionHead(ActionHeadBase)` SHALL implement DDPM training with configurable DDIM inference. Operates in readout mode.

#### Scenario: DDIM inference
- **WHEN** `predict(backbone_output, SamplingConfig(num_denoise_steps=10))` is called with DDIM scheduler
- **THEN** it SHALL denoise in 10 steps using the DDIM update rule

### Requirement: CVAE action head (ACT-style)
`CVAEHead(ActionHeadBase)` SHALL implement a conditional VAE with encoder (training) and decoder (inference). Operates in readout mode.

#### Scenario: Training with KL loss
- **WHEN** `compute_loss` is called
- **THEN** `LossDict.breakdown` SHALL contain `{"reconstruction": ..., "kl": ...}` and `total` SHALL be their weighted sum

### Requirement: VQ-BeT action head
`VQBeTHead(ActionHeadBase)` SHALL use a VQ-VAE to discretize actions, then predict discrete codes from readout states. Operates in readout mode.

#### Scenario: Codebook prediction
- **WHEN** `predict(backbone_output)` is called
- **THEN** it SHALL predict codebook indices, decode via VQ-VAE decoder, and return continuous `ActionPrediction`
