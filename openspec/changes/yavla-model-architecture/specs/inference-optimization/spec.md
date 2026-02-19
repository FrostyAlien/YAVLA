## ADDED Requirements

### Requirement: KV-cache reuse
The backbone SHALL support KV-cache for autoregressive or multi-step inference. When vision/language tokens are unchanged between calls, their KV entries SHALL be reused.

#### Scenario: Cached vision forward
- **WHEN** `forward(token_batch, kv_cache=prev_cache)` is called with unchanged vision tokens
- **THEN** only new tokens (readout/action) SHALL be computed; `BackboneOutput.kv_cache` SHALL be updated

### Requirement: Vision caching
The vision encoder SHALL cache encoded features when the input image is unchanged between inference calls.

#### Scenario: Same-image skip
- **WHEN** `encode_images` is called with the same image tensor as the previous call
- **THEN** cached `TokenBatch` SHALL be returned without re-running the ViT

### Requirement: Deployment profiles
Three profiles: `research` (no optimization), `server` (KV-cache + vision cache + batch inference), `edge` (merged LoRA + quantized + static shapes). `DeploymentProfile` enum selects the profile.

#### Scenario: Edge profile
- **WHEN** `build_policy(config, profile=DeploymentProfile.EDGE)` is called
- **THEN** LoRA SHALL be merged, model SHALL be prepared for quantization, and dynamic shapes SHALL be disabled

#### Scenario: Server profile
- **WHEN** `build_policy(config, profile=DeploymentProfile.SERVER)` is called
- **THEN** KV-cache and vision caching SHALL be enabled
