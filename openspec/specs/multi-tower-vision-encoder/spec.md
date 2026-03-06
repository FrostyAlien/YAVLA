# multi-tower-vision-encoder Specification

## Purpose
TBD - created by archiving change hybrid-vision-encoder-registry. Update Purpose after archive.

## Requirements
### Requirement: Multi-tower vision encoder fuses 2+ towers into a single token stream
The system MUST support selecting a multi-tower vision encoder that combines patch tokens from two or more vision towers into a single vision token stream with shape `[B, N_img, D]`.

For a multi-tower encoder, fusion MUST preserve the patch-token count per camera view:

- `N_img = K * N_patch_per_camera` for `K` camera views

Fusion MAY increase intermediate feature dimensionality, but the final output token dimension MUST match the backbone embedding dimension (`backbone.hidden_dim`).

#### Scenario: Two towers fuse via concatenation and projection
- **WHEN** a multi-tower vision encoder with two towers produces per-tower tokens shaped `[B, N_img, D1]` and `[B, N_img, D2]`
- **THEN** the fused vision tokens have shape `[B, N_img, backbone.hidden_dim]` (e.g., concat to `[B, N_img, D1 + D2]` followed by projection)

### Requirement: Multi-tower fusion requires aligned patch grids
All towers participating in multi-tower fusion MUST produce the same per-image patch-token count (`num_patches`) so fusion is well-defined without resampling.

#### Scenario: Patch-count mismatch is rejected
- **WHEN** a multi-tower vision encoder is configured with towers that produce different `num_patches`
- **THEN** building the vision encoder fails with a `ValueError` describing the mismatch

### Requirement: Multi-tower encoders preserve multi-camera determinism
For multi-camera inputs, multi-tower encoders MUST follow the same determinism guarantees as single-tower encoders:

- camera ordering MUST be canonical and independent of input dict insertion order
- output tokens MUST be concatenated by camera in canonical order

#### Scenario: Input dict order does not change output tokens
- **WHEN** the same camera tensors are provided with different key insertion orders
- **THEN** the multi-tower vision encoder returns the same `[B, N_img, backbone.hidden_dim]` vision token tensor corresponding to canonical camera ordering
