# multi-camera-vision-encoding Specification

## Purpose
TBD - created by archiving change multi-camera-vlm-input. Update Purpose after archive.

## Requirements
### Requirement: Vision encoder supports 1+ camera images per observation
The system MUST accept `ObservationBatch.images` containing one or more camera entries (`dict[str, Tensor]`), where each camera tensor has shape `[B, C, H, W]`. The vision encoder MUST return a single vision-token tensor with shape `[B, N_img, D]`.

For multi-camera observations with `K` cameras, the output token dimension MUST scale linearly with the number of cameras:

- `N_img = K * N_patch_per_camera`

Where `N_patch_per_camera` is the per-image patch-token count for the configured vision tower (e.g., PaliGemma’s ViT patch grid).

#### Scenario: Two cameras produce concatenated vision tokens
- **WHEN** `ObservationBatch.images` contains `{"cam0": [B, C, H, W], "cam1": [B, C, H, W]}` with the same `B,C,H,W`
- **THEN** the vision encoder returns a tensor shaped `[B, 2 * N_patch_per_camera, D]`

### Requirement: Multi-camera tokenization is deterministic via canonical camera ordering
The system MUST define a canonical camera ordering that is independent of input dict insertion order. For v1, the canonical order MUST be ascending lexicographic sort of camera names (`sorted(images.keys())`).

The vision encoder MUST concatenate camera patch tokens in canonical camera order. This makes multi-camera tokenization stable across runs and across different upstream dict construction.

#### Scenario: Input dict order does not change output tokens
- **WHEN** the same camera tensors are provided in different key insertion orders
- **THEN** the returned vision-token tensor is identical (up to floating-point determinism) and corresponds to concatenation in canonical camera order

### Requirement: Vision encoder validates multi-camera input consistency
The system MUST reject invalid multi-camera inputs:

- If `images` is empty, the vision encoder MUST raise `ValueError`.
- If camera tensors do not share the same batch size `B`, channel count `C`, height `H`, and width `W`, the vision encoder MUST raise `ValueError`.

#### Scenario: Empty camera dict is rejected
- **WHEN** `ObservationBatch.images` is empty
- **THEN** `encode_images()` raises `ValueError`

#### Scenario: Mismatched camera tensor shapes are rejected
- **WHEN** `ObservationBatch.images` contains camera tensors with different `[B, C, H, W]` shapes
- **THEN** `encode_images()` raises `ValueError`
