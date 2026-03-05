## Context

YAVLA represents vision input as `ObservationBatch.images: dict[str, Tensor]` where each entry is a camera view (`camera_name -> [B, C, H, W]`). The training collate already collects multiple camera keys (`observation.images.<cam>`), but the current PaliGemma vision encoder hard-fails when more than one camera is present:

- `PaliGemmaVisionEncoder.encode_images()` raises for `len(images) > 1`

This blocks the first full end-to-end training run on real LeRobot datasets like `lerobot/aloha_sim`, which commonly include multiple cameras.

The intended multi-camera behavior is:

- accept 1+ cameras per observation
- produce a single, stable vision-token tensor suitable for the existing merger/backbone pipeline
- make camera ordering deterministic so tokenization is stable across runs

## Goals / Non-Goals

**Goals:**
- Support multi-camera `ObservationBatch.images` end-to-end through:
  - `TrainingCollate` (already collects multiple cameras)
  - `VLAPolicy.encode_observations()` (no API change)
  - `PaliGemmaVisionEncoder.encode_images()` (remove single-camera restriction)
- Define a deterministic camera ordering contract so multi-camera tokenization is stable across runs.
- Keep downstream pipeline modules unchanged:
  - merger consumes a single `[B, N_img, D]` tensor
  - backbone consumes merged embeddings and masks without special-casing cameras
- Add unit tests and at least one integration test to prevent regressions.

**Non-Goals:**
- Adding new backbone types or changing the 7-module pipeline.
- Introducing model-internal image preprocessing (handled by `siglip-image-preprocess`).
- Learning camera calibration / geometry, or adding explicit multi-view fusion modules.
- Supporting variable camera sets within a single batch (we assume camera keys are consistent across samples in a batch).

## Decisions

### D1: Represent multi-camera as concatenated patch tokens in a single vision-token stream

**Choice**: Encode each camera image into patch tokens and concatenate along the token dimension:

- per camera: `get_image_features([B, C, H, W]) -> [B, N_patch, D]`
- multi-camera output: `[B, K * N_patch, D]` where `K=len(cameras)`

This keeps the merger and backbone interfaces unchanged (they already accept variable-length `N_img`).

**Alternative considered**: Pool/average camera features into a single `N_patch` set. Rejected for v1 because it discards information and complicates later extensions (camera-specific reasoning, attention across views).

### D2: Make camera ordering deterministic by canonicalizing keys inside the vision encoder

**Choice**: Define the canonical camera order as `sorted(images.keys())` and use that order to build the concatenated vision-token sequence.

**Why**:
- `ObservationBatch.images` is a dict and may be created from unordered sources (e.g., `set` iteration inside the current `TrainingCollate`), so relying on insertion order is unsafe.
- Sorting inside the encoder guarantees stable tokenization for both training and inference callers, regardless of upstream dict construction.

**Alternative considered**: Canonicalize ordering in `TrainingCollate`. Useful for debugging, but insufficient alone because inference may bypass `TrainingCollate`.

### D3: Implement multi-camera encoding efficiently by flattening cameras into the batch dimension

**Choice**: For PaliGemma, avoid Python loops by stacking cameras into the batch dimension, calling `get_image_features` once, then reshaping back:

```
ordered cams: [c0, c1, ..., c(K-1)]
pixel_values: [K*B, C, H, W]
features:     [K*B, N_patch, D]
reshape ->    [B, K*N_patch, D]
```

**Why**: One encoder forward is simpler and typically faster than `K` separate calls, especially for small `K` (2-4).

**Alternative considered**: Loop per camera and `torch.cat` the resulting features. Acceptable but slower and easier to accidentally produce device/dtype mismatches.

### D4: Camera “identity” is v1 = stable ordering; explicit camera embeddings are deferred

**Choice**: In v1, we do not add new learnable camera-id embeddings or separator tokens. Camera identity is represented implicitly by stable position in the concatenated token stream (driven by the canonical key order).

**Why**: It meets the immediate need (stable multi-camera tokenization, no crashes) with minimal surface area. It also avoids introducing new config plumbing before we have evidence it is necessary.

**Alternative considered**: Add per-camera embeddings (by name) and add them to patch features, or insert per-camera “summary tokens”. Deferred because it requires new parameters, initialization rules, and a policy-level contract for mapping camera names to indices.

## Risks / Trade-offs

**[Risk] Increased sequence length increases VRAM and runtime** → Multi-camera concatenation scales `N_img` linearly with number of cameras.
Mitigation: Document expected token count (`K * num_patches`) and note PaliGemma’s max sequence/position constraints; keep v1 targeted at typical `K<=4`.

**[Risk] Lexicographic camera order may be surprising** → Users may expect semantic ordering (e.g., front vs wrist).
Mitigation: Document the rule clearly; add an explicit `camera_order` config later if this becomes a common pain point.

**[Risk] Inconsistent camera presence across samples** → Current `TrainingCollate` intersects keys across samples; if a camera key is missing in some sample, it silently drops it for that batch, changing `N_img`.
Mitigation: Call this out in docs/tests; if needed later, add a “required camera keys” option that raises when a configured camera is missing.

## Migration Plan

1. Update `PaliGemmaVisionEncoder.encode_images()` to support 1+ cameras using deterministic key ordering.
2. Update unit tests to expect multi-camera success (replace the current “reject multi-camera” expectation).
3. Add a small integration test that constructs a `TrainingBatch` with multiple cameras and verifies policy forward passes through `encode_observations -> merge -> backbone -> head`.
4. Update docs/config examples to show expected shapes and the camera ordering rule.

## Open Questions

- Do we want a first-class `camera_order` configuration (and where should it live: dataset config vs policy/backbone config)?
- Should we add optional camera-id embeddings once we have multiple camera layouts in the wild (e.g., wrist + overhead + side)?
- Should `TrainingCollate` also sort camera keys for readability and batch-to-batch stability (even though the encoder will canonicalize)?
