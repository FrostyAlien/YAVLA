## 1. Config + Registry Plumbing

- [x] 1.1 Expand `VisionEncoderConfig` to support `type="from_backbone"` as the canonical default and capture the fields needed by registry-built encoders
- [x] 1.2 Add backward-compatible handling for legacy `vision_encoder.type` values (e.g., `paligemma_siglip` → `from_backbone`) with a warning path
- [x] 1.3 Activate `vision_registry` as the source of truth for non-backbone vision encoder types (list/build/default-config behavior)

## 2. `build_policy()` Wiring (Hybrid Selection)

- [x] 2.1 Update `build_policy()` to select the vision encoder based on `PolicyConfig.vision_encoder.type` (`from_backbone` vs registry-built)
- [x] 2.2 Add build-time validation that the final vision token dimension matches `backbone.hidden_dim`
- [x] 2.3 Implement a standard projection wrapper for registry-built encoders that emit `D_vision != backbone.hidden_dim`
- [x] 2.4 Ensure unknown vision encoder types fail with a clear `KeyError` listing available registry entries

## 3. Minimal Standalone Vision Encoders (for testing + examples)

- [x] 3.1 Implement one lightweight, dependency-free vision encoder (e.g., simple conv/patch encoder) that satisfies `VisionEncoderBase` and register it in `vision_registry`
- [x] 3.2 Ensure standalone encoders preserve existing multi-camera requirements (canonical camera ordering, empty/mismatched-shape validation)

## 4. Multi-Tower Vision Encoder

- [x] 4.1 Define a multi-tower vision encoder config shape (2+ towers + fusion + projector) that is compatible with the registry pattern
- [x] 4.2 Implement `MultiTowerVisionEncoder(VisionEncoderBase)` that builds towers, fuses per-patch tokens, and outputs `[B, N_img, backbone.hidden_dim]`
- [x] 4.3 Enforce patch-grid alignment across towers (`num_patches` must match) and raise `ValueError` on mismatch
- [x] 4.4 Verify multi-tower encoders preserve multi-camera determinism (canonical camera ordering) and input validation

## 5. Tests

- [x] 5.1 Add unit tests for `from_backbone` selection (including legacy alias behavior)
- [x] 5.2 Add unit tests for registry selection and unknown-type rejection
- [x] 5.3 Add unit tests for projection behavior when `D_vision != backbone.hidden_dim`
- [x] 5.4 Add unit tests for multi-tower fusion (shape + dim) and patch-count mismatch rejection
- [x] 5.5 Ensure tests do not require heavyweight HF model downloads (use synthetic encoders/backbones)

## 6. Docs + Validation

- [x] 6.1 Update architecture/docs to reflect the explicit `from_backbone` default and how to configure registry/multi-tower encoders
- [x] 6.2 Run `pixi run -e dev test`, `pixi run -e dev lint`, and `pixi run -e dev typecheck` and fix any issues caused by the change
