## 1. Vision Encoder Multi-Camera Support

- [x] 1.1 Update `PaliGemmaVisionEncoder.encode_images()` to accept 1+ cameras (remove single-camera restriction).
- [x] 1.2 Canonicalize camera ordering inside the vision encoder using `sorted(images.keys())` and concatenate patch tokens in that order.
- [x] 1.3 Validate multi-camera tensor consistency (`[B, C, H, W]` matches across cameras) and raise `ValueError` on empty `images` or mismatched shapes.
- [x] 1.4 Implement multi-camera encoding efficiently by flattening cameras into the batch dimension (single `get_image_features` call) and reshaping back to `[B, K*N_patch, D]`.

## 2. Unit Tests (Vision Encoder)

- [x] 2.1 Update `tests/models/test_encoders.py` to assert multi-camera encoding succeeds and output shape scales as `K * num_patches`.
- [x] 2.2 Add a deterministic ordering regression test: different input dict insertion orders produce the same concatenated token ordering (canonical key sort).
- [x] 2.3 Add a validation test for mismatched camera tensor shapes raising `ValueError`.
- [x] 2.4 Search for and update any other tests that expect multi-camera rejection (e.g., match strings like `single-camera`).

## 3. Pipeline Coverage

- [x] 3.1 Add a unit test that passes an `ObservationBatch` with multiple cameras through `VLAPolicy` (mock vision encoder output with `N_img=K*num_patches`) to ensure no downstream single-camera assumptions.
- [x] 3.2 Add an integration test variant (stub VLM) that uses two camera keys and verifies `build_policy() -> forward -> backward -> step` still works with multi-camera images.

## 4. Docs / Examples

- [x] 4.1 Document multi-camera support and the canonical camera ordering rule (lexicographic sort) in `docs/training-guide.md` (and/or the most relevant architecture doc).
- [x] 4.2 Add a short note about VRAM/runtime scaling with number of cameras (token count grows linearly).

## 5. Verification

- [x] 5.1 Run unit tests covering the change (at minimum `pytest tests/models/test_encoders.py -v`).
- [x] 5.2 Run the full unit test suite (`pytest tests/ -v`) and ensure no regressions.
