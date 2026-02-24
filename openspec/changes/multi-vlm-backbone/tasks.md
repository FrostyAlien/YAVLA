## 1. BackboneBase Contract Changes

- [x] 1.1 Add abstract `embed_language(texts: list[str]) -> tuple[Tensor, Tensor]` method to `BackboneBase` in `protocols.py`
- [x] 1.2 Make `BackboneBase.tokenizer` a concrete property that raises `NotImplementedError` (remove from abstract contract)
- [x] 1.3 Update `_StubBackbone` in `test_trainer.py` and any other test stubs to implement `embed_language()` (no-op returning zero tensors)
- [x] 1.4 Run existing tests to confirm no regressions from protocol changes

## 2. Create VLM Registry

- [x] 2.1 Create `src/yavla/models/vlm_registry.py` with a `vlm_registry` that maps `type` string to builder functions returning `tuple[VisionEncoderBase, BackboneBase]`
- [x] 2.2 Add unit test for vlm_registry: build known type, error on unknown type

## 3. Extract PaliGemma into Dedicated Module

- [x] 3.1 Create `src/yavla/models/backbones/` package with `__init__.py`
- [x] 3.2 Create `src/yavla/models/backbones/paligemma.py` — move `VLMBackbone` (renamed `PaliGemmaBackbone`) and `PaliGemmaVisionEncoder` from current locations
- [x] 3.3 Implement `embed_language()` on `PaliGemmaBackbone` (extract logic from `VLAPolicy.encode_observations()` lines 80-87)
- [x] 3.4 Create `build_paligemma_vlm(config: BackboneConfig) -> tuple[VisionEncoderBase, BackboneBase]` builder function (extract from `build_policy()` lines 264-310)
- [x] 3.5 Register builder with `vlm_registry.register("paligemma")`
- [x] 3.6 Keep backward-compat imports in old locations (`backbone.py`, `encoders/vision.py`) if needed, or update all internal references

## 4. Refactor build_policy and VLAPolicy

- [x] 4.1 Refactor `build_policy()` to call `vlm_registry.build(config.backbone)` for vision encoder + backbone, then build remaining modules generically
- [x] 4.2 Simplify `VLAPolicy.encode_observations()` to call `self.backbone.embed_language()` instead of reaching into tokenizer/embeddings
- [x] 4.3 Update `from_pretrained()` to read `config.backbone.type` for dispatch, defaulting to `"paligemma"` for pre-refactor checkpoints
- [ ] 4.4 Run full unit test suite — verify no regressions

## 5. Update Existing Tests

- [ ] 5.1 Update `tests/models/test_policy.py` for refactored `build_policy()` and `encode_observations()`
- [ ] 5.2 Update `tests/models/test_backbone.py` for renamed `PaliGemmaBackbone` and new `embed_language()` method
- [ ] 5.3 Verify `tests/models/test_policy_serialization.py` passes (serialization compat)
- [ ] 5.4 Run lint (`pixi run -e dev lint`) and typecheck (`pixi run -e dev typecheck`)

## 6. Training Integration Test

- [ ] 6.1 Create `tests/integration/test_training_loop.py` with `@pytest.mark.integration`
- [ ] 6.2 Implement synthetic `TrainingBatch` factory (random images, proprio, language, actions matching policy config shapes)
- [ ] 6.3 Implement test: `build_policy()` → `policy.forward(batch)` → assert finite loss
- [ ] 6.4 Implement test: forward → backward → optimizer step → assert parameters changed
- [ ] 6.5 Verify test is excluded from default `pixi run -e dev test` and runs with `-m ""`
