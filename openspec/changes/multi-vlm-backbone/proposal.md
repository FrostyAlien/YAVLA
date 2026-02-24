## Why

`build_policy()` and `VLAPolicy.encode_observations()` are hardcoded to PaliGemma's interface (`get_image_features()`, `base_model.get_input_embeddings()`, PaliGemma-specific config paths). Adding any new VLM backbone requires modifying these shared functions rather than just registering a new component. This blocks integration testing with lightweight models (e.g., SmolVLM2-256M at ~1GB vs PaliGemma-3B at ~6GB) and limits YAVLA to a single VLM family.

## What Changes

- **BREAKING**: `build_policy()` delegates VLM-specific loading to backbone-type-specific builders instead of inline PaliGemma code
- **BREAKING**: `BackboneBase` gains `embed_language(texts) -> (embeddings, attention_mask)` abstract method; existing backbone must implement it
- `VLAPolicy.encode_observations()` calls `self.backbone.embed_language()` instead of reaching into `backbone.tokenizer` and `backbone.base_model.get_input_embeddings()`
- Current PaliGemma code moves into a `PaliGemmaBackbone` builder (no behavior change for existing users)
- Backlog: SmolVLM2 backbone support (256M/500M/2.2B) — first new VLM, pending research on interface details
- Backlog: Qwen3-VL-2B backbone support — pending research on HF Transformers integration
- New integration test using a lightweight VLM backbone (SmolVLM2-256M target)

## Capabilities

### New Capabilities
- `multi-vlm-backbone`: Abstraction layer enabling multiple VLM backbone types via the registry pattern. Covers the `BackboneBase` contract changes, backbone-specific builder protocol, and `build_policy()` delegation logic.
- `training-integration-test`: End-to-end integration test exercising data loading → policy forward → loss → backward → optimizer step with a real (lightweight) model.

### Modified Capabilities
<!-- No existing spec-level requirements change — all current specs are data/training layer -->

## Impact

- `src/yavla/models/policy.py` — `build_policy()` refactored, `VLAPolicy.encode_observations()` simplified
- `src/yavla/models/protocols.py` — `BackboneBase` gains `embed_language()` abstract method
- `src/yavla/models/backbones/paligemma.py` — `PaliGemmaBackbone` (renamed from `VLMBackbone`) with `embed_language()`, `PaliGemmaVisionEncoder`, and `build_paligemma_vlm` builder
- `src/yavla/models/vlm_registry.py` — `VLMRegistry` mapping `type` → builder returning `(VisionEncoderBase, BackboneBase)`
- `src/yavla/models/backbone.py` — reduced to `BackboneConfig` only (implementation moved to `backbones/`)
- `src/yavla/models/encoders/vision.py` — reduced to `VisionEncoderConfig` only (implementation moved to `backbones/`)
- `src/yavla/models/config.py` — `BackboneConfig` may need VLM-type-specific sub-configs
- `tests/models/` — existing tests may need updates for new `embed_language()` on backbone mocks
- `tests/integration/` — new training integration test file
- No dependency changes for the refactor itself; new VLM backends would add optional deps later
