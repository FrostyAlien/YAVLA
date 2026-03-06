## Why

YAVLA’s current `VisionEncoderConfig` is effectively a dead knob: it only carries a string type and is not used by `build_policy()`, which always builds the vision encoder via the selected VLM backbone. This makes it hard to (a) swap vision encoders cleanly, and (b) represent real-world “multi-tower” vision encoders (e.g., OpenVLA/Prismatic-style dual towers) without hardcoding logic into backbone builders.

## What Changes

- Make `PolicyConfig.vision_encoder` semantically meaningful by introducing an explicit `from_backbone` mode (default) and a registry-driven mode for standalone vision encoders.
- Introduce a functional `vision_registry` to build vision encoders from typed configs, mirroring the existing registry pattern used elsewhere in the codebase.
- Add first-class support for **multi-tower** vision encoders (2+ vision towers) with configurable fusion (e.g., concat + projection), while preserving the existing multi-camera tokenization contract (`encode_images(images) -> [B, N_img, D]`).
- Add build-time validation that `vision_encoder.encode_images()` produces tokens in the backbone embedding space (`D == backbone.hidden_dim`), with a standard projection path when towers emit different feature dims.
- Keep the existing VLM-coupled build path intact (`vlm_registry` continues to return `(VisionEncoderBase, BackboneBase)` for VLM backbones). The new vision registry is used when the backbone supports external vision towers or when using non-VLM backbones in the future.

## Capabilities

### New Capabilities
- `vision-encoder-registry`: A config-driven mechanism to select/build vision encoders (including a `from_backbone` mode) with clear shape/dimension guarantees for downstream modules.
- `multi-tower-vision-encoder`: A composite vision encoder that builds multiple towers and fuses their patch tokens into a single `[B, N_img, backbone_dim]` token stream.

### Modified Capabilities
<!-- No existing spec-level requirements change; multi-camera tokenization behavior is preserved. -->

## Impact

- `src/yavla/models/encoders/vision.py` — expand `VisionEncoderConfig`, activate `vision_registry`, add composite encoder(s)
- `src/yavla/models/config.py` — clarify/configure `vision_encoder` defaults and parameters
- `src/yavla/models/policy.py` — update `build_policy()` wiring to respect `vision_encoder` config while preserving VLM-coupled backbones
- `tests/models/` — add unit tests covering registry selection, multi-tower fusion, and dimension validation (no heavyweight HF downloads)
- `docs/architecture/` — align implementation with the documented modular design space and encoder swappability goals
