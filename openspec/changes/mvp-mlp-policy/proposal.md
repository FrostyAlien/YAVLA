## Why

YAVLA has a dataset layer but no model — we can load data but can't train anything. The architecture doc defines 7 core modules, but implementing everything at once is too risky. We need the simplest possible end-to-end policy that exercises all 7 module slots, proving the interfaces work before adding complexity. OpenVLA-OFT demonstrates that a frozen VLM + MLP regression head achieves competitive results — this is our MVP target architecture.

## What Changes

- Add typed data containers (`ObservationBatch`, `TokenBatch`, `BackboneOutput`, `ActionPrediction`, `ActionChunk`, `LossDict`, `TrainingBatch`) as module boundary contracts
- Add Protocol interfaces and ABC base classes for all 7 modules
- Add generic `Registry[T]` with decorator registration
- Add capability negotiation (`IntegrationMode`, `BackboneCapabilities`, `ActionHeadRequirements`) — MVP uses readout mode only
- Add SigLIP vision encoder wrapping PaliGemma's built-in SigLIP via `get_image_features()` — NOT frozen by default, configurable via `FreezeConfig`
- Add configurable freeze + LoRA support via `peft` library (`FreezeConfig` with `freeze_modules`, `lora_modules`, `lora_r`, `lora_alpha`)
- Add MLP proprio encoder (robot-specific projection to backbone dim)
- Add simple concat token merger (no Perceiver resampler — that's post-MVP)
- Add standard VLM backbone in readout mode, wrapping PaliGemma via `transformers` with learned readout tokens (Octo-style)
- Add MLP regression action head (`L1RegressionActionHead`, referencing OpenVLA-OFT's MLPResNet pattern) with L1 loss
- Add basic action decoder (unnormalize via `ActionSpaceSpec`, no temporal ensembling)
- Add `VLAPolicy(nn.Module)` composing all modules, with `save_pretrained` / `from_pretrained`
- Add `build_policy(PolicyConfig)` factory with build-time capability validation, freeze application, and LoRA wiring via `peft`
- Add `PolicyConfig` and sub-config dataclasses (including `FreezeConfig`) compatible with tyro CLI
- Add contract tests for each module Protocol and end-to-end smoke test

## Capabilities

### New Capabilities

- `model-types`: Typed data containers for all module boundaries (ObservationBatch, TokenBatch, BackboneOutput, ActionPrediction, ActionChunk, LossDict, TrainingBatch) and multi-embodiment specs (ActionSpaceSpec, ProprioSpec)
- `model-registry`: Generic Registry[T] with decorator registration and config-driven build()
- `model-protocols`: Protocol interfaces, ABC base classes, and capability negotiation for all 7 modules
- `mvp-vision-encoder`: SigLIP So400m/14 vision encoder via PaliGemma's `get_image_features()`, configurable freeze + LoRA via `peft`
- `mvp-token-merger`: Simple concatenation merger combining vision, proprio, language, and readout tokens into a single sequence
- `mvp-backbone`: Standard VLM backbone (PaliGemma) in readout mode with learned readout tokens
- `mvp-action-head`: MLP regression action head (L1 loss, chunked prediction) following OpenVLA-OFT's MLPResNet pattern
- `mvp-policy`: VLAPolicy nn.Module composing all modules, build_policy() factory with freeze/LoRA wiring via `peft`, PolicyConfig dataclasses, save_pretrained/from_pretrained with safetensors

### Modified Capabilities

_(none — no existing model specs)_

## Impact

- **New code**: `src/yavla/models/` — types, registry, protocols, encoders, merger, backbone, head, decoder, policy, factory, configs
- **Dependencies**: No new external deps — uses transformers, peft, safetensors, einops, tyro (already in pyproject.toml)
- **APIs**: Public `build_policy()`, `VLAPolicy`, all Protocol interfaces
- **Training loop**: Provides `VLAPolicy.forward(batch) → LossDict` and `VLAPolicy.predict(obs) → ActionChunk` for the training pipeline to consume
- **Config**: New `PolicyConfig` dataclass tree compatible with tyro CLI + YAML
- **Testing**: Contract tests per module, serialization round-trip, end-to-end forward pass smoke test
