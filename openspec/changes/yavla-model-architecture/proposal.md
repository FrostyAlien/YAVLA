## Why

YAVLA has a complete dataset layer but no model architecture. The research doc (`docs/architecture/vla-design-space-research.md`) defines 7 core modules with Protocol-based interfaces, config-driven composition, and a two-level integration strategy (readout mode + joint-token mode). This change tracks the full implementation roadmap — from MVP through production-ready — so that incremental changes can be planned and verified against the target architecture.

## What Changes

- Define the complete model layer in `src/yavla/models/` with 7 core modules: Observation Adapter, Vision Encoder, Proprio/Language Encoders, Token Merger, VL Backbone, Action Head, Action Decoder
- Implement typed data containers (`ObservationBatch`, `TokenBatch`, `BackboneOutput`, `ActionPrediction`, `ActionChunk`, `LossDict`, `TrainingBatch`) as the module boundary contracts
- Implement `@runtime_checkable` Protocol interfaces and `*Base(nn.Module, ABC)` base classes for all modules
- Implement a generic `Registry[T]` with decorator-based registration and optional entry-point plugin discovery
- Implement capability negotiation (`BackboneCapabilities`, `ActionHeadRequirements`, `IntegrationMode`) for backbone–head compatibility validation
- Implement `build_policy()` factory with build-time validation and config-driven composition via tyro
- Implement `VLAPolicy(nn.Module)` top-level policy with `save_pretrained` / `from_pretrained` (safetensors, versioned config)
- Implement multiple action heads: MLP regression, Flow Matching (standalone + π0-style), Diffusion (DDPM/DDIM), CVAE (ACT-style), VQ-BeT
- Implement multiple vision encoders: SigLIP, DINOv2, Dual (Prismatic)
- Implement multiple backbones: Standard VLM (readout mode), Dual-Expert (joint-token mode)
- Implement token mergers: Concatenation, Perceiver resampler
- Implement PEFT wrappers (LoRA/IA³ application + merge for deployment)
- Implement inference optimization: KV-cache reuse, vision caching, deployment profiles (edge/server/research)
- Implement multi-embodiment support: `ActionSpaceSpec`, `ProprioSpec`, per-robot stems/heads
- Wire training pipeline: canonical `train_step`, optimizer config, gradient checkpointing
- Integrate with existing dataset layer via `TrainingBatch` contract

## Capabilities

### New Capabilities

- `model-types`: Typed data containers for all module boundaries (ObservationBatch, TokenBatch, BackboneOutput, ActionPrediction, ActionChunk, LossDict, TrainingBatch)
- `model-registry`: Generic Registry[T] with decorator registration, config-driven build(), and entry-point plugin discovery
- `model-protocols`: Protocol interfaces and ABC base classes for all 7 modules, plus capability negotiation (IntegrationMode, BackboneCapabilities, ActionHeadRequirements)
- `vision-encoder`: Swappable vision encoders (SigLIP, DINOv2, Dual/Prismatic) with configurable adaptation (frozen, LoRA, full fine-tune)
- `token-merger`: Token merger/resampler module (concat, Perceiver) that combines vision, proprio, language, and context tokens into a single sequence
- `vl-backbone`: VL backbone module supporting readout mode (standard VLM) and joint-token mode (dual-expert), with PEFT wrappers
- `action-head`: Swappable action heads (MLP, Flow Matching, Diffusion, CVAE, VQ-BeT) with unified loss/predict interface
- `action-decoder`: Action decoder that unnormalizes predictions to physical units, applies temporal ensembling, and produces executable ActionChunks
- `policy-factory`: build_policy() factory with config-driven composition, build-time validation, and capability negotiation
- `policy-checkpoint`: VLAPolicy save_pretrained/from_pretrained with versioned configs, safetensors, embodiment metadata, and load-time validation
- `inference-optimization`: KV-cache reuse, vision caching, deployment profiles (edge/server/research), receding horizon control
- `multi-embodiment`: ActionSpaceSpec, ProprioSpec, per-robot observation adapters and action decoders

### Modified Capabilities

_(none — no existing model specs)_

## Impact

- **New code**: `src/yavla/models/` — all modules per the directory structure in the architecture doc
- **Dependencies**: No new external dependencies — uses transformers, peft, safetensors, einops, tyro (all already in pyproject.toml)
- **APIs**: Public `build_policy()` factory, `VLAPolicy` class, all Protocol interfaces
- **Training loop**: Will consume `TrainingBatch` from the dataset layer's `create_dataloader()` and call the canonical `train_step`
- **Config**: New model config section (`PolicyConfig` and sub-configs) compatible with tyro CLI + YAML
- **Phased delivery**: MVP (MLP head, readout mode) → Flow Matching → PEFT → Dual-Expert → Additional heads → Multi-embodiment → Inference optimization
