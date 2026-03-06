## Context

Today, `PolicyConfig.vision_encoder` exists but is effectively unused: `build_policy()` always obtains `(vision_encoder, backbone)` from `vlm_registry.build(...)` and never consults the vision config. The only concrete vision encoder implementation (`PaliGemmaVisionEncoder`) lives inside the PaliGemma backbone module and is implicitly selected by `BackboneConfig.type="paligemma"`.

This is consistent with “VLM-coupled” architectures (vision tower is part of the VLM checkpoint), but it breaks the modular mental model and makes it difficult to represent:

- **standalone** vision encoders used with non-VLM backbones (e.g., ResNet-style encoders in ACT/Diffusion Policy families),
- **multi-tower** vision encoders (two or more vision towers fused into one token stream, e.g., Prismatic/OpenVLA-style dual towers),
- an explicit “source of truth” for how the vision encoder is chosen (from backbone vs. registry).

Constraints:

- Preserve the existing `VisionEncoderBase` contract: `encode_images(images: dict[str, Tensor]) -> Tensor` returning `[B, N_img, D]`.
- Preserve existing multi-camera behavior and determinism requirements (canonical camera ordering, input validation), as already specified in `openspec/specs/multi-camera-vision-encoding/spec.md`.
- Avoid heavyweight downloads in unit tests; prefer small, synthetic encoders for shape/contract tests.

Stakeholders:

- Model/infra developers extending YAVLA to new VLA families (ACT/diffusion/AR) and new backbones.
- Researchers iterating on vision towers (single vs. dual tower, fusion strategies).

## Goals / Non-Goals

**Goals:**

- Make `PolicyConfig.vision_encoder` semantically meaningful with a clear default behavior: **use the vision encoder that comes with the configured VLM backbone**.
- Add a **vision encoder registry** that can build standalone vision encoders from typed configs.
- Add first-class support for **multi-tower** vision encoders (2+ towers) that fuse patch tokens and project into the backbone embedding space.
- Keep VLM-coupled backbones working unchanged: `vlm_registry` remains the primary path for building VLM checkpoints that own their vision tower.
- Provide build-time checks to prevent silent dimension mismatches between vision tokens and the backbone embedding dimension.

**Non-Goals:**

- Implement a full OpenVLA/Prismatic backbone in this change (can be layered on after the architecture is in place).
- Add new action heads / token mergers / token-budget resamplers (merger improvements are orthogonal).
- Add temporal/video vision encoding support (e.g., `[B, T, C, H, W]`) in this change.
- Guarantee that “external vision tokens into a frozen VLM” are semantically compatible; the goal here is architecture and correctness checks, not performance guarantees.

## Decisions

### 1) Adopt a hybrid vision-selection approach (Option C)

**Decision:** Keep the existing VLM-coupled build path (`vlm_registry` returns `(VisionEncoderBase, BackboneBase)`) and add a second, registry-driven path for vision encoders, with an explicit config-controlled selection in `build_policy()`.

**Why:** This preserves the simplest/most-common case (VLM checkpoint owns its tower) while enabling standalone encoders for non-VLM or research settings. It also aligns with the rest of the codebase’s registry pattern.

**Alternatives considered:**

- **Option A (remove vision config):** Simplifies configuration but blocks standalone/multi-tower evolution.
- **Option B (fully pluggable vision only):** Forces refactors of current VLM builders and increases integration surface area (harder to keep stable).

### 2) Make the default explicit: `vision_encoder.type = "from_backbone"`

**Decision:** Introduce a canonical default mode where the vision encoder is sourced from the backbone builder. Concretely, treat `VisionEncoderConfig.type == "from_backbone"` as “ignore registry and use the encoder returned by `vlm_registry`”.

**Compatibility:** Accept existing values (e.g., `paligemma_siglip`) as a deprecated alias for `"from_backbone"` for at least one release cycle, with a warning.

**Why:** This turns the config from a misleading knob into an explicit declaration of intent.

### 3) Standardize the dimension contract around `backbone.hidden_dim`

**Decision:** The backbone embedding dimension (`backbone.hidden_dim`) is the single source of truth. The selected `vision_encoder` MUST produce tokens with `D == backbone.hidden_dim`.

**Implementation approach:**

- `build_policy()` validates `vision_encoder.output_dim == backbone.hidden_dim`.
- For registry-built encoders that naturally emit another dim, wrap them with a small projection module (e.g., `nn.Linear` or 2-layer MLP) that maps to `backbone.hidden_dim`.

**Why:** Downstream modules (merger, backbone forward) assume all modalities already live in the same embedding space.

**Alternatives considered:**

- Let merger handle per-modality projection. Rejected because it complicates merger implementations and obscures the vision/backbone interface.

### 4) Represent multi-tower encoders as a composite `VisionEncoderBase`

**Decision:** Implement `multi_tower` (or `dual`) as a `VisionEncoderBase` that owns multiple sub-encoders (“towers”), fuses their per-patch features, and projects to `backbone.hidden_dim`.

**Config shape (conceptual):**

- `VisionEncoderConfig(type="multi_tower", towers=[{...}, {...}], fusion="concat", projector="mlp")`

**Fusion behaviors:**

- `concat`: concatenate tower features on the channel dimension and project to `backbone.hidden_dim` (Prismatic-style).
- (Optional future) `sum` / `gated_sum`: require equal tower dims; project each to common dim if needed.

**Patch alignment constraint (initial):**

- All towers MUST produce the same per-image patch token count (`num_patches`) so fusion is well-defined without resampling.

**Why:** This keeps the external interface stable (`encode_images` still returns a single `[B, N_img, D]` stream) while supporting the main real-world “two towers” pattern.

**Alternatives considered:**

- Fuse at the backbone level (e.g., cross-attention). Rejected for this change because it couples vision choices to backbone implementations and complicates registry boundaries.

### 5) Keep multi-camera behavior inside the vision encoder boundary

**Decision:** `encode_images(images: dict[str, Tensor])` remains responsible for:

- validating non-empty input and consistent shapes,
- enforcing canonical camera ordering,
- concatenating per-camera patch tokens into one stream.

Multi-tower encoders apply “tower fusion” per camera view, then concatenate across cameras (or equivalently, operate on a flattened camera-batch and unflatten).

**Why:** This matches the existing spec and keeps the merger/backbone agnostic to camera count.

## Risks / Trade-offs

- **[External encoder into VLM may be meaningless]** → Mitigation: default to `"from_backbone"`; optionally require an explicit opt-in flag or a `BackboneCapabilities` indicator (future) before allowing overrides for certain backbones.
- **[Config complexity increases]** → Mitigation: keep a single default path; provide concise examples in docs and clear error messages when unsupported combinations are selected.
- **[Multi-tower patch mismatch]** → Mitigation: enforce equal `num_patches` initially; add resampling/token-merger support later if needed.
- **[Performance/memory overhead]** (two towers) → Mitigation: make multi-tower opt-in and expose freeze/PEFT knobs per tower in config (future extension).

## Migration Plan

1. Introduce `VisionEncoderConfig.type = "from_backbone"` as the new default.
2. Keep backward compatibility by mapping legacy `vision_encoder.type` values (e.g., `paligemma_siglip`) to `"from_backbone"` with a warning.
3. Update unit tests and docs to reflect the explicit default.
4. Add minimal registry-backed encoders for contract testing (no external downloads).
5. (Optional later) Add “real” towers (SigLIP/DINOv2) behind optional dependencies or heavyweight tests.

Rollback strategy: revert `build_policy()` to always use the `(vision_encoder, backbone)` pair returned by `vlm_registry` and ignore `config.vision_encoder`, leaving the registry code unused but harmless.

## Open Questions

- Should we extend `BackboneCapabilities` with something like `accepts_external_vision_tokens: bool` to gate overrides more explicitly?
- Do we want multi-tower fusion to support towers with different patch grids (requires resampling or token alignment)?
- Where should vision-tower tuning live long-term (`FreezeConfig` vs a dedicated `VisionTuningConfig`), especially when towers are not part of the VLM checkpoint?
