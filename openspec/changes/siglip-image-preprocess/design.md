## Context

YAVLA currently expects all image preprocessing to happen in the dataset layer via `DataConfig.image_transforms` (torchvision v2 transforms applied per-camera key) and expects the model to receive already-preprocessed `pixel_values` (no model-internal processors).

Two issues block correct SigLIP/PaliGemma training defaults:

1. **Missing canonical preprocessing**: `configs/train.yaml` currently sets `image_transforms: []`, so images are not resized to 224 and are not SigLIP-normalized before reaching the PaliGemma vision tower.
2. **Accidental dataset-stat normalization of images**: when `DataConfig.normalize=True` and `normalize_keys=None`, the factory wires `NormalizeTransform` with `keys=None`, which normalizes all keys present in both the sample and dataset stats. If camera keys have stats, this can silently normalize images using dataset stats and conflict with SigLIP’s expected pixel distribution.

Additionally, torchvision v2 `Normalize` requires float tensors; some datasets/backends may yield `uint8` images (especially for non-video image modalities), which would make a SigLIP normalization transform fail unless we coerce dtype first.

This change defines and enforces a dataset-layer preprocessing contract for SigLIP-based backbones (PaliGemma first) without introducing model-internal image processing.

## Goals / Non-Goals

**Goals:**
- Provide a canonical SigLIP/PaliGemma image preprocessing contract at the dataset boundary:
  - channel-first tensors
  - resize to 224x224
  - SigLIP normalization (mean/std = 0.5/0.5 per channel, mapping [0,1] to [-1,1])
- Make it configuration-friendly to enable (YAML/tyro) while keeping preprocessing out of the model forward.
- Prevent dataset-stat normalization from touching camera keys by default (unless explicitly requested).
- Add tests that validate preprocessing correctness (shape/dtype/range) and guard against regressions.

**Non-Goals:**
- Multi-camera support (handled in `multi-camera-vlm-input`).
- Adding new VLM backbones or changing the VLA pipeline structure.
- Implementing letterbox/pad strategies for arbitrary aspect ratios (we follow SigLIP defaults for now).
- Introducing new external dependencies (we stay within torchvision transforms + existing code).

## Decisions

### D1: Keep image preprocessing in the dataset layer (model assumes preprocessed pixel_values)

**Choice**: All resizing/normalization happens before the model input; the VLM vision encoder expects preprocessed tensors.

**Why**: Keeps the model codepath simple, avoids backbone-specific processors inside `forward`, and matches the project’s intended pipeline split (data layer owns preprocessing).

**Alternative considered**: Call HuggingFace `AutoProcessor` inside the model/backbone to preprocess images. Rejected because it duplicates work, complicates batching/multi-camera, and makes preprocessing harder to audit and test across backends.

### D2: Express SigLIP preprocessing via `DataConfig.image_transforms` with a canonical recipe

**Choice**: Use the existing `image_transforms: list[str]` mechanism to configure SigLIP preprocessing, and document/test a canonical transform list for SigLIP/PaliGemma:

- `Resize([224, 224], 3)` (force 224x224, bicubic)
- `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`

**Why**: Reuses the existing config-driven transform system, avoids adding new config unions/preset types, and provides an explicit, reviewable preprocessing path.

**Notes**:
- The transform-spec parser treats a top-level tuple as `*args`. For 2D resize, prefer a list (`[224, 224]`) or an extra-nested tuple, otherwise `Resize((224, 224))` would be mis-parsed as `Resize(224, 224)`.
- LeRobot video decoding typically already returns `float32` in `[0, 1]`. This recipe assumes that range.

**Alternative considered**: Add a dedicated `image_preprocess_preset="siglip_224"` field that expands into transforms. Deferred until we see repeated user misconfiguration; the explicit list is sufficient for v1.

### D3: Exclude camera keys from dataset-stat normalization by default

**Choice**: When `DataConfig.normalize=True` and `normalize_keys is None`, the factory computes an effective default key list that excludes `metadata.camera_keys` and passes it to `NormalizeTransform(keys=...)`. If a user explicitly sets `normalize_keys`, we respect it (including camera keys if specified).

**Why**: Dataset stats are appropriate for proprio/action normalization, but image normalization should be model-specific for pretrained VLM backbones. Making “don’t normalize images with dataset stats” the default prevents silent distribution mismatch.

**Alternative considered**: Change `NormalizeTransform(keys=None)` to internally skip image keys. Rejected because `NormalizeTransform` should remain a generic stat normalizer; the dataset factory has the necessary modality context (camera keys).

### D4: Ensure `ImageTransform` can support normalization by handling uint8 inputs safely

**Choice**: `ImageTransform` (or a small wrapper in the factory pipeline) coerces `torch.uint8` camera tensors to `float32` in `[0, 1]` before applying torchvision transforms, so that `Normalize(...)` works reliably.

**Why**: torchvision v2 `Normalize` requires float input; without coercion, a valid config can fail depending on dataset modality storage. Coercing only camera tensors keeps behavior localized.

**Alternative considered**: Require all backends to emit float images. Rejected because default backend and non-video image modalities can reasonably produce `uint8`.

### D5: Tests validate preprocessing contract rather than model internals

**Choice**: Add tests that assert:
- the canonical transform list parses and executes via `build_torchvision_transforms`
- output camera tensors have shape `[3, 224, 224]`
- dtype is float (float32) after preprocessing
- values are approximately within `[-1, 1]` after SigLIP normalization (range check, not exact equality)
- dataset-stat normalization does not touch camera keys unless explicitly requested

**Why**: The correctness boundary is the dataset output passed to the model; testing there is stable and avoids coupling to specific backbone implementations.

**Alternative considered**: Add model-forward golden tests for pixel preprocessing. Rejected because it is slower, more brittle, and overlaps with backbone tests.

## Risks / Trade-offs

**[Risk] Behavior change for users relying on dataset-stat-normalized images** → If any existing runs intentionally normalized camera keys via dataset stats, excluding them by default changes behavior.
→ Mitigation: Allow explicit `normalize_keys` to include camera keys; document the rationale and the opt-in path.

**[Risk] Resize strategy mismatch** → SigLIP defaults warp images to 224x224; some VLA stacks use aspect-ratio preserving resize + crop/letterbox.
→ Mitigation: Start by matching SigLIP’s default processor behavior; if needed, add additional documented recipes later (e.g., resize-shorter-edge + center crop).

**[Risk] uint8 coercion assumptions** → Converting `uint8` to float `[0,1]` assumes 0-255 encoding; malformed inputs could still produce unexpected ranges.
→ Mitigation: Restrict coercion to `uint8` only; add lightweight range diagnostics in tests and docs.

## Migration Plan

1. Update default/example training config (`configs/train.yaml`) to include the canonical SigLIP preprocessing transform list.
2. Land the dataset-factory default normalization-key change and the image dtype handling.
3. Add/adjust tests to cover preprocessing correctness and prevent regressions.
4. Document the new defaults and the explicit opt-in path for normalizing images with dataset stats.

## Open Questions

- Should we add a first-class preprocessing preset field (e.g., `image_preprocess_preset`) once we support multiple backbones with different image expectations?
- Do we want to standardize interpolation/antialias settings across torchvision and HF processor defaults (beyond “bicubic + 224”)?
