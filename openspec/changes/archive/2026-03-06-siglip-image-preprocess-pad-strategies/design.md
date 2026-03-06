## Context

YAVLA currently wires SigLIP/PaliGemma preprocessing in the **dataset layer** via `DataConfig.image_transforms` (torchvision v2 transform specs applied by `ImageTransform`). The canonical SigLIP recipe today is:

- `Resize([H, W], 3)` (bicubic) → **warps** non-square inputs to a fixed `HxW`
- `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))` (maps `[0, 1] → ~[-1, 1]`)

This change adds one **aspect-ratio-preserving** resize-and-pad strategy, while keeping the final tensor shape fixed to the checkpoint-derived `S_ckpt` (or configured override), so SigLIP-based vision towers still receive the expected `pixel_values` shape.

Constraints:
- Preprocessing must stay in the dataset pipeline (no model-internal processors).
- Must remain config-driven (YAML/tyro) and compatible with the existing `image_transforms: list[str]` mechanism.
- Padding must be dynamic (depends on each image’s aspect ratio), so it cannot be expressed as a static torchvision `Pad(...)` following a fixed `Resize(...)`.

## Goals / Non-Goals

**Goals:**
- Add one new SigLIP preprocessing strategy that preserves aspect ratio and pads to a fixed `HxW`:
  - **letterbox** (resize-to-fit + symmetric padding)
- Make the strategy selectable via training config when auto-wiring SigLIP preprocessing (i.e., when `dataset.image_transforms is None`) so users can choose between `warp` and `letterbox`.
- Keep the output contract unchanged at the model boundary: per-camera tensors are `[3, H, W]`, `float32`, SigLIP-normalized.
- Add unit tests that validate shapes/dtypes/value ranges and basic padding correctness for representative aspect ratios (e.g., 3:2, 16:9, portrait).

**Non-Goals:**
- Changing any backbone forward pass behavior (no `AutoProcessor` / HF image processor in-model).
- Adding crop-based strategies (center-crop, random-crop) in this change.
- Producing/consuming a padding mask as an additional model input (may be revisited later).
- Re-tuning normalization constants (still SigLIP `mean=std=0.5`).

## Decisions

### D1: Add a first-class resize strategy knob to `TrainingConfig` (used only for auto-wiring)

**Choice:** Introduce a `TrainingConfig` field (e.g., `vlm_image_resize_strategy`) with a small enum-like string set such as:
- `warp` (current behavior; default)
- `letterbox`

The training entrypoint will consult this knob only when `dataset.image_transforms is None` (auto-wiring path). If users explicitly set `dataset.image_transforms`, we continue to respect it verbatim.

**Why:** Preserves the existing “explicit transforms win” rule while making A/B comparisons easy via a single YAML/CLI switch.

**Alternatives considered:**
- Put the knob on `DataConfig`: rejected because the auto-wiring logic lives in training, and `image_transforms` are already the dataset-layer contract. We avoid introducing multiple sources of truth.
- Encode everything directly in `image_transforms`: possible, but makes the default config noisy and makes it harder to run controlled experiments across many configs.

### D2: Implement letterbox as a custom transform and extend `build_torchvision_transforms` to construct it

**Choice:** Implement one custom, torch-tensor-first transform (callable object) in the data layer:
- `LetterboxPad(...)`

Then extend `build_torchvision_transforms(...)` to recognize this name in addition to torchvision v2 transforms.

**Why:** Dynamic pad amounts depend on the per-sample image size, so we need a transform that can compute the resize scale and padding at call time. Keeping them in the same transform list preserves the existing pipeline shape (`ImageTransform` applies a list of callables to camera tensors).

**Alternatives considered:**
- Compose torchvision `Resize(...)` + `Pad(...)`: rejected because `Pad` would need dynamic per-sample values.
- Add a new non-`ImageTransform` dataset transform stage: rejected to avoid expanding the surface area of the dataset factory and to keep SigLIP preprocessing expressed in `image_transforms`.

### D3: Keep padding “neutral” under SigLIP normalization

**Choice:** Perform pad fill in **pre-normalized space** (`[0, 1]`) using fill value `0.5` per channel, and then apply the standard SigLIP `Normalize(0.5, 0.5)` step.

**Why:** With this choice, padded regions become ~0 after normalization, minimizing unintended bias from the padding pixels and matching the intuition used in other VLA stacks.

**Alternatives considered:**
- Pad with black (`0.0`) or white (`1.0`): rejected because it creates large constant regions at the extremes after normalization.
- Normalize first, then pad with `0.0`: equivalent in effect but would require padding after normalization; keeping all padding before normalization is simpler and consistent across uint8/float inputs.

### D4: Keep the pad strategy simple and aligned with SigLIP preprocessing

**Choice:** Implement letterbox padding directly in the data layer while enforcing:
- Output shape exactly `HxW`
- Resize interpolation consistent with SigLIP expectations (bicubic for all supported strategies)
- Deterministic symmetric padding placement

**Why:** The goal is to compare the current warp behavior against one clear aspect-ratio-preserving alternative without carrying multiple nearly identical pad variants.

**Letterbox (expected behavior):**
- Scale image uniformly until it fits within `HxW` (“resize-to-fit”)
- Pad remaining pixels on both sides (symmetric; distribute odd pixel remainder consistently)

## Risks / Trade-offs

- **[Risk] Strategy mismatch across datasets** → Different datasets may still respond differently to warp vs letterbox preprocessing.
  - **Mitigation:** Provide a simple config switch, keep `warp` as default, and add evaluation notes in docs/config examples.
- **[Risk] Throughput regression** → More complex transforms may reduce dataloader throughput.
  - **Mitigation:** Implement transforms in pure torch ops (`interpolate` + `pad`), avoid PIL, and keep them vectorized per-image. Benchmark later if needed.
- **[Risk] Users bypass auto-wiring** → If users set `dataset.image_transforms` explicitly, the new strategy knob won’t apply.
  - **Mitigation:** Document precedence rules clearly; log a warning if a strategy knob is set but `image_transforms` is explicitly provided (similar to existing override warnings).

## Migration Plan

1. Add the new `TrainingConfig` strategy knob with default `warp` (no behavior change).
2. Add the custom transform + transform builder support; update SigLIP auto-wiring to emit the appropriate transform spec strings for each strategy.
3. Update `openspec/specs/siglip-image-preprocessing` and `openspec/specs/data-transforms` with the new contract and transform-spec support.
4. Add tests for `warp` and `letterbox`, plus a config example (or update `configs/train.yaml`) showing how to switch strategies.

## Open Questions

- Should we expose pad fill value as a config knob (advanced), or keep it fixed to `0.5` for SigLIP?
