## Why

YAVLA’s current SigLIP preprocessing defaults warp any camera image to a fixed square `HxW` size via `Resize([H, W], 3)`, which distorts common non-square camera aspect ratios (e.g., 3:2) and may degrade VLA performance. We want to add a simpler aspect-ratio-preserving letterbox option and make it switchable in training configs so we can empirically choose between warp and letterbox per dataset/model.

## What Changes

- Add one aspect-ratio-preserving, pad-to-target SigLIP preprocessing strategy that still outputs the fixed `HxW` tensor required by SigLIP/PaliGemma vision towers:
  - **letterbox** (resize-to-fit then symmetric pad)
- Add a training config switch used by the training entrypoint when auto-wiring SigLIP preprocessing (when `dataset.image_transforms is None`) so users can select between `warp` and `letterbox` without hand-editing transform strings.
- Update tests/specs/docs to cover the new letterbox strategy and keep dataset-layer preprocessing as the source of truth for `pixel_values`.

## Capabilities

### New Capabilities
<!-- No new capabilities; this extends existing preprocessing + transform-building contracts. -->

### Modified Capabilities
- `siglip-image-preprocessing`: Extend the preprocessing contract to support selectable resize strategies (including letterbox padding) while still producing fixed-size `pixel_values` aligned with the checkpoint-declared `S_ckpt`.
- `data-transforms`: Extend config-driven torchvision transform building to support the new letterbox transform used by SigLIP preprocessing recipes.

## Impact

- **Training config / CLI**: new knob(s) to select SigLIP resize strategy; affects default auto-wired preprocessing when `dataset.image_transforms is None`.
- **Data pipeline**: new transform/helper to implement letterbox padding; modifies SigLIP preprocessing auto-wiring.
- **Compatibility**: preserves existing “warp-to-fixed-size” behavior unless a new strategy is selected (no new external dependencies).
