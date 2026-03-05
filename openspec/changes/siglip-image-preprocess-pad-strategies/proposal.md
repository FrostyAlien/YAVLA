## Why

YAVLA’s current SigLIP preprocessing defaults warp any camera image to a fixed square `HxW` size via `Resize([H, W], 3)`, which distorts common non-square camera aspect ratios (e.g., 3:2) and may degrade VLA performance. Other VLA stacks often preserve aspect ratio and pad to the model’s fixed input size; we want to add those options (OpenVLA-style letterbox and OpenPI-style resize-with-pad) and make them switchable in training configs so we can empirically choose the best strategy per dataset/model.

## What Changes

- Add two aspect-ratio-preserving, pad-to-target SigLIP preprocessing strategies that still output the fixed `HxW` tensor required by SigLIP/PaliGemma vision towers:
  - OpenVLA-style **letterbox** (resize-to-fit then symmetric pad)
  - OpenPI-style **resize-with-pad** (their reference implementation)
- Add a training config switch used by the training entrypoint when auto-wiring SigLIP preprocessing (when `dataset.image_transforms is None`) so users can select between the strategies without hand-editing transform strings.
- Update tests/specs/docs to cover the new resize strategies and keep dataset-layer preprocessing as the source of truth for `pixel_values`.

## Capabilities

### New Capabilities
<!-- No new capabilities; this extends existing preprocessing + transform-building contracts. -->

### Modified Capabilities
- `siglip-image-preprocessing`: Extend the preprocessing contract to support selectable resize strategies (including aspect-ratio-preserving pad strategies) while still producing fixed-size `pixel_values` aligned with the checkpoint-declared `S_ckpt`.
- `data-transforms`: Extend config-driven torchvision transform building to support the new pad strategies used by SigLIP preprocessing recipes.

## Impact

- **Training config / CLI**: new knob(s) to select SigLIP resize strategy; affects default auto-wired preprocessing when `dataset.image_transforms is None`.
- **Data pipeline**: new transforms/helpers to implement letterbox and resize-with-pad; modifies SigLIP preprocessing auto-wiring.
- **Compatibility**: preserves existing “warp-to-fixed-size” behavior unless a new strategy is selected (no new external dependencies).
