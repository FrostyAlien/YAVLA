## 1. Training Config + SigLIP Auto-wiring

- [ ] 1.1 Add `TrainingConfig.vlm_image_resize_strategy` with allowed values `warp|openvla_letterbox|openpi_resize_with_pad` (default: `warp`)
- [ ] 1.2 Update `src/yavla/training/siglip_preprocess.py` to auto-wire `Resize(...)` vs `LetterboxPad(...)` vs `ResizeWithPad(...)` based on `vlm_image_resize_strategy`
- [ ] 1.3 Fail fast on unknown `vlm_image_resize_strategy` and warn when strategy is set but `dataset.image_transforms` is explicitly provided (auto-wiring disabled)

## 2. Data Transforms (Letterbox + Resize-with-Pad)

- [ ] 2.1 Implement `LetterboxPad([H, W], interpolation)` transform (OpenVLA-style: resize-to-fit + symmetric padding, pad fill=0.5 in `[0,1]` space)
- [ ] 2.2 Implement `ResizeWithPad([H, W], interpolation)` transform matching OpenPI `resize_with_pad_torch` semantics (scale/rounding/pad placement) with pad fill=0.5
- [ ] 2.3 Extend `build_torchvision_transforms(...)` to recognize and construct `LetterboxPad(...)` and `ResizeWithPad(...)` specs alongside torchvision v2 transforms

## 3. Tests

- [ ] 3.1 Add unit tests that `build_torchvision_transforms` accepts `LetterboxPad([H, W], 3)` and `ResizeWithPad([H, W], 3)` and the resulting callables execute on `torch.Tensor` images
- [ ] 3.2 Add SigLIP preprocessing tests for `openvla_letterbox` and `openpi_resize_with_pad`: output shape `[3, H, W]`, dtype `float32`, range approximately `[-1.05, 1.05]`, and padded border normalizes to ~0
- [ ] 3.3 Run `pytest tests/data/test_transforms.py -v` in the Pixi dev environment

## 4. Docs + Config Examples

- [ ] 4.1 Update `docs/training-guide.md` to document `vlm_image_resize_strategy` and its precedence relative to `dataset.image_transforms`
- [ ] 4.2 Update `docs/dataset-layer/usage.md` to document the new SigLIP pad strategies and the padding fill behavior (0.5 → 0 after normalization)
- [ ] 4.3 Add/update a training config example (e.g., `configs/train.yaml`) showing how to switch between the three strategies for A/B testing

## 5. Validation

- [ ] 5.1 Run formatting/lint/typecheck (`ruff`, `mypy`) and the unit tests; ensure the default strategy remains backward-compatible (`warp`)
