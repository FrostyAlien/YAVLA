## 1. Training Config + SigLIP Auto-wiring

- [x] 1.1 Add `TrainingConfig.vlm_image_resize_strategy` with allowed values `warp|letterbox` (default: `warp`)
- [x] 1.2 Update `src/yavla/training/siglip_preprocess.py` to auto-wire `Resize(...)` vs `LetterboxPad(...)` based on `vlm_image_resize_strategy`
- [x] 1.3 Fail fast on unknown `vlm_image_resize_strategy` and warn when strategy is set but `dataset.image_transforms` is explicitly provided (auto-wiring disabled)

## 2. Data Transforms (Letterbox)

- [x] 2.1 Implement `LetterboxPad([H, W], interpolation)` transform (resize-to-fit + symmetric padding, pad fill=0.5 in `[0,1]` space)
- [x] 2.2 Extend `build_torchvision_transforms(...)` to recognize and construct `LetterboxPad(...)` specs alongside torchvision v2 transforms

## 3. Tests

- [x] 3.1 Add unit tests that `build_torchvision_transforms` accepts `LetterboxPad([H, W], 3)` and the resulting callable executes on `torch.Tensor` images
- [x] 3.2 Add SigLIP preprocessing tests for `letterbox`: output shape `[3, H, W]`, dtype `float32`, range approximately `[-1.05, 1.05]`, and padded border normalizes to ~0
- [x] 3.3 Run `pytest tests/data/test_transforms.py -v` in the Pixi dev environment

## 4. Docs + Config Examples

- [x] 4.1 Update `docs/training-guide.md` to document `vlm_image_resize_strategy` and its precedence relative to `dataset.image_transforms`
- [x] 4.2 Update `docs/dataset-layer/usage.md` to document the new SigLIP letterbox strategy and the padding fill behavior (0.5 → 0 after normalization)
- [x] 4.3 Add/update a training config example (e.g., `configs/train.yaml`) showing how to switch between `warp` and `letterbox`

## 5. Validation

- [x] 5.1 Run formatting/lint/typecheck (`ruff`, `mypy`) and the unit tests; ensure the default strategy remains `warp`
