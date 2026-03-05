## 1. Dataset Normalization Defaults (Exclude Camera Keys)

- [ ] 1.1 Update `build_transform_pipeline()` in `src/yavla/data/factory.py` so that when `normalize=True`, `metadata.stats` is present, and `normalize_keys is None`, the factory passes an explicit key list to `NormalizeTransform` derived from stats keys but excluding all camera keys from metadata.
- [ ] 1.2 Add/adjust a unit test in `tests/data/test_factory.py` that fails if a camera key with stats would be normalized under the default (`normalize_keys=None`) path.
- [ ] 1.3 Update `docs/dataset-layer/usage.md` to reflect the new default: `normalize_keys=None` excludes camera keys, and show how to explicitly include camera keys via `normalize_keys`.

## 2. ImageTransform uint8 Handling + SigLIP Transform Recipe

- [ ] 2.1 Update `ImageTransform` in `src/yavla/data/transforms.py` to coerce `torch.uint8` camera tensors to `torch.float32` and rescale by `1/255` before applying torchvision transforms.
- [ ] 2.2 Add unit tests in `tests/data/test_transforms.py` covering uint8 camera tensors with a normalization transform (assert: no error; output dtype is float32).
- [ ] 2.3 Add a unit test that builds and runs the canonical SigLIP transform list (`Resize([H, W], 3)` + `Normalize((0.5,...),(0.5,...))`) on a synthetic camera tensor and asserts output shape `[3, H, W]`, dtype float32, and approximate value range `[-1.05, 1.05]` (test at least `(H, W)=(224, 224)`; optionally add `(448, 448)`).

## 3. Size-Aware SigLIP Preprocessing (Model-Derived `S_ckpt` + Optional Override)

- [ ] 3.1 Add optional training config fields to support a simple size override:
  - `TrainingConfig.siglip_image_height_override: int | None`
  - `TrainingConfig.siglip_image_width_override: int | None`
- [ ] 3.2 Add a small helper (e.g., `src/yavla/training/siglip_preprocess.py`) that generates the canonical SigLIP transform list for target size `(H, W)` (default `(S_ckpt, S_ckpt)`, overridden by the training config fields when both are set).
- [ ] 3.3 Update `scripts/train.py` to derive `S_ckpt` from the loaded backbone checkpoint config (`vision_config.image_size`, **not** from parsing `vlm_name` strings), then:
  - if exactly one of the override fields is set → error (both-or-none)
  - if both override fields are set and `(H, W) != (S_ckpt, S_ckpt)` → log a warning that the checkpoint size is being overridden (user responsibility to verify VLM compatibility)
  - `dataset.image_transforms is None` → auto-wire canonical SigLIP transforms for `(H, W)`
  - `dataset.image_transforms == []` → respect explicit “disable preprocessing”
  - non-empty `dataset.image_transforms` → respect user-provided transforms (optionally warn if an override is set but `image_transforms` is explicit)
- [ ] 3.4 Add unit tests for the helper and training wiring covering: default auto-wire, explicit disable (`[]`), override warning path, and the both-or-none override error.
- [ ] 3.5 Update `configs/train.yaml` so a first end-to-end run uses correct preprocessing by default (e.g., omit `dataset.image_transforms` so it is `None` and auto-wire triggers for SigLIP backbones).
- [ ] 3.6 Update `docs/training-guide.md` and `docs/dataset-layer/usage.md` to document:
  - SigLIP/PaliGemma expects dataset-layer preprocessed `pixel_values` (no model-internal processor)
  - canonical transform list uses `Resize([H, W], 3)` list form (avoid tuple-parsing pitfalls)
  - `image_transforms: null`/omitted = auto-wire for SigLIP backbones; `[]` = disable preprocessing
  - size override fields: if set and mismatched vs checkpoint, training logs a warning (user responsibility to verify compatibility)

## 4. Verification

- [ ] 4.1 Run `pixi run -e dev pytest tests/data/test_transforms.py -v` and fix any failures.
- [ ] 4.2 Run `pixi run -e dev pytest tests/data/test_factory.py -v` and fix any failures.
- [ ] 4.3 Run `pixi run -e dev pytest tests/training/test_siglip_preprocess.py -v` (or equivalent) and fix any failures.
- [ ] 4.4 Run `pixi run -e dev lint` and `pixi run -e dev typecheck`.
