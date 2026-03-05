## 1. Dataset Normalization Defaults (Exclude Camera Keys)

- [ ] 1.1 Update `build_transform_pipeline()` in `src/yavla/data/factory.py` so that when `normalize=True`, `metadata.stats` is present, and `normalize_keys is None`, the factory passes an explicit key list to `NormalizeTransform` derived from stats keys but excluding all camera keys from metadata.
- [ ] 1.2 Add/adjust a unit test in `tests/data/test_factory.py` that fails if a camera key with stats would be normalized under the default (`normalize_keys=None`) path.
- [ ] 1.3 Update `docs/dataset-layer/usage.md` to reflect the new default: `normalize_keys=None` excludes camera keys, and show how to explicitly include camera keys via `normalize_keys`.

## 2. ImageTransform uint8 Handling + SigLIP Transform Recipe

- [ ] 2.1 Update `ImageTransform` in `src/yavla/data/transforms.py` to coerce `torch.uint8` camera tensors to `torch.float32` and rescale by `1/255` before applying torchvision transforms.
- [ ] 2.2 Add unit tests in `tests/data/test_transforms.py` covering uint8 camera tensors with a normalization transform (assert: no error; output dtype is float32).
- [ ] 2.3 Add a unit test that builds and runs the canonical SigLIP transform list (`Resize([224, 224], 3)` + `Normalize((0.5,...),(0.5,...))`) on a synthetic camera tensor and asserts output shape `[3, 224, 224]`, dtype float32, and approximate value range `[-1, 1]`.

## 3. Defaults in Configs + Documentation

- [ ] 3.1 Update `configs/train.yaml` to set `dataset.image_transforms` to the canonical SigLIP preprocessing list (resize 224 + normalize mean/std 0.5).
- [ ] 3.2 Update `docs/training-guide.md` “Full Config Reference” to show the canonical SigLIP transform list instead of `image_transforms: []`.
- [ ] 3.3 Add a short note in docs that SigLIP/PaliGemma expects preprocessed `pixel_values` from the dataset layer (no model-internal image processor), and document the `Resize([224, 224], 3)` list form to avoid tuple-parsing pitfalls.

## 4. Verification

- [ ] 4.1 Run `pixi run -e dev pytest tests/data/test_transforms.py -v` and fix any failures.
- [ ] 4.2 Run `pixi run -e dev pytest tests/data/test_factory.py -v` and fix any failures.
- [ ] 4.3 Run `pixi run -e dev lint` and `pixi run -e dev typecheck`.

