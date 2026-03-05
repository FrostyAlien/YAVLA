## Why

SigLIP-based VLM backbones (including PaliGemma) are sensitive to input resolution and normalization. Today YAVLA does not provide a clear, correct default image preprocessing path (resize + model-specific normalization), and our dataset-stat normalization can accidentally normalize camera tensors using dataset stats, producing silently-wrong pixel distributions that degrade or break training.

## What Changes

- Define a canonical image preprocessing contract for SigLIP/PaliGemma inputs: decode to float tensors, resize to the backbone’s expected resolution (224), then apply SigLIP-style normalization.
- Add a configuration-friendly way to enable this preprocessing in the dataset layer (no image processing inside the model forward).
- Ensure dataset statistical normalization defaults do not apply to camera/image keys unless explicitly requested, preventing double-normalization or mismatched stats.
- Add tests that validate dtype/shape/range expectations for preprocessed camera tensors.
- Update docs and example configs so a first end-to-end training run uses the correct preprocessing by default.

## Capabilities

### New Capabilities

- `siglip-image-preprocessing`: Dataset-layer resize + SigLIP normalization preset/contract for SigLIP-based VLM backbones (PaliGemma first).

### Modified Capabilities

- `dataset-factory`: Default normalization behavior and/or transform wiring SHALL avoid normalizing camera keys with dataset stats by default, to preserve model-specific image preprocessing.
- `data-transforms`: Image transform support SHALL be sufficient to express SigLIP preprocessing (resize + mean/std normalization) via config-driven transforms.

## Impact

- Data layer: `src/yavla/data/factory.py`, `src/yavla/data/transforms.py` (transform defaults/presets, key selection for normalization, camera preprocessing).
- Model integration: `src/yavla/models/backbones/paligemma.py` (assumes pixel_values are already SigLIP-preprocessed).
- Tests: new/updated unit + integration coverage for camera preprocessing correctness.
- Docs/config: `docs/dataset-layer/*`, `docs/training-guide.md`, `configs/train.yaml`.
