## Why

SigLIP-based VLM backbones (including PaliGemma) are sensitive to input resolution and normalization. Today YAVLA does not provide a clear, correct default image preprocessing path (resize + model-specific normalization), and our dataset-stat normalization can accidentally normalize camera tensors using dataset stats, producing silently-wrong pixel distributions that degrade or break training.

## What Changes

- Define a canonical image preprocessing contract for SigLIP/PaliGemma inputs: ensure camera tensors are float, resize to the backbone checkpoint’s expected resolution by default (PaliGemma: `vision_config.image_size` → `(S_ckpt, S_ckpt)`), optionally overridden to a user-specified `(H, W)` in training config, then apply SigLIP-style normalization.
- Add a configuration-friendly way to enable this preprocessing in the dataset layer (no image processing inside the model forward).
- Ensure dataset statistical normalization defaults do not apply to camera/image keys unless explicitly requested, preventing double-normalization or mismatched stats.
- Ensure training defaults stay aligned with the selected backbone checkpoint: by default derive the expected image size from the loaded model config (no parsing of `vlm_name` strings). Optionally allow an explicit training-time override of the resize target (height/width) for expert workflows; when set, training SHOULD log a warning that the checkpoint size is being overridden and that the user is responsible for verifying compatibility with the selected VLM.
- Add tests that validate dtype/shape/range expectations for preprocessed camera tensors.
- Update docs and example configs so a first end-to-end training run uses the correct preprocessing by default.

## Capabilities

### New Capabilities

- `siglip-image-preprocessing`: Dataset-layer resize + SigLIP normalization contract for SigLIP-based VLM backbones (PaliGemma first), parameterized by the backbone’s expected image size **S**.

### Modified Capabilities

- `dataset-factory`: Default normalization behavior and/or transform wiring SHALL avoid normalizing camera keys with dataset stats by default, to preserve model-specific image preprocessing.
- `data-transforms`: Image transform support SHALL be sufficient to express SigLIP preprocessing (resize + mean/std normalization) via config-driven transforms.

## Impact

- Data layer: `src/yavla/data/factory.py`, `src/yavla/data/transforms.py` (transform defaults/presets, key selection for normalization, camera preprocessing).
- Training entrypoint: `scripts/train.py`, `src/yavla/training/*` (auto-wire canonical preprocessing from loaded backbone config; optionally accept a user override and warn when it differs).
- Model integration: `src/yavla/models/backbones/paligemma.py` (assumes pixel_values are already SigLIP-preprocessed).
- Tests: new/updated unit + integration coverage for camera preprocessing correctness.
- Docs/config: `docs/dataset-layer/*`, `docs/training-guide.md`, `configs/train.yaml`.

## Background references (non-normative)

- Transformers `SiglipImageProcessor` / `SiglipImageProcessorFast` (resize to explicit size with bicubic + rescale + normalize):  
  <https://github.com/huggingface/transformers/raw/refs/heads/main/src/transformers/models/siglip/image_processing_siglip.py>  
  <https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/siglip/image_processing_siglip_fast.py>
- Transformers `PaliGemmaConfig` (checkpoint-derived `vision_config.image_size`):  
  <https://github.com/huggingface/transformers/raw/refs/heads/main/src/transformers/models/paligemma/configuration_paligemma.py>
- `google/paligemma-3b-mix-448` model card (448×448 evidence):  
  <https://huggingface.co/google/paligemma-3b-mix-448>
