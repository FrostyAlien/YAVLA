## ADDED Requirements

### Requirement: SigLIP image preprocessing contract (model-derived size) for camera tensors
The data pipeline SHALL support a SigLIP-style image preprocessing contract suitable for SigLIP-based vision towers (including PaliGemma).

Let `S_ckpt` be the expected image size declared by the selected backbone checkpoint, sourced from the loaded checkpoint config (for PaliGemma, `vision_config.image_size`). `S_ckpt` is a single integer and implies a square checkpoint expectation of `(S_ckpt, S_ckpt)`.

Let `(H, W)` be the *effective* preprocessing resize target used for SigLIP preprocessing:

- By default: `H = S_ckpt` and `W = S_ckpt`
- If `TrainingConfig.vlm_image_height_override` and `TrainingConfig.vlm_image_width_override` are both set: `H = vlm_image_height_override`, `W = vlm_image_width_override`

In config transform strings, `H`/`W` are placeholders for those concrete integer values.

For each *per-sample* camera tensor produced by the dataset pipeline, the preprocessing output SHALL be:

- channel-first with 3 channels (`[3, H, W]`) (collates to `[B, 3, H, W]` as model input)
- dtype `float32`
- resized to exactly `HxW` using bicubic interpolation (in our torchvision v2 transform specs, `Resize([H, W], 3)` uses bicubic; `3` corresponds to bicubic)
- normalized with `mean=(0.5, 0.5, 0.5)` and `std=(0.5, 0.5, 0.5)` (mapping `[0, 1]` to approximately `[-1, 1]`)

#### Scenario: SigLIP preprocessing on float images
- **WHEN** a dataset sample contains a camera tensor with dtype `float32` and values in `[0, 1]`, and the configured image transforms include `Resize([224, 224], 3)` followed by `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`
- **AND** the selected backbone checkpoint expects `S_ckpt=224`
- **THEN** the output camera tensor SHALL have shape `[3, 224, 224]`, dtype `float32`, and values approximately within `[-1.05, 1.05]`

#### Scenario: SigLIP preprocessing on float images (general `H`, `W`)
- **WHEN** a dataset sample contains a camera tensor with dtype `float32` and values in `[0, 1]`, and the configured image transforms include `Resize([H, W], 3)` followed by `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`
- **THEN** the output camera tensor SHALL have shape `[3, H, W]`, dtype `float32`, and values approximately within `[-1.05, 1.05]`

#### Scenario: SigLIP preprocessing on uint8 images
- **WHEN** a dataset sample contains a camera tensor with dtype `uint8` and values in `[0, 255]`, and the configured image transforms include `Resize([H, W], 3)` followed by `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`
- **THEN** the transform pipeline SHALL NOT error, and the output camera tensor SHALL have shape `[3, H, W]`, dtype `float32`, and values approximately within `[-1.05, 1.05]`

### Requirement: Preprocessing occurs before the model forward pass
SigLIP image preprocessing (resize + normalization) SHALL occur in the dataset/data-loader pipeline. SigLIP-based backbones (including PaliGemma) SHALL treat incoming camera tensors as already-preprocessed `pixel_values` and SHALL NOT invoke HuggingFace image processors or apply additional resizing/rescaling/normalization inside the model forward pass.

#### Scenario: No model-internal SigLIP preprocessing
- **WHEN** a training run configures SigLIP preprocessing in the dataset layer (via `image_transforms`) and executes a forward pass through a SigLIP-based backbone
- **THEN** the backbone forward pass SHALL accept preprocessed camera tensors directly as `pixel_values`, without applying a second resize/rescale/normalize step inside the model

### Requirement: Training entrypoint keeps preprocessing aligned with checkpoint expected image size
The training entrypoint SHALL keep dataset image preprocessing aligned with the selected SigLIP-based backbone checkpoint image size.
The checkpoint-declared expected size `S_ckpt` SHALL be derived from the loaded checkpoint config and MUST NOT be inferred by parsing `vlm_name` / model-id strings.
If a training-time size override is configured (both height and width), the training entrypoint SHALL use that override as the effective resize target `(H, W)` for auto-wiring and SHALL log a warning that the checkpoint-declared size is being overridden. If `(H, W) != (S_ckpt, S_ckpt)`, the warning SHOULD include both values.

#### Scenario: Auto-wire preprocessing when unset
- **WHEN** the selected backbone checkpoint expects `S_ckpt` and `DataConfig.image_transforms is None` and no size override is configured
- **THEN** the training entrypoint SHALL wire the canonical SigLIP preprocessing transform list using `Resize([S_ckpt, S_ckpt], 3)` + `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`

#### Scenario: Auto-wire preprocessing with override (warning-only)
- **WHEN** the selected backbone checkpoint expects `S_ckpt` and `DataConfig.image_transforms is None`
- **AND** `TrainingConfig.vlm_image_height_override` and `TrainingConfig.vlm_image_width_override` are set to `(H, W)`
- **THEN** the training entrypoint SHALL wire the canonical SigLIP preprocessing transform list using `Resize([H, W], 3)` + `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`
- **AND** the training entrypoint SHALL log a warning that the checkpoint-declared size is being overridden and that the user is responsible for verifying VLM compatibility

#### Scenario: Explicitly disabling preprocessing is respected
- **WHEN** `DataConfig.image_transforms == []`
- **THEN** the training entrypoint SHALL NOT auto-wire image preprocessing

#### Scenario: Explicit image_transforms are respected
- **WHEN** `DataConfig.image_transforms` is explicitly provided (non-empty)
- **THEN** the training entrypoint SHALL NOT auto-wire image preprocessing (the user is responsible for ensuring those transforms match the selected backbone’s expectations)

#### Scenario: Override requires both height and width
- **WHEN** exactly one of `TrainingConfig.vlm_image_height_override` or `TrainingConfig.vlm_image_width_override` is set
- **THEN** training SHALL fail fast with an error explaining that both height and width overrides must be set together

## Non-normative references

- Transformers `SiglipImageProcessor` / `SiglipImageProcessorFast` (resize to explicit size with bicubic + rescale + normalize):  
  <https://github.com/huggingface/transformers/raw/refs/heads/main/src/transformers/models/siglip/image_processing_siglip.py>  
  <https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/siglip/image_processing_siglip_fast.py>
- Transformers `PaliGemmaConfig` (checkpoint-derived `vision_config.image_size`):  
  <https://github.com/huggingface/transformers/raw/refs/heads/main/src/transformers/models/paligemma/configuration_paligemma.py>
- `google/paligemma-3b-mix-448` model card (448×448 evidence):  
  <https://huggingface.co/google/paligemma-3b-mix-448>
- Transformers `SiglipVisionModel` positional interpolation (`interpolate_pos_encoding`):  
  <https://github.com/huggingface/transformers/raw/refs/heads/main/src/transformers/models/siglip/modeling_siglip.py>
- Torchvision interpolation mapping (bicubic):  
  <https://docs.pytorch.org/vision/main/_modules/torchvision/transforms/functional.html>
