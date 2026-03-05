## ADDED Requirements

### Requirement: SigLIP-224 image preprocessing contract for camera tensors
The data pipeline SHALL support a SigLIP-style image preprocessing contract suitable for SigLIP-based vision towers (including PaliGemma). For each camera tensor passed to the model, the preprocessing output SHALL be:

- channel-first with 3 channels (`[3, 224, 224]`)
- dtype `float32`
- resized to exactly `224x224` using bicubic interpolation
- normalized with `mean=(0.5, 0.5, 0.5)` and `std=(0.5, 0.5, 0.5)` (mapping `[0, 1]` to `[-1, 1]`)

#### Scenario: SigLIP preprocessing on float images
- **WHEN** a dataset sample contains a camera tensor with dtype `float32` and values in `[0, 1]`, and the configured image transforms include `Resize([224, 224], 3)` followed by `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`
- **THEN** the output camera tensor SHALL have shape `[3, 224, 224]`, dtype `float32`, and values approximately within `[-1, 1]`

#### Scenario: SigLIP preprocessing on uint8 images
- **WHEN** a dataset sample contains a camera tensor with dtype `uint8` and values in `[0, 255]`, and the configured image transforms include `Resize([224, 224], 3)` followed by `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`
- **THEN** the transform pipeline SHALL NOT error, and the output camera tensor SHALL have shape `[3, 224, 224]`, dtype `float32`, and values approximately within `[-1, 1]`

### Requirement: Preprocessing occurs before the model forward pass
SigLIP-224 image preprocessing (resize + normalization) SHALL occur in the dataset/data-loader pipeline. SigLIP-based backbones (including PaliGemma) SHALL treat incoming camera tensors as already-preprocessed `pixel_values` and SHALL NOT invoke HuggingFace image processors or apply additional resizing/rescaling/normalization inside the model forward pass.

#### Scenario: No model-internal SigLIP preprocessing
- **WHEN** a training run configures SigLIP preprocessing in the dataset layer (via `image_transforms`) and executes a forward pass through a SigLIP-based backbone
- **THEN** the backbone forward pass SHALL accept preprocessed camera tensors directly as `pixel_values`, without applying a second resize/rescale/normalize step inside the model

