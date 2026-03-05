## MODIFIED Requirements

### Requirement: SigLIP image preprocessing contract (model-derived size) for camera tensors
The data pipeline SHALL support a SigLIP-style image preprocessing contract suitable for SigLIP-based vision towers (including PaliGemma), producing fixed-size `pixel_values` while supporting multiple resize strategies.

Let `S_ckpt` be the expected image size declared by the selected backbone checkpoint, sourced from the loaded checkpoint config (for PaliGemma, `vision_config.image_size`). `S_ckpt` is a single integer and implies a square checkpoint expectation of `(S_ckpt, S_ckpt)`.

Let `(H, W)` be the *effective* preprocessing resize target used for SigLIP preprocessing:

- By default: `H = S_ckpt` and `W = S_ckpt`
- If `TrainingConfig.vlm_image_height_override` and `TrainingConfig.vlm_image_width_override` are both set: `H = vlm_image_height_override`, `W = vlm_image_width_override`

Let `resize_strategy` be the configured SigLIP image resize strategy. The pipeline MUST support at least the following strategies:

- `warp`: resize directly to `HxW` with bicubic interpolation (config transform spec: `Resize([H, W], 3)`).
- `openvla_letterbox`: preserve aspect ratio by resizing to fit within `HxW` (bicubic) and then padding to exactly `HxW` (config transform spec: `LetterboxPad([H, W], 3)`).
- `openpi_resize_with_pad`: preserve aspect ratio by resizing to fit within `HxW` and then padding to exactly `HxW`, following the OpenPI reference semantics (config transform spec: `ResizeWithPad([H, W], 3)`).

In config transform strings, `H`/`W` are placeholders for those concrete integer values.

For each *per-sample* camera tensor produced by the dataset pipeline, the preprocessing output SHALL be:

- channel-first with 3 channels (`[3, H, W]`) (collates to `[B, 3, H, W]` as model input)
- dtype `float32`
- resized/padded to exactly `HxW` according to the selected `resize_strategy` (the resize step SHALL use bicubic interpolation)
- if padding is applied, padded pixels SHALL correspond to value `0.5` per channel in `[0, 1]` space before normalization (so they normalize to approximately `0.0`)
- normalized with `mean=(0.5, 0.5, 0.5)` and `std=(0.5, 0.5, 0.5)` (mapping `[0, 1]` to approximately `[-1, 1]`)

#### Scenario: SigLIP preprocessing on float images (warp)
- **WHEN** a dataset sample contains a camera tensor with dtype `float32` and values in `[0, 1]`, and the configured image transforms include `Resize([224, 224], 3)` followed by `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`
- **AND** the selected backbone checkpoint expects `S_ckpt=224`
- **THEN** the output camera tensor SHALL have shape `[3, 224, 224]`, dtype `float32`, and values approximately within `[-1.05, 1.05]`

#### Scenario: SigLIP preprocessing on float images (general `H`, `W`) (warp)
- **WHEN** a dataset sample contains a camera tensor with dtype `float32` and values in `[0, 1]`, and the configured image transforms include `Resize([H, W], 3)` followed by `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`
- **THEN** the output camera tensor SHALL have shape `[3, H, W]`, dtype `float32`, and values approximately within `[-1.05, 1.05]`

#### Scenario: SigLIP preprocessing on float images (letterbox)
- **WHEN** a dataset sample contains a camera tensor with dtype `float32` and values in `[0, 1]`, and the configured image transforms include `LetterboxPad([224, 224], 3)` followed by `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`
- **AND** the input camera tensor has shape `[3, 300, 450]` (3:2 aspect ratio)
- **THEN** the output camera tensor SHALL have shape `[3, 224, 224]`, dtype `float32`
- **AND** the output camera tensor SHALL include padded pixels with values approximately `0.0` after normalization

#### Scenario: SigLIP preprocessing on float images (resize-with-pad)
- **WHEN** a dataset sample contains a camera tensor with dtype `float32` and values in `[0, 1]`, and the configured image transforms include `ResizeWithPad([224, 224], 3)` followed by `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`
- **AND** the input camera tensor has shape `[3, 300, 450]` (3:2 aspect ratio)
- **THEN** the output camera tensor SHALL have shape `[3, 224, 224]`, dtype `float32`
- **AND** the output camera tensor SHALL include padded pixels with values approximately `0.0` after normalization

#### Scenario: SigLIP preprocessing on uint8 images (all strategies)
- **WHEN** a dataset sample contains a camera tensor with dtype `uint8` and values in `[0, 255]`
- **AND** the configured image transforms include one of `Resize([H, W], 3)`, `LetterboxPad([H, W], 3)`, or `ResizeWithPad([H, W], 3)`, followed by `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`
- **THEN** the transform pipeline SHALL NOT error, and the output camera tensor SHALL have shape `[3, H, W]`, dtype `float32`, and values approximately within `[-1.05, 1.05]`

### Requirement: Training entrypoint keeps preprocessing aligned with checkpoint expected image size
The training entrypoint SHALL keep dataset image preprocessing aligned with the selected SigLIP-based backbone checkpoint image size.
The checkpoint-declared expected size `S_ckpt` SHALL be derived from the loaded checkpoint config and MUST NOT be inferred by parsing `vlm_name` / model-id strings.

Let `(H, W)` be the effective preprocessing size resolved from `S_ckpt` and optional override fields, as defined in the SigLIP preprocessing contract.

Let `resize_strategy` be the SigLIP image resize strategy selected by `TrainingConfig.vlm_image_resize_strategy`. The default strategy SHALL be `warp`.
If `resize_strategy` is not a supported value, training SHALL fail fast with an error describing the allowed options.

If a training-time size override is configured (both height and width), the training entrypoint SHALL use that override as the effective preprocessing target `(H, W)` for auto-wiring and SHALL log a warning that the checkpoint-declared size is being overridden. If `(H, W) != (S_ckpt, S_ckpt)`, the warning SHOULD include both values.

#### Scenario: Auto-wire preprocessing when unset (warp default)
- **WHEN** the selected backbone checkpoint expects `S_ckpt`
- **AND** `DataConfig.image_transforms is None`
- **AND** `TrainingConfig.vlm_image_resize_strategy` is unset or set to `warp`
- **AND** no size override is configured
- **THEN** the training entrypoint SHALL wire the canonical SigLIP preprocessing transform list using `Resize([S_ckpt, S_ckpt], 3)` + `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`

#### Scenario: Auto-wire preprocessing when unset (OpenVLA letterbox)
- **WHEN** the selected backbone checkpoint expects `S_ckpt`
- **AND** `DataConfig.image_transforms is None`
- **AND** `TrainingConfig.vlm_image_resize_strategy == "openvla_letterbox"`
- **THEN** the training entrypoint SHALL wire the canonical SigLIP preprocessing transform list using `LetterboxPad([S_ckpt, S_ckpt], 3)` + `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`

#### Scenario: Auto-wire preprocessing when unset (OpenPI resize-with-pad)
- **WHEN** the selected backbone checkpoint expects `S_ckpt`
- **AND** `DataConfig.image_transforms is None`
- **AND** `TrainingConfig.vlm_image_resize_strategy == "openpi_resize_with_pad"`
- **THEN** the training entrypoint SHALL wire the canonical SigLIP preprocessing transform list using `ResizeWithPad([S_ckpt, S_ckpt], 3)` + `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`

#### Scenario: Auto-wire preprocessing with override (warning-only)
- **WHEN** the selected backbone checkpoint expects `S_ckpt` and `DataConfig.image_transforms is None`
- **AND** `TrainingConfig.vlm_image_height_override` and `TrainingConfig.vlm_image_width_override` are set to `(H, W)`
- **THEN** the training entrypoint SHALL wire the canonical SigLIP preprocessing transform list using the configured `resize_strategy` at size `(H, W)` followed by `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`
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
