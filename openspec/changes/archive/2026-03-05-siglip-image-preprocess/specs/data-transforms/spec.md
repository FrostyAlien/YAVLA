## MODIFIED Requirements

### Requirement: ImageTransform wraps torchvision v2 augmentations
`ImageTransform` SHALL apply torchvision v2 transforms to all camera keys in the sample.

If a camera tensor is dtype `torch.uint8`, `ImageTransform` SHALL first convert it to dtype `torch.float32` and rescale by `1/255` so that pixel values are in `[0, 1]` before applying the configured torchvision transforms. For non-`uint8` camera tensors, `ImageTransform` SHALL pass values to torchvision transforms unchanged.

#### Scenario: Resize and normalize images
- **WHEN** `ImageTransform(transforms=[Resize([H, W], 3), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))], camera_keys=["image"])` is applied to a sample with `image` as a float tensor
- **THEN** the output `image` tensor SHALL be resized to `[3, H, W]`, have dtype `float32`, and be normalized approximately to `[-1, 1]`

#### Scenario: Uint8 images are supported for normalization transforms
- **WHEN** `ImageTransform(transforms=[Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))], camera_keys=["image"])` is applied to a sample with `image` as a `uint8` tensor in `[0, 255]`
- **THEN** the transform pipeline SHALL NOT error, and the output `image` tensor SHALL have dtype `float32`

#### Scenario: Multiple camera keys
- **WHEN** the sample contains multiple camera keys (`image_left`, `image_right`) and both are in `camera_keys`
- **THEN** the same transform pipeline SHALL be applied independently to each camera key

## ADDED Requirements

### Requirement: build_torchvision_transforms supports SigLIP preprocessing specs
The transform system SHALL support config-driven torchvision v2 transforms sufficient to express SigLIP preprocessing via string specs.

At minimum, the following transform specifications SHALL be accepted and built into usable torchvision v2 transforms:

- `Resize([H, W], 3)` (for any integers `H`, `W`; `3` corresponds to bicubic)
- `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`

#### Scenario: Canonical SigLIP transform list builds successfully
- **WHEN** `build_torchvision_transforms(["Resize([H, W], 3)", "Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))"])` is called for some integers `H`, `W`
- **THEN** it SHALL return a list of callables that can be applied sequentially to camera tensors without error
