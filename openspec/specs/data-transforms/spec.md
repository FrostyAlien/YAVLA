## ADDED Requirements

### Requirement: DataTransformFn protocol
The transform system SHALL define a `DataTransformFn` protocol: any callable with signature `__call__(self, sample: dict[str, Any]) -> dict[str, Any]` qualifies as a transform. No base class inheritance required.

#### Scenario: Plain function as transform
- **WHEN** a plain function `def my_transform(sample: dict[str, Any]) -> dict[str, Any]` is used as a transform
- **THEN** it SHALL be accepted by `compose()` and by all dataset backends without wrapping or adaptation

#### Scenario: Class-based transform
- **WHEN** a class with `__call__(self, sample: dict[str, Any]) -> dict[str, Any]` is used as a transform
- **THEN** it SHALL be accepted identically to a plain function transform

### Requirement: compose() chains transforms sequentially
`compose(*transforms)` SHALL return a single `DataTransformFn` that applies each transform in order, passing the output of one as the input to the next.

#### Scenario: Composition of three transforms
- **WHEN** `composed = compose(repack, normalize, image_transform)` is called with a sample
- **THEN** the sample SHALL first pass through `repack`, then `normalize`, then `image_transform`, and the final output SHALL be returned

#### Scenario: Empty composition
- **WHEN** `compose()` is called with no arguments
- **THEN** it SHALL return an identity transform that passes the sample through unchanged

### Requirement: RepackTransform remaps dictionary keys
`RepackTransform` SHALL remap keys in the sample dictionary according to a provided mapping, enabling translation between dataset-native key names and model-expected key names.

#### Scenario: Key remapping
- **WHEN** `RepackTransform({"observation.images.laptop": "image", "observation.state": "state"})` is applied to a sample with key `observation.images.laptop`
- **THEN** the output SHALL contain key `image` with the same value, and `observation.images.laptop` SHALL be removed

#### Scenario: Unmapped keys preserved
- **WHEN** a sample contains keys not in the mapping (e.g., `timestamp`, `episode_index`)
- **THEN** those keys SHALL be preserved unchanged in the output

### Requirement: NormalizeTransform applies statistical normalization
`NormalizeTransform` SHALL normalize specified feature keys using statistics (mean, std or min, max) from `LeRobotDatasetMetadata.stats`. Output SHALL always be `torch.Tensor` with dtype `float32`.

#### Scenario: Z-score normalization
- **WHEN** `NormalizeTransform(stats, mode="z-score", keys=["observation.state"])` is applied
- **THEN** `observation.state` SHALL be transformed to `(value - mean) / std` using the corresponding stats, output as `torch.Tensor`

#### Scenario: Min-max normalization
- **WHEN** `NormalizeTransform(stats, mode="min-max", keys=["action"])` is applied
- **THEN** `action` SHALL be transformed to `(value - min) / (max - min)`, mapping to [0, 1] range, output as `torch.Tensor`

#### Scenario: Keys without stats are skipped
- **WHEN** a key in the sample has no corresponding entry in `stats`
- **THEN** that key SHALL be passed through unchanged without error

#### Scenario: Z-score normalization with zero std
- **WHEN** `NormalizeTransform(stats, mode="z-score", keys=[...])` is applied and a target key has `std == 0`
- **THEN** that key SHALL normalize to zeros for those elements (no division by zero)

#### Scenario: Min-max normalization with zero range
- **WHEN** `NormalizeTransform(stats, mode="min-max", keys=[...])` is applied and a target key has `max == min`
- **THEN** that key SHALL normalize to zeros for those elements (no division by zero)

#### Scenario: Numpy input produces tensor output
- **WHEN** `NormalizeTransform` is applied to a sample where `action` is a `numpy.ndarray`
- **THEN** the output `action` SHALL be a `torch.Tensor` with dtype `float32`, NOT converted back to numpy

#### Scenario: Numpy scalar input produces tensor output
- **WHEN** `NormalizeTransform` is applied to a sample where a key's value is a numpy scalar (`np.int64`, `np.bool_`, etc.)
- **THEN** `_to_tensor` SHALL convert it via `value.item()` before `torch.as_tensor`, producing a `torch.Tensor`

#### Scenario: Non-normalized keys pass through unchanged
- **WHEN** a key has no matching stats entry or is not in the target key set
- **THEN** that key's value SHALL pass through with its original type unchanged

### Requirement: UnnormalizeTransform inverts normalization
`UnnormalizeTransform` SHALL invert the normalization applied by `NormalizeTransform`, for use during inference to convert model outputs back to original scale. Output SHALL always be `torch.Tensor` with dtype `float32`.

#### Scenario: Z-score unnormalization roundtrip
- **WHEN** a value is normalized with z-score and then unnormalized with the same stats
- **THEN** the result SHALL equal the original value within float32 tolerance (`atol=1e-6`), output as `torch.Tensor`

#### Scenario: Min-max unnormalization roundtrip
- **WHEN** a value is normalized with min-max and then unnormalized with the same stats
- **THEN** the result SHALL equal the original value within float32 tolerance (`atol=1e-6`), output as `torch.Tensor`

#### Scenario: Zero-variance and zero-range unnormalization
- **WHEN** `UnnormalizeTransform` is applied for stats with `std == 0` (z-score) or `max == min` (min-max)
- **THEN** it SHALL return the corresponding constant original-scale value (`mean` for z-score, `min` for min-max) without numerical instability

### Requirement: ImageTransform wraps torchvision v2 augmentations
`ImageTransform` SHALL apply a configured list of image transforms to all camera keys in the sample.
The configured list MAY include torchvision v2 transforms and YAVLA-provided custom image transforms (e.g., aspect-ratio-preserving resize+pad transforms used for SigLIP preprocessing).

If a camera tensor is dtype `torch.uint8`, `ImageTransform` SHALL first convert it to dtype `torch.float32` and rescale by `1/255` so that pixel values are in `[0, 1]` before applying the configured transforms. For non-`uint8` camera tensors, `ImageTransform` SHALL pass values to the configured transforms unchanged.

#### Scenario: Resize and normalize images
- **WHEN** `ImageTransform(transforms=[Resize([H, W], 3), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))], camera_keys=["image"])` is applied to a sample with `image` as a float tensor
- **THEN** the output `image` tensor SHALL be resized to `[3, H, W]`, have dtype `float32`, and be normalized approximately to `[-1, 1]`

#### Scenario: Uint8 images are supported for normalization transforms
- **WHEN** `ImageTransform(transforms=[Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))], camera_keys=["image"])` is applied to a sample with `image` as a `uint8` tensor in `[0, 255]`
- **THEN** the transform pipeline SHALL NOT error, and the output `image` tensor SHALL have dtype `float32`

#### Scenario: Multiple camera keys
- **WHEN** the sample contains multiple camera keys (`image_left`, `image_right`) and both are in `camera_keys`
- **THEN** the same transform pipeline SHALL be applied independently to each camera key

#### Scenario: Custom SigLIP pad transforms are supported
- **WHEN** `ImageTransform(transforms=[LetterboxPad([H, W], 3)], camera_keys=["image"])` is applied to a sample with `image` as a float tensor
- **THEN** the transform pipeline SHALL NOT error

### Requirement: build_torchvision_transforms supports SigLIP preprocessing specs
The transform system SHALL support config-driven image transforms sufficient to express SigLIP preprocessing via string specs.

At minimum, the following transform specifications SHALL be accepted and built into usable transform callables:

- `Resize([H, W], 3)` (bicubic warp to fixed size)
- `LetterboxPad([H, W], 3)` (aspect-ratio-preserving resize-to-fit + symmetric pad)
- `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`

#### Scenario: Canonical SigLIP transform list builds successfully (warp)
- **WHEN** `build_torchvision_transforms(["Resize([H, W], 3)", "Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))"])` is called for some integers `H`, `W`
- **THEN** it SHALL return a list of callables that can be applied sequentially to camera tensors without error

#### Scenario: Canonical SigLIP transform list builds successfully (letterbox)
- **WHEN** `build_torchvision_transforms(["LetterboxPad([H, W], 3)", "Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))"])` is called for some integers `H`, `W`
- **THEN** it SHALL return a list of callables that can be applied sequentially to camera tensors without error

### Requirement: Smart key filtering when keys=None
When `NormalizeTransform.keys` is `None`, the transform SHALL iterate only keys present in both `self.stats` and the sample dict, using order-preserving filtering over `self.stats` keys.

#### Scenario: Default keys filters to stats intersection
- **WHEN** `NormalizeTransform(stats, keys=None)` is applied to a sample with 12 keys, where only `action` and `observation.state` have stats entries
- **THEN** only `action` and `observation.state` SHALL be processed

#### Scenario: Iteration order follows stats key order
- **WHEN** `keys=None` and stats contains keys `["action", "observation.state"]`
- **THEN** the iteration order SHALL follow the stats key order, not the sample key order

#### Scenario: Explicit keys override still works
- **WHEN** `NormalizeTransform(stats, keys=["action"])` is applied
- **THEN** only `action` SHALL be processed, regardless of what other keys exist in stats
