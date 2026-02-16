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
`NormalizeTransform` SHALL normalize specified feature keys using statistics (mean, std or min, max) from `LeRobotDatasetMetadata.stats`.

#### Scenario: Z-score normalization
- **WHEN** `NormalizeTransform(stats, mode="z-score", keys=["observation.state"])` is applied
- **THEN** `observation.state` SHALL be transformed to `(value - mean) / std` using the corresponding stats

#### Scenario: Min-max normalization
- **WHEN** `NormalizeTransform(stats, mode="min-max", keys=["action"])` is applied
- **THEN** `action` SHALL be transformed to `(value - min) / (max - min)`, mapping to [0, 1] range

#### Scenario: Keys without stats are skipped
- **WHEN** a key in the sample has no corresponding entry in `stats`
- **THEN** that key SHALL be passed through unchanged without error

#### Scenario: Z-score normalization with zero std
- **WHEN** `NormalizeTransform(stats, mode="z-score", keys=[...])` is applied and a target key has `std == 0`
- **THEN** that key SHALL normalize to zeros for those elements (no division by zero)

#### Scenario: Min-max normalization with zero range
- **WHEN** `NormalizeTransform(stats, mode="min-max", keys=[...])` is applied and a target key has `max == min`
- **THEN** that key SHALL normalize to zeros for those elements (no division by zero)

### Requirement: UnnormalizeTransform inverts normalization
`UnnormalizeTransform` SHALL invert the normalization applied by `NormalizeTransform`, for use during inference to convert model outputs back to original scale.

#### Scenario: Z-score unnormalization roundtrip
- **WHEN** a value is normalized with z-score and then unnormalized with the same stats
- **THEN** the result SHALL equal the original value (within floating-point tolerance)

#### Scenario: Min-max unnormalization roundtrip
- **WHEN** a value is normalized with min-max and then unnormalized with the same stats
- **THEN** the result SHALL equal the original value (within floating-point tolerance)

#### Scenario: Zero-variance and zero-range unnormalization
- **WHEN** `UnnormalizeTransform` is applied for stats with `std == 0` (z-score) or `max == min` (min-max)
- **THEN** it SHALL return the corresponding constant original-scale value (`mean` for z-score, `min` for min-max) without numerical instability

### Requirement: ImageTransform wraps torchvision v2 augmentations
`ImageTransform` SHALL apply torchvision v2 transforms to all camera keys in the sample.

#### Scenario: Resize and normalize images
- **WHEN** `ImageTransform(transforms=[Resize(224), Normalize(mean, std)], camera_keys=["image"])` is applied
- **THEN** `image` tensor SHALL be resized to 224x224 and normalized with the given mean/std

#### Scenario: Multiple camera keys
- **WHEN** the sample contains multiple camera keys (`image_left`, `image_right`) and both are in `camera_keys`
- **THEN** the same transform pipeline SHALL be applied independently to each camera key
