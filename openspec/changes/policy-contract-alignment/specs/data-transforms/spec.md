## MODIFIED Requirements

<!--
Refs (bounds normalization to [-1, 1]):
- Octo supports bounds normalization mapping to [-1, 1] (NormalizationType.BOUNDS):
  https://raw.githubusercontent.com/octo-models/octo/main/octo/data/utils/data_utils.py
- OpenPI quantile normalization maps to [-1, 1] with eps for stability (illustrates the common convention):
  https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/transforms.py
-->

### Requirement: NormalizeTransform applies statistical normalization
`NormalizeTransform` SHALL normalize specified feature keys using statistics (mean, std or min, max) from `LeRobotDatasetMetadata.stats`. Output SHALL always be `torch.Tensor` with dtype `float32`.

It SHALL support the following `mode` values:
- `z-score` (mean/std normalization)
- `min-max` (map to `[0, 1]`)
- `bounds` (map to `[-1, 1]`)

#### Scenario: Z-score normalization
- **WHEN** `NormalizeTransform(stats, mode="z-score", keys=["observation.state"])` is applied
- **THEN** `observation.state` SHALL be transformed to `(value - mean) / std` using the corresponding stats, output as `torch.Tensor`

#### Scenario: Min-max normalization
- **WHEN** `NormalizeTransform(stats, mode="min-max", keys=["action"])` is applied
- **THEN** `action` SHALL be transformed to `(value - min) / (max - min)`, mapping to [0, 1] range, output as `torch.Tensor`

#### Scenario: Bounds normalization
- **WHEN** `NormalizeTransform(stats, mode="bounds", keys=["action"])` is applied
- **THEN** `action` SHALL be transformed to `2 * (value - min) / (max - min) - 1`, mapping to [-1, 1] range, output as `torch.Tensor`

#### Scenario: Keys without stats are skipped
- **WHEN** a key in the sample has no corresponding entry in `stats`
- **THEN** that key SHALL be passed through unchanged without error

#### Scenario: Z-score normalization with zero std
- **WHEN** `NormalizeTransform(stats, mode="z-score", keys=[...])` is applied and a target key has `std == 0`
- **THEN** that key SHALL normalize to zeros for those elements (no division by zero)

#### Scenario: Min-max normalization with zero range
- **WHEN** `NormalizeTransform(stats, mode="min-max", keys=[...])` is applied and a target key has `max == min`
- **THEN** that key SHALL normalize to zeros for those elements (no division by zero)

#### Scenario: Bounds normalization with zero range
- **WHEN** `NormalizeTransform(stats, mode="bounds", keys=[...])` is applied and a target key has `max == min`
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

It SHALL support the same `mode` values as `NormalizeTransform`.

#### Scenario: Z-score unnormalization roundtrip
- **WHEN** a value is normalized with z-score and then unnormalized with the same stats
- **THEN** the result SHALL equal the original value within float32 tolerance (`atol=1e-6`), output as `torch.Tensor`

#### Scenario: Min-max unnormalization roundtrip
- **WHEN** a value is normalized with min-max and then unnormalized with the same stats
- **THEN** the result SHALL equal the original value within float32 tolerance (`atol=1e-6`), output as `torch.Tensor`

#### Scenario: Bounds unnormalization roundtrip
- **WHEN** a value is normalized with bounds mode and then unnormalized with the same stats
- **THEN** the result SHALL equal the original value within float32 tolerance (`atol=1e-6`), output as `torch.Tensor`

#### Scenario: Zero-variance and zero-range unnormalization
- **WHEN** `UnnormalizeTransform` is applied for stats with `std == 0` (z-score) or `max == min` (min-max/bounds)
- **THEN** it SHALL return the corresponding constant original-scale value (`mean` for z-score, `min` for min-max/bounds) without numerical instability

