## MODIFIED Requirements

### Requirement: NormalizeTransform outputs torch.Tensor always
`NormalizeTransform` SHALL output `torch.Tensor` for all normalized keys, removing the `_restore_type` conversion step. The `_restore_type` function SHALL be deleted. The `_to_tensor` helper SHALL be retained for input conversion.

#### Scenario: Numpy input produces tensor output
- **WHEN** `NormalizeTransform` is applied to a sample where `action` is a `numpy.ndarray`
- **THEN** the output `action` SHALL be a `torch.Tensor` with dtype `float32`, NOT converted back to numpy

#### Scenario: Python scalar input produces tensor output
- **WHEN** `NormalizeTransform` is applied to a sample where a key's value is a Python `float`
- **THEN** the output SHALL be a `torch.Tensor`, NOT converted back to `float`

#### Scenario: Tensor input cast to float32
- **WHEN** `NormalizeTransform` is applied to a sample where a key's value is already a `torch.Tensor` (any dtype, e.g., `float64`)
- **THEN** the output SHALL be a `torch.Tensor` with dtype `float32` (normalization always operates in float32)

#### Scenario: Non-normalized keys pass through unchanged
- **WHEN** a key has no matching stats entry or is not in the target key set
- **THEN** that key's value SHALL pass through with its original type unchanged

### Requirement: UnnormalizeTransform outputs torch.Tensor always
`UnnormalizeTransform` SHALL output `torch.Tensor` for all unnormalized keys, matching the same output-type contract as `NormalizeTransform`.

#### Scenario: Roundtrip preserves values
- **WHEN** a value is normalized with `NormalizeTransform` and then unnormalized with `UnnormalizeTransform` using the same stats and mode
- **THEN** the result SHALL equal the original value within float32 tolerance (`atol=1e-6`), and the output SHALL be a `torch.Tensor` with dtype `float32`. Note: precision loss is expected when the original input was float64 or a Python float.

### Requirement: Smart key filtering when keys=None
When `NormalizeTransform.keys` is `None`, the transform SHALL iterate only keys present in both `self.stats` and the sample dict, using order-preserving filtering over `self.stats` keys.

#### Scenario: Default keys filters to stats intersection
- **WHEN** `NormalizeTransform(stats, keys=None)` is applied to a sample with 12 keys, where only `action` and `observation.state` have stats entries
- **THEN** only `action` and `observation.state` SHALL be processed; the remaining 10 keys SHALL not be iterated

#### Scenario: Iteration order is deterministic
- **WHEN** `keys=None` and stats contains keys `["action", "observation.state"]`
- **THEN** the iteration order SHALL follow the stats key order, not the sample key order

#### Scenario: Explicit keys override still works
- **WHEN** `NormalizeTransform(stats, keys=["action"])` is applied
- **THEN** only `action` SHALL be processed, regardless of what other keys exist in stats
