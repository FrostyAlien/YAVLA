## MODIFIED Requirements

### Requirement: Transform pipeline wiring
The factory SHALL compose and attach the transform pipeline to the dataset based on `DataConfig` settings.

#### Scenario: Normalization enabled with default keys excludes camera keys
- **WHEN** `DataConfig.normalize=True`, dataset stats are available via `LeRobotDatasetMetadata.stats`, and `DataConfig.normalize_keys is None`
- **THEN** the factory SHALL include a `NormalizeTransform` in the pipeline configured with an explicit key list derived from stats keys that EXCLUDES all camera keys from dataset metadata

#### Scenario: Normalization enabled with explicit keys
- **WHEN** `DataConfig.normalize=True`, dataset stats are available via `LeRobotDatasetMetadata.stats`, and `DataConfig.normalize_keys` is explicitly provided
- **THEN** the factory SHALL include a `NormalizeTransform(keys=DataConfig.normalize_keys)` in the pipeline (even if that list includes camera/image keys)

#### Scenario: Normalization disabled
- **WHEN** `DataConfig.normalize=False`
- **THEN** no `NormalizeTransform` SHALL be included in the pipeline

#### Scenario: Custom repack mapping
- **WHEN** `DataConfig.repack_keys` specifies a key mapping
- **THEN** a `RepackTransform` with that mapping SHALL be prepended to the pipeline

#### Scenario: Image transforms configured
- **WHEN** `DataConfig.image_transforms` specifies transform names
- **THEN** an `ImageTransform` with the corresponding torchvision v2 transforms SHALL be included in the pipeline, applied to all camera keys from dataset metadata

