## ADDED Requirements

### Requirement: Simple action decoder
`SimpleActionDecoder(ActionDecoderBase)` SHALL unnormalize `ActionPrediction` via `ActionSpaceSpec.limits` and return `ActionChunk`. No temporal ensembling.

#### Scenario: Unnormalize actions
- **WHEN** `decode(prediction, meta)` is called with normalized actions in `[-1, 1]`
- **THEN** `ActionChunk.actions` SHALL be scaled to `ActionSpaceSpec.limits` range

### Requirement: Temporal ensembling decoder
`EnsemblingDecoder(ActionDecoderBase)` SHALL maintain a buffer of overlapping action chunks and average predictions at each timestep, with configurable exponential weighting.

#### Scenario: Overlapping chunk averaging
- **WHEN** two consecutive chunks overlap by 3 timesteps
- **THEN** the overlapping timesteps SHALL be averaged (optionally with exponential decay weighting)

#### Scenario: Contact-aware disable
- **WHEN** `meta.force_magnitude` exceeds a threshold
- **THEN** temporal ensembling SHALL be disabled and the latest chunk used directly

### Requirement: Receding horizon control
The decoder SHALL support receding horizon: execute only the first K steps of each chunk, then re-predict. `horizon_length` SHALL be configurable (default: `chunk_len`).

#### Scenario: Partial chunk execution
- **WHEN** `horizon_length=3` and `chunk_len=10`
- **THEN** only the first 3 actions SHALL be returned per decode call
