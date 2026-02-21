## ADDED Requirements

<!--
Refs (prior art / pitfalls):
- SB3 action rescaling utilities ([-1, 1] ↔ [low, high]): https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/policies.html
- SpinningUp SAC actor tanh-squash + scale (bounded actions): https://raw.githubusercontent.com/openai/spinningup/master/spinup/algos/pytorch/sac/core.py
- Octo supports both mean/std and bounds normalization: https://raw.githubusercontent.com/octo-models/octo/main/octo/data/utils/data_utils.py
- OpenPI normalizes actions/proprio with dataset stats (mean/std) and recommends storing/reloading stats: https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/docs/norm_stats.md
- LeRobot normalization pipeline supports multiple modes and stores stats alongside policy: https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/processor/normalize_processor.py
-->

### Requirement: Policy-declared action normalization mode
Every `PolicyConfig`-backed policy SHALL declare an `action_normalization` mode that specifies the numeric space the
action head predicts in, and the numeric space expected in `TrainingBatch.actions`.

Supported modes for this change:
- `bounds`: normalized action space is `[-1, 1]` per dimension.
- `z-score`: normalized action space is mean/std normalized.

#### Scenario: Default mode is bounds
- **WHEN** an MVP MLP policy config is constructed with defaults
- **THEN** the action normalization mode SHALL be `bounds`

#### Scenario: Mode is persisted for checkpoint roundtrip
- **WHEN** `VLAPolicy.save_pretrained(path)` is called
- **THEN** the saved artifacts SHALL include the action normalization mode such that `VLAPolicy.from_pretrained(path)` can reconstruct identical normalization behavior without relying on external context

### Requirement: Bounds normalization uses `ActionSpaceSpec.limits`
In `bounds` mode, action normalization SHALL be defined relative to `ActionSpaceSpec.limits` (shape `[action_dim, 2]`
with `(min, max)` per dimension):

- **Normalize (training targets):** `a_norm = 2 * (a - lo) / (hi - lo) - 1`
- **Unnormalize (inference outputs):** `a = (a_norm + 1) / 2 * (hi - lo) + lo`

Implementations MUST guard division by zero for any dimension where `hi == lo`.

#### Scenario: Decoder unnormalizes in bounds mode
- **WHEN** `SimpleActionDecoder.decode(pred)` is called with `pred.mean` in `[-1, 1]` and `ActionSpaceSpec.limits` set
- **THEN** `ActionChunk.actions` SHALL be unnormalized into `[lo, hi]` using the linear mapping above

#### Scenario: Missing limits is an error in bounds mode
- **WHEN** the action normalization mode is `bounds` but `ActionSpaceSpec.limits` is `None`
- **THEN** policy construction or first use SHALL raise a `ValueError` explaining that bounds mode requires limits

#### Scenario: No implicit clamping in decoder (performance contract)
- **WHEN** `decode(pred)` receives an out-of-range normalized action value (e.g., `2.0`)
- **THEN** the decoder SHALL NOT clamp it to `[-1, 1]` by default and SHALL apply the linear mapping as-is

<!-- Rationale: avoiding extra ops/sync points in latency-critical inference; clamping belongs in env/policy wrapper. -->

### Requirement: Z-score normalization uses stored action statistics
In `z-score` mode, action normalization SHALL use per-dimension statistics stored alongside the checkpoint (e.g.,
`action_stats.json`), with an epsilon for stability:

- **Normalize:** `a_norm = (a - mean) / (std + eps)`
- **Unnormalize:** `a = a_norm * (std + eps) + mean`

Implementations MUST treat `std == 0` as a special case and avoid division-by-zero (e.g., map that dimension to zero in
normalized space and to `mean` in unnormalized space).

#### Scenario: Z-score normalization roundtrip
- **WHEN** `z-score` normalization is used with non-degenerate stats and an action tensor `a`
- **THEN** normalizing then unnormalizing SHALL reproduce `a` within float tolerance

#### Scenario: Missing stats is an error in z-score mode
- **WHEN** the action normalization mode is `z-score` but required stats (`mean`, `std`) are missing
- **THEN** policy construction or first use SHALL raise a `ValueError` describing which keys are missing

### Requirement: ActionSpaceSpec remains the embodiment-level bounds contract
`ActionSpaceSpec` SHALL continue to represent embodiment-level action semantics (names/units/bounds), and MUST NOT
introduce ad-hoc decoder behavior toggles in its constructor (e.g., `clip_unnormalized`).

#### Scenario: Spec construction does not accept `clip_unnormalized`
- **WHEN** an `ActionSpaceSpec` is constructed
- **THEN** it SHALL accept only the fields defined in the model-types capability (`names`, `units`, `limits`, `frame`, `control_mode`)

