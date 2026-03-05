## MODIFIED Requirements

### Requirement: Backend selection and guardrails
The factory SHALL use `DataConfig.backend` directly and validate unsupported feature/backend combinations.

When `backend="default"` and `action_chunk_size` is configured, the factory SHALL support action chunking and MUST assemble chunked actions through LeRobot temporal queries (via `delta_timestamps` on the `action` key).

`action_chunk_size` is a convenience alias for contiguous forward action deltas (including the current frame at step 0). For custom/non-contiguous action deltas, users MAY configure `delta_timestamps["action"]` directly and leave `action_chunk_size` unset.

If `action_chunk_size` is set and `delta_timestamps` is also set with an explicit `"action"` entry, the factory SHALL raise a `ValueError` indicating the configuration is ambiguous (choose exactly one way to specify action chunking).

#### Scenario: Default backend is used when backend is omitted
- **WHEN** `DataConfig` is constructed with only `repo_id` (and no explicit backend)
- **THEN** the factory SHALL use the `default` (LeRobotDataset) backend

#### Scenario: Explicit lazy backend selection
- **WHEN** `backend="lazy"` is configured
- **THEN** the factory SHALL use `LazyLeRobotDataset`

#### Scenario: Explicit streaming backend selection
- **WHEN** `backend="streaming"` is configured
- **THEN** the factory SHALL use `ShardInterleavedDataset`

#### Scenario: Default backend supports action chunking via LeRobot temporal queries
- **WHEN** `backend="default"` and `action_chunk_size=4` is configured and `delta_timestamps` is unset (or does not contain an `"action"` entry)
- **THEN** `create_dataloader()` SHALL succeed (no `ValueError`), and default-backend samples SHALL include `action` as a stacked tensor with first dimension `4` and `action_is_pad` as a boolean tensor of shape `(4,)`

#### Scenario: Default backend rejects conflicting action chunk configuration
- **WHEN** `backend="default"`, `action_chunk_size` is configured, and `delta_timestamps` is configured with an explicit `"action"` entry
- **THEN** `create_dataloader()` SHALL raise a `ValueError` explaining that `"action"` chunking cannot be specified by both `action_chunk_size` and `delta_timestamps["action"]`

#### Scenario: Streaming backend rejects temporal features
- **WHEN** `backend="streaming"` and `delta_timestamps` or `action_chunk_size` is configured
- **THEN** `create_dataloader()` SHALL raise a `ValueError` indicating temporal features require the `lazy` or `default` backend

#### Scenario: Backend selection logging
- **WHEN** a backend is selected
- **THEN** the factory SHALL log the chosen backend and the reason for the selection
