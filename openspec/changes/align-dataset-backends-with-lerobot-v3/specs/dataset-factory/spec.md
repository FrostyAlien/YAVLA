## MODIFIED Requirements

### Requirement: create_dataloader factory function
`create_dataloader()` SHALL accept a `DataConfig` dataclass and return a fully configured `torch.utils.data.DataLoader` with the appropriate dataset backend and transforms wired up.

#### Scenario: Explicit lazy backend selection
- **WHEN** Something like `DataConfig(repo_id="lerobot/aloha_sim", backend="lazy")` is passed to `create_dataloader()`
- **THEN** it SHALL return a `DataLoader` wrapping a `LazyLeRobotDataset` instance

#### Scenario: Default backend selection
- **WHEN** Something like `DataConfig(repo_id="lerobot/aloha_sim", backend="default")` is passed
- **THEN** it SHALL return a `DataLoader` wrapping a standard `LeRobotDataset` instance

#### Scenario: Streaming backend remains selectable
- **WHEN** Something like `DataConfig(repo_id="lerobot/aloha_sim", backend="streaming")` is passed
- **THEN** it SHALL continue to return a `DataLoader` for the currently configured streaming backend implementation

### Requirement: Backend roles are documented accurately
The public backend matrix SHALL describe backend roles according to actual training guarantees rather than presenting all backends as equal peers.

This means:

- `default` SHALL be documented as the upstream LeRobot-native baseline
- `lazy` SHALL be documented as the supported YAVLA-owned backend for workloads that need map-style random access or sampler-driven training behavior
- `streaming` SHALL be documented as experimental and SHALL NOT be presented as the recommended backend for workloads that require true random access, strong sampler semantics, or stable distributed epoch behavior

#### Scenario: Default backend is described as upstream-native
- **WHEN** factory-facing docs or guides describe backend choices
- **THEN** `backend="default"` SHALL be identified as the already-aligned upstream path

#### Scenario: Random-access guidance recommends default or lazy
- **WHEN** docs explain how to choose a backend for workloads that need true random access or sampler-driven shuffling
- **THEN** they SHALL direct users to `default` or `lazy`, not to `streaming`

#### Scenario: Streaming backend is labeled experimental
- **WHEN** docs or guides mention `backend="streaming"`
- **THEN** they SHALL label it as experimental rather than as a primary supported training backend

#### Scenario: Future upstream replacement is documented as follow-up only
- **WHEN** docs mention possible future ownership changes for the streaming backend
- **THEN** they MAY state that the experimental streaming backend could later be replaced by upstream `StreamingLeRobotDataset`, but SHALL NOT claim that replacement is already part of this change
