## ADDED Requirements

### Requirement: Default LeRobotDataset backend decodes video with real data
Integration tests SHALL verify that direct default `LeRobotDataset` access decodes video/media keys into tensors for real `lerobot/pusht` data when baseline decoding is available in the environment.

#### Scenario: Default backend decodes media tensors
- **WHEN** `LeRobotDataset` is constructed with `repo_id="lerobot/pusht"` and iterated over a bounded sample window
- **THEN** at least one media key SHALL resolve to a `torch.Tensor` with 3 dimensions `(C, H, W)` and dtype `torch.float32` or `torch.uint8`

#### Scenario: Default backend decode checks are baseline-gated for environment variability
- **WHEN** the environment cannot decode media tensors for direct `LeRobotDataset` samples in the bounded window
- **THEN** decode-specific integration assertions SHALL be skipped with an explicit reason message

### Requirement: Factory default backend path works on real data
Integration tests SHALL verify that `create_dataloader()` with `backend="default"` yields valid real-data batches and decoded media tensors when baseline decode is available.

#### Scenario: Factory default dataloader yields required metadata batch keys
- **WHEN** `create_dataloader()` is called with `DataConfig(repo_id="lerobot/pusht", backend="default", batch_size=2, num_workers=0)`
- **THEN** the first batch SHALL be a dict containing required metadata keys (`episode_index`, `index`, `timestamp`, `frame_index`, `task_index`) with leading batch dimension `2`

#### Scenario: Factory default dataloader yields decoded batched media tensor
- **WHEN** baseline default decode is available and `create_dataloader()` is called with `backend="default"`
- **THEN** at least one media key in the first batch SHALL be a `torch.Tensor` with 4 dimensions `(B, C, H, W)` and dtype `torch.float32` or `torch.uint8`
