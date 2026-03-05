## MODIFIED Requirements

### Requirement: Factory default backend path works on real data
Integration tests SHALL verify that `create_dataloader()` with `backend="default"` yields valid real-data batches and decoded media tensors when baseline decode is available.

#### Scenario: Factory default dataloader yields required metadata batch keys
- **WHEN** `create_dataloader()` is called with `DataConfig(repo_id="lerobot/pusht", backend="default", batch_size=2, num_workers=0)`
- **THEN** the first batch SHALL be a dict containing required metadata keys (`episode_index`, `index`, `timestamp`, `frame_index`, `task_index`) with leading batch dimension `2`

#### Scenario: Factory default dataloader yields decoded batched media tensor
- **WHEN** baseline default decode is available and `create_dataloader()` is called with `backend="default"`
- **THEN** at least one media key in the first batch SHALL be a `torch.Tensor` with 4 dimensions `(B, C, H, W)` and dtype `torch.float32` or `torch.uint8`

#### Scenario: Factory default backend supports action_chunk_size with end-of-episode padding
- **WHEN** `create_dataloader()` is called with `DataConfig(repo_id="lerobot/pusht", backend="default", action_chunk_size=4, batch_size=2, num_workers=0)` and a sample corresponding to the final frame of an episode is accessed from the returned dataset
- **THEN** the sample SHALL contain `action` as a 2-D tensor with first dimension `4`, and `action_is_pad` as a boolean tensor of shape `(4,)` with at least one `True` entry for padded future steps, and the step-0 `action_is_pad` entry SHALL be `False`
