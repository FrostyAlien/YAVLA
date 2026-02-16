## Why

Current integration coverage for `lerobot/pusht` is strict for lazy/streaming media decoding, but default backend coverage is still limited to load/schema smoke checks. This leaves a parity gap for:

- direct default `LeRobotDataset` media decoding assertions
- `create_dataloader(..., backend="default")` behavior on real data

The gap matters because lazy/streaming tests use default LeRobot decoding as their baseline gate, and factory behavior should be validated for default backend the same way it is validated for lazy backend.

## What Changes

- Add integration checks in `tests/integration/test_lerobot_pusht.py` for default backend media decoding:
  - at least one decoded media tensor from real `lerobot/pusht`
  - strict decoded tensor requirement: 3-D `(C, H, W)` and dtype in `{torch.float32, torch.uint8}`
- Add integration check for `create_dataloader()` with `backend="default"`:
  - batch includes required metadata keys
  - batch media includes at least one decoded tensor with batched shape `(B, C, H, W)` and dtype in `{torch.float32, torch.uint8}` when baseline decode is available
- Update `dataset-integration-testing` delta spec to encode these default-backend checks as normative requirements.
- Add a small dataset-layer caveat note clarifying default-backend integration parity checks and environment-dependent decode behavior.

## Capabilities

### New Capabilities

- _(none)_

### Modified Capabilities

- `dataset-integration-testing`: default backend acceptance now includes strict media decode and factory-default dataloader parity checks

## Impact

- **Modified tests**: `tests/integration/test_lerobot_pusht.py`
- **Modified docs**: `docs/dataset-layer/caveats.md`
- **New change specs**:
  - `openspec/changes/dataset-default-backend-checks/specs/dataset-integration-testing/spec.md`
- **Public API**: no constructor signature changes
- **Runtime behavior**: no production backend logic changes; test/doc/spec alignment only
