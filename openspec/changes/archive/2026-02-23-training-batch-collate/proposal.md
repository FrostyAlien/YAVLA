## Why

The data layer (`create_dataloader`) produces raw `dict[str, Tensor]` batches (LeRobot convention), but `VLAPolicy.forward()` expects typed `TrainingBatch` dataclass instances. Without a bridge, `scripts/train.py` crashes immediately — `batch.observations` fails on a dict. Every VLA framework (OpenVLA, LeRobot, Octo) solves this; YAVLA currently does not.

## What Changes

- Add a `training_collate_fn` that converts `list[dict]` samples into a `TrainingBatch` (with nested `ObservationBatch`), using convention-based key mapping from LeRobot dict keys
- Wire the collate into `create_training_dataloader()` via the existing `collate_fn` parameter, with explicit `dt_hz` and `chunk_len` keyword arguments sourced by the caller from `PolicyConfig` / action head config

## Capabilities

### New Capabilities
- `training-batch-collate`: Collate function that converts raw LeRobot dict batches into typed `TrainingBatch` / `ObservationBatch` dataclass instances for policy consumption

### Modified Capabilities
- `dataset-factory`: `create_dataloader()` gains the ability to accept and pass through a `collate_fn`, which `create_training_dataloader()` will use

## Impact

- **Code**: `src/yavla/training/data.py` (new collate function + wiring), `scripts/train.py` (pass `dt_hz`/`chunk_len` at call site)
- **APIs**: `create_training_dataloader()` now returns a DataLoader yielding `TrainingBatch` instead of raw dicts
- **Unblocks**: `scripts/train.py` end-to-end execution, training integration tests
- **No breaking changes**: `create_dataloader()` already accepts `collate_fn=None`; default behavior unchanged
