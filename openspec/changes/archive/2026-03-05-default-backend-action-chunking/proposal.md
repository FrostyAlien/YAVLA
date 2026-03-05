## Why

YAVLA’s training pipeline requires chunked action targets (`[B, chunk_len, action_dim]`), but today the `default` backend (`LeRobotDataset`) rejects `action_chunk_size`, forcing users onto the `lazy` backend even when their dataset is small enough for default loading. This blocks the first real end-to-end training run on datasets like `lerobot/aloha_sim` unless the user changes backends for an otherwise “default” setup.

## What Changes

- Enable `action_chunk_size` when `backend="default"` by deriving action sequences via the underlying LeRobot `delta_timestamps` mechanism (future actions at fixed step offsets).
- Ensure `default` and `lazy` backends produce consistent `action` and `action_is_pad` shapes/semantics when `action_chunk_size` is configured.
- Make `action_chunk_size` an explicit convenience alias for contiguous future-action `delta_timestamps["action"]`. If both are provided, raise an actionable error; for custom/non-contiguous action deltas, use `delta_timestamps["action"]` and leave `action_chunk_size` unset.
- Update factory guardrails and error messages so only truly unsupported combinations are rejected (streaming remains incompatible with temporal features).
- Add/extend integration tests to cover action chunking on the `default` backend with real data.
- Update documentation and example configs so users can run “default backend + action_chunk_size” without surprises.

## Capabilities

### New Capabilities

<!-- None -->

### Modified Capabilities

- `dataset-factory`: Default backend SHALL support `action_chunk_size` (no longer raises `ValueError`) and MUST assemble chunked actions through LeRobot temporal queries.
- `dataset-integration-testing`: Integration tests SHALL cover `action_chunk_size` behavior for the default backend, including presence and polarity of `action_is_pad`.

## Impact

- Data layer: `src/yavla/data/factory.py` (backend selection/validation, `delta_timestamps` composition for action chunking).
- Tests: `tests/integration/` (real-data assertions for default backend action chunking).
- Docs/config: dataset backend guidance and training config examples (`docs/`, `configs/`).
