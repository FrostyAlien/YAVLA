## Why

The original `dataset-integration-tests` change introduced runtime compatibility behavior in `lazy` and `streaming` backends while fixing failing real-data tests. That behavior is currently implemented and validated, but not yet reflected in formal OpenSpec requirements and targeted docs.

## What Changes

- Add follow-up spec deltas for `lazy-dataset` and `streaming-dataset` to codify:
  - container-agnostic episode metadata ingestion (`datasets.Dataset`, pandas DataFrame, list-like records)
  - dual media decode source resolution (row payload and canonical LeRobot v3 episode metadata)
  - timestamp fallback and `from_timestamp` shift behavior
- Tighten integration media assertions in `tests/integration/test_lerobot_pusht.py` back to strict spec-level checks:
  - decoded media must be a 3-D tensor `(C, H, W)`
  - dtype must be `torch.float32` or `torch.uint8`
  - keep baseline gate/skip behavior when upstream `LeRobotDataset` cannot decode in the current environment
- Add a focused unit regression test for HF `datasets.Dataset` metadata record normalization to prevent reintroducing `to_dict(orient=...)` coupling.
- Update targeted dataset-layer documentation in `docs/dataset-layer/caveats.md` (and a small index pointer in `docs/dataset-layer/README.md`) to document media source resolution and comparison with default LeRobot behavior.

## Capabilities

### New Capabilities
- _(none)_

### Modified Capabilities
- `lazy-dataset`: specify metadata container compatibility and dual-path media resolution semantics without constructor/API changes
- `streaming-dataset`: specify metadata container compatibility and dual-path media resolution semantics without constructor/API changes

## Impact

- **Modified code/tests**: `tests/integration/test_lerobot_pusht.py`, `tests/data/test_lazy_dataset.py`
- **Modified docs**: `docs/dataset-layer/caveats.md`, `docs/dataset-layer/README.md`
- **New change specs**:
  - `openspec/changes/dataset-media-compat-and-video-assertions/specs/lazy-dataset/spec.md`
  - `openspec/changes/dataset-media-compat-and-video-assertions/specs/streaming-dataset/spec.md`
- **Public API**: no constructor signature changes for `LazyLeRobotDataset` or `ShardInterleavedDataset`
- **Memory behavior**: no new eager frame-level parquet loading during dataset initialization
