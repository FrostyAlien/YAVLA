## Why

YAVLA currently presents three dataset backends, but the picture is more uneven than that surface suggests:

- `backend="default"` already delegates to upstream `LeRobotDataset`, so this path is effectively LeRobot-native today.
- The pinned LeRobot v3 revision already includes `StreamingLeRobotDataset`, which overlaps with YAVLA's custom `ShardInterleavedDataset` more than our current proposal wording acknowledges.
- `LazyLeRobotDataset` still provides the clearest YAVLA-specific value: map-style random access, sampler-driven uniform shuffling, and tighter control over decoder/cache behavior.

That means the real simplification opportunity is not "migrate all three backends", but to stop treating `default` as a custom path and decide whether YAVLA should keep owning a separate streaming implementation.

Because YAVLA is not currently using the streaming backend in training configs, we can replace or remove the custom streaming path with relatively low migration risk if we explicitly accept the contract change.

## What Changes

- Stop framing `backend="default"` as a migration target and document it as the already-aligned LeRobot-native baseline.
- Replace YAVLA's `backend="streaming"` implementation with upstream `StreamingLeRobotDataset`, or remove the YAVLA-owned `ShardInterleavedDataset` surface entirely if the factory no longer needs to expose a separate custom streaming backend.
- Keep `LazyLeRobotDataset` as the custom backend for capabilities that still require YAVLA-owned map-style random access, sampler-compatible uniform shuffling, or decoder/cache behavior that upstream streaming does not provide.
- Update `create_dataloader()` backend selection, guardrails, and logging so the supported backend matrix reflects this narrower custom surface.
- Update specs, tests, and dataset-layer docs to reflect that the main contract shift is the streaming backend, not the default backend.
- **BREAKING**: `backend="streaming"` will adopt upstream LeRobot streaming semantics if retained. This removes the current YAVLA-specific contract around temporal-feature rejection and may also change guarantees around decoder choice and worker/rank behavior. Because YAVLA is not currently using the streaming backend, the expected migration risk is low.

## Capabilities

### New Capabilities
- _(none)_

### Modified Capabilities
- `dataset-factory`: backend selection and public backend guarantees will be updated to clarify that `default` is already LeRobot-native, while `streaming` is replaced, deprecated, or removed as a YAVLA-owned implementation.
- `streaming-dataset`: streaming behavior will be redefined around upstream `StreamingLeRobotDataset` semantics, or the custom streaming capability will be retired from YAVLA's public surface.
- `dataset-integration-testing`: integration expectations will shift so upstream LeRobot backends are the primary compatibility target, with YAVLA-specific coverage focused on the retained lazy backend and factory wiring.

## Impact

- **Modified code**: `src/yavla/data/factory.py`, `src/yavla/data/streaming.py`, related dataset tests/specs, and dataset-layer documentation.
- **Potential removals**: `ShardInterleavedDataset` implementation and its direct factory exposure.
- **Retained custom surface**: `LazyLeRobotDataset` and any transform/collate logic that still provides YAVLA-specific map-style access, sampling, or decoder/cache control.
- **Dependencies**: stronger reliance on pinned LeRobot v3 dataset APIs, including upstream streaming interfaces; no new runtime dependency is expected.
- **Operational risk**: limited expected migration cost because current configs and training flows do not rely on `backend="streaming"`.
