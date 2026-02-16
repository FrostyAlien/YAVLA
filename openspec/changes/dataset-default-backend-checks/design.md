## Context

`tests/integration/test_lerobot_pusht.py` currently enforces strict media decode requirements for `LazyLeRobotDataset` and `ShardInterleavedDataset`, but default backend coverage is weaker:

- `test_default_backend_loads` checks only load/schema
- `test_factory_creates_dataloader` validates only `backend="lazy"`

This creates an acceptance asymmetry around the baseline backend that lazy/streaming decode tests depend on.

## Goals / Non-Goals

**Goals:**
- Enforce strict decoded-media assertions for direct default backend integration.
- Add real-data integration coverage for factory default backend dataloader path.
- Encode the new checks in `dataset-integration-testing` spec deltas.
- Document baseline-gated decode behavior in dataset-layer caveats.

**Non-Goals:**
- No changes to `LazyLeRobotDataset` or `ShardInterleavedDataset` runtime code.
- No changes to `create_dataloader()` behavior.
- No OOM/performance tuning changes.

## Decisions

1. **Default backend decode check is strict when camera keys exist**
- Require at least one decoded 3-D media tensor in first bounded sample window.
- Use the same tensor constraints as lazy/streaming checks (`ndim == 3`, dtype `{float32,uint8}`).

2. **Factory default check validates both metadata and media outputs**
- Confirm default backend selection metadata (`yavla_backend`) and required batch keys.
- Confirm at least one decoded media tensor exists in batch with shape `(B, C, H, W)` and dtype `{float32,uint8}` when baseline decode is available.

3. **Environment-dependent decode remains skip-gated**
- If upstream default decode does not produce decoded tensors in the bounded scan, tests requiring decode assertions skip with clear reason.
- This mirrors existing lazy/streaming baseline gate behavior and avoids false negatives from codec/environment issues.

4. **Keep change additive and narrowly scoped**
- New follow-up change only updates integration tests, integration-testing spec delta, and a targeted caveat note.

## Risks / Trade-offs

- **[Risk] Additional decode checks can increase integration runtime**
  - **Mitigation:** use bounded scans (max 256 samples) and reuse existing helper patterns.
- **[Risk] Skip logic could hide regressions if overused**
  - **Mitigation:** skip only on explicit “baseline decode unavailable” condition; fail on raw-only payloads or missing media evidence where appropriate.
- **[Risk] Requirement drift if specs are not updated with tests**
  - **Mitigation:** include `dataset-integration-testing` delta in this same change.
