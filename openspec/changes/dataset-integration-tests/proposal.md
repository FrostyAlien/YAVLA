## Why

All existing dataset tests use synthetic data via `FakeMetadata` and monkeypatched constructors. This means we never exercise the real LeRobot metadata loading, actual Parquet I/O, or video decoding against production-shaped data. An integration test suite that fetches a real (small) HuggingFace dataset catches format drift, schema mismatches, and decoding regressions that unit tests structurally cannot.

## What Changes

- Add a `tests/integration/` directory with a `conftest.py` that provides a session-scoped `pusht_root` fixture, downloads/caches `lerobot/pusht` (v3.0, ~674KB parquet + ~50MB video) under `~/.cache/yavla-test-data/`, and skips integration tests with a clear reason if the first-time download fails.
- Add `tests/integration/test_lerobot_pusht.py` with ~12 integration tests covering all three dataset backends (lazy, streaming, default/LeRobotDataset), the factory path, transforms pipeline, episode filtering, delta_timestamps, action_chunk_size, batched access, and video frame decoding.
- Mark all tests in `tests/integration/` with `@pytest.mark.integration` via collection-time hook in `conftest.py` so marker behavior is centralized and reliable.
- Update `pyproject.toml` to register the `integration` marker and exclude it from default `pytest` runs via `-m "not integration"`.

## Capabilities

### New Capabilities
- `dataset-integration-testing`: Integration test infrastructure and test cases that validate all three dataset backends against real HuggingFace LeRobot v3.0 data.

### Modified Capabilities
_(none — no existing spec requirements change)_

## Impact

- **New files**: `tests/integration/conftest.py`, `tests/integration/test_lerobot_pusht.py`
- **Modified files**: `pyproject.toml` (marker registration + default marker filter)
- **Dependencies**: No new runtime dependencies. Tests rely on existing `lerobot`, `datasets`, `pyarrow`, `torch` packages already in the project.
- **CI**: Integration tests are opt-in via `-m integration` and require network access on first run. Subsequent runs use the local cache.
