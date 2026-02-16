## Context

All existing dataset tests in `tests/data/` use `FakeMetadata` with monkeypatched `LeRobotDatasetMetadata` constructors. This validates internal logic but never exercises real HuggingFace Hub metadata resolution, actual Parquet row-group I/O, or video frame decoding against production data. The dataset layer has three backends (`default`, `lazy`, `streaming`) plus a factory and transform pipeline — all untested against real data.

`lerobot/pusht` is a LeRobot v3.0 dataset on HuggingFace Hub with 206 episodes, 25,650 frames at 10fps, 96×96 video, and ~674KB of Parquet data. It's small enough for local caching and fast iteration while still exercising the full data path including video decoding.

The existing `pyproject.toml` has `[tool.pytest.ini_options]` with `testpaths = ["tests"]` and `addopts = "-v --tb=short"` but no custom markers. The pixi dev task runs `pytest tests/ -v`.

## Goals / Non-Goals

**Goals:**
- Validate all three dataset backends against real `lerobot/pusht` data without monkeypatching
- Verify delta_timestamps, action_chunk_size, episode filtering, video decoding, transforms, factory, and batched access work end-to-end
- Keep integration tests isolated from the default `pytest` run via marker-based opt-in
- Cache downloaded data across sessions to avoid repeated network calls

**Non-Goals:**
- Testing against multiple datasets or non-v3.0 formats (future work)
- Logging/visualization infrastructure for human review (backlog)
- CI pipeline integration (tests are opt-in; CI config is out of scope)
- Performance benchmarking or profiling
- Testing distributed (multi-GPU) code paths

## Decisions

### 1. Single dataset: `lerobot/pusht`

Use only `lerobot/pusht` for all integration tests. It's v3.0, has video + tabular + task metadata, and is small (~674KB parquet, ~50MB video).

**Alternative considered:** Multiple small datasets (e.g., `lerobot/aloha_sim_insertion_human`) to cover more shapes. Rejected because the user explicitly requested minimal disk usage and fewer datasets. One dataset that exercises all code paths is sufficient for now.

### 2. Download via `LeRobotDataset`

Use lerobot's own `LeRobotDataset(repo_id="lerobot/pusht", root=<cache_root>)` to trigger download and materialize the canonical on-disk layout. The fixture then returns the resolved dataset root path for all tests. This reuses lerobot's existing download/cache logic rather than reimplementing HuggingFace Hub API calls.

**Alternative considered:** Direct `huggingface_hub.snapshot_download()`. Rejected because `LeRobotDataset` already handles the full download including metadata, parquet shards, and video files in the correct directory layout. Using it avoids duplicating path resolution logic.

### 3. Cache location: `~/.cache/yavla-test-data/`

Store downloaded data under a YAVLA-specific cache root rather than lerobot's default `~/.cache/lerobot/`. The fixture is responsible for resolving the concrete dataset root path inside this cache root and returning it as `pusht_root`.

**Alternative considered:** Use lerobot's default cache (`~/.cache/lerobot/`). Rejected because coupling test infrastructure to lerobot's internal cache layout is fragile — if lerobot changes its cache structure, tests break for unrelated reasons.

### 4. Session-scoped fixture with existence check and skip-on-failure behavior

The download fixture is `@pytest.fixture(scope="session")`. It checks whether the cache already contains a usable dataset root (metadata + parquet layout). If present, it returns the cached path immediately. If absent, it downloads via `LeRobotDataset`. If download fails (network/rate limit), integration tests are skipped with a clear reason instead of hard-failing unrelated local test runs.

**Alternative considered:** `autouse=True` module-scoped fixture. Rejected because session scope is more efficient (download once per `pytest` invocation) and `autouse` would make the fixture implicit rather than explicit in test signatures.

### 5. Marker-based test isolation

Register `integration` as a custom pytest marker in `pyproject.toml`. Add `addopts = "-m 'not integration'"` to the existing pytest config so the default `pytest` run skips integration tests. Developers opt in with `pytest -m integration`.

Apply markers centrally in `tests/integration/conftest.py` using a `pytest_collection_modifyitems` hook for items under `tests/integration/`, rather than relying on `pytestmark` in `conftest.py`.

**Alternative considered:** Separate `tests/integration/` directory excluded via `--ignore`. Rejected because marker-based filtering is more flexible (can combine with other markers, run subsets) and is the standard pytest pattern. The tests still live in `tests/integration/` for organizational clarity, but the marker is what controls execution.

### 6. Episode filtering for speed, with factory-path constraints

Most direct `LazyLeRobotDataset` tests construct datasets with `episodes=[0]` to limit I/O to a single episode. This keeps individual tests fast while still exercising the real data path. Full-dataset tests (e.g., `len(dataset)` check) use complete metadata but access only minimal samples.

The factory-path test does not use episode filtering because `DataConfig` currently has no `episodes` field. For that test, use `num_workers=0` and assert one collated batch has required schema keys and expected batch dimension.

### 7. Column pruning for non-video tests

For tests that are not explicitly about video decoding, pass `feature_columns` that exclude video keys. This avoids unnecessary MP4 decode overhead and keeps integration runtime stable.

### 8. Video backend: `pyav` only

All video decoding tests use the `pyav` backend. `torchcodec` is an optional backend that requires additional system dependencies and is not guaranteed to be available in all dev environments.

**Alternative considered:** Test both `pyav` and `torchcodec` via parametrize. Rejected because `torchcodec` availability is environment-dependent and the integration tests should be reliably runnable. `pyav` is the default and always available.

### 9. Test file structure

```
tests/
  integration/
    __init__.py
    conftest.py          # session-scoped download fixture, marker auto-application
    test_lerobot_pusht.py  # all integration tests for lerobot/pusht
```

`conftest.py` applies `@pytest.mark.integration` to all tests in the directory via `pytest_collection_modifyitems`, so individual test functions don't need the decorator.

## Risks / Trade-offs

**Network dependency on first run** → The first test run requires internet access to download from HuggingFace Hub. Mitigation: session-scoped caching means this is a one-time cost; first-run failures skip integration tests with a clear reason.

**HuggingFace Hub availability** → If HuggingFace is down or rate-limits, first-run download is unavailable. Mitigation: explicit skip behavior plus cache persistence across runs.

**Dataset format drift** → If `lerobot/pusht` is updated on HuggingFace (e.g., re-encoded, schema change), cached data becomes stale. Mitigation: the cache uses a fixed snapshot. A future enhancement could pin to a specific dataset revision/commit.

**Disk usage (~50MB)** → Video files for `lerobot/pusht` are ~50MB. Acceptable for a dev machine cache. The parquet data is negligible (~674KB).

**Video decoding speed** → Video frame decoding is the slowest part of the test suite. Mitigation: non-video tests prune video columns; dedicated video tests decode only 1-2 frames.
