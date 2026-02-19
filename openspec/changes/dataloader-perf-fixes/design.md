## Context

The YAVLA data pipeline (transforms.py, streaming.py) was built following openpi's pattern: per-sample transforms in `__getitem__`, composed via a `DataTransformFn` protocol. Cross-project analysis against openpi, LeRobot v3, Octo, and InternVLA-A1 confirmed this architecture is sound. However, three implementation-level inefficiencies were identified:

1. `NormalizeTransform` converts every value to torch, normalizes, then converts back to the original type (`_to_tensor` → normalize → `_restore_type`). No other VLA project does this round-trip.
2. `ShardInterleavedDataset._read_shard_rows` calls `batch.to_pylist()` which converts Arrow batches to Python dicts row-by-row — losing Arrow's columnar efficiency.
3. When `normalize_keys=None`, `NormalizeTransform` iterates every key in the sample dict, including metadata keys (episode_index, timestamp, etc.) that will never have stats.

Additionally, there is no benchmark infrastructure to measure whether changes improve or regress data pipeline throughput.

## Goals / Non-Goals

**Goals:**
- Eliminate the `_to_tensor`/`_restore_type` round-trip — output torch tensors directly
- Optimize Arrow→dict conversion in streaming backend to use columnar numpy conversion
- Add smart key filtering to `NormalizeTransform` to skip keys without stats
- Provide a benchmark script that measures samples/sec and per-transform latency
- Maintain backward compatibility for the `DataTransformFn` protocol contract (`dict[str, Any] → dict[str, Any]`)

**Non-Goals:**
- Moving normalization to GPU/batched post-DataLoader (LeRobot v3 pattern) — our architecture intentionally matches openpi's per-sample CPU pattern, and the overhead for small action/state vectors is negligible
- Changing the `DataTransformFn` protocol signature
- Optimizing video decoding or Parquet I/O (separate concerns)
- Adding `auto` backend selection (deferred per prior decision)

## Decisions

### D1: NormalizeTransform and UnnormalizeTransform output torch.Tensor always

**Choice:** Remove `_restore_type`. After normalization, the value stays as `torch.Tensor`. Remove the `_restore_type` function entirely. Keep `_to_tensor` for input conversion but do not convert back.

**Why:** The downstream consumer is always a PyTorch model that needs tensors. The current round-trip (numpy→torch→numpy or float→torch→float) is pure waste. openpi avoids this by staying in numpy throughout; we stay in torch throughout — same principle, different framework.

**Why not stay in numpy (like openpi):** openpi uses JAX, so numpy is the natural intermediate. YAVLA uses PyTorch end-to-end. Staying in torch avoids a second conversion at collation/model input time.

**Impact:** Callers that previously received numpy arrays from normalized keys will now receive `torch.Tensor`. The `dict[str, Any]` contract is preserved (tensors are valid `Any`). Existing tests that assert numpy output types will need updating.

### D2: Smart key filtering via stats-intersection default

**Choice:** When `NormalizeTransform.keys` is `None`, iterate the intersection of stats keys and sample keys instead of `sample.keys()`. Implementation uses order-preserving filtering: `[k for k in self.stats if k in sample]` — iterating stats keys (smaller set, deterministic order from metadata) and checking membership in the sample dict.

**Why:** A typical sample has ~10-15 keys (episode_index, index, timestamp, frame_index, task_index, task, action, observation.state, image keys...). Only 2-3 have stats entries (action, observation.state). Iterating all keys wastes cycles on the `key not in self.stats` guard for ~80% of keys.

**Why not require explicit keys always:** That would be a breaking API change. The intersection approach is backward-compatible — same observable behavior, fewer iterations.

**Key representation:** Keys are flat dot-separated strings (e.g., `"observation.state"`, `"action"`), matching the LeRobot v3 metadata format. Both `stats` and `sample` use the same flat key namespace — no nested dict traversal needed.

### D3: Streaming shard reader uses per-column Arrow conversion

**Choice:** Replace `batch.to_pylist()` (which creates one Python dict per row, converting each cell individually) with per-column conversion: for each column in the batch, convert the entire column array at once, then zip into per-row dicts.

Conversion strategy per column type:
- **Primitive numeric columns** (int, float): `column.to_numpy(zero_copy_only=False)` → numpy array, then index per row. Zero-copy when possible, single-copy otherwise.
- **List/array columns** (e.g., action vectors stored as Arrow list arrays): `column.to_pylist()` per column — still Python objects, but amortized across the batch rather than per-cell.
- **String columns** (e.g., task): `column.to_pylist()` per column — numpy object arrays for strings offer no advantage.

Per-row dicts are then assembled by indexing into these column arrays: `{col: arrays[col][i] for col in columns}` for each row `i`.

**Why:** `to_pylist()` on the full batch creates N×M Python objects (N rows × M columns) through individual cell conversion. Per-column conversion amortizes the conversion overhead: numeric columns get zero-copy numpy arrays (indexing is pointer arithmetic), and even list/string columns benefit from batch-level `to_pylist()` on a single column array vs. cell-by-cell extraction.

**Why not yield columnar batches directly:** The shuffle buffer and transform pipeline operate on individual samples (`dict[str, Any]`). Changing to batch-level processing would require rewriting the shuffle buffer, transforms, and the `DataTransformFn` protocol — far beyond scope.

**Leaf value types after conversion:** Numeric metadata columns (episode_index, frame_index, etc.) will produce numpy scalars (`np.int64`, `np.float64`) instead of Python `int`/`float`. The existing `_to_tensor` helper and schema validation already handle `np.integer`/`np.floating` types. Action/state vectors stored as Arrow list arrays will remain Python lists (same as `to_pylist()` today).

### D4: Benchmark script using torch.utils.benchmark.Timer

**Choice:** A standalone script `scripts/bench_dataloader.py` that:
1. Creates a DataLoader for each backend (default, lazy, streaming) with a configurable dataset
2. Measures end-to-end throughput (samples/sec) using `torch.utils.benchmark.Timer` for stable, statistically sound measurements
3. Measures per-transform latency with `num_workers=0` so timing runs in the main process
4. Outputs structured results (JSON) suitable for CI comparison

**Benchmark methodology:**
- **Warmup:** First N batches (configurable, default 10) are excluded from timing to let workers initialize and caches warm.
- **Default mode:** `num_workers=0` for reproducible, low-noise measurements. Optional `--workers N` flag for multi-worker throughput testing.
- **Multi-worker runs:** Create the DataLoader once, discard warmup batches, then time steady-state iteration only. This avoids measuring worker spawn/teardown noise.
- **Per-transform timing:** Wraps each transform's `__call__` with `time.perf_counter_ns` instrumentation in the main process (`num_workers=0`). This captures actual transform cost without worker IPC overhead.

**Why `torch.utils.benchmark.Timer` over raw `time.perf_counter`:** Timer handles warmup, multiple runs, statistical aggregation (median, IQR), and avoids common benchmarking pitfalls (JIT warmup, GC interference). It's the PyTorch-recommended approach.

**Why not `torch.profiler`:** The profiler is complementary (for deep-dive traces) but too heavyweight for a quick regression check. The script will support an optional `--profile` flag that enables `torch.profiler` output for detailed investigation, but the default mode uses Timer for fast, comparable numbers.

**Why a script, not a pytest benchmark:** pytest-benchmark adds a dependency and is designed for micro-benchmarks. DataLoader throughput measurement needs control over warmup batches, worker spawning, and epoch boundaries that don't fit the pytest fixture model cleanly.

**`--synthetic` scope:** Synthetic mode generates random tensors matching a configurable schema (action dim, state dim, image shape). It exercises the transform pipeline and collation path but does NOT exercise the Parquet I/O or Arrow conversion path. This is intentional — synthetic mode measures compute overhead; real dataset mode measures end-to-end including I/O.

## Risks / Trade-offs

**[Output type change breaks downstream code]** → The `NormalizeTransform` output type change from "preserves input type" to "always torch.Tensor" is a behavioral change. Mitigation: the only known downstream consumers are the model input pipeline (which needs tensors) and `UnnormalizeTransform` (which will also be updated). Tests will be updated. The `dict[str, Any]` protocol contract is preserved.

**[Streaming optimization changes leaf value types]** → Per-column conversion produces numpy scalars (`np.int64`, `np.float64`) for metadata columns instead of Python `int`/`float`. Mitigation: (1) `_to_tensor` already handles `np.integer`/`np.floating`, (2) schema validation uses `isinstance(v, (int, np.integer))` checks, (3) PyTorch's default collate handles numpy scalars natively. Add a test verifying metadata column types are acceptable through the full pipeline.

**[Benchmark results vary across hardware]** → Absolute numbers aren't comparable across machines. Mitigation: the benchmark outputs relative comparisons (before/after on same hardware) and includes hardware metadata in JSON output. CI can track relative regressions, not absolute thresholds.

**[Benchmark requires a real dataset]** → Can't run in CI without downloading data. Mitigation: `--synthetic` flag generates random tensors matching a configurable schema. Note: synthetic mode exercises transforms and collation only — not the Parquet/Arrow I/O path (see D4).

## Acceptance Criteria

- **D1:** `NormalizeTransform` and `UnnormalizeTransform` output `torch.Tensor` for all normalized keys; non-normalized keys pass through unchanged. `_restore_type` is removed. Roundtrip test (normalize → unnormalize) passes within float tolerance.
- **D2:** When `keys=None`, only keys present in both `stats` and `sample` are iterated. Iteration order is deterministic (stats-key order). Existing behavior unchanged when `keys` is explicitly provided.
- **D3:** `_read_shard_rows` yields dicts with the same key set as before. Numeric metadata values are numpy scalars; list-type values (action vectors) remain Python lists. Full pipeline test (streaming → transforms → collate) produces valid batches.
- **D4:** Benchmark script runs with `--synthetic` flag without network access. JSON output includes samples/sec, per-transform latency breakdown, and hardware metadata. `--profile` flag produces a `torch.profiler` trace file.
