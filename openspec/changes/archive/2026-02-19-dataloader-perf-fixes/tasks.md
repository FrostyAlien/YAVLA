## Tasks

### Task 1: Remove `_restore_type` and smart key filtering (D1 + D2)
**Specs:** data-transforms — "NormalizeTransform outputs torch.Tensor always" + "Smart key filtering when keys=None"
**File:** `src/yavla/data/transforms.py`

**Steps:**
1. Delete the `_restore_type` function entirely
2. In both `NormalizeTransform.__call__` and `UnnormalizeTransform.__call__`:
   a. Change `target_keys` from `tuple(sample.keys())` to `[k for k in self.stats if k in sample]` when `self.keys is None`
   b. Remove `value_reference = output[key]` — use `output[key]` directly in `_to_tensor()`
   c. Replace `output[key] = _restore_type(value_reference, normalized/unnormalized)` with `output[key] = normalized/unnormalized`
3. Guard handling (SAFETY-CRITICAL):
   - When `keys is None`: the stats-intersection already filters, so `key not in self.stats` is redundant. Keep only `key not in output`.
   - When explicit `self.keys` provided: KEEP both `key not in output` and `key not in self.stats` guards.

**Verify:** `lsp_diagnostics` clean on transforms.py.

---

### Task 2: Per-column Arrow conversion in streaming shard reader (D3)
**Spec:** streaming-dataset — "Per-column Arrow batch conversion"
**File:** `src/yavla/data/streaming.py`

**Steps:**
1. Add `import pyarrow as pa` at top
2. Replace `_read_shard_rows` to extract columns first, then assemble rows by index:
   - For each column: use `col.to_numpy(zero_copy_only=False)` when `pa.types.is_primitive(col.type)` AND type is not string/binary
   - Otherwise: use `col.to_pylist()`
   - Assemble per-row dicts via index loop

**Verify:** `lsp_diagnostics` clean on streaming.py.

---

### Task 3: Benchmark script (D4)
**Spec:** dataloader-benchmark — all requirements
**File:** `scripts/bench_dataloader.py` (NEW)

**Steps:**
1. Create argparse CLI with: `--repo-id`, `--backends`, `--workers`, `--warmup-batches`, `--synthetic`, `--profile`, `--output`
2. Throughput measurement via `torch.utils.benchmark.Timer`
3. Per-transform timing with `time.perf_counter_ns` wrapper
4. Synthetic dataset mode (random tensors)
5. `--profile` flag with `torch.profiler` trace export
6. JSON output with hardware metadata (torch/pyarrow versions, CPU/GPU, num_workers, batch_size)
7. Warmup batches excluded from timing; Timer measures stable callable (not dataset construction)

**Verify:** Script runs with `--synthetic` flag without errors. JSON output is valid.

---

### Task 4: Update existing tests
**Spec:** data-transforms — all modified scenarios
**Files:** `tests/data/test_transforms.py`, `tests/integration/test_lerobot_pusht.py`

**Steps:**
1. `test_transforms.py`: Update `test_unnormalize_minmax_roundtrip_and_zero_range` — input numpy, expect `torch.Tensor` output with dtype `float32`. Use `torch.allclose` with `atol=1e-6` instead of `np.testing.assert_allclose`.
2. `test_lerobot_pusht.py`: Already uses `torch.as_tensor()` on output — no changes needed.

**Verify:** `pixi run -e dev test` passes.
