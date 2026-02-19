## Why

The data transform pipeline has unnecessary overhead identified through cross-project analysis against openpi, LeRobot v3, and Octo. Specifically: (1) `NormalizeTransform` does a numpy→torch→numpy round-trip per key per sample that no other VLA project does, (2) the streaming backend converts Arrow batches to Python dicts row-by-row instead of batch-wise, (3) `NormalizeTransform` iterates all sample keys when most don't need normalization, and (4) there is no way to measure whether changes improve or regress data pipeline throughput. Fixing these before the training loop lands ensures the data layer scales cleanly.

## What Changes

- Eliminate the `_to_tensor` / `_restore_type` round-trip in `NormalizeTransform` and `UnnormalizeTransform` — stay in torch tensors throughout (the downstream model needs tensors anyway). This aligns with the design doc's intent and removes per-key per-sample type conversion overhead.
- Optimize `ShardInterleavedDataset._read_shard_rows` to process Arrow batches as columnar numpy/torch arrays instead of converting to Python dicts row-by-row via `to_pylist()`.
- Add default `normalize_keys` filtering so `NormalizeTransform` only iterates keys that actually have stats entries, instead of trying every key in the sample.
- Introduce a data pipeline benchmark script (`scripts/bench_dataloader.py`) that measures samples/sec throughput, per-transform latency breakdown, and per-backend comparison. Uses `torch.utils.benchmark.Timer` for stable measurements and optionally `torch.profiler` for detailed traces. This becomes the regression gate for future data layer changes.

## Capabilities

### New Capabilities

- `dataloader-benchmark`: A benchmark script and utilities for measuring data pipeline throughput (samples/sec), per-transform latency, and per-backend comparison across lazy, streaming, and default backends. Produces structured output suitable for CI regression tracking.

### Modified Capabilities

- `data-transforms`: `NormalizeTransform` and `UnnormalizeTransform` will output torch tensors instead of preserving the input type. Keys without matching stats entries will be skipped without attempting conversion. This is a behavioral change: callers that relied on getting numpy arrays back from normalization will now receive torch tensors.
- `streaming-dataset`: `_read_shard_rows` will yield samples with numpy arrays (from Arrow columnar conversion) instead of Python-native types from `to_pylist()`. The sample dict contract is preserved but leaf value types may change from Python lists/floats to numpy arrays.

## Impact

- **Modified files**: `src/yavla/data/transforms.py` (NormalizeTransform, UnnormalizeTransform), `src/yavla/data/streaming.py` (_read_shard_rows)
- **New files**: `scripts/bench_dataloader.py`
- **APIs**: `NormalizeTransform` / `UnnormalizeTransform` output type changes from "same as input" to "always torch.Tensor". Downstream consumers (model input pipeline, `UnnormalizeTransform` at inference) must accept tensors.
- **Tests**: Existing transform tests will need updates to expect torch tensor outputs instead of numpy/scalar outputs.
- **Dependencies**: No new dependencies. `torch.utils.benchmark` and `torch.profiler` are part of PyTorch.
