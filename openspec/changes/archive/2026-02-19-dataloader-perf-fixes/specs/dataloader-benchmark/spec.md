## ADDED Requirements

### Requirement: End-to-end throughput measurement
`scripts/bench_dataloader.py` SHALL measure samples/sec throughput for each dataset backend by timing steady-state DataLoader iteration.

#### Scenario: Default throughput measurement
- **WHEN** the script is run with `--repo-id <dataset>`
- **THEN** it SHALL create a DataLoader with `num_workers=0`, exclude warmup batches (default 10), and report median samples/sec with IQR

#### Scenario: Multi-worker throughput
- **WHEN** `--workers N` is specified with N > 0
- **THEN** the DataLoader SHALL use `num_workers=N`, discard warmup batches, and time only steady-state iteration

#### Scenario: Backend comparison
- **WHEN** `--backends lazy,streaming` is specified
- **THEN** throughput SHALL be measured independently for each listed backend and results reported side-by-side

### Requirement: Per-transform latency breakdown
The script SHALL measure per-transform latency by wrapping each transform's `__call__` with timing instrumentation.

#### Scenario: Transform timing with num_workers=0
- **WHEN** the script runs in default mode (num_workers=0)
- **THEN** each transform in the compose chain SHALL be individually timed using `time.perf_counter_ns`, and per-transform median latency SHALL be included in the output

### Requirement: Structured JSON output
The script SHALL output results as structured JSON suitable for CI comparison.

#### Scenario: JSON output format
- **WHEN** the script completes
- **THEN** it SHALL write JSON containing: samples_per_sec (median, IQR), per_transform_latency_ns (per transform name), hardware metadata (CPU, GPU if available, RAM), and script configuration

### Requirement: Synthetic dataset mode
The script SHALL support a `--synthetic` flag for CI runs without network access.

#### Scenario: Synthetic mode runs offline
- **WHEN** `--synthetic` is specified
- **THEN** the script SHALL generate random tensors matching a configurable schema (action dim, state dim, image shape) and measure transform + collation overhead only — NOT Parquet I/O

### Requirement: Optional torch.profiler integration
The script SHALL support an optional `--profile` flag for detailed trace output.

#### Scenario: Profile trace generation
- **WHEN** `--profile` is specified
- **THEN** the script SHALL run with `torch.profiler` enabled and write a Chrome-compatible trace file to the output directory
