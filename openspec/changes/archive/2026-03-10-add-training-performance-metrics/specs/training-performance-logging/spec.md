## ADDED Requirements

### Requirement: Trainer logs optimizer-step performance metrics
The Accelerate-backed trainer SHALL log training-time performance metrics under the `perf/` namespace whenever it emits training metrics. The logged metric set SHALL include:

- `perf/step_time_s`
- `perf/samples_per_sec`
- `perf/data_wait_time_s`
- `perf/compute_time_s`
- `perf/data_wait_fraction`

These metrics SHALL be derived from optimizer steps, not individual micro-batches.

#### Scenario: Log performance metrics on a normal training step
- **WHEN** a training run reaches a logging boundary with `config.wandb=True`
- **THEN** `accelerator.log(...)` SHALL include all required `perf/*` metrics alongside the existing training metrics

#### Scenario: Gradient accumulation uses optimizer-step semantics
- **WHEN** `gradient_accumulation_steps > 1`
- **THEN** the logged `perf/step_time_s` and `perf/samples_per_sec` values SHALL describe the full optimizer step spanning all accumulated micro-batches

### Requirement: Performance metrics use trainer-visible timing semantics
The trainer SHALL measure `perf/data_wait_time_s` as wall-clock time spent waiting for the training iterator to yield the next micro-batch. It SHALL measure `perf/step_time_s` as end-to-end optimizer-step wall time observed by the trainer. `perf/compute_time_s` SHALL be the non-wait remainder of step time, and `perf/data_wait_fraction` SHALL equal data wait time divided by step time.

#### Scenario: Data wait time excludes non-fetch work
- **WHEN** the trainer spends part of an optimizer step blocked on `next(data_iter)` and the rest on forward, backward, clipping, stepping, and scheduler work
- **THEN** `perf/data_wait_time_s` SHALL reflect only the fetch wait portion and `perf/compute_time_s` SHALL reflect the remaining portion

#### Scenario: Throughput derives from effective batch progress
- **WHEN** the trainer logs a performance window
- **THEN** `perf/samples_per_sec` SHALL be derived from the effective global samples processed during that window divided by the corresponding window step time

### Requirement: Performance logging stays minimal and aligned with existing cadence
The trainer SHALL emit performance metrics on the existing `config.log_freq` cadence and SHALL aggregate them over the optimizer steps since the previous log event. This capability SHALL NOT require per-transform latency instrumentation or new training configuration fields.

#### Scenario: Log frequency controls performance metric cadence
- **WHEN** `config.log_freq=100`
- **THEN** performance metrics SHALL be emitted only at steps 100, 200, 300, and so on, using data accumulated since the prior performance log

#### Scenario: Per-transform latency remains out of scope
- **WHEN** performance metrics are added to the trainer
- **THEN** the training loop SHALL NOT instrument individual dataset transforms or worker internals for W&B logging
