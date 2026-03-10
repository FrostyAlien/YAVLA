## 1. Trainer Instrumentation

- [x] 1.1 Add a small performance accumulator in `src/yavla/training/trainer.py` that tracks windowed optimizer-step wall time, trainer-visible data wait time, and processed sample counts between log events.
- [x] 1.2 Instrument `Trainer.run()` so batch-fetch wait is timed around iterator reads and optimizer-step wall time covers the full accumulated step.
- [x] 1.3 Extend the logged metrics payload to include `perf/step_time_s`, `perf/samples_per_sec`, `perf/data_wait_time_s`, `perf/compute_time_s`, and `perf/data_wait_fraction` on the existing `log_freq` cadence.

## 2. Validation

- [x] 2.1 Add trainer tests that verify the new `perf/*` metrics are emitted when W&B logging is enabled.
- [x] 2.2 Add trainer tests that verify optimizer-step semantics remain correct under `gradient_accumulation_steps > 1`.
- [x] 2.3 Add trainer tests that verify performance metrics are windowed on the existing log cadence rather than emitted every micro-batch.

## 3. Documentation

- [x] 3.1 Update the training guide W&B metrics section to document the new `perf/*` metrics and clarify that `data_wait_time` is trainer-visible wait rather than per-transform latency.
- [x] 3.2 Keep the standalone dataloader benchmark documentation unchanged except for any cross-reference needed to distinguish offline benchmarking from training-time performance logging.
