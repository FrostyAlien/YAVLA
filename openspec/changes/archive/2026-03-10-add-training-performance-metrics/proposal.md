## Why

YAVLA's current W&B integration logs optimization state such as loss, learning rate, gradient norm, and sample progress, but it does not expose training-time performance signals. This makes it difficult to tell whether a run is bottlenecked by model compute or input delivery, and slows down iteration on dataset, dataloader, and distributed-training settings.

## What Changes

- Add training-time performance metric logging for the Accelerate-backed trainer.
- Log end-to-end optimizer-step wall time and derived throughput in W&B.
- Log trainer-visible dataloader wait time so input-pipeline stalls can be distinguished from model compute time.
- Document the new metrics in the training guide and keep the scope intentionally small; do not add per-transform latency or offline profiling features to the training loop.

## Capabilities

### New Capabilities
- `training-performance-logging`: Log concise training-time performance metrics from the trainer, including step time, samples per second, dataloader wait time, compute time, and data-wait fraction.

### Modified Capabilities
- None.

## Impact

- Affected code: `src/yavla/training/trainer.py`, trainer tests, and training documentation.
- Affected systems: W&B experiment tracking and console-visible training observability.
- No new runtime dependencies or config surface are required if the metrics follow the existing `log_freq` cadence.
