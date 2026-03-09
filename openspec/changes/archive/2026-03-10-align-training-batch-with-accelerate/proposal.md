## Why

The first real one-step smoke run through `scripts/train.py` exposed a gap between YAVLA's typed batch containers and HuggingFace Accelerate: the model is prepared onto the accelerator device, but nested tensors inside plain `TrainingBatch` / `ObservationBatch` dataclasses remain on CPU, causing a device mismatch before the first optimizer step. YAVLA is committed to keeping both typed batch boundaries and Accelerate, so the missing piece is an explicit runtime transport contract for typed batches rather than a move away from either design.

## What Changes

- Define runtime tensor-transport semantics for typed training batch containers so `TrainingBatch` and `ObservationBatch` can move across devices without losing their structured fields or non-tensor metadata.
- Align the Accelerate-backed training runtime around explicit typed-batch handling instead of assuming dataloader device placement works for plain dataclasses.
- Add regression coverage for the real `Trainer` + `Accelerate` + typed batch path so a one-step smoke training run reaches the first optimizer step on supported accelerator devices.
- Clarify trainer ownership of batch movement and related runtime handling so future backends do not depend on implicit framework heuristics for custom batch objects.

## Capabilities

### New Capabilities

- `training-batch-transport`: runtime contract for `ObservationBatch` and `TrainingBatch` as typed tensor containers, including device movement and related batch transport semantics needed by training.
- `accelerate-training-runtime`: requirements for the Accelerate-backed training runtime to consume typed YAVLA batches correctly and complete real training steps without device mismatch failures.

### Modified Capabilities

- _(none)_

## Impact

- **Modified code**: `src/yavla/models/types.py`, `src/yavla/training/trainer.py`, and any trainer/data helpers that currently rely on implicit framework batch placement.
- **Tests**: training/runtime coverage will need to exercise the actual `Trainer` + `Accelerate` path with typed batches, not only direct `policy.forward(...)` or trainer stubs that ignore batch contents.
- **Dependencies**: no new runtime dependency is expected; this change keeps HuggingFace Accelerate as the training foundation.
- **Contract**: typed batch containers stop being passive DTOs only and become runtime-aware objects at the training boundary, while preserving the typed model interface inside YAVLA.
