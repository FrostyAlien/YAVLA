## 1. Define Typed Batch Transport

- [x] 1.1 Add a tensor-transport helper to the typed training batch classes so `ObservationBatch` and `TrainingBatch` can return same-type copies with tensor leaves transformed while preserving non-tensor metadata.
- [x] 1.2 Expose the supported public transport operation for typed training batches and cover nested image tensors, proprio, actions, and optional mask fields in unit tests.

## 2. Align Trainer Runtime With Accelerate

- [x] 2.1 Update the training runtime so the trainer explicitly moves typed batches onto the accelerator-compatible device before invoking `policy(batch)`.
- [x] 2.2 Keep the Accelerate-based training loop behavior intact for forward, backward, clipping, optimizer step, logging, and checkpointing while removing dependence on implicit dataloader placement for custom dataclasses.

## 3. Add Regression Coverage

- [x] 3.1 Strengthen trainer/runtime tests so at least one Accelerate-backed path uses a policy that actually reads tensors from a typed `TrainingBatch`.
- [x] 3.2 Add regression coverage for the original failure mode, asserting that a one-step trainer run with typed batches completes without device mismatch errors.

## 4. Validate End-to-End Smoke Path

- [x] 4.1 Re-run the targeted training tests and any relevant model/trainer tests covering typed batch transport and the Accelerate runtime seam.
- [x] 4.2 Re-run the real one-step smoke training path to confirm the first optimizer step succeeds with typed batches under Accelerate.
