## ADDED Requirements

### Requirement: Accelerate-backed training moves typed batches before policy execution
The Accelerate-backed training runtime SHALL accept `TrainingBatch` values yielded by the training dataloader and SHALL ensure their tensor fields are device-compatible with the prepared policy before invoking `policy(batch)`.

#### Scenario: Host batch reaches accelerator-backed policy without device mismatch
- **WHEN** the policy has been prepared onto an accelerator device and the training dataloader yields a typed `TrainingBatch` whose tensors originated on the host
- **THEN** the training runtime SHALL invoke the policy with an equivalent typed `TrainingBatch` whose tensor fields are on the accelerator-compatible device and SHALL NOT fail because batch tensors and model weights are on different devices

### Requirement: One-step training works with typed batches under Accelerate
The training runtime SHALL complete a real optimizer step with typed YAVLA batches under HuggingFace Accelerate without requiring the caller to manually move batch tensors.

#### Scenario: First optimizer step completes with typed batch consumption
- **WHEN** `Trainer` runs for one step with an Accelerate-prepared policy whose forward pass reads tensors from a typed `TrainingBatch`
- **THEN** the run SHALL complete forward, backward, gradient clipping, and optimizer step successfully without the caller manually moving the batch to `accelerator.device`
