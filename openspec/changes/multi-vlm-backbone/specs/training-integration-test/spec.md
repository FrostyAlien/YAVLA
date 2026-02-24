## ADDED Requirements

### Requirement: End-to-end training forward-backward test
An integration test SHALL exercise the full training path: synthetic data → `TrainingCollate` → `build_policy()` → `policy.forward()` → `LossDict` → `backward()` → optimizer step. The test MUST use a real VLM backbone (not mocks/stubs).

#### Scenario: Forward pass produces finite loss
- **WHEN** a policy built with `build_policy()` receives a synthetic `TrainingBatch` with random images, proprio, language, and actions
- **THEN** `policy.forward(batch)` returns a `LossDict` with finite `total` loss (not NaN, not Inf)

#### Scenario: Backward pass updates parameters
- **WHEN** `loss.total.backward()` is called followed by `optimizer.step()`
- **THEN** at least one trainable parameter has changed from its initial value

#### Scenario: Correct tensor shapes flow through pipeline
- **WHEN** synthetic data has images `[B, 3, 224, 224]`, proprio `[B, D_proprio]`, actions `[B, chunk_len, action_dim]`
- **THEN** the pipeline completes without shape mismatch errors at any module boundary

### Requirement: Integration test uses pytest integration marker
The training integration test SHALL be marked with `@pytest.mark.integration` and SHALL be excluded from the default test run (`pixi run -e dev test`). It SHALL run when integration tests are explicitly included (`pixi run -e dev pytest tests/ -v -m ""`).

#### Scenario: Default test run excludes integration test
- **WHEN** `pixi run -e dev test` is executed
- **THEN** the training integration test is not collected or executed

#### Scenario: Explicit inclusion runs integration test
- **WHEN** `pixi run -e dev pytest tests/integration/test_training_loop.py -v -m ""` is executed
- **THEN** the training integration test runs and passes (requires GPU and HF model access)

### Requirement: Synthetic training data for integration test
The integration test SHALL construct synthetic `TrainingBatch` data with random tensors matching the expected shapes for the configured policy, without requiring any real dataset download.

#### Scenario: Synthetic batch matches policy config
- **WHEN** a policy is built with `action_dim=7`, `chunk_len=5`, `proprio_dim=14`
- **THEN** the synthetic batch has `actions` shape `[B, 5, 7]`, `proprio` shape `[B, 14]`, and images matching the VLM's expected resolution
