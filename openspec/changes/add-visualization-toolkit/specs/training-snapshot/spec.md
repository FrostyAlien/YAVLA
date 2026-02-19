## ADDED Requirements

### Requirement: Initialize snapshot state with fixed samples
The system SHALL provide an `init_snapshot_state` function that selects `snapshot_num_samples` dataset **indices** (not decoded tensors) using `snapshot_seed` for deterministic selection. It SHALL return a `SnapshotState` dataclass (`@dataclass(slots=True)`) holding the fixed indices and a deterministic DataLoader, or `None` if disabled or non-zero rank.

#### Scenario: Snapshot disabled
- **WHEN** `init_snapshot_state(config, dataset)` is called with `config.snapshot_enabled == False`
- **THEN** the function SHALL return `None` without accessing the dataset

#### Scenario: Non-zero rank in DDP/FSDP
- **WHEN** `init_snapshot_state(config, dataset, rank=1)` is called (or `torch.distributed.get_rank() != 0`)
- **THEN** the function SHALL return `None` — snapshots run on rank 0 only

#### Scenario: Deterministic sample selection
- **WHEN** `init_snapshot_state` is called twice with the same `snapshot_seed` and dataset
- **THEN** both calls SHALL select the identical sample indices via `torch.Generator(device='cpu').manual_seed(seed)` + `torch.randperm`

#### Scenario: Deterministic DataLoader for fixed samples
- **WHEN** `SnapshotState` is created
- **THEN** it SHALL contain a DataLoader with `num_workers=0`, no shuffle, and a transform pipeline that excludes stochastic augmentations (no random crop/flip/color jitter — only resize + normalize)

#### Scenario: Dataset smaller than requested samples
- **WHEN** the dataset has fewer samples than `snapshot_num_samples`
- **THEN** the function SHALL use all available samples without error

### Requirement: Interval-based snapshot generation
The system SHALL provide a `maybe_generate_snapshot` function that accepts `SnapshotState | None`. If `state is None`, it SHALL return immediately (no-op). Otherwise, it SHALL check whether the current training step is a multiple of `snapshot_interval_steps`. If so, it SHALL generate attention heatmaps for all configured methods on the fixed samples and log them to wandb.

#### Scenario: State is None (disabled or non-rank-0)
- **WHEN** `maybe_generate_snapshot(state=None, model=model, step=10000)` is called
- **THEN** the function SHALL return immediately without any computation or logging

#### Scenario: Step matches interval
- **WHEN** `maybe_generate_snapshot(state, model, step=10000)` is called with `snapshot_interval_steps=10000`
- **THEN** the function SHALL generate heatmaps using each method in `snapshot_methods`, log each as a `wandb.Image`, and update `last_snapshot_step`

#### Scenario: Step does not match interval
- **WHEN** `maybe_generate_snapshot` is called at step 9999 with interval 10000
- **THEN** the function SHALL return immediately without generating any heatmaps or logging

#### Scenario: W&B logging contract
- **WHEN** heatmaps are generated and `wandb.run` is active
- **THEN** images SHALL be logged with key `"viz/{method}/{sample_idx}"` (e.g., `"viz/grad_cam/0"`), as `wandb.Image(heatmap_overlay, caption=f"step={step} sample={idx} method={method}")`, with `step=step` for W&B x-axis alignment

#### Scenario: W&B not initialized
- **WHEN** `maybe_generate_snapshot` is called with `wandb_run=None` and no active wandb run
- **THEN** the function SHALL generate heatmaps but skip logging, emitting `logger.warning("W&B not initialized, skipping snapshot logging")`

### Requirement: Snapshot does not interfere with training
The snapshot callback SHALL NOT modify model weights, optimizer state, or the training gradient graph. It SHALL operate in a fully isolated context.

#### Scenario: Gradient isolation mechanism
- **WHEN** a snapshot is generated mid-training
- **THEN** the function SHALL:
  1. Save current training gradients: `{p: p.grad for p in model.parameters() if p.grad is not None}`
  2. Switch to `model.eval()`
  3. Run heatmap generation under `torch.enable_grad()` + `torch.autocast(device_type, enabled=False)`
  4. Call `model.zero_grad(set_to_none=True)` to discard snapshot gradients
  5. Restore saved training gradients
  6. Restore `model.train()` mode

#### Scenario: Training gradients preserved
- **WHEN** a snapshot is generated mid-training during gradient accumulation
- **THEN** the model's accumulated gradients from the training batch SHALL be identical before and after the snapshot

#### Scenario: AMP scaler not corrupted
- **WHEN** a snapshot runs while AMP (automatic mixed precision) is active
- **THEN** the snapshot SHALL disable autocast during its forward+backward pass, ensuring the GradScaler state is not affected

### Requirement: Testability
Unit tests SHALL mock `wandb` and attention visualization functions. Core logic (interval checking, sample selection, gradient save/restore) SHALL be testable without GPU. Integration tests requiring a real model SHALL be marked `@pytest.mark.integration`.
