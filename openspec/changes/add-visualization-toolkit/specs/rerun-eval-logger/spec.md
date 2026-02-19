## ADDED Requirements

### Requirement: RerunEvalLogger context manager for episode logging
The system SHALL provide a `RerunEvalLogger` class that acts as a context manager. On entry it SHALL initialize a Rerun recording for the episode. On exit it SHALL flush and close the recording. Rerun SHALL be lazy-imported (raising `ImportError` with install instructions if missing).

#### Scenario: Basic episode logging
- **WHEN** `with RerunEvalLogger(config, episode_id="ep_001") as logger:` is entered
- **THEN** Rerun SHALL be initialized with application ID "yavla-eval" and the recording SHALL be associated with the given episode_id

#### Scenario: Context manager cleanup on exception
- **WHEN** an exception occurs inside the `with` block
- **THEN** the logger SHALL still flush and close the Rerun recording without suppressing the exception

### Requirement: Log eval steps with synchronized timeline
The `RerunEvalLogger.log_step` method SHALL accept a step index, optional image, optional predicted action, optional ground-truth action, and optional attention map. It SHALL set the Rerun time sequence to the step index before logging any data. The timeline is keyed by **environment step index** (0, 1, 2, ..., T) — not wall time or dataset timestamp.

#### Scenario: Log image and actions
- **WHEN** `logger.log_step(step=42, image=img, pred_action=pred, gt_action=gt)` is called
- **THEN** `rr.set_time_sequence("step", 42)` SHALL be called first, then the image SHALL be logged as `rr.Image` under "vision/camera", and each action dimension SHALL be logged as `rr.Scalar` under "action/pred/dim_{i}" and "action/gt/dim_{i}"

#### Scenario: Action values are raw (denormalized)
- **WHEN** action tensors are passed to `log_step`
- **THEN** they SHALL be logged as-is in the model's output space (denormalized, unclipped). The caller is responsible for any normalization/denormalization before passing to the logger.

#### Scenario: Multiple cameras share timeline
- **WHEN** an episode has multiple camera views
- **THEN** each camera SHALL be logged under `"vision/{camera_key}"` and all share the same step-indexed timeline via `rr.set_time_sequence("step", step)`

#### Scenario: Log with attention overlay
- **WHEN** `log_step` is called with `attention_map` provided and `config.rerun_log_attention == True`
- **THEN** the attention heatmap SHALL be logged as `rr.Image` under "vision/attention" at the same time step

#### Scenario: Selective logging via config
- **WHEN** `config.rerun_log_images == False`
- **THEN** `log_step` SHALL skip image logging even if an image tensor is provided

### Requirement: Headless server support via .rrd file saving
The `RerunEvalLogger` SHALL always save recordings to `.rrd` files. When a display is available, it SHALL also spawn the Rerun viewer for live streaming.

#### Scenario: Headless environment
- **WHEN** `RerunEvalLogger` is created on a server without a display (`DISPLAY` env var is unset/empty on Linux)
- **THEN** the recording SHALL be saved to `{rerun_output_dir}/{episode_id}.rrd` (default `rerun_output_dir="./rerun_logs"`) and no viewer process SHALL be spawned

#### Scenario: Local environment with display
- **WHEN** `RerunEvalLogger` is created on a machine with a display
- **THEN** the Rerun viewer SHALL be spawned AND the recording SHALL also be saved to `.rrd` for later replay

#### Scenario: Output file size expectations
- **WHEN** an episode of 100 steps with 224×224 images is logged
- **THEN** the `.rrd` file SHALL be approximately 1-5MB (JPEG-compressed images + action scalars)

### Requirement: Rerun is eval-only
The `RerunEvalLogger` SHALL NOT be used during training. The training snapshot system uses wandb for logging, not Rerun.

#### Scenario: No Rerun imports at training time
- **WHEN** `VizConfig(rerun_enabled=False)` is used
- **THEN** the `rerun` package SHALL NOT be imported (lazy import), avoiding import errors on systems where rerun-sdk is not installed

#### Scenario: Rerun not installed but disabled
- **WHEN** `rerun_enabled=False` and `rerun-sdk` is not installed
- **THEN** no error SHALL occur — the code path is never entered and no import is attempted

### Requirement: Testability
Unit tests SHALL mock `rerun` imports and test timeline sequencing logic, file path construction, and config-driven skip behavior as pure functions. Integration tests requiring `rerun-sdk` SHALL be marked `@pytest.mark.integration`.
