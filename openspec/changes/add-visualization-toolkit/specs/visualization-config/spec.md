## ADDED Requirements

### Requirement: VizConfig dataclass with slots=True convention
The system SHALL provide a `VizConfig` dataclass at `src/yavla/visualization/config.py` using `@dataclass(slots=True)` matching the project convention. All visualization features SHALL default to disabled (False/off) so that importing or composing VizConfig adds zero runtime overhead.

#### Scenario: Default config has all features disabled
- **WHEN** a `VizConfig()` is instantiated with no arguments
- **THEN** `snapshot_enabled` SHALL be `False`, `rerun_enabled` SHALL be `False`, and no visualization code executes

#### Scenario: Config is composable into TrainingConfig
- **WHEN** a future `TrainingConfig` includes a `viz: VizConfig` field
- **THEN** the VizConfig SHALL be accessible as `config.viz` and all its fields SHALL be independently configurable

#### Scenario: Importing VizConfig never triggers heavy imports
- **WHEN** `from yavla.visualization import VizConfig` is executed
- **THEN** no optional dependencies (`rerun`, `pytorch_grad_cam`, `captum`, `umap`) SHALL be imported — `config.py` is pure Python with no heavy deps

### Requirement: Snapshot configuration fields
The `VizConfig` SHALL expose the following snapshot fields with specified defaults: `snapshot_enabled: bool = False`, `snapshot_interval_steps: int = 10_000`, `snapshot_num_samples: int = 4`, `snapshot_methods: list[str] = ["attention_rollout", "grad_cam"]`, `snapshot_layers: list[str] = ["last"]`, `snapshot_seed: int = 42`.

#### Scenario: Custom snapshot interval
- **WHEN** `VizConfig(snapshot_enabled=True, snapshot_interval_steps=5000)` is created
- **THEN** the config SHALL have `snapshot_interval_steps == 5000` and all other fields at defaults

### Requirement: Rerun configuration fields
The `VizConfig` SHALL expose: `rerun_enabled: bool = False`, `rerun_log_images: bool = True`, `rerun_log_actions: bool = True`, `rerun_log_attention: bool = False`, `rerun_output_dir: str = "./rerun_logs"`.

#### Scenario: Rerun with attention logging enabled
- **WHEN** `VizConfig(rerun_enabled=True, rerun_log_attention=True)` is created
- **THEN** the config SHALL enable both Rerun logging and attention overlay logging

#### Scenario: Custom Rerun output directory
- **WHEN** `VizConfig(rerun_enabled=True, rerun_output_dir="/data/rerun")` is created
- **THEN** `.rrd` files SHALL be saved to `/data/rerun/{episode_id}.rrd`

### Requirement: FiftyOne configuration fields
The `VizConfig` SHALL expose: `fiftyone_subsample_rate: int = 10`, `fiftyone_umap_pca_dims: int = 50`.

#### Scenario: High subsample rate for large datasets
- **WHEN** `VizConfig(fiftyone_subsample_rate=100)` is created
- **THEN** the FiftyOne loader SHALL use every 100th frame when loading data

### Requirement: "Disabled" semantics — config flag vs missing dependency
A feature is "disabled" when EITHER the config flag is off OR the dependency is missing. These are distinct failure modes.

#### Scenario: Config off, dep installed
- **WHEN** `snapshot_enabled=False` and `pytorch-grad-cam` is installed
- **THEN** no snapshot code runs, no imports attempted, zero overhead

#### Scenario: Config off, dep missing
- **WHEN** `snapshot_enabled=False` and `pytorch-grad-cam` is NOT installed
- **THEN** no error occurs — the code path is never entered

#### Scenario: Config on, dep missing
- **WHEN** `snapshot_enabled=True` and `pytorch-grad-cam` is NOT installed
- **THEN** the function SHALL raise `ImportError` with message: `"pytorch-grad-cam is required for attention heatmaps. Install with: pip install yavla[viz]"`

### Requirement: Testability
`VizConfig` is a pure Python dataclass with no external dependencies. Unit tests SHALL verify default values, field types, and composition into parent configs without any optional deps installed.
