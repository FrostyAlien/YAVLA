## ADDED Requirements

### Requirement: OptimizerConfig dataclass
`OptimizerConfig` SHALL be a `@dataclass` in `src/yavla/training/config.py` with fields: `name: str = "AdamW"`, `lr: float = 1e-4`, `weight_decay: float = 0.01`, `betas: tuple[float, float] = (0.9, 0.999)`, `eps: float = 1e-8`, `grad_clip_norm: float = 1.0`, `backbone_lr_scale: float = 0.1`.

#### Scenario: Default construction
- **WHEN** `OptimizerConfig()` is constructed
- **THEN** `name` SHALL be `"AdamW"`, `lr` SHALL be `1e-4`, `grad_clip_norm` SHALL be `1.0`, `backbone_lr_scale` SHALL be `0.1`

### Requirement: SchedulerConfig dataclass
`SchedulerConfig` SHALL be a `@dataclass` with fields: `name: str = "cosine"`, `warmup_steps: int = 1000`, `min_lr_ratio: float = 0.1`.

#### Scenario: Default construction
- **WHEN** `SchedulerConfig()` is constructed
- **THEN** `name` SHALL be `"cosine"`, `warmup_steps` SHALL be `1000`

### Requirement: Expanded TrainingConfig
`TrainingConfig` in `src/yavla/training/config.py` SHALL be expanded with: `optimizer: OptimizerConfig`, `scheduler: SchedulerConfig`, `precision: str = "bf16"` (maps to Accelerate's `mixed_precision` parameter), `num_steps: int = 100_000`, `log_freq: int = 100`, `save_freq: int = 5000`, `output_dir: str = "outputs/train"`, `resume: bool = False`, `gradient_checkpointing: bool = True`, `use_policy_preset: bool = True`, `wandb: bool = False`, `gradient_accumulation_steps: int = 1`.

#### Scenario: YAML round-trip
- **WHEN** a `TrainingConfig` is serialized to dict and reconstructed
- **THEN** all fields SHALL be preserved with correct types

#### Scenario: Precision maps to Accelerate
- **WHEN** `config.precision = "bf16"`
- **THEN** `Accelerator(mixed_precision="bf16")` SHALL be constructed

### Requirement: get_optimizer_preset on PolicyBase
`PolicyBase` in `src/yavla/models/protocols.py` SHALL have `get_optimizer_preset() -> OptimizerConfig | None` returning `None` by default. Concrete policies MAY override to return policy-specific defaults.

#### Scenario: Default returns None
- **WHEN** `VLAPolicy().get_optimizer_preset()` is called on a policy that does not override
- **THEN** it SHALL return `None`
