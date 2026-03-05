"""Training configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field

from yavla.data.factory import DataConfig
from yavla.visualization.config import VizConfig


@dataclass(slots=True)
class OptimizerConfig:
    """Optimizer hyperparameters."""

    name: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    grad_clip_norm: float = 1.0
    backbone_lr_scale: float = 0.1


@dataclass(slots=True)
class SchedulerConfig:
    """LR scheduler hyperparameters."""

    name: str = "cosine"
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1


@dataclass(slots=True)
class TrainingConfig:
    """Full training configuration."""

    dataset: DataConfig = field(default_factory=lambda: DataConfig(repo_id="lerobot/aloha_sim"))
    viz: VizConfig = field(default_factory=VizConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    precision: str = "bf16"
    num_steps: int = 100_000
    log_freq: int = 100
    save_freq: int = 5000
    output_dir: str = "outputs/train"
    resume: bool = False
    gradient_checkpointing: bool = True
    use_policy_preset: bool = True
    vlm_image_height_override: int | None = None
    vlm_image_width_override: int | None = None
    wandb: bool = False
    gradient_accumulation_steps: int = 1
