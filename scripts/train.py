"""Train a VLA policy (CLI entry point).

Composes ``TrainingConfig`` and ``PolicyConfig`` into a single
``TrainConfig`` dataclass parsed by **tyro**.  An optional ``--config``
flag loads YAML defaults (e.g. ``configs/train.yaml``) before tyro
applies CLI overrides, giving the standard YAML-first / CLI-override
workflow.

Compatible with both single-process and distributed launches::

    # Single GPU
    python scripts/train.py --training.num-steps 50000

    # Distributed via Accelerate
    accelerate launch scripts/train.py --training.num-steps 50000

    # YAML defaults + CLI overrides
    python scripts/train.py --config configs/train.yaml --training.wandb True
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import tyro

from yavla.models.config import PolicyConfig
from yavla.models.policy import build_policy
from yavla.training import Trainer, TrainingConfig, create_training_dataloader

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Top-level config composing training and policy settings."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)


def _filter_fields(cls: type, raw: dict[str, object]) -> dict[str, object]:
    """Keep only keys that match dataclass fields."""
    return {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}  # type: ignore[attr-defined]


def _pop_config_flag() -> TrainConfig | None:
    """Extract ``--config <path>`` before tyro sees it, return YAML defaults."""
    if "--config" not in sys.argv:
        return None
    idx = sys.argv.index("--config")
    if idx + 1 >= len(sys.argv):
        sys.exit("error: --config requires a path argument")
    path = Path(sys.argv[idx + 1])
    sys.argv = sys.argv[:idx] + sys.argv[idx + 2 :]

    if not path.exists():
        sys.exit(f"error: config file not found: {path}")

    import yaml

    with open(path) as f:
        loaded = yaml.safe_load(f)
    raw: dict[str, object] = loaded if isinstance(loaded, dict) else {}

    from yavla.training.config import OptimizerConfig, SchedulerConfig

    opt_raw = raw.pop("optimizer", {})
    if isinstance(opt_raw, dict) and "betas" in opt_raw:
        opt_raw["betas"] = tuple(opt_raw["betas"])
    opt = OptimizerConfig(**_filter_fields(OptimizerConfig, opt_raw))  # type: ignore[arg-type]
    sched = SchedulerConfig(**_filter_fields(SchedulerConfig, raw.pop("scheduler", {})))  # type: ignore[arg-type]

    dataset_raw = raw.pop("dataset", None)
    viz_raw = raw.pop("viz", None)
    training_kwargs: dict[str, object] = {"optimizer": opt, "scheduler": sched}
    if isinstance(dataset_raw, dict):
        from yavla.data.factory import DataConfig

        training_kwargs["dataset"] = DataConfig(**{str(k): v for k, v in dataset_raw.items()})
    if isinstance(viz_raw, dict):
        from yavla.visualization.config import VizConfig

        training_kwargs["viz"] = VizConfig(**_filter_fields(VizConfig, viz_raw))  # type: ignore[arg-type]
    for k, v in raw.items():
        if k in TrainingConfig.__dataclass_fields__:
            training_kwargs[k] = v
    return TrainConfig(training=TrainingConfig(**training_kwargs))  # type: ignore[arg-type]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    default = _pop_config_flag() or TrainConfig()
    cfg = tyro.cli(TrainConfig, default=default)

    LOGGER.info("Building policy...")
    policy = build_policy(cfg.policy)

    LOGGER.info("Creating dataloader...")
    dataloader = create_training_dataloader(
        cfg.training, dt_hz=cfg.policy.dt_hz, chunk_len=cfg.policy.action_head.chunk_len
    )

    trainer = Trainer(policy, cfg.training, dataloader)
    LOGGER.info("Starting training for %d steps", cfg.training.num_steps)
    trainer.run()


if __name__ == "__main__":
    main()
