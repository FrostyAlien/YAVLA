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
from pathlib import Path
from typing import Any

import tyro

from yavla.models.policy import build_policy
from yavla.models.types import TrainingBatch
from yavla.training import Trainer, create_training_dataloader
from yavla.training.entry_config import TrainConfig, load_train_config_file
from yavla.training.siglip_preprocess import autowire_siglip_image_transforms

LOGGER = logging.getLogger(__name__)

def _pop_config_flag() -> TrainConfig | None:
    """Extract ``--config <path>`` before tyro sees it, return YAML defaults."""
    if "--config" not in sys.argv:
        return None
    idx = sys.argv.index("--config")
    if idx + 1 >= len(sys.argv):
        sys.exit("error: --config requires a path argument")
    path = Path(sys.argv[idx + 1])
    sys.argv = sys.argv[:idx] + sys.argv[idx + 2 :]

    try:
        return load_train_config_file(path)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"error: {exc}")


def _peek_training_batch(dataloader: Any) -> TrainingBatch:
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        sys.exit("error: training dataloader is empty; cannot validate action/proprio dimensions")

    if not isinstance(batch, TrainingBatch):
        sys.exit(f"error: training dataloader yielded {type(batch).__name__}, expected TrainingBatch")
    return batch


def _validate_training_dimensions(cfg: TrainConfig, batch: TrainingBatch) -> None:
    configured_chunk_len = cfg.policy.action_head.chunk_len
    configured_action_dim = cfg.policy.action_dim
    configured_proprio_dim = cfg.policy.proprio_dim
    dataset_chunk_len = cfg.training.dataset.action_chunk_size

    if dataset_chunk_len is not None and dataset_chunk_len != configured_chunk_len:
        sys.exit(
            "error: training.dataset.action_chunk_size="
            f"{dataset_chunk_len} does not match policy.action_head.chunk_len={configured_chunk_len}; "
            "fix the dataset action chunk size or the policy action head chunk length"
        )

    actual_chunk_len = batch.actions.shape[1]
    if actual_chunk_len != configured_chunk_len:
        sys.exit(
            "error: first batch action chunk length mismatch: "
            f"expected policy.action_head.chunk_len={configured_chunk_len}, got {actual_chunk_len}; "
            "fix policy.action_head.chunk_len or training.dataset.action_chunk_size"
        )

    actual_action_dim = batch.actions.shape[2]
    if actual_action_dim != configured_action_dim:
        sys.exit(
            "error: first batch action dimension mismatch: "
            f"expected policy.embodiment.action_dim={configured_action_dim}, got {actual_action_dim}; "
            "fix policy.embodiment.action_dim to match the dataset embodiment"
        )

    actual_proprio_dim = batch.observations.proprio.shape[-1]
    if actual_proprio_dim != configured_proprio_dim:
        sys.exit(
            "error: first batch proprio dimension mismatch: "
            f"expected policy.embodiment.proprio_dim={configured_proprio_dim}, got {actual_proprio_dim}; "
            "fix policy.embodiment.proprio_dim to match the dataset embodiment"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    default = _pop_config_flag() or TrainConfig()
    cfg = tyro.cli(TrainConfig, default=default, config=(tyro.conf.CascadeSubcommandArgs,))
    LOGGER.info("Effective train config: %s", cfg)

    from accelerate.utils import set_seed  # type: ignore[import-untyped]

    set_seed(cfg.training.dataset.seed)

    LOGGER.info("Building policy...")
    policy = build_policy(cfg.policy)

    if cfg.policy.backbone.type == "paligemma":
        try:
            ckpt_image_size = int(policy.backbone.base_model.config.vision_config.image_size)
        except Exception as exc:  # pragma: no cover
            sys.exit(f"error: failed to derive SigLIP checkpoint image_size from loaded model config: {exc}")
        try:
            autowire_siglip_image_transforms(cfg.training, ckpt_image_size=ckpt_image_size)
        except ValueError as exc:
            sys.exit(f"error: {exc}")

    LOGGER.info("Creating dataloader...")
    dataloader = create_training_dataloader(
        cfg.training, dt_hz=cfg.policy.dt_hz, chunk_len=cfg.policy.action_head.chunk_len
    )
    _validate_training_dimensions(cfg, _peek_training_batch(dataloader))

    trainer = Trainer(policy, cfg.training, dataloader)
    LOGGER.info("Starting training for %d steps", cfg.training.num_steps)
    trainer.run()


if __name__ == "__main__":
    main()
