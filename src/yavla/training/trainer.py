"""Training loop and Trainer class (Accelerate-first).

Provides ``train_step`` (a single forward/backward/clip/step) and ``Trainer``
(the full training loop with checkpoint save/resume and WandB logging).

All distributed-training concerns — AMP, gradient clipping, gradient
accumulation, and checkpointing — are delegated to HuggingFace Accelerate.
The loop counts *optimizer steps* (not micro-batches); logging and checkpoint
saves are gated on ``accelerator.sync_gradients`` so they only fire after a
real parameter update.

Gradient accumulation flow::

    while completed_steps < num_steps:
        batch = next(data_iter)
        with accelerator.accumulate(policy):   # no-ops opt.step on non-sync
            train_step(policy, batch, ...)
        scheduler.step()                       # AcceleratedScheduler gates internally
        if not sync_gradients: continue
        completed_steps += 1                   # ← optimizer step boundary
        log / checkpoint here

On resume, ``skip_first_batches`` fast-forwards the dataloader by
``start_step * gradient_accumulation_steps`` micro-batches so the model
does not replay data it already trained on.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from torch import nn

if TYPE_CHECKING:
    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    from yavla.models.protocols import PolicyBase
    from yavla.models.types import LossDict
    from yavla.training.config import TrainingConfig

LOGGER = logging.getLogger(__name__)


def train_step(
    policy: PolicyBase,
    batch: Any,
    accelerator: Accelerator,
    optimizer: Any,
    config: TrainingConfig,
) -> tuple[LossDict, float]:
    """Execute one training step: forward, backward, clip, step.

    AMP is handled automatically by Accelerate — no manual autocast/GradScaler.
    When using gradient accumulation, call this inside ``accelerator.accumulate()``;
    Accelerate will no-op ``optimizer.step()``/``zero_grad()`` on non-sync steps.

    Returns:
        (loss_dict, grad_norm) where grad_norm is the total gradient norm
        (computed before clipping is applied). Returns 0.0 on accumulation
        micro-steps where gradients are not yet synced.
    """
    loss_dict = policy(batch)
    accelerator.backward(loss_dict.total)
    grad_norm = 0.0
    if accelerator.sync_gradients:
        norm = accelerator.clip_grad_norm_(policy.parameters(), config.optimizer.grad_clip_norm)
        grad_norm = float(norm)  # type: ignore[arg-type]
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss_dict, grad_norm


class Trainer:
    """Accelerate-first training loop for VLA policies."""

    def __init__(
        self,
        policy: PolicyBase,
        config: TrainingConfig,
        train_dataloader: DataLoader[Any],
    ) -> None:
        from accelerate import Accelerator

        from yavla.training.optim import make_optimizer_and_scheduler

        log_with = "wandb" if config.wandb else None
        self.accelerator = Accelerator(
            mixed_precision=config.precision,
            log_with=log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
        self.config = config

        # Gradient checkpointing (before prepare)
        if config.gradient_checkpointing:
            backbone = cast(nn.Module, policy.backbone)
            enable_fn = getattr(backbone, "gradient_checkpointing_enable", None)
            if enable_fn is not None:
                enable_fn(gradient_checkpointing_kwargs={"use_reentrant": False})

        optimizer, scheduler = make_optimizer_and_scheduler(
            policy, config, config.num_steps
        )

        self.policy, self.optimizer, self.train_dataloader, self.scheduler = (
            self.accelerator.prepare(policy, optimizer, train_dataloader, scheduler)
        )

    # ------------------------------------------------------------------
    # Checkpoint save / resume (Phase 4)
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int) -> None:
        """Save full training state via Accelerate."""
        path = f"{self.config.output_dir}/checkpoint-{step}"
        self.accelerator.save_state(path)
        self.accelerator.print(f"Saved checkpoint at step {step}")

    def _load_latest_checkpoint(self) -> int:
        """Scan output_dir for highest-step checkpoint, restore state, return step."""
        output = Path(self.config.output_dir)
        if not output.exists():
            return 0
        steps = []
        for d in output.iterdir():
            m = re.match(r"checkpoint-(\d+)$", d.name)
            if m and d.is_dir():
                steps.append(int(m.group(1)))
        if not steps:
            return 0
        latest = max(steps)
        self.accelerator.load_state(str(output / f"checkpoint-{latest}"))
        self.accelerator.print(f"Resumed from checkpoint-{latest}")
        return latest

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the training loop for ``config.num_steps`` optimizer steps.

        With ``gradient_accumulation_steps > 1``, each optimizer step consumes
        multiple micro-batches.  Logging and checkpointing are gated on
        ``sync_gradients`` so they only fire on real optimizer updates.
        """
        cfg = self.config
        start_step = 0
        if cfg.resume:
            start_step = self._load_latest_checkpoint()

        if cfg.wandb:
            self.accelerator.init_trackers("yavla")

        self.policy.train()

        # Fast-forward dataloader on resume (account for accumulation micro-batches)
        if start_step > 0:
            from accelerate.data_loader import skip_first_batches

            skip = start_step * cfg.gradient_accumulation_steps
            data_iter = iter(skip_first_batches(self.train_dataloader, skip))
        else:
            data_iter = iter(self.train_dataloader)

        completed_steps = start_step

        while completed_steps < cfg.num_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            with self.accelerator.accumulate(self.policy):
                loss_dict, grad_norm = train_step(
                    self.policy, batch, self.accelerator, self.optimizer, cfg
                )

            self.scheduler.step()

            if not self.accelerator.sync_gradients:
                continue

            completed_steps += 1

            # Logging (only on optimizer steps)
            if completed_steps % cfg.log_freq == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                metrics: dict[str, float] = {
                    "train/loss": float(loss_dict.total),
                    "train/lr": lr,
                    "train/grad_norm": grad_norm,
                }
                for k, v in loss_dict.breakdown.items():
                    metrics[f"train/{k}"] = float(v)
                self.accelerator.print(
                    f"step {completed_steps}/{cfg.num_steps}"
                    f"  loss={loss_dict.total:.4f}"
                    f"  lr={lr:.2e}  grad_norm={grad_norm:.2f}"
                )
                if cfg.wandb:
                    self.accelerator.log(metrics, step=completed_steps)

            # Checkpoint (only on optimizer steps)
            if completed_steps % cfg.save_freq == 0:
                self.save_checkpoint(completed_steps)

        # Always save final checkpoint
        if completed_steps % cfg.save_freq != 0:
            self.save_checkpoint(completed_steps)

        if cfg.wandb:
            self.accelerator.end_training()
