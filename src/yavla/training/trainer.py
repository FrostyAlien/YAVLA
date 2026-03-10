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

On resume, the trainer derives the dataloader epoch and in-epoch offset
from the skipped micro-batch count when ``len(train_dataloader)`` is
available. This keeps epoch-aware shuffling and logged ``train/epoch``
aligned with the actual dataloader rollover points.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from torch import Tensor, nn

from yavla.training.data import advance_data_epoch

if TYPE_CHECKING:
    from accelerate import Accelerator  # type: ignore[import-untyped]
    from torch.utils.data import DataLoader

    from yavla.models.protocols import PolicyBase
    from yavla.models.types import LossDict
    from yavla.training.config import TrainingConfig

LOGGER = logging.getLogger(__name__)


def _to_float(value: Tensor | float | int) -> float:
    """Detach tensor scalars before converting them to Python floats."""
    if isinstance(value, Tensor):
        return float(value.detach())
    return float(value)


def _move_batch_to_device(batch: Any, accelerator: Accelerator) -> Any:
    """Move typed YAVLA batches onto the prepared accelerator device."""
    from yavla.models.types import TrainingBatch

    if isinstance(batch, TrainingBatch):
        return batch.to(accelerator.device, non_blocking=False)
    return batch


def _infer_batch_size(batch: Any) -> int:
    """Best-effort batch-size inference for logging."""
    from yavla.models.types import TrainingBatch

    if isinstance(batch, TrainingBatch):
        return int(batch.actions.shape[0])
    if isinstance(batch, Tensor):
        return int(batch.shape[0]) if batch.ndim > 0 else 1
    if isinstance(batch, dict):
        for value in batch.values():
            size = _infer_batch_size(value)
            if size > 0:
                return size
        return 1
    if isinstance(batch, (list, tuple)):
        for value in batch:
            size = _infer_batch_size(value)
            if size > 0:
                return size
        return len(batch) if len(batch) > 0 else 1
    return 1


def _get_optimizer_lrs(optimizer: Any) -> tuple[float, float]:
    """Return (main_lr, backbone_lr) from the optimizer param groups."""
    main_lr = float(optimizer.param_groups[0]["lr"])
    backbone_lr = float(optimizer.param_groups[1]["lr"]) if len(optimizer.param_groups) > 1 else main_lr
    return main_lr, backbone_lr


def _count_model_parameters(policy: PolicyBase) -> dict[str, float]:
    """Collect static parameter-count metrics for one training run."""
    backbone = cast(nn.Module, policy.backbone)
    backbone_ids = {id(param) for param in backbone.parameters()}

    params_total = 0
    params_trainable = 0
    params_trainable_backbone = 0
    params_trainable_non_backbone = 0

    for param in policy.parameters():
        numel = int(param.numel())
        params_total += numel
        if not param.requires_grad:
            continue
        params_trainable += numel
        if id(param) in backbone_ids:
            params_trainable_backbone += numel
        else:
            params_trainable_non_backbone += numel

    fraction = params_trainable / params_total if params_total > 0 else 0.0
    return {
        "model/params_total": float(params_total),
        "model/params_trainable": float(params_trainable),
        "model/params_trainable_backbone": float(params_trainable_backbone),
        "model/params_trainable_non_backbone": float(params_trainable_non_backbone),
        "model/params_trainable_fraction": float(fraction),
    }


def _try_num_batches_per_epoch(dataloader: DataLoader[Any]) -> int | None:
    """Return the prepared dataloader length when available."""
    try:
        return int(len(dataloader))
    except TypeError:
        return None


def _policy_action_dim_ratio(policy: PolicyBase) -> tuple[int, int] | None:
    """Return (active_dims, total_dims) when the policy exposes embodiment widths."""
    config = getattr(policy, "config", None)
    candidates = [config, getattr(config, "embodiment", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        action_dim = getattr(candidate, "action_dim", None)
        max_action_dim = getattr(candidate, "max_action_dim", None)
        if (
            isinstance(action_dim, Integral)
            and isinstance(max_action_dim, Integral)
            and 0 < int(action_dim) <= int(max_action_dim)
        ):
            return int(action_dim), int(max_action_dim)
    return None


def _estimate_start_samples(
    dataloader: DataLoader[Any],
    accelerator: Accelerator,
    config: TrainingConfig,
    start_step: int,
) -> int:
    """Approximate cumulative samples for resumed runs when batch_size is known."""
    batch_size = getattr(dataloader, "batch_size", None)
    if not isinstance(batch_size, int) or batch_size <= 0:
        return 0
    effective_global_batch = batch_size * config.gradient_accumulation_steps * accelerator.num_processes
    return int(start_step * effective_global_batch)


def _format_epoch_value(epoch: float) -> str:
    """Format epoch progress for concise console logging."""
    return f"{epoch:.3f}".rstrip("0").rstrip(".")


@dataclass(slots=True)
class _StepMetricAccumulator:
    """Aggregate metrics across all micro-batches in one optimizer step."""

    weighted_sums: dict[str, float] = field(default_factory=dict)
    sample_count: int = 0
    action_valid_numerator: int = 0
    action_valid_denominator: int = 0
    action_valid_seen: bool = False
    action_dim_active_numerator: int = 0
    action_dim_active_denominator: int = 0
    action_dim_active_seen: bool = False

    def add(self, *, batch: Any, loss_dict: LossDict, policy: PolicyBase) -> None:
        from yavla.models.types import TrainingBatch

        batch_size = max(_infer_batch_size(batch), 1)
        self.sample_count += batch_size

        self.weighted_sums["train/loss"] = (
            self.weighted_sums.get("train/loss", 0.0) + _to_float(loss_dict.total) * batch_size
        )
        for name, value in loss_dict.breakdown.items():
            key = f"train/{name}"
            self.weighted_sums[key] = self.weighted_sums.get(key, 0.0) + _to_float(value) * batch_size

        if not isinstance(batch, TrainingBatch):
            return

        total_actions = int(batch.actions.shape[0] * batch.actions.shape[1])
        invalid_actions = int(batch.action_mask.sum().item()) if batch.action_mask is not None else 0
        self.action_valid_numerator += total_actions - invalid_actions
        self.action_valid_denominator += total_actions
        self.action_valid_seen = True

        if batch.action_dim_mask is not None:
            total_dims = int(batch.action_dim_mask.numel())
            inactive_dims = int(batch.action_dim_mask.sum().item())
            active_dims = total_dims - inactive_dims
        else:
            ratio = _policy_action_dim_ratio(policy)
            if ratio is None:
                return
            active_dims, total_dims = ratio

        self.action_dim_active_numerator += active_dims * int(batch.actions.shape[0])
        self.action_dim_active_denominator += total_dims * int(batch.actions.shape[0])
        self.action_dim_active_seen = True

    def global_batch_size(self, accelerator: Accelerator) -> int:
        """Return the effective global batch size for this optimizer step."""
        return max(self.sample_count, 1) * accelerator.num_processes

    def to_metrics(
        self,
        *,
        accelerator: Accelerator,
        lr_main: float,
        lr_backbone: float,
        grad_norm: float,
        epoch: int,
        samples_seen: int,
    ) -> dict[str, float]:
        """Emit finalized scalar metrics for W&B and console logging."""
        weight = max(self.sample_count, 1)
        metrics = {name: total / weight for name, total in self.weighted_sums.items()}
        metrics["train/lr"] = float(lr_main)
        metrics["train/lr_main"] = float(lr_main)
        metrics["train/lr_backbone"] = float(lr_backbone)
        metrics["train/grad_norm"] = float(grad_norm)
        metrics["train/global_batch_size"] = float(self.global_batch_size(accelerator))
        metrics["train/samples_seen"] = float(samples_seen)
        metrics["train/epoch"] = float(epoch)

        if self.action_valid_seen and self.action_valid_denominator > 0:
            metrics["train/action_valid_fraction"] = (
                self.action_valid_numerator / self.action_valid_denominator
            )
        if self.action_dim_active_seen and self.action_dim_active_denominator > 0:
            metrics["train/action_dim_active_fraction"] = (
                self.action_dim_active_numerator / self.action_dim_active_denominator
            )
        return metrics


@dataclass(slots=True)
class _PerformanceWindowAccumulator:
    """Aggregate trainer-visible performance metrics across a log window."""

    step_count: int = 0
    total_step_time_s: float = 0.0
    total_data_wait_time_s: float = 0.0
    total_samples: int = 0

    def add_step(self, *, step_time_s: float, data_wait_time_s: float, global_samples: int) -> None:
        self.step_count += 1
        self.total_step_time_s += float(step_time_s)
        self.total_data_wait_time_s += float(data_wait_time_s)
        self.total_samples += int(global_samples)

    def to_metrics(self) -> dict[str, float]:
        """Emit averaged performance metrics for the completed log window."""
        steps = max(self.step_count, 1)
        total_step_time = max(self.total_step_time_s, 0.0)
        total_data_wait = max(min(self.total_data_wait_time_s, total_step_time), 0.0)
        total_compute_time = max(total_step_time - total_data_wait, 0.0)

        return {
            "perf/step_time_s": total_step_time / steps,
            "perf/samples_per_sec": (self.total_samples / total_step_time) if total_step_time > 0.0 else 0.0,
            "perf/data_wait_time_s": total_data_wait / steps,
            "perf/compute_time_s": total_compute_time / steps,
            "perf/data_wait_fraction": (total_data_wait / total_step_time) if total_step_time > 0.0 else 0.0,
        }


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
    batch = _move_batch_to_device(batch, accelerator)
    loss_dict = policy(batch)
    accelerator.backward(loss_dict.total)
    grad_norm = 0.0
    if accelerator.sync_gradients:
        norm = accelerator.clip_grad_norm_(policy.parameters(), config.optimizer.grad_clip_norm)
        if norm is not None:
            grad_norm = float(norm)
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
        *,
        tracker_config: dict[str, Any] | None = None,
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
        self.tracker_config = tracker_config
        self._model_metrics = _count_model_parameters(policy)
        self._data_epoch = 0
        self._micro_batches_in_epoch = 0
        self._samples_seen = 0

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
        self._num_batches_per_epoch = _try_num_batches_per_epoch(self.train_dataloader)

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

    def _build_data_iterator(self, *, epoch_index: int, batches_to_skip: int = 0) -> Any:
        """Create an iterator aligned to the requested dataloader epoch."""
        advance_data_epoch(self.train_dataloader, epoch_index)
        self._data_epoch = epoch_index
        self._micro_batches_in_epoch = batches_to_skip

        data_iter = iter(self.train_dataloader)
        if batches_to_skip > 0:
            for _ in range(batches_to_skip):
                next(data_iter)
        return data_iter

    def _initial_data_iterator(self, *, start_step: int) -> Any:
        """Build the starting iterator, preserving epoch position on resume."""
        skipped_micro_batches = start_step * self.config.gradient_accumulation_steps
        if skipped_micro_batches <= 0:
            return self._build_data_iterator(epoch_index=0)

        if self._num_batches_per_epoch is not None and self._num_batches_per_epoch > 0:
            epoch_index = skipped_micro_batches // self._num_batches_per_epoch
            batches_to_skip = skipped_micro_batches % self._num_batches_per_epoch
            return self._build_data_iterator(epoch_index=epoch_index, batches_to_skip=batches_to_skip)

        from accelerate.data_loader import skip_first_batches  # type: ignore[import-untyped]

        advance_data_epoch(self.train_dataloader, 0)
        self._data_epoch = 0
        self._micro_batches_in_epoch = 0
        return iter(skip_first_batches(self.train_dataloader, skipped_micro_batches))

    def _next_batch(self, data_iter: Any) -> tuple[Any, Any]:
        """Fetch the next micro-batch, advancing epoch state on rollover."""
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = self._build_data_iterator(epoch_index=self._data_epoch + 1)
            batch = next(data_iter)

        self._micro_batches_in_epoch += 1
        return batch, data_iter

    def _logged_epoch(self) -> float:
        """Return fractional epoch progress when the dataloader length is known."""
        if self._num_batches_per_epoch is not None and self._num_batches_per_epoch > 0:
            return self._data_epoch + (self._micro_batches_in_epoch / self._num_batches_per_epoch)
        return float(self._data_epoch + 1)

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
            tracker_kwargs: dict[str, Any] = {}
            if self.tracker_config is not None:
                tracker_kwargs["config"] = self.tracker_config
            self.accelerator.init_trackers("yavla", **tracker_kwargs)
            self.accelerator.log(self._model_metrics, step=start_step)

        self.policy.train()

        completed_steps = start_step
        self._samples_seen = _estimate_start_samples(self.train_dataloader, self.accelerator, cfg, start_step)
        data_iter = self._initial_data_iterator(start_step=start_step)
        perf_window = _PerformanceWindowAccumulator()

        while completed_steps < cfg.num_steps:
            step_metrics = _StepMetricAccumulator()
            grad_norm = 0.0
            step_start_time = time.perf_counter()
            data_wait_time_s = 0.0

            while True:
                wait_start_time = time.perf_counter()
                batch, data_iter = self._next_batch(data_iter)
                data_wait_time_s += time.perf_counter() - wait_start_time

                with self.accelerator.accumulate(self.policy):
                    loss_dict, grad_norm = train_step(
                        self.policy, batch, self.accelerator, self.optimizer, cfg
                    )

                self.scheduler.step()
                step_metrics.add(batch=batch, loss_dict=loss_dict, policy=self.policy)

                if self.accelerator.sync_gradients:
                    break

            step_time_s = time.perf_counter() - step_start_time
            completed_steps += 1
            global_samples = step_metrics.global_batch_size(self.accelerator)
            self._samples_seen += global_samples
            perf_window.add_step(
                step_time_s=step_time_s,
                data_wait_time_s=data_wait_time_s,
                global_samples=global_samples,
            )

            # Logging (only on optimizer steps)
            if completed_steps % cfg.log_freq == 0:
                lr_main, lr_backbone = _get_optimizer_lrs(self.optimizer)
                metrics = step_metrics.to_metrics(
                    accelerator=self.accelerator,
                    lr_main=lr_main,
                    lr_backbone=lr_backbone,
                    grad_norm=grad_norm,
                    epoch=self._logged_epoch(),
                    samples_seen=self._samples_seen,
                )
                metrics.update(perf_window.to_metrics())
                self.accelerator.print(
                    f"step {completed_steps}/{cfg.num_steps}"
                    f"  epoch={_format_epoch_value(metrics['train/epoch'])}"
                    f"  loss={metrics['train/loss']:.4f}"
                    f"  lr={lr_main:.2e}  grad_norm={grad_norm:.2f}"
                )
                if cfg.wandb:
                    self.accelerator.log(metrics, step=completed_steps)
                perf_window = _PerformanceWindowAccumulator()

            # Checkpoint (only on optimizer steps)
            if completed_steps % cfg.save_freq == 0:
                self.save_checkpoint(completed_steps)

        # Always save final checkpoint
        if completed_steps % cfg.save_freq != 0:
            self.save_checkpoint(completed_steps)

        if cfg.wandb:
            self.accelerator.end_training()
