"""Optimizer and LR scheduler factory.

Builds AdamW with discriminative learning rates (backbone at ``lr * backbone_lr_scale``,
all other params at full ``lr``) and a warmup-then-cosine scheduler via built-in PyTorch
``SequentialLR([LinearLR, CosineAnnealingLR])``.

Discriminative LR is standard practice for fine-tuning VLA models: the pretrained
backbone (vision + language) uses a lower LR to preserve general features, while the
action head trains at full LR for fast task adaptation. Typical ratio is 0.1×.

The returned optimizer/scheduler are **not** yet wrapped by Accelerate — the ``Trainer``
calls ``accelerator.prepare()`` on them.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from yavla.models.protocols import PolicyBase
    from yavla.training.config import OptimizerConfig, TrainingConfig


def make_optimizer_and_scheduler(
    policy: PolicyBase,
    config: TrainingConfig,
    num_training_steps: int,
) -> tuple[Optimizer, LRScheduler]:
    """Build optimizer with per-module LR groups and warmup+cosine scheduler.

    Args:
        policy: The policy module (must have a ``backbone`` attribute).
        config: Full training config.
        num_training_steps: Total number of training steps (for cosine decay).

    Returns:
        (optimizer, scheduler) — not yet wrapped by Accelerate.
    """
    opt_cfg = config.optimizer
    if config.use_policy_preset:
        preset = policy.get_optimizer_preset()
        if preset is not None:
            opt_cfg = _merge_preset(config.optimizer, preset)

    if opt_cfg.name != "AdamW":
        raise NotImplementedError(f"Only AdamW is supported, got {opt_cfg.name!r}")

    # Split params: backbone vs rest
    backbone = cast(nn.Module, policy.backbone)
    backbone_ids = {id(p) for p in backbone.parameters()}
    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    other_params = [p for _, p in policy.named_parameters() if p.requires_grad and id(p) not in backbone_ids]

    param_groups = [
        {"params": other_params, "lr": opt_cfg.lr},
        {"params": backbone_params, "lr": opt_cfg.lr * opt_cfg.backbone_lr_scale},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=opt_cfg.weight_decay,
        betas=opt_cfg.betas,
        eps=opt_cfg.eps,
    )

    # Warmup (linear) + cosine decay via built-in PyTorch schedulers
    # eta_min uses backbone floor so no group is clamped above its start LR
    sched_cfg = config.scheduler
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=sched_cfg.warmup_steps)
    remaining = max(num_training_steps - sched_cfg.warmup_steps, 1)
    eta_min = opt_cfg.lr * opt_cfg.backbone_lr_scale * sched_cfg.min_lr_ratio
    cosine = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=eta_min)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[sched_cfg.warmup_steps])

    return optimizer, scheduler


def _merge_preset(
    base: OptimizerConfig,
    preset: OptimizerConfig,
) -> OptimizerConfig:
    """Overlay non-default preset fields onto base config.

    Fields in *preset* that differ from ``OptimizerConfig()`` defaults win;
    all others fall through to *base*. This means a preset cannot explicitly
    set a field to the default value — an acceptable trade-off for simplicity.
    """
    from yavla.training.config import OptimizerConfig

    default = OptimizerConfig()
    merged = {}
    for f in dataclasses.fields(OptimizerConfig):
        preset_val = getattr(preset, f.name)
        if preset_val != getattr(default, f.name):
            merged[f.name] = preset_val
        else:
            merged[f.name] = getattr(base, f.name)
    return OptimizerConfig(**merged)
