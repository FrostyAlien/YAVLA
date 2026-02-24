"""Tests for optimizer and scheduler factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from yavla.models.protocols import (
    BackboneBase,
    BackboneCapabilities,
    BackboneOutput,
    IntegrationMode,
    PolicyBase,
)
from yavla.models.types import ActionChunk, LossDict, ObservationBatch, TrainingBatch
from yavla.training.config import OptimizerConfig, SchedulerConfig, TrainingConfig
from yavla.training.optim import make_optimizer_and_scheduler


# -- Minimal stubs for testing ------------------------------------------------


@dataclass
class _StubPolicyConfig:
    pass


class _StubBackbone(BackboneBase):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    @property
    def capabilities(self) -> BackboneCapabilities:
        return BackboneCapabilities(supported_modes={IntegrationMode.READOUT})

    @property
    def hidden_dim(self) -> int:
        return 4

    def embed_language(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        B = len(texts)
        return torch.zeros(B, 1, 4), torch.ones(B, 1)

    def forward(self, inputs_embeds: Tensor, attention_mask: Tensor, token_type_ids: Tensor) -> BackboneOutput:
        return BackboneOutput(readout_states=None, token_states=inputs_embeds, attention_mask=attention_mask)


class _StubPolicy(PolicyBase):
    name = "stub"
    config_class = _StubPolicyConfig

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _StubBackbone()
        self.head = nn.Linear(4, 2)

    def forward(self, batch: TrainingBatch) -> LossDict:
        return LossDict(total=torch.tensor(0.0))

    def predict(self, obs: ObservationBatch) -> ActionChunk:
        return ActionChunk(actions=torch.zeros(1, 1, 2), dt_hz=10.0, chunk_len=1)


# -- Tests --------------------------------------------------------------------


class TestMakeOptimizerAndScheduler:
    def test_returns_optimizer_and_scheduler(self) -> None:
        policy = _StubPolicy()
        config = TrainingConfig()
        opt, sched = make_optimizer_and_scheduler(policy, config, num_training_steps=10_000)
        assert opt is not None
        assert sched is not None

    def test_two_param_groups(self) -> None:
        policy = _StubPolicy()
        config = TrainingConfig()
        opt, _ = make_optimizer_and_scheduler(policy, config, num_training_steps=10_000)
        assert len(opt.param_groups) == 2
        # Check initial_lr (scheduler modifies current lr at construction)
        assert opt.param_groups[0]["initial_lr"] == config.optimizer.lr
        assert opt.param_groups[1]["initial_lr"] == config.optimizer.lr * config.optimizer.backbone_lr_scale

    def test_warmup_increases_lr(self) -> None:
        policy = _StubPolicy()
        config = TrainingConfig(scheduler=SchedulerConfig(warmup_steps=100))
        opt, sched = make_optimizer_and_scheduler(policy, config, num_training_steps=10_000)
        lrs = []
        for _ in range(100):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        # LR should increase during warmup
        assert lrs[-1] > lrs[0]

    def test_cosine_decay_after_warmup(self) -> None:
        policy = _StubPolicy()
        config = TrainingConfig(scheduler=SchedulerConfig(warmup_steps=10))
        opt, sched = make_optimizer_and_scheduler(policy, config, num_training_steps=1000)
        # Step through warmup
        for _ in range(10):
            opt.step()
            sched.step()
        lr_at_warmup_end = opt.param_groups[0]["lr"]
        # Step further into cosine decay
        for _ in range(500):
            opt.step()
            sched.step()
        assert opt.param_groups[0]["lr"] < lr_at_warmup_end

    def test_policy_preset_overrides(self) -> None:
        class _PresetPolicy(_StubPolicy):
            name = "preset_stub"

            def get_optimizer_preset(self) -> OptimizerConfig | None:
                return OptimizerConfig(lr=5e-5)

        policy = _PresetPolicy()
        config = TrainingConfig(optimizer=OptimizerConfig(lr=1e-4))
        opt, _ = make_optimizer_and_scheduler(policy, config, num_training_steps=10_000)
        # Preset lr=5e-5 should override config lr=1e-4
        assert opt.param_groups[0]["initial_lr"] == 5e-5

    def test_policy_preset_ignored_when_disabled(self) -> None:
        class _PresetPolicy(_StubPolicy):
            name = "preset_stub2"

            def get_optimizer_preset(self) -> OptimizerConfig | None:
                return OptimizerConfig(lr=5e-5)

        policy = _PresetPolicy()
        config = TrainingConfig(optimizer=OptimizerConfig(lr=1e-4), use_policy_preset=False)
        opt, _ = make_optimizer_and_scheduler(policy, config, num_training_steps=10_000)
        # Preset should be ignored, config lr used
        assert opt.param_groups[0]["initial_lr"] == 1e-4
