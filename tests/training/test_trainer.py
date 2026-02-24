"""Tests for training loop and Trainer class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

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
from yavla.training.config import TrainingConfig
from yavla.training.trainer import train_step


# -- Stubs --------------------------------------------------------------------


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
        return BackboneOutput(
            readout_states=None,
            token_states=inputs_embeds,
            attention_mask=attention_mask,
        )


class _StubPolicy(PolicyBase):
    name = "stub"
    config_class = _StubPolicyConfig

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _StubBackbone()
        self.head = nn.Linear(4, 2)

    def forward(self, batch: TrainingBatch) -> LossDict:
        # Simple loss that depends on parameters so gradients flow
        x = self.head(torch.ones(1, 4))
        return LossDict(total=x.sum())

    def predict(self, obs: ObservationBatch) -> ActionChunk:
        return ActionChunk(actions=torch.zeros(1, 1, 2), dt_hz=10.0, chunk_len=1)


# -- Tests --------------------------------------------------------------------


class TestTrainStep:
    def test_returns_loss_and_grad_norm(self) -> None:
        from accelerate import Accelerator

        accelerator = Accelerator(cpu=True)
        policy = _StubPolicy()
        config = TrainingConfig()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
        policy = accelerator.prepare(policy)

        batch = MagicMock()  # policy.forward ignores batch in stub
        loss_dict, grad_norm = train_step(policy, batch, accelerator, optimizer, config)

        assert isinstance(loss_dict, LossDict)
        assert loss_dict.total.ndim == 0  # scalar
        assert isinstance(grad_norm, float)
        assert grad_norm >= 0.0

    def test_parameters_updated(self) -> None:
        from accelerate import Accelerator

        accelerator = Accelerator(cpu=True)
        policy = _StubPolicy()
        config = TrainingConfig()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-2)
        policy = accelerator.prepare(policy)

        params_before = {n: p.clone() for n, p in policy.named_parameters()}
        batch = MagicMock()
        train_step(policy, batch, accelerator, optimizer, config)

        changed = False
        for n, p in policy.named_parameters():
            if not torch.equal(p, params_before[n]):
                changed = True
                break
        assert changed, "At least one parameter should have been updated"


class TestTrainer:
    def test_init_creates_accelerator(self) -> None:
        from yavla.training.trainer import Trainer

        policy = _StubPolicy()
        config = TrainingConfig(
            num_steps=10,
            precision="no",  # CPU, no mixed precision
            wandb=False,
        )
        dl = torch.utils.data.DataLoader([torch.ones(4)] * 20, batch_size=2)
        trainer = Trainer(policy, config, dl)

        assert trainer.accelerator is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_run_executes_steps(self) -> None:
        from yavla.training.trainer import Trainer

        policy = _StubPolicy()
        config = TrainingConfig(
            num_steps=5,
            log_freq=2,
            save_freq=100,  # don't save during this short test
            precision="no",
            wandb=False,
        )
        dl = torch.utils.data.DataLoader([torch.ones(4)] * 20, batch_size=2)
        trainer = Trainer(policy, config, dl)

        # Patch train_step to count calls
        call_count = 0
        original_train_step = train_step

        def counting_train_step(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            return original_train_step(*args, **kwargs)

        with patch("yavla.training.trainer.train_step", side_effect=counting_train_step):
            trainer.run()

        assert call_count == 5

    def test_save_checkpoint_creates_dir(self, tmp_path: Any) -> None:
        from yavla.training.trainer import Trainer

        policy = _StubPolicy()
        config = TrainingConfig(
            num_steps=10, precision="no", wandb=False,
            output_dir=str(tmp_path / "out"),
        )
        dl = torch.utils.data.DataLoader([torch.ones(4)] * 20, batch_size=2)
        trainer = Trainer(policy, config, dl)
        trainer.save_checkpoint(100)
        assert (tmp_path / "out" / "checkpoint-100").is_dir()

    def test_load_latest_checkpoint_no_dir_returns_zero(self, tmp_path: Any) -> None:
        from yavla.training.trainer import Trainer

        policy = _StubPolicy()
        config = TrainingConfig(
            num_steps=10, precision="no", wandb=False,
            output_dir=str(tmp_path / "nonexistent"),
        )
        dl = torch.utils.data.DataLoader([torch.ones(4)] * 20, batch_size=2)
        trainer = Trainer(policy, config, dl)
        assert trainer._load_latest_checkpoint() == 0

    def test_load_latest_checkpoint_picks_highest(self, tmp_path: Any) -> None:
        from yavla.training.trainer import Trainer

        policy = _StubPolicy()
        out = tmp_path / "out"
        config = TrainingConfig(
            num_steps=10, precision="no", wandb=False,
            output_dir=str(out),
        )
        dl = torch.utils.data.DataLoader([torch.ones(4)] * 20, batch_size=2)
        trainer = Trainer(policy, config, dl)

        # Save two checkpoints
        trainer.save_checkpoint(100)
        trainer.save_checkpoint(200)

        # Should pick 200
        with patch.object(trainer.accelerator, "load_state") as mock_load:
            step = trainer._load_latest_checkpoint()
        assert step == 200
        mock_load.assert_called_once_with(str(out / "checkpoint-200"))

    def test_resume_continues_from_checkpoint(self, tmp_path: Any) -> None:
        from yavla.training.trainer import Trainer

        policy = _StubPolicy()
        out = tmp_path / "out"
        config = TrainingConfig(
            num_steps=5, save_freq=3, log_freq=100,
            precision="no", wandb=False, output_dir=str(out),
        )
        dl = torch.utils.data.DataLoader([torch.ones(4)] * 40, batch_size=2)

        # Run first trainer for 5 steps (saves at step 3)
        Trainer(policy, config, dl).run()
        assert (out / "checkpoint-3").is_dir()

        # Resume with more steps — checkpoint-5 exists, so run steps 6-8
        policy2 = _StubPolicy()
        config2 = TrainingConfig(
            num_steps=8, save_freq=100, log_freq=100,
            precision="no", wandb=False, output_dir=str(out), resume=True,
        )
        trainer2 = Trainer(policy2, config2, dl)

        call_count = 0
        original = train_step

        def counting(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        with patch("yavla.training.trainer.train_step", side_effect=counting):
            trainer2.run()

        # Steps 6, 7, 8 remain (5 already done)
        assert call_count == 3

    def test_gradient_accumulation_step_counting(self) -> None:
        from yavla.training.trainer import Trainer

        policy = _StubPolicy()
        config = TrainingConfig(
            num_steps=4, gradient_accumulation_steps=2,
            log_freq=100, save_freq=100,
            precision="no", wandb=False,
        )
        dl = torch.utils.data.DataLoader([torch.ones(4)] * 40, batch_size=2)
        trainer = Trainer(policy, config, dl)

        call_count = 0
        original = train_step

        def counting(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        with patch("yavla.training.trainer.train_step", side_effect=counting):
            trainer.run()

        # 4 optimizer steps * 2 accumulation = 8 micro-batches
        assert call_count == 8
