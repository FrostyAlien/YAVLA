"""Tests for training loop and Trainer class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
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
from yavla.training.config import OptimizerConfig, TrainingConfig
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
        batch_size = len(texts)
        return torch.zeros(batch_size, 1, 4), torch.ones(batch_size, 1)

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


class _BatchReadingPolicy(PolicyBase):
    name = "batch-reading"
    config_class = _StubPolicyConfig

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _StubBackbone()
        self.head = nn.Linear(4, 2)
        self.last_batch: TrainingBatch | None = None

    def forward(self, batch: TrainingBatch) -> LossDict:
        self.last_batch = batch

        device = self.head.weight.device
        assert batch.actions.device == device
        assert batch.observations.proprio.device == device
        assert all(image.device == device for image in batch.observations.images.values())
        if batch.action_mask is not None:
            assert batch.action_mask.device == device

        camera = batch.observations.images["cam0"]
        image_feature = camera.mean(dim=(1, 2, 3))
        proprio_feature = batch.observations.proprio.mean(dim=1)
        action_feature = batch.actions.mean(dim=(1, 2))
        mask_feature = (
            batch.action_mask.float().mean(dim=1)
            if batch.action_mask is not None
            else torch.zeros_like(action_feature)
        )
        x = torch.stack([image_feature, proprio_feature, action_feature, mask_feature], dim=1)
        return LossDict(total=self.head(x).pow(2).mean())

    def predict(self, obs: ObservationBatch) -> ActionChunk:
        return ActionChunk(actions=torch.zeros(1, 1, 2), dt_hz=10.0, chunk_len=1)


class _ScalarLossPolicy(PolicyBase):
    name = "scalar-loss"
    config_class = _StubPolicyConfig

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _StubBackbone()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: TrainingBatch) -> LossDict:
        marker = batch.observations.proprio[:, 0].mean()
        loss = self.scale * marker
        return LossDict(total=loss, breakdown={"marker": loss})

    def predict(self, obs: ObservationBatch) -> ActionChunk:
        return ActionChunk(actions=torch.zeros(1, 1, 2), dt_hz=10.0, chunk_len=1)


def _make_training_batch() -> TrainingBatch:
    return TrainingBatch(
        observations=ObservationBatch(
            images={"cam0": torch.randn(2, 3, 8, 8)},
            proprio=torch.randn(2, 4),
            language=["pick up the cube", "place the cube"],
            timestamps=torch.tensor([0.0, 0.1]),
            masks=torch.tensor([True, False]),
        ),
        actions=torch.randn(2, 3, 2),
        dt_hz=10.0,
        chunk_len=3,
        action_mask=torch.tensor([[False, False, True], [False, False, False]]),
        action_dim_mask=torch.tensor([False, True]),
    )


def _make_value_batch(
    *,
    batch_size: int,
    marker: float,
    chunk_len: int = 3,
    action_dim: int = 2,
    include_masks: bool = False,
) -> TrainingBatch:
    action_mask = None
    action_dim_mask = None
    if include_masks:
        action_mask = torch.zeros(batch_size, chunk_len, dtype=torch.bool)
        action_mask[0, -1] = True
        action_dim_mask = torch.tensor([False, True], dtype=torch.bool)

    return TrainingBatch(
        observations=ObservationBatch(
            images={},
            proprio=torch.full((batch_size, 4), marker),
            language=["task"] * batch_size,
        ),
        actions=torch.zeros(batch_size, chunk_len, action_dim),
        dt_hz=10.0,
        chunk_len=chunk_len,
        action_mask=action_mask,
        action_dim_mask=action_dim_mask,
    )


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

    def test_moves_typed_batch_to_accelerator_device(self) -> None:
        from accelerate import Accelerator

        accelerator = Accelerator(cpu=True)
        policy = _BatchReadingPolicy()
        config = TrainingConfig()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
        policy = accelerator.prepare(policy)

        batch = _make_training_batch()
        with patch.object(batch, "to", wraps=batch.to) as moved:
            loss_dict, grad_norm = train_step(policy, batch, accelerator, optimizer, config)

        moved.assert_called_once_with(accelerator.device, non_blocking=False)
        assert isinstance(loss_dict, LossDict)
        assert isinstance(grad_norm, float)

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

    def test_run_completes_one_step_with_typed_batch_consumption(self, tmp_path: Any) -> None:
        from yavla.training.trainer import Trainer

        policy = _BatchReadingPolicy()
        config = TrainingConfig(
            num_steps=1,
            log_freq=1,
            save_freq=100,
            precision="no",
            wandb=False,
            output_dir=str(tmp_path / "out"),
        )
        dl = torch.utils.data.DataLoader([_make_training_batch(), _make_training_batch()], batch_size=None)

        trainer = Trainer(policy, config, dl)
        trainer.run()

        assert policy.last_batch is not None
        assert policy.last_batch.actions.device == trainer.accelerator.device
        assert policy.last_batch.observations.proprio.device == trainer.accelerator.device

    def test_wandb_logs_tracker_config_and_model_metrics(self, tmp_path: Any) -> None:
        from yavla.training.trainer import Trainer

        policy = _ScalarLossPolicy()
        config = TrainingConfig(
            num_steps=1,
            log_freq=1,
            save_freq=100,
            precision="no",
            wandb=True,
            output_dir=str(tmp_path / "out"),
            optimizer=OptimizerConfig(lr=0.0),
        )
        tracker_config = {"training": {"num_steps": 1}, "policy": {"name": "scalar-loss"}}
        dl = torch.utils.data.DataLoader([_make_value_batch(batch_size=2, marker=1.0)], batch_size=None)
        trainer = Trainer(policy, config, dl, tracker_config=tracker_config)

        with (
            patch.object(trainer.accelerator, "init_trackers") as init_trackers,
            patch.object(trainer.accelerator, "log") as log_metrics,
            patch.object(trainer.accelerator, "end_training"),
            patch.object(trainer, "save_checkpoint"),
        ):
            trainer.run()

        init_trackers.assert_called_once_with("yavla", config=tracker_config)
        first_log = log_metrics.call_args_list[0]
        assert first_log.kwargs["step"] == 0
        assert "model/params_total" in first_log.args[0]
        assert "model/params_trainable_fraction" in first_log.args[0]

    def test_wandb_logs_epoch_and_advances_data_epoch_on_rollover(self, tmp_path: Any) -> None:
        from yavla.training.trainer import Trainer

        policy = _ScalarLossPolicy()
        config = TrainingConfig(
            num_steps=3,
            log_freq=1,
            save_freq=100,
            precision="no",
            wandb=True,
            output_dir=str(tmp_path / "out"),
            optimizer=OptimizerConfig(lr=0.0),
        )
        batches = [
            _make_value_batch(batch_size=2, marker=1.0),
            _make_value_batch(batch_size=2, marker=2.0),
        ]
        dl = torch.utils.data.DataLoader(batches, batch_size=None)
        trainer = Trainer(policy, config, dl)

        with (
            patch("yavla.training.trainer.advance_data_epoch") as advance_epoch,
            patch.object(trainer.accelerator, "init_trackers"),
            patch.object(trainer.accelerator, "log") as log_metrics,
            patch.object(trainer.accelerator, "end_training"),
            patch.object(trainer, "save_checkpoint"),
        ):
            trainer.run()

        train_logs = [call.args[0] for call in log_metrics.call_args_list if "train/epoch" in call.args[0]]
        assert [metrics["train/epoch"] for metrics in train_logs] == [0.5, 1.0, 1.5]
        assert advance_epoch.call_args_list == [
            call(trainer.train_dataloader, 0),
            call(trainer.train_dataloader, 1),
        ]

    def test_resume_derives_epoch_and_in_epoch_offset_from_dataloader_length(self, tmp_path: Any) -> None:
        from yavla.training.trainer import Trainer

        policy = _ScalarLossPolicy()
        config = TrainingConfig(
            num_steps=5,
            log_freq=1,
            save_freq=100,
            precision="no",
            wandb=True,
            output_dir=str(tmp_path / "out"),
            optimizer=OptimizerConfig(lr=0.0),
            resume=True,
        )
        batches = [
            _make_value_batch(batch_size=2, marker=1.0),
            _make_value_batch(batch_size=2, marker=2.0),
        ]
        dl = torch.utils.data.DataLoader(batches, batch_size=None)
        trainer = Trainer(policy, config, dl)

        with (
            patch.object(trainer, "_load_latest_checkpoint", return_value=3),
            patch("yavla.training.trainer.advance_data_epoch") as advance_epoch,
            patch.object(trainer.accelerator, "init_trackers"),
            patch.object(trainer.accelerator, "log") as log_metrics,
            patch.object(trainer.accelerator, "end_training"),
            patch.object(trainer, "save_checkpoint"),
        ):
            trainer.run()

        train_logs = [call.args[0] for call in log_metrics.call_args_list if "train/epoch" in call.args[0]]
        assert [metrics["train/epoch"] for metrics in train_logs] == [2.0, 2.5]
        assert advance_epoch.call_args_list == [
            call(trainer.train_dataloader, 1),
            call(trainer.train_dataloader, 2),
        ]

    def test_gradient_accumulation_logs_weighted_losses(self, tmp_path: Any) -> None:
        from yavla.training.trainer import Trainer

        policy = _ScalarLossPolicy()
        config = TrainingConfig(
            num_steps=1,
            log_freq=1,
            save_freq=100,
            precision="no",
            wandb=True,
            output_dir=str(tmp_path / "out"),
            gradient_accumulation_steps=2,
            optimizer=OptimizerConfig(lr=0.0),
        )
        dl = torch.utils.data.DataLoader(
            [
                _make_value_batch(batch_size=1, marker=1.0),
                _make_value_batch(batch_size=3, marker=3.0),
            ],
            batch_size=None,
        )
        trainer = Trainer(policy, config, dl)

        with (
            patch.object(trainer.accelerator, "init_trackers"),
            patch.object(trainer.accelerator, "log") as log_metrics,
            patch.object(trainer.accelerator, "end_training"),
            patch.object(trainer, "save_checkpoint"),
        ):
            trainer.run()

        train_log = next(call.args[0] for call in log_metrics.call_args_list if "train/loss" in call.args[0])
        assert train_log["train/loss"] == pytest.approx(2.5)
        assert train_log["train/marker"] == pytest.approx(2.5)
        assert train_log["train/global_batch_size"] == pytest.approx(4.0)
        assert train_log["train/samples_seen"] == pytest.approx(4.0)
        assert train_log["train/action_valid_fraction"] == pytest.approx(1.0)

    def test_wandb_logs_performance_metrics(self, tmp_path: Any) -> None:
        from yavla.training.trainer import Trainer

        policy = _ScalarLossPolicy()
        config = TrainingConfig(
            num_steps=1,
            log_freq=1,
            save_freq=100,
            precision="no",
            wandb=True,
            output_dir=str(tmp_path / "out"),
            optimizer=OptimizerConfig(lr=0.0),
        )
        dl = torch.utils.data.DataLoader([_make_value_batch(batch_size=2, marker=1.0)], batch_size=None)
        trainer = Trainer(policy, config, dl)

        with (
            patch.object(trainer.accelerator, "init_trackers"),
            patch.object(trainer.accelerator, "log") as log_metrics,
            patch.object(trainer.accelerator, "end_training"),
            patch.object(trainer, "save_checkpoint"),
            patch("yavla.training.trainer.time.perf_counter", side_effect=[0.0, 0.1, 0.3, 1.0]),
        ):
            trainer.run()

        train_log = next(call.args[0] for call in log_metrics.call_args_list if "train/loss" in call.args[0])
        assert train_log["perf/step_time_s"] == pytest.approx(1.0)
        assert train_log["perf/samples_per_sec"] == pytest.approx(2.0)
        assert train_log["perf/data_wait_time_s"] == pytest.approx(0.2)
        assert train_log["perf/compute_time_s"] == pytest.approx(0.8)
        assert train_log["perf/data_wait_fraction"] == pytest.approx(0.2)

    def test_gradient_accumulation_logs_optimizer_step_performance(self, tmp_path: Any) -> None:
        from yavla.training.trainer import Trainer

        policy = _ScalarLossPolicy()
        config = TrainingConfig(
            num_steps=1,
            log_freq=1,
            save_freq=100,
            precision="no",
            wandb=True,
            output_dir=str(tmp_path / "out"),
            gradient_accumulation_steps=2,
            optimizer=OptimizerConfig(lr=0.0),
        )
        dl = torch.utils.data.DataLoader(
            [
                _make_value_batch(batch_size=1, marker=1.0),
                _make_value_batch(batch_size=3, marker=3.0),
            ],
            batch_size=None,
        )
        trainer = Trainer(policy, config, dl)

        with (
            patch.object(trainer.accelerator, "init_trackers"),
            patch.object(trainer.accelerator, "log") as log_metrics,
            patch.object(trainer.accelerator, "end_training"),
            patch.object(trainer, "save_checkpoint"),
            patch(
                "yavla.training.trainer.time.perf_counter",
                side_effect=[0.0, 0.1, 0.3, 0.4, 0.8, 1.5],
            ),
        ):
            trainer.run()

        train_log = next(call.args[0] for call in log_metrics.call_args_list if "train/loss" in call.args[0])
        assert train_log["perf/step_time_s"] == pytest.approx(1.5)
        assert train_log["perf/samples_per_sec"] == pytest.approx(4.0 / 1.5)
        assert train_log["perf/data_wait_time_s"] == pytest.approx(0.6)
        assert train_log["perf/compute_time_s"] == pytest.approx(0.9)
        assert train_log["perf/data_wait_fraction"] == pytest.approx(0.4)

    def test_performance_metrics_are_windowed_on_log_cadence(self, tmp_path: Any) -> None:
        from yavla.training.trainer import Trainer

        policy = _ScalarLossPolicy()
        config = TrainingConfig(
            num_steps=3,
            log_freq=2,
            save_freq=100,
            precision="no",
            wandb=True,
            output_dir=str(tmp_path / "out"),
            optimizer=OptimizerConfig(lr=0.0),
        )
        dl = torch.utils.data.DataLoader(
            [
                _make_value_batch(batch_size=2, marker=1.0),
                _make_value_batch(batch_size=2, marker=2.0),
                _make_value_batch(batch_size=2, marker=3.0),
            ],
            batch_size=None,
        )
        trainer = Trainer(policy, config, dl)

        with (
            patch.object(trainer.accelerator, "init_trackers"),
            patch.object(trainer.accelerator, "log") as log_metrics,
            patch.object(trainer.accelerator, "end_training"),
            patch.object(trainer, "save_checkpoint"),
            patch(
                "yavla.training.trainer.time.perf_counter",
                side_effect=[
                    0.0,
                    0.1,
                    0.2,
                    0.8,
                    1.0,
                    1.2,
                    1.5,
                    2.0,
                    3.0,
                    3.1,
                    3.2,
                    3.8,
                ],
            ),
        ):
            trainer.run()

        train_logs = [call.args[0] for call in log_metrics.call_args_list if "train/loss" in call.args[0]]
        assert len(train_logs) == 1
        train_log = train_logs[0]
        assert train_log["perf/step_time_s"] == pytest.approx(0.9)
        assert train_log["perf/samples_per_sec"] == pytest.approx(4.0 / 1.8)
        assert train_log["perf/data_wait_time_s"] == pytest.approx(0.2)
        assert train_log["perf/compute_time_s"] == pytest.approx(0.7)
        assert train_log["perf/data_wait_fraction"] == pytest.approx(0.4 / 1.8)

    def test_typed_batch_logs_action_coverage_metrics(self, tmp_path: Any) -> None:
        from yavla.training.trainer import Trainer

        policy = _ScalarLossPolicy()
        config = TrainingConfig(
            num_steps=1,
            log_freq=1,
            save_freq=100,
            precision="no",
            wandb=True,
            output_dir=str(tmp_path / "out"),
            optimizer=OptimizerConfig(lr=0.0),
        )
        dl = torch.utils.data.DataLoader(
            [_make_value_batch(batch_size=2, marker=1.0, include_masks=True)],
            batch_size=None,
        )
        trainer = Trainer(policy, config, dl)

        with (
            patch.object(trainer.accelerator, "init_trackers"),
            patch.object(trainer.accelerator, "log") as log_metrics,
            patch.object(trainer.accelerator, "end_training"),
            patch.object(trainer, "save_checkpoint"),
        ):
            trainer.run()

        train_log = next(call.args[0] for call in log_metrics.call_args_list if "train/loss" in call.args[0])
        assert train_log["train/action_valid_fraction"] == pytest.approx(5.0 / 6.0)
        assert train_log["train/action_dim_active_fraction"] == pytest.approx(0.5)

    def test_console_step_log_adds_epoch_only(self, tmp_path: Any) -> None:
        from yavla.training.trainer import Trainer

        policy = _ScalarLossPolicy()
        config = TrainingConfig(
            num_steps=1,
            log_freq=1,
            save_freq=100,
            precision="no",
            wandb=False,
            output_dir=str(tmp_path / "out"),
            optimizer=OptimizerConfig(lr=0.0),
        )
        dl = torch.utils.data.DataLoader(
            [
                _make_value_batch(batch_size=2, marker=1.0),
                _make_value_batch(batch_size=2, marker=2.0),
            ],
            batch_size=None,
        )
        trainer = Trainer(policy, config, dl)

        with (
            patch.object(trainer.accelerator, "print") as print_step,
            patch.object(trainer, "save_checkpoint"),
        ):
            trainer.run()

        step_lines = [call.args[0] for call in print_step.call_args_list if str(call.args[0]).startswith("step ")]
        assert step_lines == ["step 1/1  epoch=0.5  loss=1.0000  lr=0.00e+00  grad_norm=1.00"]

    def test_generic_batch_logging_omits_training_batch_specific_metrics(self, tmp_path: Any) -> None:
        from yavla.training.trainer import Trainer

        policy = _StubPolicy()
        config = TrainingConfig(
            num_steps=1,
            log_freq=1,
            save_freq=100,
            precision="no",
            wandb=True,
            output_dir=str(tmp_path / "out"),
        )
        dl = torch.utils.data.DataLoader([torch.ones(4)] * 4, batch_size=2)
        trainer = Trainer(policy, config, dl)

        with (
            patch.object(trainer.accelerator, "init_trackers"),
            patch.object(trainer.accelerator, "log") as log_metrics,
            patch.object(trainer.accelerator, "end_training"),
            patch.object(trainer, "save_checkpoint"),
        ):
            trainer.run()

        train_log = next(call.args[0] for call in log_metrics.call_args_list if "train/loss" in call.args[0])
        assert "train/action_valid_fraction" not in train_log
        assert "train/action_dim_active_fraction" not in train_log
