"""Tests for training config dataclasses (Phase 1)."""

from __future__ import annotations

from yavla.training.config import OptimizerConfig, SchedulerConfig, TrainingConfig


class TestOptimizerConfig:
    def test_defaults(self) -> None:
        cfg = OptimizerConfig()
        assert cfg.name == "AdamW"
        assert cfg.lr == 1e-4
        assert cfg.weight_decay == 0.01
        assert cfg.betas == (0.9, 0.999)
        assert cfg.eps == 1e-8
        assert cfg.grad_clip_norm == 1.0
        assert cfg.backbone_lr_scale == 0.1


class TestSchedulerConfig:
    def test_defaults(self) -> None:
        cfg = SchedulerConfig()
        assert cfg.name == "cosine"
        assert cfg.warmup_steps == 1000
        assert cfg.min_lr_ratio == 0.1


class TestTrainingConfig:
    def test_defaults(self) -> None:
        cfg = TrainingConfig()
        assert isinstance(cfg.optimizer, OptimizerConfig)
        assert isinstance(cfg.scheduler, SchedulerConfig)
        assert cfg.precision == "bf16"
        assert cfg.num_steps == 100_000
        assert cfg.log_freq == 100
        assert cfg.save_freq == 5000
        assert cfg.output_dir == "outputs/train"
        assert cfg.resume is False
        assert cfg.gradient_checkpointing is True
        assert cfg.use_policy_preset is True
        assert cfg.vlm_image_height_override is None
        assert cfg.vlm_image_width_override is None
        assert cfg.wandb is False
        assert cfg.gradient_accumulation_steps == 1

    def test_backward_compat_dataset(self) -> None:
        """TrainingConfig still has dataset and viz fields."""
        cfg = TrainingConfig()
        assert cfg.dataset.repo_id == "lerobot/aloha_sim"
        assert cfg.viz is not None


class TestPolicyBasePreset:
    def test_default_returns_none(self) -> None:
        from yavla.models.protocols import PolicyBase

        # PolicyBase.get_optimizer_preset default returns None
        assert PolicyBase.get_optimizer_preset(None) is None  # type: ignore[arg-type]
