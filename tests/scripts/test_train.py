"""Tests for the training CLI helpers in scripts/train.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import scripts.train as train_script
import torch

from yavla.data.factory import DataConfig
from yavla.models.config import EmbodimentConfig, PolicyConfig
from yavla.models.heads.mlp import MLPHeadConfig
from yavla.models.types import ObservationBatch, TrainingBatch
from yavla.training.config import TrainingConfig


def _make_train_config(
    *,
    dataset_chunk_len: int | None = 5,
    chunk_len: int = 5,
    action_dim: int = 7,
    proprio_dim: int = 7,
) -> train_script.TrainConfig:
    return train_script.TrainConfig(
        training=TrainingConfig(
            dataset=DataConfig(
                repo_id="dummy/repo",
                action_chunk_size=dataset_chunk_len,
            )
        ),
        policy=PolicyConfig(
            action_head=MLPHeadConfig(chunk_len=chunk_len),
            embodiment=EmbodimentConfig(action_dim=action_dim, proprio_dim=proprio_dim),
        ),
    )


def _make_batch(*, chunk_len: int = 5, action_dim: int = 7, proprio_dim: int = 7) -> TrainingBatch:
    return TrainingBatch(
        observations=ObservationBatch(images={}, proprio=torch.zeros(2, proprio_dim)),
        actions=torch.zeros(2, chunk_len, action_dim),
        dt_hz=10.0,
        chunk_len=chunk_len,
    )


def test_pop_config_flag_loads_nested_train_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "nested.yaml"
    config_path.write_text(
        """
training:
  dataset:
    repo_id: "lerobot/aloha_sim"
    action_chunk_size: 5
    batch_size: 2
  optimizer:
    betas: [0.8, 0.88]
  num_steps: 10
policy:
  embodiment:
    mode: max_padded
    action_dim: 14
    proprio_dim: 14
    max_action_dim: 32
    max_proprio_dim: 32
  action_head:
    chunk_len: 5
  dt_hz: 20.0
""".strip()
    )
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])

    cfg = train_script._pop_config_flag()

    assert cfg is not None
    assert cfg.training.dataset.batch_size == 2
    assert cfg.training.optimizer.betas == (0.8, 0.88)
    assert cfg.training.num_steps == 10
    assert cfg.policy.action_dim == 14
    assert cfg.policy.proprio_dim == 14
    assert cfg.policy.max_action_dim == 32
    assert cfg.policy.max_proprio_dim == 32
    assert cfg.policy.action_head.action_dim == 32
    assert cfg.policy.proprio_encoder.proprio_dim == 32
    assert cfg.policy.dt_hz == 20.0


def test_pop_config_flag_rejects_legacy_flat_training_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "flat.yaml"
    config_path.write_text(
        """
dataset:
  repo_id: "lerobot/aloha_sim"
  action_chunk_size: 5
  batch_size: 4
optimizer:
  betas: [0.7, 0.95]
num_steps: 12
""".strip()
    )
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])

    with pytest.raises(SystemExit, match="legacy flat train config format is not supported"):
        train_script._pop_config_flag()


@pytest.mark.parametrize(
    ("cfg", "batch", "message"),
    [
        (
            _make_train_config(chunk_len=5),
            _make_batch(chunk_len=4),
            "policy.action_head.chunk_len=5, got 4",
        ),
        (
            _make_train_config(action_dim=7),
            _make_batch(action_dim=8),
            "policy.embodiment.action_dim=7, got 8",
        ),
        (
            _make_train_config(proprio_dim=7),
            _make_batch(proprio_dim=9),
            "policy.embodiment.proprio_dim=7, got 9",
        ),
    ],
)
def test_validate_training_dimensions_exits_on_first_batch_mismatch(
    cfg: train_script.TrainConfig,
    batch: TrainingBatch,
    message: str,
) -> None:
    with pytest.raises(SystemExit, match=message):
        train_script._validate_training_dimensions(cfg, batch)
