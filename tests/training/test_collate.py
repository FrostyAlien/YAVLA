"""Tests for TrainingCollate and create_training_dataloader collate wiring."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

from yavla.models.types import TrainingBatch
from yavla.training.data import TrainingCollate, create_training_dataloader


def _sample(
    *,
    proprio_dim: int = 7,
    action_shape: tuple[int, ...] = (5, 7),
    cam_names: tuple[str, ...] = ("top",),
    img_shape: tuple[int, ...] = (3, 64, 64),
    task: str | None = "pick up the cube",
    action_is_pad: tuple[int, ...] | None = None,
    extras: dict[str, object] | None = None,
) -> dict[str, object]:
    s: dict[str, object] = {
        "observation.state": torch.randn(proprio_dim),
        "action": torch.randn(*action_shape),
    }
    for cam in cam_names:
        s[f"observation.images.{cam}"] = torch.randn(*img_shape)
    if task is not None:
        s["task"] = task
    if action_is_pad is not None:
        s["action_is_pad"] = torch.tensor(action_is_pad, dtype=torch.bool)
    if extras:
        s.update(extras)
    return s


class TestTrainingCollate:
    """Tasks 3.1–3.5, 3.7, 3.8."""

    def _collate(self, samples: list[dict[str, object]], **kw: object) -> TrainingBatch:
        c = TrainingCollate(dt_hz=kw.get("dt_hz", 10.0), chunk_len=kw.get("chunk_len", 5))  # type: ignore[arg-type]
        return c(samples)  # type: ignore[arg-type]

    def test_standard_collation(self) -> None:
        """3.1: images, proprio, language, actions → correct shapes/types."""
        batch = self._collate([_sample(), _sample()])
        assert isinstance(batch, TrainingBatch)
        assert batch.observations.proprio.shape == (2, 7)
        assert batch.observations.images["top"].shape == (2, 3, 64, 64)
        assert batch.observations.language == ["pick up the cube", "pick up the cube"]
        assert batch.actions.shape == (2, 5, 7)

    def test_multiple_cameras(self) -> None:
        """3.2: two image keys → both in observations.images."""
        batch = self._collate([_sample(cam_names=("cam_left", "cam_right"))])
        assert "cam_left" in batch.observations.images
        assert "cam_right" in batch.observations.images

    def test_missing_optional_fields(self) -> None:
        """3.3: no task → language is None; no action_is_pad → action_mask is None."""
        batch = self._collate([_sample(task=None)])
        assert batch.observations.language is None
        assert batch.action_mask is None

    def test_dt_hz_and_chunk_len_passthrough(self) -> None:
        """3.4: constructor params appear on output."""
        batch = self._collate([_sample()], dt_hz=20.0, chunk_len=10)
        assert batch.dt_hz == 20.0
        assert batch.chunk_len == 10

    def test_extra_keys_ignored(self) -> None:
        """3.5: unknown keys silently ignored."""
        batch = self._collate([_sample(extras={"timestamp": torch.tensor(1.0), "episode_index": torch.tensor(0)})])
        assert isinstance(batch, TrainingBatch)

    def test_2d_actions_raise(self) -> None:
        """3.7: 2D actions raise ValueError mentioning action_chunk_size."""
        with pytest.raises(ValueError, match="action_chunk_size"):
            self._collate([_sample(action_shape=(7,))])

    def test_missing_proprio_raises(self) -> None:
        """3.8: missing observation.state raises ValueError."""
        s: dict[str, object] = {"action": torch.randn(5, 7), "observation.images.top": torch.randn(3, 64, 64)}
        with pytest.raises(ValueError, match="observation.state"):
            self._collate([s])

    def test_action_mask_present(self) -> None:
        """action_is_pad → action_mask with correct polarity."""
        batch = self._collate([_sample(action_is_pad=(True, False, False, True, False))])
        assert batch.action_mask is not None
        assert batch.action_mask.shape == (1, 5)
        assert batch.action_mask[0, 0].item() is True


class _DictDataset(torch.utils.data.Dataset):  # type: ignore[type-arg]
    """Synthetic dataset returning LeRobot-style dicts."""

    def __init__(self, n: int = 4) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, _idx: int) -> dict[str, object]:
        return _sample()


def test_create_training_dataloader_yields_training_batch() -> None:
    """3.6: create_training_dataloader yields TrainingBatch instances."""
    from yavla.training.config import TrainingConfig

    def _fake_create(_config, *, collate_fn=None):  # type: ignore[no-untyped-def]
        return DataLoader(_DictDataset(4), batch_size=2, collate_fn=collate_fn)

    cfg = TrainingConfig()
    with patch("yavla.training.data.create_dataloader", side_effect=_fake_create):
        loader = create_training_dataloader(cfg, dt_hz=10.0, chunk_len=5)
        batch = next(iter(loader))
        assert isinstance(batch, TrainingBatch)
