"""Unit tests for the MLP action head."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from yavla.models.heads.mlp import (
    MLPHeadConfig,
    MLPRegressionHead,
    ResidualMLP,
    head_registry,
)
from yavla.models.protocols import IntegrationMode
from yavla.models.types import (
    ActionPrediction,
    BackboneOutput,
    LossDict,
    ObservationBatch,
    TrainingBatch,
)


class TestMLPHeadConfig:
    def test_defaults(self) -> None:
        cfg = MLPHeadConfig()
        assert cfg.type == "mlp"
        assert cfg.hidden_dim == 1024
        assert cfg.num_blocks == 2
        assert cfg.chunk_len == 5
        assert cfg.action_dim == 7


class TestResidualMLP:
    def test_output_shape(self) -> None:
        net = ResidualMLP(num_blocks=2, input_dim=128, hidden_dim=64, output_dim=35)
        x = torch.randn(4, 128)
        out = net(x)
        assert out.shape == (4, 35)

    def test_zero_blocks(self) -> None:
        net = ResidualMLP(num_blocks=0, input_dim=32, hidden_dim=16, output_dim=7)
        x = torch.randn(2, 32)
        out = net(x)
        assert out.shape == (2, 7)

    def test_gradient_flow(self) -> None:
        net = ResidualMLP(num_blocks=2, input_dim=64, hidden_dim=32, output_dim=10)
        x = torch.randn(2, 64, requires_grad=True)
        out = net(x).sum()
        out.backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 64)


class TestMLPRegressionHead:
    def _make_head(self, backbone_dim: int = 128, chunk_len: int = 5, action_dim: int = 7) -> MLPRegressionHead:
        cfg = MLPHeadConfig(hidden_dim=64, num_blocks=2, chunk_len=chunk_len, action_dim=action_dim)
        return MLPRegressionHead(config=cfg, backbone_dim=backbone_dim)

    def _make_backbone_output(self, batch_size: int = 2, n_readout: int = 64, dim: int = 128) -> BackboneOutput:
        return BackboneOutput(
            readout_states=torch.randn(batch_size, n_readout, dim),
            token_states=None,
            attention_mask=torch.ones(batch_size, n_readout),
        )

    def _make_training_batch(
        self, batch_size: int = 2, chunk_len: int = 5, action_dim: int = 7
    ) -> TrainingBatch:
        obs = ObservationBatch(images={}, proprio=torch.zeros(batch_size, 7))
        return TrainingBatch(
            observations=obs,
            actions=torch.randn(batch_size, chunk_len, action_dim),
            dt_hz=10.0,
            chunk_len=chunk_len,
        )

    def _assert_backward_safe_zero_loss(
        self,
        head: MLPRegressionHead,
        bo: BackboneOutput,
        batch: TrainingBatch,
        predicted: torch.Tensor,
    ) -> None:
        with patch.object(head, "_pool_and_predict", return_value=predicted):
            loss = head.compute_loss(bo, batch)

        assert torch.isfinite(loss.total)
        assert loss.total.item() == pytest.approx(0.0, abs=1e-6)
        assert loss.total.requires_grad

        loss.total.backward()

        assert predicted.grad is not None
        assert predicted.grad.shape == predicted.shape
        assert torch.allclose(predicted.grad, torch.zeros_like(predicted.grad))

    def test_requirements(self) -> None:
        head = self._make_head()
        req = head.requirements
        assert req.required_mode == IntegrationMode.READOUT
        assert req.accepts_readout is True

    def test_predict_shape(self) -> None:
        head = self._make_head()
        bo = self._make_backbone_output()
        pred = head.predict(bo)
        assert isinstance(pred, ActionPrediction)
        assert pred.mean.shape == (2, 5, 7)

    def test_compute_loss(self) -> None:
        head = self._make_head()
        bo = self._make_backbone_output()
        batch = self._make_training_batch()
        loss = head.compute_loss(bo, batch)
        assert isinstance(loss, LossDict)
        assert loss.total.shape == ()
        assert "l1" in loss.breakdown

    def test_loss_zero_on_perfect_prediction(self) -> None:
        head = self._make_head(backbone_dim=128, chunk_len=5, action_dim=7)
        # Force net to output zeros by zeroing all parameters
        with torch.no_grad():
            for p in head.parameters():
                p.zero_()
        bo = self._make_backbone_output()
        batch = self._make_training_batch()
        batch.actions.zero_()
        loss = head.compute_loss(bo, batch)
        assert loss.total.item() == pytest.approx(0.0, abs=1e-6)

    def test_compute_loss_masks_padded_timesteps(self) -> None:
        head = self._make_head(backbone_dim=128, chunk_len=3, action_dim=2)
        bo = self._make_backbone_output(batch_size=1)
        batch = TrainingBatch(
            observations=ObservationBatch(images={}, proprio=torch.zeros(1, 7)),
            actions=torch.tensor([[[1.0, 3.0], [5.0, 7.0], [100.0, 100.0]]]),
            dt_hz=10.0,
            chunk_len=3,
            action_mask=torch.tensor([[False, False, True]]),
        )
        predicted = torch.zeros(1, 3, 2)

        with patch.object(head, "_pool_and_predict", return_value=predicted):
            loss = head.compute_loss(bo, batch)

        assert loss.total.item() == pytest.approx(4.0, abs=1e-6)

    def test_compute_loss_returns_zero_for_fully_masked_chunk(self) -> None:
        head = self._make_head(backbone_dim=128, chunk_len=3, action_dim=2)
        bo = self._make_backbone_output(batch_size=1)
        batch = TrainingBatch(
            observations=ObservationBatch(images={}, proprio=torch.zeros(1, 7)),
            actions=torch.ones(1, 3, 2),
            dt_hz=10.0,
            chunk_len=3,
            action_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        predicted = torch.zeros(1, 3, 2, requires_grad=True)

        self._assert_backward_safe_zero_loss(head, bo, batch, predicted)

    def test_compute_loss_masks_inactive_action_dimensions(self) -> None:
        head = self._make_head(backbone_dim=128, chunk_len=2, action_dim=4)
        bo = self._make_backbone_output(batch_size=1)
        batch = TrainingBatch(
            observations=ObservationBatch(images={}, proprio=torch.zeros(1, 7)),
            actions=torch.tensor([[[1.0, 3.0, 100.0, 100.0], [5.0, 7.0, 100.0, 100.0]]]),
            dt_hz=10.0,
            chunk_len=2,
            action_dim_mask=torch.tensor([False, False, True, True]),
        )

        with patch.object(head, "_pool_and_predict", return_value=torch.zeros(1, 2, 4)):
            loss = head.compute_loss(bo, batch)

        assert loss.total.item() == pytest.approx(4.0, abs=1e-6)

    def test_compute_loss_combines_timestep_and_dimension_masks(self) -> None:
        head = self._make_head(backbone_dim=128, chunk_len=3, action_dim=4)
        bo = self._make_backbone_output(batch_size=1)
        batch = TrainingBatch(
            observations=ObservationBatch(images={}, proprio=torch.zeros(1, 7)),
            actions=torch.tensor(
                [
                    [
                        [1.0, 3.0, 100.0, 100.0],
                        [5.0, 7.0, 100.0, 100.0],
                        [9.0, 11.0, 100.0, 100.0],
                    ]
                ]
            ),
            dt_hz=10.0,
            chunk_len=3,
            action_mask=torch.tensor([[False, True, False]]),
            action_dim_mask=torch.tensor([False, False, True, True]),
        )

        with patch.object(head, "_pool_and_predict", return_value=torch.zeros(1, 3, 4)):
            loss = head.compute_loss(bo, batch)

        assert loss.total.item() == pytest.approx(6.0, abs=1e-6)

    def test_compute_loss_returns_zero_for_fully_dimension_masked_chunk(self) -> None:
        head = self._make_head(backbone_dim=128, chunk_len=2, action_dim=3)
        bo = self._make_backbone_output(batch_size=1)
        batch = TrainingBatch(
            observations=ObservationBatch(images={}, proprio=torch.zeros(1, 7)),
            actions=torch.ones(1, 2, 3),
            dt_hz=10.0,
            chunk_len=2,
            action_dim_mask=torch.ones(3, dtype=torch.bool),
        )
        predicted = torch.zeros(1, 2, 3, requires_grad=True)

        self._assert_backward_safe_zero_loss(head, bo, batch, predicted)

    def test_compute_loss_returns_zero_for_fully_combined_masked_chunk(self) -> None:
        head = self._make_head(backbone_dim=128, chunk_len=2, action_dim=3)
        bo = self._make_backbone_output(batch_size=1)
        batch = TrainingBatch(
            observations=ObservationBatch(images={}, proprio=torch.zeros(1, 7)),
            actions=torch.ones(1, 2, 3),
            dt_hz=10.0,
            chunk_len=2,
            action_mask=torch.tensor([[False, True]]),
            action_dim_mask=torch.ones(3, dtype=torch.bool),
        )
        predicted = torch.zeros(1, 2, 3, requires_grad=True)

        self._assert_backward_safe_zero_loss(head, bo, batch, predicted)

    def test_compute_loss_rejects_chunk_length_mismatch(self) -> None:
        head = self._make_head(chunk_len=5, action_dim=7)
        bo = self._make_backbone_output()
        batch = self._make_training_batch(chunk_len=4, action_dim=7)

        with pytest.raises(ValueError, match="chunk length mismatch: expected 5, got 4"):
            head.compute_loss(bo, batch)

    def test_compute_loss_rejects_action_dimension_mismatch(self) -> None:
        head = self._make_head(chunk_len=5, action_dim=7)
        bo = self._make_backbone_output()
        batch = self._make_training_batch(chunk_len=5, action_dim=8)

        with pytest.raises(ValueError, match="action dimension mismatch: expected 7, got 8"):
            head.compute_loss(bo, batch)

    def test_compute_loss_rejects_action_dimension_mask_shape_mismatch(self) -> None:
        head = self._make_head(chunk_len=5, action_dim=7)
        bo = self._make_backbone_output()
        batch = self._make_training_batch(chunk_len=5, action_dim=7)
        batch.action_dim_mask = torch.zeros(5, dtype=torch.bool)

        with pytest.raises(
            ValueError,
            match="dimension mask shape mismatch: expected torch.Size\\(\\[7\\]\\), got \\(5,\\)",
        ):
            head.compute_loss(bo, batch)

    def test_no_readout_raises(self) -> None:
        head = self._make_head()
        bo = BackboneOutput(readout_states=None, token_states=None, attention_mask=torch.ones(2, 10))
        with pytest.raises(ValueError, match="requires readout_states"):
            head.predict(bo)

    def test_empty_readout_raises(self) -> None:
        head = self._make_head()
        bo = BackboneOutput(readout_states=torch.empty(2, 0, 128), token_states=None, attention_mask=torch.ones(2, 0))
        with pytest.raises(ValueError, match="0 tokens"):
            head.predict(bo)

    def test_registry(self) -> None:
        assert "mlp" in head_registry.list()
