"""Unit tests for the MLP action head."""

from __future__ import annotations

import torch
import pytest

from yavla.models.heads.mlp import (
    MLPHeadConfig,
    MLPRegressionHead,
    ResidualMLP,
    head_registry,
)
from yavla.models.protocols import ActionHeadRequirements, IntegrationMode
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

    def _make_backbone_output(self, B: int = 2, N_readout: int = 64, D: int = 128) -> BackboneOutput:
        return BackboneOutput(
            readout_states=torch.randn(B, N_readout, D),
            token_states=None,
            attention_mask=torch.ones(B, N_readout),
        )

    def _make_training_batch(self, B: int = 2, chunk_len: int = 5, action_dim: int = 7) -> TrainingBatch:
        obs = ObservationBatch(images={}, proprio=torch.zeros(B, 7))
        return TrainingBatch(
            observations=obs,
            actions=torch.randn(B, chunk_len, action_dim),
            dt_hz=10.0,
            chunk_len=chunk_len,
        )

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
