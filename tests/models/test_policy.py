"""Unit tests for SimpleActionDecoder and VLAPolicy (mocked components)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
import pytest

from yavla.models.decoder import SimpleActionDecoder
from yavla.models.config import PolicyConfig
from yavla.models.types import (
    ActionChunk,
    ActionPrediction,
    ActionSpaceSpec,
    BackboneOutput,
    LossDict,
    ObservationBatch,
    ProprioSpec,
    TrainingBatch,
)


class TestSimpleActionDecoder:
    def test_passthrough_no_limits(self) -> None:
        spec = ActionSpaceSpec(names=["x"], units=["m"], limits=None)
        dec = SimpleActionDecoder(action_space_spec=spec, dt_hz=10.0)
        pred = ActionPrediction(mean=torch.tensor([[[0.5, -0.3]]]))
        chunk = dec.decode(pred)
        assert isinstance(chunk, ActionChunk)
        assert torch.allclose(chunk.actions, pred.mean)
        assert chunk.dt_hz == 10.0
        assert chunk.chunk_len == 1

    def test_unnormalize_with_limits(self) -> None:
        # limits: dim0 in [0, 10], dim1 in [-5, 5]
        limits = torch.tensor([[0.0, 10.0], [-5.0, 5.0]])
        spec = ActionSpaceSpec(names=["a", "b"], units=["m", "rad"], limits=limits)
        dec = SimpleActionDecoder(action_space_spec=spec, dt_hz=5.0)
        # Input in [-1, 1]: -1 → lo, +1 → hi
        pred = ActionPrediction(mean=torch.tensor([[[-1.0, 1.0]]]))
        chunk = dec.decode(pred)
        assert chunk.actions[0, 0, 0].item() == pytest.approx(0.0, abs=1e-5)
        assert chunk.actions[0, 0, 1].item() == pytest.approx(5.0, abs=1e-5)

    def test_unnormalize_midpoint(self) -> None:
        limits = torch.tensor([[0.0, 10.0]])
        spec = ActionSpaceSpec(names=["x"], units=["m"], limits=limits)
        dec = SimpleActionDecoder(action_space_spec=spec)
        pred = ActionPrediction(mean=torch.tensor([[[0.0]]]))  # midpoint
        chunk = dec.decode(pred)
        assert chunk.actions[0, 0, 0].item() == pytest.approx(5.0, abs=1e-5)

    def test_action_space_spec_property(self) -> None:
        spec = ActionSpaceSpec(names=[], units=[], limits=None)
        dec = SimpleActionDecoder(action_space_spec=spec)
        assert dec.action_space_spec is spec



    def test_unnormalize_without_clamping(self) -> None:
        limits = torch.tensor([[0.0, 10.0]])
        spec = ActionSpaceSpec(names=["x"], units=["m"], limits=limits, clip_unnormalized=False)
        dec = SimpleActionDecoder(action_space_spec=spec)
        # Input 2.0 unnormalized without clamping:
        # (2.0 + 1) / 2 * 10 + 0 = 1.5 * 10 = 15.0
        pred = ActionPrediction(mean=torch.tensor([[[2.0]]]))
        chunk = dec.decode(pred)
        assert chunk.actions[0, 0, 0].item() == pytest.approx(15.0, abs=1e-5)

class TestPolicyConfig:
    def test_defaults(self) -> None:
        cfg = PolicyConfig()
        assert cfg.config_version == "1.0"
        assert cfg.dt_hz == 10.0
        assert cfg.backbone.type == "vlm"
        assert cfg.action_head.type == "mlp"
        assert cfg.merger.type == "concat"
        assert cfg.vision_encoder.type == "paligemma_siglip"
        assert cfg.proprio_encoder.type == "linear"

    def test_sub_config_independence(self) -> None:
        c1 = PolicyConfig()
        c2 = PolicyConfig()
        c1.action_head.action_dim = 99
        assert c2.action_head.action_dim == 7  # default, not mutated


class TestVLAPolicy:
    """Test VLAPolicy with mocked subcomponents (no real PaliGemma)."""

    B, D, N_IMG, N_READOUT, CHUNK, ADIM = 2, 64, 4, 8, 5, 7

    def _make_policy(self) -> "VLAPolicy":
        from yavla.models.policy import VLAPolicy
        from yavla.models.encoders.vision import PaliGemmaVisionEncoder, VisionEncoderConfig
        from yavla.models.encoders.proprio import ProprioEncoder, ProprioEncoderConfig
        from yavla.models.merger import ConcatMerger, TokenMergerConfig
        from yavla.models.heads.mlp import MLPRegressionHead, MLPHeadConfig

        D = self.D

        # Mock vision encoder
        vision = MagicMock(spec=PaliGemmaVisionEncoder)
        vision.encode_images = MagicMock(return_value=torch.randn(self.B, self.N_IMG, D))

        # Real proprio encoder
        proprio = ProprioEncoder(ProprioEncoderConfig(proprio_dim=7, backbone_dim=D))

        # Real merger
        merger = ConcatMerger(TokenMergerConfig(num_readout_tokens=self.N_READOUT), backbone_dim=D)

        # Mock backbone
        backbone = MagicMock()
        backbone.base_model = MagicMock()
        backbone.base_model.get_input_embeddings.return_value = MagicMock(
            side_effect=lambda ids: torch.randn(ids.shape[0], ids.shape[1], D)
        )
        tok_output = {"input_ids": torch.ones(self.B, 3, dtype=torch.long), "attention_mask": torch.ones(self.B, 3)}
        backbone.tokenizer = MagicMock(return_value=tok_output)

        total_seq = self.N_IMG + 1 + 3 + self.N_READOUT
        backbone.return_value = BackboneOutput(
            readout_states=torch.randn(self.B, self.N_READOUT, D),
            token_states=torch.randn(self.B, total_seq, D),
            attention_mask=torch.ones(self.B, total_seq),
        )

        # Real action head
        head = MLPRegressionHead(
            MLPHeadConfig(hidden_dim=32, num_blocks=1, chunk_len=self.CHUNK, action_dim=self.ADIM),
            backbone_dim=D,
        )

        # Real decoder
        spec = ActionSpaceSpec(names=[], units=[], limits=None)
        decoder = SimpleActionDecoder(action_space_spec=spec, dt_hz=10.0)

        cfg = PolicyConfig()
        return VLAPolicy(vision, proprio, merger, backbone, head, decoder, cfg)

    def _make_obs(self) -> ObservationBatch:
        return ObservationBatch(
            images={"cam": torch.randn(self.B, 3, 224, 224)},
            proprio=torch.randn(self.B, 7),
            language=["pick up the cup"] * self.B,
        )

    def test_forward_returns_loss(self) -> None:
        policy = self._make_policy()
        obs = self._make_obs()
        batch = TrainingBatch(
            observations=obs,
            actions=torch.randn(self.B, self.CHUNK, self.ADIM),
            dt_hz=10.0,
            chunk_len=self.CHUNK,
        )
        loss = policy.forward(batch)
        assert isinstance(loss, LossDict)
        assert loss.total.shape == ()
        assert "l1" in loss.breakdown

    def test_predict_returns_action_chunk(self) -> None:
        policy = self._make_policy()
        obs = self._make_obs()
        chunk = policy.predict(obs)
        assert isinstance(chunk, ActionChunk)
        assert chunk.actions.shape == (self.B, self.CHUNK, self.ADIM)
        assert chunk.dt_hz == 10.0

    def test_submodule_access(self) -> None:
        policy = self._make_policy()
        assert policy.vision_encoder is not None
        assert policy.proprio_encoder is not None
        assert policy.backbone is not None
        assert policy.action_head is not None
        assert policy.decoder is not None

    def test_is_policy_base(self) -> None:
        from yavla.models.protocols import PolicyBase

        policy = self._make_policy()
        assert isinstance(policy, PolicyBase)

    def test_has_overridable_steps(self) -> None:
        policy = self._make_policy()
        assert callable(getattr(policy, "encode_observations", None))
        assert callable(getattr(policy, "merge_tokens", None))
        assert callable(getattr(policy, "run_backbone", None))
        assert callable(getattr(policy, "compute_loss", None))
        assert callable(getattr(policy, "decode_prediction", None))

    def test_reset_is_noop(self) -> None:
        policy = self._make_policy()
        # Should not raise
        policy.reset()

    def test_name_and_config_class(self) -> None:
        from yavla.models.policy import VLAPolicy
        from yavla.models.config import PolicyConfig

        assert VLAPolicy.name == "vla"
        assert VLAPolicy.config_class is PolicyConfig


class TestPolicyBaseEnforcement:
    """Test __init_subclass__ contract enforcement."""

    def test_missing_name_raises(self) -> None:
        from yavla.models.protocols import PolicyBase

        with pytest.raises(TypeError, match="must define 'name'"):

            class BadPolicy(PolicyBase):
                config_class = PolicyConfig

                def forward(self, batch):
                    ...

                def predict(self, obs):
                    ...

    def test_missing_config_class_raises(self) -> None:
        from yavla.models.protocols import PolicyBase

        with pytest.raises(TypeError, match="must define 'config_class'"):

            class BadPolicy(PolicyBase):
                name = "bad"

                def forward(self, batch):
                    ...

                def predict(self, obs):
                    ...

