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
        spec = ActionSpaceSpec(names=["x"], units=["m"], limits=limits)
        dec = SimpleActionDecoder(action_space_spec=spec)
        # Input 2.0 unnormalized without clamping:
        # (2.0 + 1) / 2 * 10 + 0 = 1.5 * 10 = 15.0
        pred = ActionPrediction(mean=torch.tensor([[[2.0]]]))
        chunk = dec.decode(pred)
        assert chunk.actions[0, 0, 0].item() == pytest.approx(15.0, abs=1e-5)

    def test_unnormalize_negative_out_of_range(self) -> None:
        limits = torch.tensor([[0.0, 10.0]])
        spec = ActionSpaceSpec(names=["x"], units=["m"], limits=limits)
        dec = SimpleActionDecoder(action_space_spec=spec)
        # Input -2.0 without clamping: (-2.0 + 1) / 2 * 10 + 0 = -0.5 * 10 = -5.0
        pred = ActionPrediction(mean=torch.tensor([[[-2.0]]]))
        chunk = dec.decode(pred)
        assert chunk.actions[0, 0, 0].item() == pytest.approx(-5.0, abs=1e-5)

    def test_action_space_spec_rejects_clip_unnormalized(self) -> None:
        with pytest.raises(TypeError):
            ActionSpaceSpec(names=["x"], units=["m"], limits=None, clip_unnormalized=False)


class TestPolicyConfig:
    def test_defaults(self) -> None:
        cfg = PolicyConfig()
        assert cfg.config_version == "1.0"
        assert cfg.dt_hz == 10.0
        assert cfg.backbone.type == "paligemma"
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

    def _make_policy(self, n_img: int | None = None) -> "VLAPolicy":
        from yavla.models.policy import VLAPolicy
        from yavla.models.backbones.paligemma import PaliGemmaVisionEncoder
        from yavla.models.encoders.proprio import ProprioEncoder, ProprioEncoderConfig
        from yavla.models.merger import ConcatMerger, TokenMergerConfig
        from yavla.models.heads.mlp import MLPRegressionHead, MLPHeadConfig

        D = self.D
        n_img = self.N_IMG if n_img is None else n_img

        # Mock vision encoder
        vision = MagicMock(spec=PaliGemmaVisionEncoder)
        vision.encode_images = MagicMock(return_value=torch.randn(self.B, n_img, D))

        # Real proprio encoder
        proprio = ProprioEncoder(ProprioEncoderConfig(proprio_dim=7, backbone_dim=D))

        # Real merger
        merger = ConcatMerger(TokenMergerConfig(num_readout_tokens=self.N_READOUT), backbone_dim=D)

        # Mock backbone
        backbone = MagicMock()
        backbone.embed_language = MagicMock(
            return_value=(torch.randn(self.B, 3, D), torch.ones(self.B, 3))
        )

        total_seq = n_img + 1 + 3 + self.N_READOUT
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

    def test_forward_accepts_multi_camera_images(self) -> None:
        num_patches_per_camera = 4
        images = {
            "cam_left": torch.randn(self.B, 3, 224, 224),
            "cam_right": torch.randn(self.B, 3, 224, 224),
        }
        n_img = len(images) * num_patches_per_camera

        policy = self._make_policy(n_img=n_img)
        obs = ObservationBatch(
            images=images,
            proprio=torch.randn(self.B, 7),
            language=["pick up the cup"] * self.B,
        )
        batch = TrainingBatch(
            observations=obs,
            actions=torch.randn(self.B, self.CHUNK, self.ADIM),
            dt_hz=10.0,
            chunk_len=self.CHUNK,
        )

        loss = policy.forward(batch)
        assert isinstance(loss, LossDict)

        policy.vision_encoder.encode_images.assert_called_once()
        passed_images = policy.vision_encoder.encode_images.call_args[0][0]
        assert set(passed_images.keys()) == set(images.keys())

        expected_seq = n_img + 1 + 3 + self.N_READOUT
        inputs_embeds = policy.backbone.call_args[0][0]
        assert inputs_embeds.shape == (self.B, expected_seq, self.D)

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

    def test_encode_observations_delegates_to_embed_language(self) -> None:
        """encode_observations() must call backbone.embed_language(), not access tokenizer."""
        policy = self._make_policy()
        obs = self._make_obs()
        policy.encode_observations(obs)
        policy.backbone.embed_language.assert_called_once()
        args = policy.backbone.embed_language.call_args[0][0]
        assert isinstance(args, list)
        assert len(args) == self.B

    def test_encode_observations_broadcasts_str_language(self) -> None:
        """A single string language instruction is broadcast to batch size."""
        policy = self._make_policy()
        obs = ObservationBatch(
            images={"cam": torch.randn(self.B, 3, 224, 224)},
            proprio=torch.randn(self.B, 7),
            language="pick up the cup",
        )
        policy.encode_observations(obs)
        args = policy.backbone.embed_language.call_args[0][0]
        assert args == ["pick up the cup"] * self.B

    def test_encode_observations_handles_none_language(self) -> None:
        """language=None is normalized to [''] * batch_size."""
        policy = self._make_policy()
        obs = ObservationBatch(
            images={"cam": torch.randn(self.B, 3, 224, 224)},
            proprio=torch.randn(self.B, 7),
            language=None,
        )
        policy.encode_observations(obs)
        args = policy.backbone.embed_language.call_args[0][0]
        assert args == [""] * self.B

    def test_encode_observations_passes_list_through(self) -> None:
        """A pre-formed list[str] is forwarded to embed_language unchanged."""
        policy = self._make_policy()
        langs = ["pick up the cup", "open the drawer"]
        obs = ObservationBatch(
            images={"cam": torch.randn(self.B, 3, 224, 224)},
            proprio=torch.randn(self.B, 7),
            language=langs,
        )
        policy.encode_observations(obs)
        args = policy.backbone.embed_language.call_args[0][0]
        assert args == langs

    def test_encode_observations_empty_string_language(self) -> None:
        """language='' (falsy string) is broadcast to [''] * batch_size."""
        policy = self._make_policy()
        obs = ObservationBatch(
            images={"cam": torch.randn(self.B, 3, 224, 224)},
            proprio=torch.randn(self.B, 7),
            language="",
        )
        policy.encode_observations(obs)
        args = policy.backbone.embed_language.call_args[0][0]
        assert args == [""] * self.B


class TestPolicyBaseEnforcement:
    """Test __init_subclass__ contract enforcement."""

    def test_missing_name_raises(self) -> None:
        from yavla.models.protocols import PolicyBase

        with pytest.raises(TypeError, match="must define 'name'"):

            class BadPolicy(PolicyBase):
                config_class = PolicyConfig

                def forward(self, batch): ...

                def predict(self, obs): ...

    def test_missing_config_class_raises(self) -> None:
        from yavla.models.protocols import PolicyBase

        with pytest.raises(TypeError, match="must define 'config_class'"):

            class BadPolicy(PolicyBase):
                name = "bad"

                def forward(self, batch): ...

                def predict(self, obs): ...
