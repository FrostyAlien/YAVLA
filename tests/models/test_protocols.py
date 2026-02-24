"""Unit tests for protocols, ABCs, and capability validation."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from yavla.models.types import (
    ActionChunk,
    ActionPrediction,
    ActionSpaceSpec,
    BackboneOutput,
    LossDict,
    TrainingBatch,
)
from yavla.models.protocols import (
    ActionDecoderProto,
    ActionHeadBase,
    ActionHeadProto,
    ActionHeadRequirements,
    BackboneBase,
    BackboneCapabilities,
    BackboneProto,
    IncompatibleError,
    IntegrationMode,
    ProprioEncoderBase,
    ProprioEncoderProto,
    TokenMergerProto,
    VisionEncoderBase,
    VisionEncoderProto,
    validate_integration,
)
from typing import Any


# --- Concrete stubs for ABC tests ---


class _StubVisionEncoder(VisionEncoderBase):
    @property
    def output_dim(self) -> int:
        return 2048

    @property
    def num_patches(self) -> int:
        return 256

    def encode_images(self, images: dict[str, Tensor]) -> Tensor:
        v = next(iter(images.values()))
        return torch.randn(v.shape[0], 256, 2048)


class _StubBackbone(BackboneBase):
    @property
    def capabilities(self) -> BackboneCapabilities:
        return BackboneCapabilities(supported_modes={IntegrationMode.READOUT})

    @property
    def hidden_dim(self) -> int:
        return 2048

    def embed_language(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        B = len(texts)
        return torch.zeros(B, 1, 2048), torch.ones(B, 1)

    def forward(self, inputs_embeds: Tensor, attention_mask: Tensor, token_type_ids: Tensor) -> BackboneOutput:
        B = inputs_embeds.shape[0]
        return BackboneOutput(
            readout_states=torch.randn(B, 64, 2048),
            token_states=None,
            attention_mask=attention_mask,
        )


class _StubActionHead(ActionHeadBase):
    @property
    def requirements(self) -> ActionHeadRequirements:
        return ActionHeadRequirements(required_mode=IntegrationMode.READOUT)

    def compute_loss(self, backbone_output: BackboneOutput, batch: TrainingBatch) -> LossDict:
        return LossDict(total=torch.tensor(0.0))

    def predict(self, backbone_output: BackboneOutput) -> ActionPrediction:
        return ActionPrediction(mean=torch.randn(1, 5, 7))


class _StubProprioEncoder(ProprioEncoderBase):
    @property
    def output_dim(self) -> int:
        return 2048

    def encode_proprio(self, proprio: Tensor) -> Tensor:
        return torch.randn(proprio.shape[0], 1, 2048)


class TestIntegrationMode:
    def test_values(self) -> None:
        assert IntegrationMode.READOUT.value == "readout"
        assert IntegrationMode.JOINT_TOKENS.value == "joint_tokens"


class TestProtocolConformance:
    def test_vision_encoder_proto(self) -> None:
        enc = _StubVisionEncoder()
        assert isinstance(enc, VisionEncoderProto)

    def test_backbone_proto(self) -> None:
        bb = _StubBackbone()
        assert isinstance(bb, BackboneProto)

    def test_backbone_embed_language_contract(self) -> None:
        bb = _StubBackbone()
        embeds, mask = bb.embed_language(["hello", "world"])
        assert embeds.shape[0] == 2
        assert embeds.shape[2] == bb.hidden_dim
        assert mask.shape == embeds.shape[:2]

    def test_action_head_proto(self) -> None:
        head = _StubActionHead()
        assert isinstance(head, ActionHeadProto)

    def test_proprio_encoder_proto(self) -> None:
        enc = _StubProprioEncoder()
        assert isinstance(enc, ProprioEncoderProto)


class TestABCEnforcement:
    def test_incomplete_action_head_raises(self) -> None:
        class _Incomplete(ActionHeadBase):
            @property
            def requirements(self) -> ActionHeadRequirements:
                return ActionHeadRequirements(required_mode=IntegrationMode.READOUT)

            def predict(self, backbone_output: BackboneOutput) -> ActionPrediction:
                return ActionPrediction(mean=torch.randn(1, 5, 7))

        with pytest.raises(TypeError):
            _Incomplete()

    def test_incomplete_backbone_raises(self) -> None:
        class _Incomplete(BackboneBase):
            pass

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]


class TestValidateIntegration:
    def test_compatible_readout(self) -> None:
        bb = _StubBackbone()
        head = _StubActionHead()
        mode = validate_integration(bb, head)
        assert mode == IntegrationMode.READOUT

    def test_incompatible_joint_tokens(self) -> None:
        bb = _StubBackbone()

        class _JointHead(ActionHeadBase):
            @property
            def requirements(self) -> ActionHeadRequirements:
                return ActionHeadRequirements(required_mode=IntegrationMode.JOINT_TOKENS, accepts_readout=False)

            def compute_loss(self, backbone_output: BackboneOutput, batch: TrainingBatch) -> LossDict:
                return LossDict(total=torch.tensor(0.0))

            def predict(self, backbone_output: BackboneOutput) -> ActionPrediction:
                return ActionPrediction(mean=torch.randn(1, 5, 7))

        with pytest.raises(IncompatibleError):
            validate_integration(bb, _JointHead())
