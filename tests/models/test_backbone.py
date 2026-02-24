"""Unit tests for the VLM backbone."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
import pytest

from yavla.models.config import BackboneConfig
from yavla.models.backbones.paligemma import PaliGemmaBackbone
from yavla.models.protocols import BackboneProto, IntegrationMode


def _make_mock_paligemma(hidden_size: int = 2048, num_readout: int = 64) -> tuple[MagicMock, MagicMock]:
    model = MagicMock()
    model.config.text_config.hidden_size = hidden_size

    def fake_forward(**kwargs):
        embeds = kwargs["inputs_embeds"]
        B, S, D = embeds.shape
        result = MagicMock()
        result.hidden_states = [torch.randn(B, S, D)]
        return result

    model.return_value = fake_forward
    model.side_effect = fake_forward
    tokenizer = MagicMock()
    return model, tokenizer


class TestBackboneConfig:
    def test_defaults(self) -> None:
        cfg = BackboneConfig()
        assert cfg.type == "paligemma"
        assert cfg.gradient_checkpointing is True


class TestPaliGemmaBackbone:
    def test_forward_readout_extraction(self) -> None:
        model, tok = _make_mock_paligemma()
        bb = PaliGemmaBackbone(model, tok, num_readout_tokens=64)
        B, S, D = 2, 341, 2048
        out = bb.forward(
            inputs_embeds=torch.randn(B, S, D),
            attention_mask=torch.ones(B, S),
            token_type_ids=torch.zeros(B, S, dtype=torch.long),
        )
        assert out.readout_states is not None
        assert out.readout_states.shape == (B, 64, D)

    def test_bypasses_vision_pipeline(self) -> None:
        model, tok = _make_mock_paligemma()
        bb = PaliGemmaBackbone(model, tok, num_readout_tokens=8)
        bb.forward(
            inputs_embeds=torch.randn(1, 20, 2048),
            attention_mask=torch.ones(1, 20),
            token_type_ids=torch.zeros(1, 20, dtype=torch.long),
        )
        assert model.call_count == 1
        _, kwargs = model.call_args
        assert kwargs["input_ids"] is None
        assert kwargs["pixel_values"] is None
        assert kwargs["output_hidden_states"] is True

    def test_capabilities(self) -> None:
        model, tok = _make_mock_paligemma()
        bb = PaliGemmaBackbone(model, tok, num_readout_tokens=64)
        caps = bb.capabilities
        assert IntegrationMode.READOUT in caps.supported_modes
        assert caps.supports_kv_cache is False

    def test_protocol_conformance(self) -> None:
        model, tok = _make_mock_paligemma()
        bb = PaliGemmaBackbone(model, tok, num_readout_tokens=64)
        assert isinstance(bb, BackboneProto)

    def test_base_model_property(self) -> None:
        model, tok = _make_mock_paligemma()
        bb = PaliGemmaBackbone(model, tok, num_readout_tokens=64)
        assert bb.base_model is model
        new_model = MagicMock()
        bb.base_model = new_model
        assert bb.base_model is new_model
