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

    def test_embed_language_returns_embeddings_and_mask(self) -> None:
        model, tok = _make_mock_paligemma()
        D = 2048
        # Mock tokenizer output
        tok.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [1, 2, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        }
        # Mock embedding layer
        embed_layer = MagicMock(return_value=torch.randn(2, 3, D))
        model.get_input_embeddings = MagicMock(return_value=embed_layer)

        bb = PaliGemmaBackbone(model, tok, num_readout_tokens=64)
        # The mock model isn't a real nn.Module, so nn.Module.parameters() is empty.
        # Register a dummy parameter so embed_language's next(self.parameters()).device works.
        bb.register_parameter("_dummy", torch.nn.Parameter(torch.empty(1)))
        embeds, mask = bb.embed_language(["hello world", "hi"])

        tok.assert_called_once()
        assert embeds.shape == (2, 3, D)
        assert mask.shape == (2, 3)
        assert mask[0].tolist() == [1, 1, 1]
        assert mask[1].tolist() == [1, 1, 0]
