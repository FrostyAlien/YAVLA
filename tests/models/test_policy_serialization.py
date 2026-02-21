"""Round-trip serialization tests for VLAPolicy save/load."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from yavla.models.config import PolicyConfig
from yavla.models.decoder import SimpleActionDecoder
from yavla.models.encoders.proprio import ProprioEncoder, ProprioEncoderConfig
from yavla.models.heads.mlp import MLPHeadConfig, MLPRegressionHead
from yavla.models.merger import ConcatMerger, TokenMergerConfig
from yavla.models.policy import VLAPolicy, _dict_to_config, _filter_known_fields, _tensor_to_list
from yavla.models.types import ActionSpaceSpec, BackboneOutput, FreezeConfig


# -- Helpers ---------------------------------------------------------------


def _make_policy(D: int = 64, N_READOUT: int = 8, CHUNK: int = 5, ADIM: int = 7) -> VLAPolicy:
    """Build a VLAPolicy with mocked backbone (no PaliGemma download)."""
    B = 1

    # Mock vision encoder
    vision = MagicMock()
    vision.encode_images = MagicMock(return_value=torch.randn(B, 4, D))

    # Real proprio encoder
    proprio = ProprioEncoder(ProprioEncoderConfig(proprio_dim=7, backbone_dim=D))

    # Real merger
    merger = ConcatMerger(TokenMergerConfig(num_readout_tokens=N_READOUT), backbone_dim=D)

    # Mock backbone (plain — no peft)
    backbone = MagicMock()
    backbone.base_model = MagicMock()
    backbone.base_model.get_input_embeddings.return_value = MagicMock(
        side_effect=lambda ids: torch.randn(ids.shape[0], ids.shape[1], D)
    )
    backbone.model = MagicMock()
    # No peft_config by default → _has_lora returns False
    del backbone.model.peft_config
    tok_output = {"input_ids": torch.ones(B, 3, dtype=torch.long), "attention_mask": torch.ones(B, 3)}
    backbone.tokenizer = MagicMock(return_value=tok_output)
    backbone.return_value = BackboneOutput(
        readout_states=torch.randn(B, N_READOUT, D),
        token_states=torch.randn(B, 16, D),
        attention_mask=torch.ones(B, 16),
    )

    # Real action head
    head = MLPRegressionHead(
        MLPHeadConfig(hidden_dim=32, num_blocks=1, chunk_len=CHUNK, action_dim=ADIM),
        backbone_dim=D,
    )

    # Real decoder
    limits = torch.tensor([[-1.0, 1.0]] * ADIM)
    spec = ActionSpaceSpec(names=["x"] * ADIM, units=["m"] * ADIM, limits=limits)
    decoder = SimpleActionDecoder(action_space_spec=spec, dt_hz=10.0)

    cfg = PolicyConfig()
    return VLAPolicy(vision, proprio, merger, backbone, head, decoder, cfg)


# -- Tests for helper functions -------------------------------------------


class TestTensorToList:
    def test_converts_tensor(self) -> None:
        d: dict = {"a": torch.tensor([1.0, 2.0]), "b": "hello"}
        _tensor_to_list(d)
        assert d["a"] == [1.0, 2.0]
        assert d["b"] == "hello"

    def test_nested(self) -> None:
        d: dict = {"outer": {"inner": torch.tensor(3)}}
        _tensor_to_list(d)
        assert d["outer"]["inner"] == 3

    def test_empty(self) -> None:
        d: dict = {}
        _tensor_to_list(d)
        assert d == {}


class TestDictToConfig:
    def test_defaults(self) -> None:
        cfg = _dict_to_config({})
        assert cfg.config_version == "1.0"
        assert cfg.dt_hz == 10.0

    def test_roundtrip(self) -> None:
        """Config → dict → Config should preserve values."""
        import dataclasses

        original = PolicyConfig(dt_hz=5.0, config_version="2.0")
        d = dataclasses.asdict(original)
        _tensor_to_list(d)
        restored = _dict_to_config(d)
        assert restored.dt_hz == original.dt_hz
        assert restored.config_version == original.config_version
        assert restored.action_head.action_dim == original.action_head.action_dim

    def test_with_limits(self) -> None:
        d = {
            "action_space": {
                "names": ["x"],
                "units": ["m"],
                "limits": torch.tensor([[0.0, 1.0]]),
            }
        }
        cfg = _dict_to_config(d)
        assert cfg.action_space.limits is not None
        assert cfg.action_space.limits.shape == (1, 2)

    def test_unknown_keys_dropped(self) -> None:
        """Extra keys from a 'future' config version should be dropped, not crash."""
        d = {
            "backbone": {"vlm_name": "google/paligemma-3b-pt-224", "future_flag": True},
            "merger": {"type": "concat", "num_readout_tokens": 32, "new_field": 42},
        }
        cfg = _dict_to_config(d)
        assert cfg.backbone.vlm_name == "google/paligemma-3b-pt-224"
        assert cfg.merger.num_readout_tokens == 32
        # Unknown keys should not appear
        assert not hasattr(cfg.backbone, "future_flag")
        assert not hasattr(cfg.merger, "new_field")

    def test_unknown_keys_logged(self, caplog) -> None:
        """Dropped keys should produce a warning log."""
        import logging

        d = {"backbone": {"unknown_key": "value"}}
        with caplog.at_level(logging.WARNING):
            _dict_to_config(d)
        assert "unknown_key" in caplog.text
        assert "BackboneConfig" in caplog.text


class TestFilterKnownFields:
    def test_keeps_known(self) -> None:
        from yavla.models.merger import TokenMergerConfig

        result = _filter_known_fields(TokenMergerConfig, {"type": "concat", "num_readout_tokens": 32})
        assert result == {"type": "concat", "num_readout_tokens": 32}

    def test_drops_unknown(self) -> None:
        from yavla.models.merger import TokenMergerConfig

        result = _filter_known_fields(TokenMergerConfig, {"type": "concat", "future_param": 99})
        assert result == {"type": "concat"}

    def test_empty_dict(self) -> None:
        from yavla.models.merger import TokenMergerConfig

        result = _filter_known_fields(TokenMergerConfig, {})
        assert result == {}


# -- Tests for save_pretrained/from_pretrained ----------------------------


class TestSavePretrainedFiles:
    """Verify save_pretrained writes the expected files."""

    def test_non_lora_files(self) -> None:
        policy = _make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            policy.save_pretrained(tmpdir)
            p = Path(tmpdir)
            assert (p / "config.json").exists()
            assert (p / "model.safetensors").exists()
            assert (p / "checkpoint_meta.json").exists()
            assert (p / "action_stats.json").exists()
            assert (p / "embodiment.json").exists()
            # No adapter dir when LoRA is not active
            assert not (p / "adapter").exists()
            assert not (p / "non_vlm_weights.safetensors").exists()

    def test_checkpoint_meta_no_lora(self) -> None:
        policy = _make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            policy.save_pretrained(tmpdir)
            with open(Path(tmpdir) / "checkpoint_meta.json") as f:
                meta = json.load(f)
            assert meta["has_lora"] is False

    def test_config_json_roundtrip(self) -> None:
        policy = _make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            policy.save_pretrained(tmpdir)
            with open(Path(tmpdir) / "config.json") as f:
                cfg_dict = json.load(f)
            # Should be JSON-serializable (no tensors)
            assert isinstance(cfg_dict, dict)
            assert cfg_dict["config_version"] == "1.0"

    def test_embodiment_json_content(self) -> None:
        policy = _make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            policy.save_pretrained(tmpdir)
            with open(Path(tmpdir) / "embodiment.json") as f:
                emb = json.load(f)
            assert emb["action_dim"] == 7
            assert emb["proprio_dim"] == 7

    def test_weights_loadable(self) -> None:
        """Verify saved safetensors can be loaded back."""
        from safetensors.torch import load_file

        policy = _make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            policy.save_pretrained(tmpdir)
            state = load_file(str(Path(tmpdir) / "model.safetensors"))
            assert len(state) > 0
            # Keys should match policy's state_dict
            assert set(state.keys()) == set(policy.state_dict().keys())


class TestHasLora:
    def test_no_lora(self) -> None:
        policy = _make_policy()
        assert policy._has_lora() is False

    def test_with_lora(self) -> None:
        policy = _make_policy()
        policy.backbone.model.peft_config = {"default": MagicMock()}
        assert policy._has_lora() is True
