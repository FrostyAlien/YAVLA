"""Round-trip serialization tests for VLAPolicy save/load."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from yavla.models.config import EmbodimentConfig, PolicyConfig
from yavla.models.decoder import SimpleActionDecoder
from yavla.models.encoders.proprio import ProprioEncoder, ProprioEncoderConfig
from yavla.models.encoders.vision import MultiTowerVisionEncoderConfig, SimplePatchVisionEncoderConfig
from yavla.models.heads.mlp import MLPHeadConfig, MLPRegressionHead
from yavla.models.merger import ConcatMerger, TokenMergerConfig
from yavla.models.policy import VLAPolicy, _dict_to_config, _filter_known_fields, _tensor_to_list
from yavla.models.types import ActionSpaceSpec, BackboneOutput, ObservationBatch

# -- Helpers ---------------------------------------------------------------


def _make_policy(
    backbone_dim: int = 64,
    num_readout_tokens: int = 8,
    chunk_len: int = 5,
    *,
    config: PolicyConfig | None = None,
) -> VLAPolicy:
    """Build a VLAPolicy with mocked backbone (no PaliGemma download)."""
    batch_size = 1
    config = PolicyConfig() if config is None else config

    # Mock vision encoder
    vision = MagicMock()
    vision.encode_images = MagicMock(return_value=torch.randn(batch_size, 4, backbone_dim))

    # Real proprio encoder
    proprio = ProprioEncoder(ProprioEncoderConfig(proprio_dim=config.max_proprio_dim, backbone_dim=backbone_dim))

    # Real merger
    merger = ConcatMerger(TokenMergerConfig(num_readout_tokens=num_readout_tokens), backbone_dim=backbone_dim)

    # Mock backbone (plain — no peft)
    backbone = MagicMock()
    backbone.base_model = MagicMock()
    backbone.base_model.get_input_embeddings.return_value = MagicMock(
        side_effect=lambda ids: torch.randn(ids.shape[0], ids.shape[1], backbone_dim)
    )
    backbone.model = MagicMock()
    # No peft_config by default → _has_lora returns False
    del backbone.model.peft_config
    backbone.embed_language = MagicMock(
        return_value=(torch.randn(batch_size, 3, backbone_dim), torch.ones(batch_size, 3))
    )
    tok_output = {"input_ids": torch.ones(batch_size, 3, dtype=torch.long), "attention_mask": torch.ones(batch_size, 3)}
    backbone.tokenizer = MagicMock(return_value=tok_output)
    backbone.return_value = BackboneOutput(
        readout_states=torch.randn(batch_size, num_readout_tokens, backbone_dim),
        token_states=torch.randn(batch_size, 16, backbone_dim),
        attention_mask=torch.ones(batch_size, 16),
    )

    # Real action head
    head = MLPRegressionHead(
        MLPHeadConfig(hidden_dim=32, num_blocks=1, chunk_len=chunk_len, action_dim=config.max_action_dim),
        backbone_dim=backbone_dim,
    )

    # Real decoder
    spec = ActionSpaceSpec(names=["x"] * config.action_dim, units=["m"] * config.action_dim, limits=None)
    decoder = SimpleActionDecoder(action_space_spec=spec, dt_hz=10.0)

    return VLAPolicy(vision, proprio, merger, backbone, head, decoder, config)


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
    def test_missing_embodiment_block_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing the required 'embodiment' block"):
            _dict_to_config({})

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
            "embodiment": {
                "action_dim": 1,
                "proprio_dim": 7,
            },
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
            "embodiment": {"action_dim": 7, "proprio_dim": 7},
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

        d = {"embodiment": {"action_dim": 7, "proprio_dim": 7}, "backbone": {"unknown_key": "value"}}
        with caplog.at_level(logging.WARNING):
            _dict_to_config(d)
        assert "unknown_key" in caplog.text
        assert "BackboneConfig" in caplog.text

    def test_roundtrip_multi_tower_vision_config(self) -> None:
        import dataclasses

        original = PolicyConfig(
            vision_encoder=MultiTowerVisionEncoderConfig(
                towers=[
                    SimplePatchVisionEncoderConfig(image_size=32, patch_size=16, hidden_dim=8),
                    SimplePatchVisionEncoderConfig(image_size=32, patch_size=16, hidden_dim=12),
                ]
            )
        )
        d = dataclasses.asdict(original)
        restored = _dict_to_config(d)
        assert isinstance(restored.vision_encoder, MultiTowerVisionEncoderConfig)
        assert len(restored.vision_encoder.towers) == 2
        assert isinstance(restored.vision_encoder.towers[0], SimplePatchVisionEncoderConfig)

    def test_legacy_width_override_without_embodiment_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing the required 'embodiment' block"):
            _dict_to_config(
                {
                    "action_head": {"action_dim": 14},
                    "proprio_encoder": {"proprio_dim": 14},
                }
            )


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
            assert cfg_dict["config_version"] == "1.1"

    def test_embodiment_json_content(self) -> None:
        policy = _make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            policy.save_pretrained(tmpdir)
            with open(Path(tmpdir) / "embodiment.json") as f:
                emb = json.load(f)
            assert emb["mode"] == "exact"
            assert emb["action_dim"] == 7
            assert emb["proprio_dim"] == 7
            assert emb["max_action_dim"] == 7
            assert emb["max_proprio_dim"] == 7

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


class TestFromPretrained:
    def test_legacy_checkpoint_without_embodiment_block_is_rejected(self, tmp_path: Path) -> None:
        policy = _make_policy()
        policy.save_pretrained(tmp_path)

        legacy_config = {
            "vision_encoder": {"type": "from_backbone"},
            "proprio_encoder": {"type": "linear", "proprio_dim": 7, "backbone_dim": 64},
            "merger": {"type": "concat", "num_readout_tokens": 64},
            "backbone": {"type": "paligemma", "vlm_name": "google/paligemma-3b-pt-224"},
            "action_head": {"type": "mlp", "hidden_dim": 1024, "num_blocks": 2, "chunk_len": 5, "action_dim": 7},
            "freeze": {},
            "action_space": {"names": [], "units": [], "limits": None},
            "proprio": {"names": [], "units": [], "limits": None},
            "dt_hz": 10.0,
            "config_version": "1.0",
        }
        (tmp_path / "config.json").write_text(json.dumps(legacy_config, indent=2))

        with pytest.raises(ValueError, match="missing the required 'embodiment' block"):
            VLAPolicy.from_pretrained(tmp_path)

    def test_checkpoint_without_full_embodiment_metadata_is_rejected(self, tmp_path: Path) -> None:
        policy = _make_policy()
        policy.save_pretrained(tmp_path)
        (tmp_path / "embodiment.json").write_text(json.dumps({"action_dim": 7, "proprio_dim": 7}, indent=2))

        with pytest.raises(ValueError, match="legacy checkpoints are not supported"):
            VLAPolicy.from_pretrained(tmp_path)

    def test_strict_load_rejects_max_width_mismatch(self, tmp_path: Path) -> None:
        config = PolicyConfig(
            embodiment=EmbodimentConfig(
                mode="max_padded",
                action_dim=6,
                proprio_dim=6,
                max_action_dim=8,
                max_proprio_dim=8,
            )
        )
        _make_policy(config=config).save_pretrained(tmp_path)

        target = PolicyConfig(
            embodiment=EmbodimentConfig(
                mode="max_padded",
                action_dim=6,
                proprio_dim=6,
                max_action_dim=10,
                max_proprio_dim=8,
            )
        )

        with pytest.raises(ValueError, match="Checkpoint max action width mismatch"):
            VLAPolicy.from_pretrained(tmp_path, config=target, strict=True)

    def test_non_strict_load_allows_smaller_compatible_embodiment(self, tmp_path: Path) -> None:
        checkpoint_config = PolicyConfig(
            embodiment=EmbodimentConfig(
                mode="max_padded",
                action_dim=6,
                proprio_dim=6,
                max_action_dim=8,
                max_proprio_dim=8,
            )
        )
        source = _make_policy(config=checkpoint_config)
        source.save_pretrained(tmp_path)

        target = PolicyConfig(
            embodiment=EmbodimentConfig(
                mode="max_padded",
                action_dim=4,
                proprio_dim=4,
                max_action_dim=8,
                max_proprio_dim=8,
            )
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("yavla.models.policy.build_policy", lambda cfg: _make_policy(config=cfg))
            restored = VLAPolicy.from_pretrained(tmp_path, config=target, strict=False)

        assert restored.config.action_dim == 4
        assert restored.config.max_action_dim == 8
        chunk = restored.predict(
            ObservationBatch(
                images={"cam0": torch.randn(1, 3, 224, 224)},
                proprio=torch.randn(1, 4),
            )
        )
        assert chunk.actions.shape == (1, 5, 4)
