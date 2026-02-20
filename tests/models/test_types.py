"""Unit tests for model data containers."""

from __future__ import annotations

import torch
import pytest

from yavla.models.types import (
    ActionChunk,
    ActionPrediction,
    ActionSpaceSpec,
    BackboneOutput,
    FreezeConfig,
    LossDict,
    ObservationBatch,
    ProprioSpec,
    TokenBatch,
    TrainingBatch,
)


class TestObservationBatch:
    def test_construction_all_fields(self) -> None:
        obs = ObservationBatch(
            images={"cam0": torch.randn(2, 3, 224, 224)},
            proprio=torch.randn(2, 7),
            language="pick up the cup",
            timestamps=torch.tensor([0.0, 0.1]),
            masks=None,
        )
        assert obs.images["cam0"].shape == (2, 3, 224, 224)
        assert obs.proprio.shape == (2, 7)
        assert obs.language == "pick up the cup"

    def test_multi_camera(self) -> None:
        obs = ObservationBatch(
            images={
                "cam0": torch.randn(2, 3, 224, 224),
                "cam1": torch.randn(2, 3, 224, 224),
            },
            proprio=torch.randn(2, 7),
        )
        assert len(obs.images) == 2

    def test_language_list(self) -> None:
        obs = ObservationBatch(
            images={"cam0": torch.randn(1, 3, 224, 224)},
            proprio=torch.randn(1, 7),
            language=["pick up", "put down"],
        )
        assert isinstance(obs.language, list)


class TestTokenBatch:
    def test_token_type_ids_assignment(self) -> None:
        B, N, D = 2, 341, 2048
        tb = TokenBatch(
            tokens=torch.randn(B, N, D),
            attention_mask=torch.ones(B, N),
            token_type_ids=torch.cat([torch.zeros(B, 256), torch.ones(B, 85)], dim=1),
            modality_ids=torch.zeros(B, N),
        )
        assert (tb.token_type_ids[:, :256] == 0).all()
        assert (tb.token_type_ids[:, 256:] == 1).all()

    def test_modality_tracking(self) -> None:
        B, D = 2, 2048
        n_img, n_lang, n_proprio, n_readout = 256, 20, 1, 64
        N = n_img + n_lang + n_proprio + n_readout
        modality = torch.cat(
            [
                torch.full((B, n_img), 0),
                torch.full((B, n_lang), 1),
                torch.full((B, n_proprio), 2),
                torch.full((B, n_readout), 3),
            ],
            dim=1,
        )
        tb = TokenBatch(
            tokens=torch.randn(B, N, D),
            attention_mask=torch.ones(B, N),
            token_type_ids=torch.zeros(B, N),
            modality_ids=modality,
        )
        assert (tb.modality_ids[:, :n_img] == 0).all()
        assert (tb.modality_ids[:, n_img : n_img + n_lang] == 1).all()


class TestBackboneOutput:
    def test_readout_mode(self) -> None:
        out = BackboneOutput(
            readout_states=torch.randn(2, 64, 2048),
            token_states=None,
            attention_mask=torch.ones(2, 341),
        )
        assert out.readout_states is not None
        assert out.token_states is None
        assert out.readout_states.shape == (2, 64, 2048)


class TestActionPrediction:
    def test_mlp_prediction(self) -> None:
        pred = ActionPrediction(mean=torch.randn(2, 5, 7))
        assert pred.mean.shape == (2, 5, 7)
        assert pred.samples is None
        assert pred.log_prob is None


class TestActionChunk:
    def test_chunk_shape(self) -> None:
        chunk = ActionChunk(
            actions=torch.randn(2, 5, 7),
            dt_hz=10.0,
            chunk_len=5,
            action_mask=torch.ones(2, 5),
        )
        assert chunk.actions.shape[1] == chunk.chunk_len


class TestLossDict:
    def test_backward(self) -> None:
        x = torch.randn(1, requires_grad=True)
        loss = LossDict(total=x.sum(), breakdown={"l1": x.sum()})
        loss.total.backward()
        assert x.grad is not None


class TestTrainingBatch:
    def test_construction(self) -> None:
        B, chunk_len, action_dim = 2, 5, 7
        obs = ObservationBatch(
            images={"cam0": torch.randn(B, 3, 224, 224)},
            proprio=torch.randn(B, 7),
            language="pick up the cup",
            timestamps=torch.tensor([0.0, 0.1]),
        )
        batch = TrainingBatch(
            observations=obs,
            actions=torch.randn(B, chunk_len, action_dim),
            dt_hz=10.0,
            chunk_len=chunk_len,
        )
        assert batch.actions.shape == (B, chunk_len, action_dim)
        assert isinstance(batch.observations, ObservationBatch)


class TestActionSpaceSpec:
    def test_limits(self) -> None:
        spec = ActionSpaceSpec(
            names=["x", "y", "z", "rx", "ry", "rz", "grip"],
            units=["m", "m", "m", "rad", "rad", "rad", "bool"],
            limits=torch.tensor([[-1.0, 1.0]] * 7),
        )
        assert spec.limits is not None
        assert spec.limits.shape == (7, 2)

    def test_no_limits(self) -> None:
        spec = ActionSpaceSpec(names=["x"], units=["m"], limits=None)
        assert spec.limits is None


class TestProprioSpec:
    def test_construction(self) -> None:
        spec = ProprioSpec(names=["j0", "j1"], units=["rad", "rad"])
        assert spec.limits is None


class TestFreezeConfig:
    def test_defaults(self) -> None:
        cfg = FreezeConfig()
        assert cfg.freeze_modules == []
        assert cfg.lora_target_modules == []
        assert cfg.lora_r == 8
        assert cfg.lora_alpha == 16
        assert cfg.lora_dropout == 0.0

    def test_custom(self) -> None:
        cfg = FreezeConfig(
            freeze_modules=["vision_tower", "multi_modal_projector"],
            lora_target_modules=["q_proj", "v_proj"],
            lora_r=16,
        )
        assert len(cfg.freeze_modules) == 2
        assert cfg.lora_r == 16
