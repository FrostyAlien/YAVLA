"""Integration tests for the training loop: build_policy → forward → backward → step."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from torch import Tensor, nn

from yavla.models.config import BackboneConfig, PolicyConfig
from yavla.models.heads.mlp import MLPHeadConfig
from yavla.models.merger import TokenMergerConfig
from yavla.models.policy import build_policy
from yavla.models.protocols import (
    BackboneBase,
    BackboneCapabilities,
    IntegrationMode,
    VisionEncoderBase,
)
from yavla.models.types import (
    BackboneOutput,
    FreezeConfig,
    ObservationBatch,
    TrainingBatch,
)
from yavla.models.vlm_registry import VLMRegistry, vlm_registry as _real_registry


# ---------------------------------------------------------------------------
# Lightweight stub VLM for integration testing (no HF download required)
# ---------------------------------------------------------------------------

_STUB_DIM = 32
_NUM_PATCHES = 4


class _StubVisionEncoder(VisionEncoderBase):
    """Minimal vision encoder that projects images to stub embeddings."""

    def __init__(self, hidden_dim: int = _STUB_DIM) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._proj = nn.Linear(3 * 8 * 8, hidden_dim)

    @property
    def output_dim(self) -> int:
        return self._hidden_dim

    @property
    def num_patches(self) -> int:
        return _NUM_PATCHES

    def encode_images(self, images: dict[str, Tensor]) -> Tensor:
        if len(images) == 0:
            raise ValueError("No camera images provided")

        ordered_cams = sorted(images.keys())
        pixel_values_list = [images[name] for name in ordered_cams]

        expected_shape = pixel_values_list[0].shape
        if len(expected_shape) != 4:
            raise ValueError(
                f"Camera tensor must have shape [B, C, H, W], got {tuple(expected_shape)} for '{ordered_cams[0]}'"
            )
        for name, pixel_values in zip(ordered_cams, pixel_values_list, strict=True):
            if pixel_values.shape != expected_shape:
                raise ValueError(
                    f"Mismatched camera tensor shapes: '{name}' has {tuple(pixel_values.shape)} but expected {tuple(expected_shape)}"
                )

        # Flatten cameras into the batch dimension:
        # [K, B, C, H, W] -> [K*B, C, H, W] -> [B, K*N_patch, D]
        stacked = torch.stack(pixel_values_list, dim=0)
        num_cams, batch_size, channels, height, width = stacked.shape
        flat = stacked.reshape(num_cams * batch_size, channels, height, width)

        pooled = nn.functional.adaptive_avg_pool2d(flat, (8, 8))  # [K*B, 3, 8, 8]
        flat_tokens = pooled.reshape(num_cams * batch_size, 1, -1).expand(num_cams * batch_size, _NUM_PATCHES, -1)
        tokens = self._proj(flat_tokens)  # [K*B, N_patch, D]
        tokens = tokens.reshape(num_cams, batch_size, tokens.shape[1], tokens.shape[2])
        return tokens.permute(1, 0, 2, 3).reshape(batch_size, num_cams * tokens.shape[2], tokens.shape[3])


class _StubBackbone(BackboneBase):
    """Minimal backbone with a single linear layer (trainable parameters)."""

    def __init__(self, hidden_dim: int = _STUB_DIM, num_readout_tokens: int = 8) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._num_readout = num_readout_tokens
        self._lang_embed = nn.Embedding(100, hidden_dim)
        self._proj = nn.Linear(hidden_dim, hidden_dim)

    @property
    def capabilities(self) -> BackboneCapabilities:
        return BackboneCapabilities(supported_modes={IntegrationMode.READOUT})

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def embed_language(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        if not texts:
            raise ValueError("texts must be non-empty")
        B = len(texts)
        # Fixed-length token sequence (3 tokens per text)
        ids = torch.zeros(B, 3, dtype=torch.long, device=self._lang_embed.weight.device)
        embeds = self._lang_embed(ids)
        mask = torch.ones(B, 3, device=embeds.device)
        return embeds, mask

    def forward(
        self, inputs_embeds: Tensor, attention_mask: Tensor, token_type_ids: Tensor
    ) -> BackboneOutput:
        B, S, D = inputs_embeds.shape
        hidden = self._proj(inputs_embeds)
        readout = hidden[:, -self._num_readout :]
        return BackboneOutput(
            readout_states=readout,
            token_states=hidden,
            attention_mask=attention_mask,
        )


def _build_stub_vlm(
    config: BackboneConfig, freeze: FreezeConfig, num_readout_tokens: int
) -> tuple[VisionEncoderBase, BackboneBase]:
    return _StubVisionEncoder(_STUB_DIM), _StubBackbone(_STUB_DIM, num_readout_tokens)


# ---------------------------------------------------------------------------
# Synthetic TrainingBatch factory
# ---------------------------------------------------------------------------

_BATCH_SIZE = 2
_CHUNK_LEN = 5
_ACTION_DIM = 7
_PROPRIO_DIM = 7
_NUM_READOUT = 8


def _make_training_batch() -> TrainingBatch:
    """Create a synthetic TrainingBatch with random data matching policy config shapes."""
    obs = ObservationBatch(
        images={"cam0": torch.randn(_BATCH_SIZE, 3, 224, 224)},
        proprio=torch.randn(_BATCH_SIZE, _PROPRIO_DIM),
        language=["pick up the block"] * _BATCH_SIZE,
    )
    return TrainingBatch(
        observations=obs,
        actions=torch.randn(_BATCH_SIZE, _CHUNK_LEN, _ACTION_DIM),
        dt_hz=10.0,
        chunk_len=_CHUNK_LEN,
    )


def _make_multi_camera_training_batch(num_cams: int = 2) -> TrainingBatch:
    """Create a synthetic TrainingBatch with multiple camera views."""
    if num_cams < 1:
        raise ValueError("num_cams must be >= 1")
    images = {f"cam{i}": torch.randn(_BATCH_SIZE, 3, 224, 224) for i in range(num_cams)}
    obs = ObservationBatch(
        images=images,
        proprio=torch.randn(_BATCH_SIZE, _PROPRIO_DIM),
        language=["pick up the block"] * _BATCH_SIZE,
    )
    return TrainingBatch(
        observations=obs,
        actions=torch.randn(_BATCH_SIZE, _CHUNK_LEN, _ACTION_DIM),
        dt_hz=10.0,
        chunk_len=_CHUNK_LEN,
    )


def _make_stub_config() -> PolicyConfig:
    """PolicyConfig wired to the stub VLM with small dimensions."""
    return PolicyConfig(
        backbone=BackboneConfig(type="stub_test"),
        action_head=MLPHeadConfig(
            hidden_dim=32, num_blocks=1, chunk_len=_CHUNK_LEN, action_dim=_ACTION_DIM,
        ),
        merger=TokenMergerConfig(num_readout_tokens=_NUM_READOUT),
    )


# ---------------------------------------------------------------------------
# Fixture: register stub VLM in a temporary registry scope
# ---------------------------------------------------------------------------


@pytest.fixture()
def stub_registry() -> Iterator[VLMRegistry]:
    """Register the stub builder on the real vlm_registry, then clean up.

    NOTE: This fixture mutates the global vlm_registry singleton.
    Tests using it must not run in parallel (no pytest-xdist thread workers).
    """
    _real_registry.register("stub_test")(_build_stub_vlm)
    try:
        yield _real_registry
    finally:
        _real_registry._builders.pop("stub_test", None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_forward_finite_loss(stub_registry: VLMRegistry) -> None:
    """build_policy() → policy.forward(batch) → assert finite loss."""
    config = _make_stub_config()
    policy = build_policy(config)
    batch = _make_training_batch()

    loss = policy.forward(batch)

    assert loss.total.isfinite(), f"Loss is not finite: {loss.total.item()}"
    assert loss.total.requires_grad
    assert "l1" in loss.breakdown
    assert loss.breakdown["l1"].isfinite()

    batch_multi = _make_multi_camera_training_batch(num_cams=2)
    loss_multi = policy.forward(batch_multi)
    assert loss_multi.total.isfinite(), f"Loss is not finite: {loss_multi.total.item()}"


def test_forward_backward_step_updates_params(stub_registry: VLMRegistry) -> None:
    """forward → backward → optimizer step → assert parameters changed."""
    config = _make_stub_config()
    policy = build_policy(config)
    batch = _make_training_batch()

    # Snapshot parameters before
    params_before = {
        name: p.clone().detach() for name, p in policy.named_parameters() if p.requires_grad
    }
    assert len(params_before) > 0, "No trainable parameters found"

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # Forward + backward + step
    optimizer.zero_grad()
    loss = policy.forward(batch)
    loss.total.backward()
    optimizer.step()

    # Assert at least some parameters changed
    changed = 0
    for name, p in policy.named_parameters():
        if name in params_before and not torch.equal(p.data, params_before[name]):
            changed += 1
    assert changed > 0, "No parameters were updated after optimizer step"

    # Multi-camera variant: ensure the same training loop works with >1 camera key.
    params_before_multi = {
        name: p.clone().detach() for name, p in policy.named_parameters() if p.requires_grad
    }
    optimizer.zero_grad()
    batch_multi = _make_multi_camera_training_batch(num_cams=2)
    loss_multi = policy.forward(batch_multi)
    loss_multi.total.backward()
    optimizer.step()

    changed_multi = 0
    for name, p in policy.named_parameters():
        if name in params_before_multi and not torch.equal(p.data, params_before_multi[name]):
            changed_multi += 1
    assert changed_multi > 0, "No parameters were updated after optimizer step (multi-camera)"
