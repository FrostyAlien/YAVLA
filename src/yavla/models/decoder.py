"""Simple action decoder with optional unnormalization."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from yavla.models.protocols import ActionDecoderBase
from yavla.models.registry import Registry
from yavla.models.types import ActionChunk, ActionPrediction, ActionSpaceSpec

decoder_registry: Registry[Any, ActionDecoderBase] = Registry("action_decoder")


@decoder_registry.register("simple", None)
class SimpleActionDecoder(ActionDecoderBase):
    def __init__(self, action_space_spec: ActionSpaceSpec, dt_hz: float = 10.0) -> None:
        super().__init__()
        self._action_space_spec = action_space_spec
        self._dt_hz = dt_hz

    @property
    def action_space_spec(self) -> ActionSpaceSpec:
        return self._action_space_spec

    def decode(self, pred: ActionPrediction) -> ActionChunk:
        actions = pred.mean
        if self._action_space_spec.limits is not None:
            # Note: We intentionally DO NOT clamp `actions` to [-1, 1] here to avoid
            # GPU sync points / graph overhead during latency-critical inference.
            # We assume the action head provides natively bounded predictions, and rely 
            # on the environment or dataset layer to handle physical clipping if needed.
            limits = self._action_space_spec.limits.to(actions.device)
            lo, hi = limits[:, 0], limits[:, 1]
            actions = (actions + 1.0) / 2.0 * (hi - lo) + lo
        chunk_len = actions.shape[1]
        return ActionChunk(actions=actions, dt_hz=self._dt_hz, chunk_len=chunk_len)
