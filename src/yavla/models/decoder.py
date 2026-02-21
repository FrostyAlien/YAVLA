"""Simple action decoder with optional unnormalization."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from yavla.models.protocols import ActionDecoderBase
from yavla.models.registry import Registry
from yavla.models.types import ActionChunk, ActionNormalizationConfig, ActionPrediction, ActionSpaceSpec

decoder_registry: Registry[Any, ActionDecoderBase] = Registry("action_decoder")


@decoder_registry.register("simple", None)
class SimpleActionDecoder(ActionDecoderBase):
    def __init__(
        self,
        action_space_spec: ActionSpaceSpec,
        dt_hz: float = 10.0,
        normalization: ActionNormalizationConfig | None = None,
        action_stats: dict[str, Tensor] | None = None,
    ) -> None:
        super().__init__()
        self._action_space_spec = action_space_spec
        self._dt_hz = dt_hz
        self._norm = normalization or ActionNormalizationConfig()
        self._action_stats = action_stats

        self._validate()

    def _validate(self) -> None:
        if self._norm.mode == "bounds":
            if self._action_space_spec.limits is None:
                raise ValueError("bounds mode requires ActionSpaceSpec.limits to be set")
            lim = self._action_space_spec.limits
            if lim.ndim != 2 or lim.shape[1] != 2:
                raise ValueError(f"bounds mode requires limits shaped [action_dim, 2], got {list(lim.shape)}")
        elif self._norm.mode == "z-score":
            missing = []
            if self._action_stats is None:
                missing = ["mean", "std"]
            else:
                for k in ("mean", "std"):
                    if k not in self._action_stats:
                        missing.append(k)
            if missing:
                raise ValueError(f"z-score mode requires action_stats keys {missing}")
        else:
            raise ValueError(f"Unknown normalization mode: {self._norm.mode!r} (expected 'bounds' or 'z-score')")

    @property
    def action_space_spec(self) -> ActionSpaceSpec:
        return self._action_space_spec

    def decode(self, pred: ActionPrediction) -> ActionChunk:
        actions = pred.mean

        if self._norm.mode == "bounds":
            # No clamping — see ActionSpaceSpec docstring for rationale.
            limits = self._action_space_spec.limits
            assert limits is not None  # guaranteed by _validate
            limits = limits.to(actions.device)
            lo, hi = limits[:, 0], limits[:, 1]
            actions = (actions + 1.0) / 2.0 * (hi - lo) + lo
        elif self._norm.mode == "z-score":
            assert self._action_stats is not None  # guaranteed by _validate
            mean = self._action_stats["mean"].to(actions.device)
            std = self._action_stats["std"].to(actions.device)
            # Mask dims with std==0 to mean (spec: no division-by-zero, stable mapping)
            safe_std = std + self._norm.eps
            zero_mask = std == 0
            actions = torch.where(zero_mask, mean, actions * safe_std + mean)

        chunk_len = actions.shape[1]
        return ActionChunk(actions=actions, dt_hz=self._dt_hz, chunk_len=chunk_len)
