"""MLP regression action head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from yavla.models.protocols import ActionHeadBase, ActionHeadRequirements, IntegrationMode
from yavla.models.registry import Registry
from yavla.models.types import ActionPrediction, BackboneOutput, LossDict, TrainingBatch

head_registry: Registry[Any, ActionHeadBase] = Registry("action_head")


@dataclass
class MLPHeadConfig:
    type: str = "mlp"
    hidden_dim: int = 1024
    num_blocks: int = 2
    chunk_len: int = 5
    action_dim: int = 7


class ResidualMLP(nn.Module):
    """LayerNorm → Linear → ReLU → [N× (LayerNorm → Linear → ReLU + residual)] → LayerNorm → Linear."""

    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_linear = nn.Linear(input_dim, hidden_dim)

        blocks: list[nn.Module] = []
        for _ in range(num_blocks):
            blocks.append(_ResBlock(hidden_dim))
        self.blocks = nn.ModuleList(blocks)

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.input_linear(self.input_norm(x)))
        for block in self.blocks:
            x = block(x)
        result: Tensor = self.output_linear(self.output_norm(x))
        return result


class _ResBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return x + F.relu(self.linear(self.norm(x)))


class MLPRegressionHead(ActionHeadBase):
    def __init__(self, config: MLPHeadConfig, backbone_dim: int) -> None:
        super().__init__()
        self._config = config
        self._requirements = ActionHeadRequirements(
            required_mode=IntegrationMode.READOUT,
            accepts_readout=True,
        )
        self.net = ResidualMLP(
            num_blocks=config.num_blocks,
            input_dim=backbone_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.chunk_len * config.action_dim,
        )

    @property
    def requirements(self) -> ActionHeadRequirements:
        return self._requirements

    def _pool_and_predict(self, backbone_output: BackboneOutput) -> Tensor:
        readout = backbone_output.readout_states
        if readout is None:
            raise ValueError("MLPRegressionHead requires readout_states")
        if readout.shape[1] == 0:
            raise ValueError("readout_states has 0 tokens, cannot mean-pool")
        pooled = readout.mean(dim=1)  # [B, N_readout, D] → [B, D]
        flat: Tensor = self.net(pooled)  # [B, chunk_len * action_dim]
        return flat.view(flat.shape[0], self._config.chunk_len, self._config.action_dim)

    def predict(self, backbone_output: BackboneOutput) -> ActionPrediction:
        return ActionPrediction(mean=self._pool_and_predict(backbone_output))

    def compute_loss(self, backbone_output: BackboneOutput, batch: TrainingBatch) -> LossDict:
        predicted = self._pool_and_predict(backbone_output)
        target = batch.actions[:, : self._config.chunk_len, : self._config.action_dim]
        l1 = F.l1_loss(predicted, target)
        return LossDict(total=l1, breakdown={"l1": l1})


head_registry.register("mlp", MLPHeadConfig)(MLPRegressionHead)
