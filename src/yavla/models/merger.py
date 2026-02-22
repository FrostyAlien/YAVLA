"""Token merger — concatenates modality embeddings into a single sequence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from yavla.models.protocols import TokenMergerBase
from yavla.models.registry import Registry

merger_registry: Registry[Any, nn.Module] = Registry("merger")


@dataclass
class TokenMergerConfig:
    type: str = "concat"
    num_readout_tokens: int = 64


class ConcatMerger(TokenMergerBase):
    def __init__(self, config: TokenMergerConfig, backbone_dim: int) -> None:
        super().__init__()
        self._config = config
        self.readout_pos_embed = nn.Parameter(torch.randn(1, config.num_readout_tokens, backbone_dim) * 0.02)

    def merge(
        self,
        vision_tokens: Tensor,
        proprio_tokens: Tensor,
        language_tokens: Tensor,
        language_attn_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        B = vision_tokens.shape[0]  # noqa: N806
        device = vision_tokens.device
        n_img = vision_tokens.shape[1]
        n_proprio = proprio_tokens.shape[1]
        n_lang = language_tokens.shape[1]
        n_readout = self._config.num_readout_tokens

        readout = torch.zeros(B, n_readout, vision_tokens.shape[2], device=device)
        readout = readout + self.readout_pos_embed

        inputs_embeds = torch.cat([vision_tokens, proprio_tokens, language_tokens, readout], dim=1)

        img_mask = torch.ones(B, n_img, device=device)
        proprio_mask = torch.ones(B, n_proprio, device=device)
        readout_mask = torch.ones(B, n_readout, device=device)
        attention_mask = torch.cat([img_mask, proprio_mask, language_attn_mask, readout_mask], dim=1)

        token_type_ids = torch.cat(
            [
                torch.zeros(B, n_img, device=device, dtype=torch.long),
                torch.ones(B, n_proprio + n_lang + n_readout, device=device, dtype=torch.long),
            ],
            dim=1,
        )

        return inputs_embeds, attention_mask, token_type_ids


merger_registry.register("concat", TokenMergerConfig)(ConcatMerger)
