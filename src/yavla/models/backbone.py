"""VLM backbone wrapping PaliGemma."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from yavla.models.protocols import BackboneBase, BackboneCapabilities, IntegrationMode
from yavla.models.registry import Registry
from yavla.models.types import BackboneOutput

backbone_registry: Registry[Any, BackboneBase] = Registry("backbone")


@dataclass
class BackboneConfig:
    type: str = "vlm"
    vlm_name: str = "google/paligemma-3b-pt-224"
    gradient_checkpointing: bool = True


class VLMBackbone(BackboneBase):
    def __init__(self, model: Any, tokenizer_instance: Any, num_readout_tokens: int) -> None:
        super().__init__()
        self._model = model  # PaliGemma (or PeftModel wrapper)
        self._base_model = model  # unwrapped ref, updated by build_policy if peft applied
        self._tokenizer = tokenizer_instance
        self._num_readout_tokens = num_readout_tokens

    @property
    def base_model(self) -> Any:
        return self._base_model

    @base_model.setter
    def base_model(self, value: Any) -> None:
        self._base_model = value

    @property
    def model(self) -> Any:
        return self._model

    @model.setter
    def model(self, value: Any) -> None:
        self._model = value

    @property
    def capabilities(self) -> BackboneCapabilities:
        return BackboneCapabilities(supported_modes={IntegrationMode.READOUT}, supports_kv_cache=False)

    @property
    def hidden_dim(self) -> int:
        return self._base_model.config.text_config.hidden_size  # type: ignore[no-any-return]

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def forward(self, inputs_embeds: Tensor, attention_mask: Tensor, token_type_ids: Tensor) -> BackboneOutput:
        outputs = self._model(
            input_ids=None,
            pixel_values=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]
        readout_states = last_hidden[:, -self._num_readout_tokens :, :]
        return BackboneOutput(
            readout_states=readout_states,
            token_states=last_hidden,
            attention_mask=attention_mask,
        )


backbone_registry.register("vlm", BackboneConfig)(VLMBackbone)
