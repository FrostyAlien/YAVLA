"""PaliGemma VLM backbone and vision encoder."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from yavla.models.protocols import BackboneBase, BackboneCapabilities, IntegrationMode, VisionEncoderBase
from yavla.models.types import BackboneOutput
from yavla.models.vlm_registry import vlm_registry


class PaliGemmaVisionEncoder(VisionEncoderBase):
    def __init__(self, base_model: Any, config: Any | None = None) -> None:
        super().__init__()
        self._base_model = base_model
        self._config = config

    @property
    def output_dim(self) -> int:
        return int(self._base_model.config.text_config.hidden_size)

    @property
    def num_patches(self) -> int:
        img_size: int = self._base_model.config.vision_config.image_size
        patch_size: int = self._base_model.config.vision_config.patch_size
        return (img_size // patch_size) ** 2

    def encode_images(self, images: dict[str, Tensor]) -> Tensor:
        if len(images) == 0:
            raise ValueError("No camera images provided")
        if len(images) > 1:
            raise ValueError(
                f"MVP encoder supports single-camera only, got {len(images)} cameras: {list(images.keys())}"
            )
        pixel_values = next(iter(images.values()))
        result: Tensor = self._base_model.get_image_features(pixel_values)
        return result


class PaliGemmaBackbone(BackboneBase):
    def __init__(self, model: Any, tokenizer_instance: Any, num_readout_tokens: int) -> None:
        super().__init__()
        self._model = model
        self._base_model = model
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
        return int(self._base_model.config.text_config.hidden_size)

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def embed_language(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        """Tokenize and embed language via PaliGemma's tokenizer and embedding layer."""
        tok_out = self._tokenizer(texts, return_tensors="pt", padding=True)
        device = next(self.parameters()).device
        embed_layer = self._base_model.get_input_embeddings()
        input_ids = tok_out["input_ids"].to(device)
        attention_mask = tok_out["attention_mask"].to(device)
        embeddings = embed_layer(input_ids)
        return embeddings, attention_mask

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


@vlm_registry.register("paligemma")
def build_paligemma_vlm(config: Any) -> tuple[VisionEncoderBase, BackboneBase]:
    """Build PaliGemma vision encoder + backbone pair.

    Handles HF model loading, freezing, LoRA wrapping, and gradient checkpointing.
    Requires ``freeze`` config from the parent ``PolicyConfig`` passed via
    ``config._freeze`` (set by ``build_policy()`` before calling the registry).
    """
    from transformers import AutoModelForVision2Seq, AutoProcessor

    base_model = AutoModelForVision2Seq.from_pretrained(  # type: ignore[no-untyped-call]
        config.vlm_name,
        torch_dtype=torch.float32,
    )
    tokenizer = AutoProcessor.from_pretrained(config.vlm_name).tokenizer  # type: ignore[no-untyped-call]

    unwrapped = base_model

    # Freeze modules (requires freeze config injected by build_policy)
    freeze = getattr(config, "_freeze", None)
    if freeze is not None:
        for name in freeze.freeze_modules:
            for param_name, param in base_model.named_parameters():
                if param_name.startswith(name):
                    param.requires_grad_(False)

    # Apply LoRA if configured
    peft_model = None
    if freeze is not None and freeze.lora_target_modules:
        import peft

        lora_config = peft.LoraConfig(
            target_modules=freeze.lora_target_modules,
            r=freeze.lora_r,
            lora_alpha=freeze.lora_alpha,
            lora_dropout=freeze.lora_dropout,
        )
        peft_model = peft.get_peft_model(base_model, lora_config)
        peft_model.enable_input_require_grads()  # pyright: ignore[reportCallIssue]

    # Gradient checkpointing
    if config.gradient_checkpointing:
        unwrapped.config.use_cache = False
        unwrapped.gradient_checkpointing_enable()

    model_for_forward = peft_model if peft_model is not None else base_model
    num_readout = getattr(config, "_num_readout_tokens", 64)

    backbone = PaliGemmaBackbone(
        model=model_for_forward,
        tokenizer_instance=tokenizer,
        num_readout_tokens=num_readout,
    )
    backbone.base_model = unwrapped

    vision_encoder = PaliGemmaVisionEncoder(base_model=unwrapped)

    return vision_encoder, backbone
