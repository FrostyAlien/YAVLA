"""PaliGemma vision encoder wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor

from yavla.models.protocols import VisionEncoderBase
from yavla.models.registry import Registry

vision_registry: Registry[Any, VisionEncoderBase] = Registry("vision_encoder")


@dataclass
class VisionEncoderConfig:
    type: str = "paligemma_siglip"


class PaliGemmaVisionEncoder(VisionEncoderBase):
    def __init__(self, base_model: Any, config: VisionEncoderConfig | None = None) -> None:
        super().__init__()
        self._base_model = base_model
        self._config = config or VisionEncoderConfig()

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


vision_registry.register("paligemma_siglip", VisionEncoderConfig)(PaliGemmaVisionEncoder)
