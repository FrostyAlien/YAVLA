"""Vision encoder configs, utilities, and registry-backed implementations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, cast

import torch
import tyro
from torch import Tensor, nn

from yavla.models.protocols import VisionEncoderBase
from yavla.models.registry import Registry

vision_registry: Registry[VisionEncoderConfig, VisionEncoderBase] = Registry("vision_encoder")

LEGACY_BACKBONE_VISION_ENCODER_TYPES = {
    "paligemma_siglip": "from_backbone",
}


@dataclass
class VisionEncoderConfig:
    type: str = "from_backbone"
    model_name: str | None = None
    extract_layer: int = -1


@dataclass
class FromBackboneVisionEncoderConfig(VisionEncoderConfig):
    type: str = "from_backbone"


@dataclass
class SimplePatchVisionEncoderConfig(VisionEncoderConfig):
    type: str = "simple_patch"
    image_size: int = 224
    patch_size: int = 16
    hidden_dim: int = 256
    in_channels: int = 3


@dataclass
class MultiTowerVisionEncoderConfig(VisionEncoderConfig):
    type: str = "multi_tower"
    towers: list[VisionEncoderConfigVariant] = field(default_factory=list)
    fusion: str = "concat"
    projector: str = "linear"


type VisionEncoderConfigVariant = (
    Annotated[FromBackboneVisionEncoderConfig, tyro.conf.subcommand("from_backbone")]
    | Annotated[SimplePatchVisionEncoderConfig, tyro.conf.subcommand("simple_patch")]
    | Annotated[
    MultiTowerVisionEncoderConfig,
    tyro.conf.subcommand("multi_tower"),
    ]
)


VISION_ENCODER_CONFIG_TYPES: dict[str, type[VisionEncoderConfig]] = {
    "from_backbone": FromBackboneVisionEncoderConfig,
    "simple_patch": SimplePatchVisionEncoderConfig,
    "multi_tower": MultiTowerVisionEncoderConfig,
}


def get_vision_config_class(type_name: str | None) -> type[VisionEncoderConfig]:
    canonical_type = LEGACY_BACKBONE_VISION_ENCODER_TYPES.get(type_name or "", type_name or "from_backbone")
    return VISION_ENCODER_CONFIG_TYPES.get(canonical_type, VisionEncoderConfig)


def canonicalize_vision_encoder_config(
    config: VisionEncoderConfig, *, warn_on_alias: bool = False
) -> VisionEncoderConfig:
    canonical_type = LEGACY_BACKBONE_VISION_ENCODER_TYPES.get(config.type, config.type)
    if canonical_type == "from_backbone":
        if isinstance(config, FromBackboneVisionEncoderConfig):
            return config
        if canonical_type != config.type and warn_on_alias:
            logging.warning(
                "Vision encoder type %r is deprecated; use %r instead.",
                config.type,
                canonical_type,
            )
        return FromBackboneVisionEncoderConfig(model_name=config.model_name, extract_layer=config.extract_layer)
    if canonical_type == config.type:
        return config
    if warn_on_alias:
        logging.warning(
            "Vision encoder type %r is deprecated; use %r instead.",
            config.type,
            canonical_type,
        )
    return replace(config, type=canonical_type)


def _ordered_camera_tensors(images: dict[str, Tensor]) -> list[Tensor]:
    if not images:
        raise ValueError("No camera images provided")

    ordered_cameras = sorted(images.keys())
    ordered_tensors = [images[name] for name in ordered_cameras]

    expected_shape = ordered_tensors[0].shape
    if len(expected_shape) != 4:
        raise ValueError(
            f"Camera tensor must have shape [B, C, H, W], got {tuple(expected_shape)} for '{ordered_cameras[0]}'"
        )

    for name, pixel_values in zip(ordered_cameras, ordered_tensors, strict=True):
        if pixel_values.shape != expected_shape:
            raise ValueError(
                f"Mismatched camera tensor shapes: '{name}' has {tuple(pixel_values.shape)} "
                f"but expected {tuple(expected_shape)}"
            )

    return ordered_tensors


def _flatten_camera_images(images: dict[str, Tensor]) -> tuple[Tensor, int]:
    ordered_tensors = _ordered_camera_tensors(images)
    stacked = torch.stack(ordered_tensors, dim=0)
    num_cams, batch_size, channels, height, width = stacked.shape
    flat = stacked.reshape(num_cams * batch_size, channels, height, width)
    return flat, num_cams


def _restore_camera_tokens(tokens: Tensor, num_cams: int) -> Tensor:
    batch_size = tokens.shape[0] // num_cams
    tokens = tokens.reshape(num_cams, batch_size, tokens.shape[1], tokens.shape[2])
    return tokens.permute(1, 0, 2, 3).reshape(batch_size, num_cams * tokens.shape[2], tokens.shape[3])


class ProjectedVisionEncoder(VisionEncoderBase):
    """Wrap a vision encoder with a projection into the backbone embedding space."""

    def __init__(self, base_encoder: VisionEncoderBase, output_dim: int) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self.projector = nn.Linear(base_encoder.output_dim, output_dim)
        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def num_patches(self) -> int:
        return self.base_encoder.num_patches

    def encode_images(self, images: dict[str, Tensor]) -> Tensor:
        projected = cast(Tensor, self.projector(self.base_encoder.encode_images(images)))
        return projected


@vision_registry.register("simple_patch", SimplePatchVisionEncoderConfig)
class SimplePatchVisionEncoder(VisionEncoderBase):
    """A lightweight patchifying encoder for tests and simple experiments."""

    def __init__(self, config: SimplePatchVisionEncoderConfig, **_: Any) -> None:
        super().__init__()
        if config.image_size % config.patch_size != 0:
            raise ValueError(
                f"image_size={config.image_size} must be divisible by patch_size={config.patch_size}"
            )
        self.config = config
        self.proj = nn.Conv2d(
            config.in_channels,
            config.hidden_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self._num_patches = (config.image_size // config.patch_size) ** 2

    @property
    def output_dim(self) -> int:
        return self.config.hidden_dim

    @property
    def num_patches(self) -> int:
        return self._num_patches

    def encode_images(self, images: dict[str, Tensor]) -> Tensor:
        flat_images, num_cams = _flatten_camera_images(images)
        tokens = self.proj(flat_images).flatten(2).transpose(1, 2)
        if tokens.shape[1] != self._num_patches:
            raise ValueError(
                f"Input images produced {tokens.shape[1]} patches, expected {self._num_patches} "
                f"from image_size={self.config.image_size} and patch_size={self.config.patch_size}"
            )
        return _restore_camera_tokens(tokens, num_cams)


@vision_registry.register("multi_tower", MultiTowerVisionEncoderConfig)
class MultiTowerVisionEncoder(VisionEncoderBase):
    """Compose multiple registry-built towers into one projected token stream."""

    def __init__(self, config: MultiTowerVisionEncoderConfig, backbone_dim: int, **_: Any) -> None:
        super().__init__()
        if len(config.towers) < 2:
            raise ValueError("Multi-tower vision encoder requires at least two tower configs")
        if config.fusion != "concat":
            raise ValueError(f"Unsupported multi-tower fusion {config.fusion!r}; expected 'concat'")
        if config.projector not in {"identity", "linear"}:
            raise ValueError(f"Unsupported multi-tower projector {config.projector!r}")

        towers: list[VisionEncoderBase] = []
        for tower_config in config.towers:
            canonical_config = canonicalize_vision_encoder_config(tower_config)
            if canonical_config.type == "from_backbone":
                raise ValueError("Multi-tower vision encoder does not support 'from_backbone' sub-towers")
            towers.append(vision_registry.build(canonical_config, backbone_dim=backbone_dim))

        patch_counts = {tower.num_patches for tower in towers}
        if len(patch_counts) != 1:
            raise ValueError(f"Multi-tower patch-count mismatch: {sorted(patch_counts)}")

        self.config = config
        self.towers = nn.ModuleList(towers)
        self._num_patches = towers[0].num_patches
        fused_dim = sum(tower.output_dim for tower in towers)
        self.projector: nn.Module
        if config.projector == "identity":
            if fused_dim != backbone_dim:
                raise ValueError(
                    f"Identity projector requires fused_dim == backbone_dim, got {fused_dim} vs {backbone_dim}"
                )
            self.projector = nn.Identity()
        elif fused_dim == backbone_dim:
            self.projector = nn.Identity()
        else:
            self.projector = nn.Linear(fused_dim, backbone_dim)
        self._output_dim = backbone_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def num_patches(self) -> int:
        return self._num_patches

    def encode_images(self, images: dict[str, Tensor]) -> Tensor:
        tower_tokens = [tower.encode_images(images) for tower in self.towers]
        token_counts = {tokens.shape[1] for tokens in tower_tokens}
        if len(token_counts) != 1:
            raise ValueError(f"Multi-tower token-count mismatch at runtime: {sorted(token_counts)}")
        fused_tokens = torch.cat(tower_tokens, dim=-1)
        projected = cast(Tensor, self.projector(fused_tokens))
        return projected
