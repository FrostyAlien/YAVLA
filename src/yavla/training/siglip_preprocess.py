"""Helpers for dataset-layer SigLIP image preprocessing.

SigLIP-style VLMs (including PaliGemma's SigLIP vision tower) expect images to be
resized to the checkpoint's configured size and normalized with mean/std = 0.5.
YAVLA keeps this preprocessing in the dataset layer via ``DataConfig.image_transforms``.
"""

from __future__ import annotations

import logging

from yavla.training.config import TrainingConfig

LOGGER = logging.getLogger(__name__)

_SIGLIP_NORMALIZE = "Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))"
_SIGLIP_RESIZE_STRATEGIES = {"warp", "letterbox"}


def _validate_siglip_resize_strategy(strategy: str) -> str:
    if strategy not in _SIGLIP_RESIZE_STRATEGIES:
        allowed = "|".join(sorted(_SIGLIP_RESIZE_STRATEGIES))
        raise ValueError(f"Unknown vlm_image_resize_strategy={strategy!r}; allowed: {allowed}")
    return strategy


def build_siglip_image_transform_specs(
    height: int,
    width: int,
    *,
    resize_strategy: str = "warp",
) -> list[str]:
    """Return the canonical SigLIP transform list for a target ``(H, W)`` size."""

    if height <= 0 or width <= 0:
        raise ValueError(f"SigLIP image size must be positive, got (H, W)=({height}, {width})")
    resize_strategy = _validate_siglip_resize_strategy(resize_strategy)

    if resize_strategy == "warp":
        resize_spec = f"Resize([{height}, {width}], 3)"
    elif resize_strategy == "letterbox":
        resize_spec = f"LetterboxPad([{height}, {width}], 3)"
    else:  # pragma: no cover
        raise AssertionError(f"Unhandled resize strategy: {resize_strategy}")

    return [
        resize_spec,
        _SIGLIP_NORMALIZE,
    ]


def resolve_siglip_target_size(
    training: TrainingConfig,
    *,
    ckpt_image_size: int,
) -> tuple[int, int]:
    """Resolve the effective SigLIP resize target.

    Defaults to the checkpoint-declared square size ``(S_ckpt, S_ckpt)``.
    If override fields are set, both MUST be set and become the effective size.
    """

    if ckpt_image_size <= 0:
        raise ValueError(f"Checkpoint image size must be positive, got {ckpt_image_size}")

    height_override = training.vlm_image_height_override
    width_override = training.vlm_image_width_override
    if (height_override is None) != (width_override is None):
        raise ValueError(
            "VLM image size overrides must be set as both-or-none: "
            "vlm_image_height_override and vlm_image_width_override"
        )

    if height_override is None or width_override is None:
        return ckpt_image_size, ckpt_image_size

    height = int(height_override)
    width = int(width_override)
    if height <= 0 or width <= 0:
        raise ValueError(f"SigLIP image size override must be positive, got (H, W)=({height}, {width})")
    return height, width


def autowire_siglip_image_transforms(
    training: TrainingConfig,
    *,
    ckpt_image_size: int,
) -> None:
    """Auto-wire canonical SigLIP preprocessing when ``dataset.image_transforms`` is unset.

    Rules:
    - If exactly one override is set → error.
    - If both overrides are set and mismatch the checkpoint size → warn.
    - If ``dataset.image_transforms is None`` → set to canonical SigLIP transform list.
    - If ``dataset.image_transforms == []`` → respect explicit disable.
    - If ``dataset.image_transforms`` is non-empty → respect user transforms (warn if overrides are set).
    """

    height, width = resolve_siglip_target_size(training, ckpt_image_size=ckpt_image_size)
    override_active = training.vlm_image_height_override is not None
    resize_strategy = _validate_siglip_resize_strategy(training.vlm_image_resize_strategy)

    image_transforms = training.dataset.image_transforms
    if image_transforms is None:
        if override_active and (height, width) != (ckpt_image_size, ckpt_image_size):
            LOGGER.warning(
                "Overriding VLM checkpoint image size: ckpt expects (%d, %d) but using (%d, %d). "
                "You are responsible for verifying VLM compatibility.",
                ckpt_image_size,
                ckpt_image_size,
                height,
                width,
            )
        training.dataset.image_transforms = build_siglip_image_transform_specs(
            height,
            width,
            resize_strategy=resize_strategy,
        )
        return

    if override_active:
        LOGGER.warning(
            "VLM image size override fields are set (%d, %d) but dataset.image_transforms is explicitly set; "
            "override will not be auto-wired.",
            height,
            width,
        )

    if resize_strategy != "warp":
        LOGGER.warning(
            "vlm_image_resize_strategy is set to %r but dataset.image_transforms is explicitly set; "
            "strategy will not be auto-wired.",
            resize_strategy,
        )
