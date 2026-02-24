"""Registry mapping VLM type strings to builder functions."""

from __future__ import annotations

from collections.abc import Callable

from yavla.models.backbone import BackboneConfig
from yavla.models.protocols import BackboneBase, VisionEncoderBase

VLMBuilder = Callable[[BackboneConfig], tuple[VisionEncoderBase, BackboneBase]]


class VLMRegistry:
    def __init__(self) -> None:
        self._builders: dict[str, VLMBuilder] = {}

    def register(self, name: str) -> Callable[[VLMBuilder], VLMBuilder]:
        def decorator(fn: VLMBuilder) -> VLMBuilder:
            if name in self._builders:
                raise ValueError(f"Duplicate VLM registration: '{name}'")
            self._builders[name] = fn
            return fn

        return decorator

    def build(self, config: BackboneConfig) -> tuple[VisionEncoderBase, BackboneBase]:
        if config.type not in self._builders:
            available = ", ".join(sorted(self._builders.keys()))
            raise KeyError(f"Unknown VLM type '{config.type}'. Available: [{available}]")
        return self._builders[config.type](config)

    def list(self) -> list[str]:
        return list(self._builders.keys())


vlm_registry = VLMRegistry()
