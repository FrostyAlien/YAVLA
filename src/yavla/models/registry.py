"""Generic registry for config-driven module instantiation."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

ConfigT = TypeVar("ConfigT")
ModuleT = TypeVar("ModuleT")


class Registry(Generic[ConfigT, ModuleT]):
    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, tuple[type[ConfigT], type[ModuleT]]] = {}

    def register(self, name: str, config_cls: type[ConfigT] | None = None) -> Any:
        def decorator(module_cls: type[ModuleT]) -> type[ModuleT]:
            if name in self._entries:
                raise ValueError(f"Duplicate registration in '{self._name}': '{name}' already registered")
            cfg = config_cls if config_cls is not None else getattr(module_cls, "Config", None)
            self._entries[name] = (cfg, module_cls)  # type: ignore[arg-type]
            return module_cls

        return decorator

    def build(self, config: ConfigT, **kwargs: Any) -> ModuleT:
        type_name = getattr(config, "type", None)
        if type_name is None:
            raise ValueError(f"Config must have a 'type' field, got {type(config)}")
        if type_name not in self._entries:
            available = ", ".join(sorted(self._entries.keys()))
            raise KeyError(f"Unknown {self._name} type '{type_name}'. Available: [{available}]")
        _, module_cls = self._entries[type_name]
        return module_cls(config=config, **kwargs)  # type: ignore[call-arg]

    def list(self) -> list[str]:
        return list(self._entries.keys())

    def get_default_config(self, name: str) -> ConfigT:
        if name not in self._entries:
            available = ", ".join(sorted(self._entries.keys()))
            raise KeyError(f"Unknown {self._name} type '{name}'. Available: [{available}]")
        config_cls, _ = self._entries[name]
        if config_cls is None:
            raise ValueError(f"No config class registered for '{name}'")
        return config_cls()  # type: ignore[call-arg]
