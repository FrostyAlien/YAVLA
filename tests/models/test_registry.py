"""Unit tests for the generic Registry."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from yavla.models.registry import Registry


@dataclass
class _DummyConfig:
    type: str = "dummy"
    value: int = 42


class _DummyModule:
    def __init__(self, config: _DummyConfig, **kwargs: object) -> None:
        self.config = config


@dataclass
class _OtherConfig:
    type: str = "other"


class _OtherModule:
    def __init__(self, config: _OtherConfig, **kwargs: object) -> None:
        self.config = config


class TestRegistry:
    def test_register_and_build(self) -> None:
        reg: Registry[_DummyConfig, _DummyModule] = Registry("test")
        reg.register("dummy", _DummyConfig)(_DummyModule)
        module = reg.build(_DummyConfig())
        assert isinstance(module, _DummyModule)
        assert module.config.value == 42

    def test_list(self) -> None:
        reg: Registry[object, object] = Registry("test")
        reg.register("a", _DummyConfig)(_DummyModule)
        reg.register("b", _OtherConfig)(_OtherModule)
        assert sorted(reg.list()) == ["a", "b"]

    def test_duplicate_raises(self) -> None:
        reg: Registry[object, object] = Registry("test")
        reg.register("dup", _DummyConfig)(_DummyModule)
        with pytest.raises(ValueError, match="already registered"):
            reg.register("dup", _DummyConfig)(_DummyModule)

    def test_unknown_type_raises(self) -> None:
        reg: Registry[_DummyConfig, _DummyModule] = Registry("test")
        cfg = _DummyConfig(type="nonexistent")
        with pytest.raises(KeyError, match="nonexistent"):
            reg.build(cfg)

    def test_get_default_config(self) -> None:
        reg: Registry[_DummyConfig, _DummyModule] = Registry("test")
        reg.register("dummy", _DummyConfig)(_DummyModule)
        cfg = reg.get_default_config("dummy")
        assert isinstance(cfg, _DummyConfig)
        assert cfg.type == "dummy"

    def test_get_default_config_unknown_raises(self) -> None:
        reg: Registry[_DummyConfig, _DummyModule] = Registry("test")
        with pytest.raises(KeyError, match="nonexistent"):
            reg.get_default_config("nonexistent")
