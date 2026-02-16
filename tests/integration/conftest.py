"""Pytest fixtures and marker plumbing for dataset integration tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[import-untyped]

REPO_ID = "lerobot/pusht"
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "yavla-test-data"
FALLBACK_CACHE_ROOT = Path.cwd() / ".cache" / "yavla-test-data"
ROOT_SENTINEL_NAME = "lerobot-pusht-root.txt"


def _is_usable_dataset_root(path: Path) -> bool:
    """Check for minimum LeRobot v3 layout signals used by local backends."""

    return (path / "meta" / "info.json").is_file() and (path / "data").is_dir()


def _extract_dataset_root(dataset: Any) -> Path | None:
    meta = getattr(dataset, "meta", None)
    meta_root = getattr(meta, "root", None) if meta is not None else None
    if meta_root is not None:
        return Path(str(meta_root)).expanduser().resolve()

    dataset_root = getattr(dataset, "root", None)
    if dataset_root is not None:
        return Path(str(dataset_root)).expanduser().resolve()

    return None


def _sentinel_path(cache_root: Path) -> Path:
    return cache_root / ROOT_SENTINEL_NAME


def _select_cache_root() -> Path:
    env_override = os.environ.get("YAVLA_TEST_DATA_CACHE")
    candidates = [Path(env_override).expanduser()] if env_override else []
    candidates.extend([DEFAULT_CACHE_ROOT, FALLBACK_CACHE_ROOT])

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate.resolve()
        except OSError:
            continue

    pytest.skip(
        f"unable to create test-data cache directory in any supported location: {[str(path) for path in candidates]}"
    )


def _cached_root_from_sentinel(cache_root: Path) -> Path | None:
    root_sentinel = _sentinel_path(cache_root)
    if not root_sentinel.is_file():
        return None

    candidate = Path(root_sentinel.read_text(encoding="utf-8").strip()).expanduser()
    if _is_usable_dataset_root(candidate):
        return candidate.resolve()

    try:
        root_sentinel.unlink()
    except OSError:
        pass
    return None


def _download_or_skip(cache_root: Path) -> Path:
    try:
        dataset = LeRobotDataset(repo_id=REPO_ID, root=cache_root, video_backend="pyav")
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"unable to download {REPO_ID} into {cache_root}: {exc}")

    resolved_root = _extract_dataset_root(dataset)
    if resolved_root is None or not _is_usable_dataset_root(resolved_root):
        pytest.skip(f"downloaded {REPO_ID} but could not resolve a usable local dataset root")

    _sentinel_path(cache_root).write_text(f"{resolved_root}\n", encoding="utf-8")
    return resolved_root


@pytest.fixture(scope="session")
def pusht_root() -> Path:
    """Return local root for a cached/downloaded ``lerobot/pusht`` dataset copy."""

    cache_root = _select_cache_root()
    cached_root = _cached_root_from_sentinel(cache_root)
    if cached_root is not None:
        return cached_root
    return _download_or_skip(cache_root)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply ``integration`` marker to all tests in this directory."""

    del config
    integration_dir = Path(__file__).parent.resolve()
    for item in items:
        path_obj = getattr(item, "path", None)
        if path_obj is None:
            continue

        item_path = Path(path_obj).resolve()
        if item_path == integration_dir or integration_dir in item_path.parents:
            item.add_marker(pytest.mark.integration)
