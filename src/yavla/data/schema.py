"""Shared dataset sample schema contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

REQUIRED_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "episode_index",
        "index",
        "timestamp",
        "frame_index",
    }
)

TASK_FIELDS: frozenset[str] = frozenset({"task_index", "task"})


def validate_sample_schema(sample: Mapping[str, Any], *, require_task_name: bool = False) -> None:
    """Validate that a sample includes required metadata/task fields.

    Args:
        sample: Candidate sample dictionary.
        require_task_name: Whether to require the human-readable `task` field.

    Raises:
        KeyError: If required metadata/task fields are missing.
    """

    missing_metadata = REQUIRED_METADATA_KEYS.difference(sample.keys())
    if missing_metadata:
        raise KeyError(f"Sample is missing required metadata keys: {sorted(missing_metadata)}")

    if "task_index" not in sample:
        raise KeyError("Sample is missing required task field: task_index")

    if require_task_name and "task" not in sample:
        raise KeyError("Sample is missing required task field: task")
