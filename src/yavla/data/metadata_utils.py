"""Helpers for interpreting LeRobot metadata structures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_task_lookup(tasks: Any) -> dict[int, str]:
    """Build ``task_index -> task_name`` lookup from metadata tasks."""

    if tasks is None:
        return {}

    lookup: dict[int, str] = {}

    if hasattr(tasks, "iterrows"):
        for index, row in tasks.iterrows():
            row_mapping = row.to_dict() if hasattr(row, "to_dict") else dict(row)

            task_index_raw = row_mapping.get("task_index", index)
            try:
                task_index = int(task_index_raw)
            except (TypeError, ValueError):
                continue

            task_name: str | None = None
            for candidate in ("task", "task_name", "name"):
                value = row_mapping.get(candidate)
                if value is not None:
                    task_name = str(value)
                    break
            if task_name is None:
                task_name = str(index)

            lookup[task_index] = task_name
        return lookup

    if isinstance(tasks, Mapping):
        for key, value in tasks.items():
            try:
                task_index = int(key)
            except (TypeError, ValueError):
                continue
            lookup[task_index] = str(value)
        return lookup

    if isinstance(tasks, list):
        for entry in tasks:
            if not isinstance(entry, Mapping):
                continue
            if "task_index" not in entry:
                continue
            try:
                task_index = int(entry["task_index"])
            except (TypeError, ValueError):
                continue
            task_name = entry.get("task", entry.get("task_name", entry.get("name", task_index)))
            lookup[task_index] = str(task_name)

    return lookup
