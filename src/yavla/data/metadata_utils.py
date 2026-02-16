"""Helpers for interpreting LeRobot metadata structures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class EpisodeMediaReference:
    """Episode-level video file reference from LeRobot v3 metadata."""

    chunk_index: int
    file_index: int
    from_timestamp: float


def metadata_records(obj: Any) -> list[dict[str, Any]]:
    """Normalize metadata tables (HF Dataset / pandas / list) into row mappings."""

    if obj is None:
        return []

    if isinstance(obj, list):
        return [dict(record) for record in obj if isinstance(record, Mapping)]

    if hasattr(obj, "iterrows"):
        records: list[dict[str, Any]] = []
        for _, row in obj.iterrows():
            row_mapping = row.to_dict() if hasattr(row, "to_dict") else dict(row)
            records.append(dict(row_mapping))
        return records

    if hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
        try:
            return [dict(obj[idx]) for idx in range(len(obj))]
        except Exception:
            pass

    if hasattr(obj, "to_dict"):
        try:
            records_obj = obj.to_dict(orient="records")
        except TypeError:
            records_obj = obj.to_dict()

        if isinstance(records_obj, list):
            return [dict(record) for record in records_obj if isinstance(record, Mapping)]

        if isinstance(records_obj, Mapping):
            sequence_keys = [
                key
                for key, value in records_obj.items()
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
            ]
            if not sequence_keys:
                return []
            length = len(records_obj[sequence_keys[0]])
            rows = []
            for idx in range(length):
                row: dict[str, Any] = {}
                for key, value in records_obj.items():
                    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                        row[key] = value[idx]
                    else:
                        row[key] = value
                rows.append(row)
            return rows

    raise TypeError(f"Unsupported metadata table type: {type(obj)!r}")


def build_episode_media_lookup(
    episodes: Any,
    media_keys: Sequence[str],
) -> dict[int, dict[str, EpisodeMediaReference]]:
    """Build ``episode_index -> media_key -> video reference`` lookup."""

    lookup: dict[int, dict[str, EpisodeMediaReference]] = {}
    records = metadata_records(episodes)
    for record in records:
        try:
            episode_index = int(record["episode_index"])
        except (KeyError, TypeError, ValueError):
            continue

        episode_media = lookup.setdefault(episode_index, {})
        for media_key in media_keys:
            chunk_key = f"videos/{media_key}/chunk_index"
            file_key = f"videos/{media_key}/file_index"
            if chunk_key not in record or file_key not in record:
                continue

            try:
                chunk_index = int(record[chunk_key])
                file_index = int(record[file_key])
            except (TypeError, ValueError):
                continue

            from_timestamp_raw = record.get(f"videos/{media_key}/from_timestamp", 0.0)
            try:
                from_timestamp = float(from_timestamp_raw)
            except (TypeError, ValueError):
                from_timestamp = 0.0

            episode_media[media_key] = EpisodeMediaReference(
                chunk_index=chunk_index,
                file_index=file_index,
                from_timestamp=from_timestamp,
            )

    return lookup


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
