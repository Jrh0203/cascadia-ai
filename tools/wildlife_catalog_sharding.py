"""Shared fail-closed task selection for distributed wildlife catalogs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "wildlife-catalog-taskset-v1"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_taskset(
    path: Path,
    *,
    scoring_cards: str,
    canonical_counts: list[tuple[int, int, int, int, int]],
) -> tuple[set[tuple[int, int, int, int, int]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported taskset schema in {path}")
    if payload.get("scoring_cards") != scoring_cards:
        raise ValueError(
            f"taskset scoring cards {payload.get('scoring_cards')!r}, expected {scoring_cards}"
        )
    rows = [tuple(int(value) for value in row) for row in payload.get("counts", [])]
    if any(len(row) != 5 for row in rows):
        raise ValueError("every taskset row must contain five counts")
    if len(rows) != len(set(rows)):
        raise ValueError("taskset contains duplicate count vectors")
    canonical = set(canonical_counts)
    unexpected = set(rows) - canonical
    if unexpected:
        raise ValueError(f"taskset contains noncanonical counts: {sorted(unexpected)}")
    expected_count = payload.get("task_count")
    if expected_count is not None and int(expected_count) != len(rows):
        raise ValueError(f"taskset count says {expected_count}, contains {len(rows)} rows")
    return set(rows), {
        "path": str(path),
        "sha256": file_sha256(path),
        "schema": SCHEMA,
        "scoring_cards": scoring_cards,
        "task_count": len(rows),
    }


def select_shard(
    tasks: list[dict[str, Any]],
    *,
    shard_index: int,
    shard_count: int,
) -> list[dict[str, Any]]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index {shard_index} outside [0, {shard_count})")
    return tasks[shard_index::shard_count]

