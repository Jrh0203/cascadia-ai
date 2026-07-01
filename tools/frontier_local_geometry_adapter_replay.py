#!/usr/bin/env python3
"""Compare ADR 0111 origin and replay scientific payloads."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import blake3

EXPERIMENT_ID = (
    "complete-action-frontier-calibrated-local-geometry-adapter-v1"
)
ARM = "calibrated-local-geometry-adapter-group"


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return blake3.blake3(payload).hexdigest()


def compare_reports(
    origin: dict[str, Any],
    replay: dict[str, Any],
) -> dict[str, Any]:
    if (
        origin.get("experiment_id") != EXPERIMENT_ID
        or replay.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("unexpected ADR 0111 experiment identity")
    left = origin["scientific"]
    right = replay["scientific"]
    group_index = int(left.get("group_index", -1))
    if (
        left.get("arm") != ARM
        or right.get("arm") != ARM
        or group_index not in range(4)
        or group_index != int(right.get("group_index", -1))
    ):
        raise ValueError("ADR 0111 origin/replay group identity differs")
    left_digest = _digest(left)
    right_digest = _digest(right)
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "arm": ARM,
        "group_index": group_index,
        "origin_host": origin["telemetry"]["host"],
        "replay_host": replay["telemetry"]["host"],
        "origin_scientific_blake3": left_digest,
        "replay_scientific_blake3": right_digest,
        "scientific_payload_identical": left_digest == right_digest,
        "origin_telemetry": origin["telemetry"],
        "replay_telemetry": replay["telemetry"],
    }


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"replay input is not an object: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_reports(_load(args.origin), _load(args.replay))
    _write(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["scientific_payload_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
