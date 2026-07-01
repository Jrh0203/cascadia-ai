#!/usr/bin/env python3
"""Compare ADR 0102 origin/replay science while excluding host wall time."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import blake3

EXPERIMENT_ID = "complete-action-frontier-fit-interference-audit-v1"


def canonical_scientific_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Remove only explicitly non-scientific trajectory timing fields."""
    if report.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected ADR 0102 experiment identity")
    scientific = report.get("scientific")
    if not isinstance(scientific, dict):
        raise ValueError("ADR 0102 report is missing scientific payload")
    return _without_elapsed_seconds(scientific)


def _without_elapsed_seconds(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_elapsed_seconds(item)
            for key, item in value.items()
            if key != "elapsed_seconds"
        }
    if isinstance(value, list):
        return [_without_elapsed_seconds(item) for item in value]
    return value


def canonical_digest(value: Any) -> str:
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
    origin_payload = canonical_scientific_payload(origin)
    replay_payload = canonical_scientific_payload(replay)
    origin_arm = origin_payload.get("arm")
    replay_arm = replay_payload.get("arm")
    if origin_arm != replay_arm:
        raise ValueError("origin and replay arm identities differ")
    origin_digest = canonical_digest(origin_payload)
    replay_digest = canonical_digest(replay_payload)
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "arm": origin_arm,
        "origin_host": origin["telemetry"]["host"],
        "replay_host": replay["telemetry"]["host"],
        "origin_scientific_blake3": origin_digest,
        "replay_scientific_blake3": replay_digest,
        "scientific_payload_identical": origin_digest == replay_digest,
        "origin_telemetry": origin["telemetry"],
        "replay_telemetry": replay["telemetry"],
    }


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read replay input {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"replay input is not an object: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
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
    report = compare_reports(load_json(args.origin), load_json(args.replay))
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["scientific_payload_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
