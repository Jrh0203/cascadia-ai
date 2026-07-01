#!/usr/bin/env python3
"""Compare ADR 0114 origin and replay scientific payloads."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import blake3
from cascadia_mlx.full_legal_hierarchical_factor_oracle import ARMS

EXPERIMENT_ID = "full-legal-hierarchical-factor-oracle-v1"


def _digest(value: Any) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def compare_reports(
    origin: dict[str, Any],
    replay: dict[str, Any],
) -> dict[str, Any]:
    if (
        origin.get("experiment_id") != EXPERIMENT_ID
        or replay.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("unexpected ADR 0114 experiment identity")
    left = origin["scientific"]
    right = replay["scientific"]
    index = int(left.get("arm_index", -1))
    if (
        index not in range(4)
        or left.get("arm") != ARMS[index]
        or right.get("arm") != ARMS[index]
        or index != int(right.get("arm_index", -1))
    ):
        raise ValueError("ADR 0114 origin/replay arm identity differs")
    left_digest = _digest(left)
    right_digest = _digest(right)
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "arm": ARMS[index],
        "arm_index": index,
        "origin_host": origin["telemetry"]["host"],
        "replay_host": replay["telemetry"]["host"],
        "origin_scientific_blake3": left_digest,
        "replay_scientific_blake3": right_digest,
        "scientific_payload_identical": left_digest == right_digest,
        "origin_telemetry": origin["telemetry"],
        "replay_telemetry": replay["telemetry"],
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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
