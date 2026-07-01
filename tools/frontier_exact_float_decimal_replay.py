#!/usr/bin/env python3
"""Compare ADR 0106 exact-float Decimal origin and replay science."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from frontier_arbitrary_precision_replay import compare_reports

EXPERIMENT_ID = "complete-action-frontier-exact-float-decimal-control-v1"
SCIENTIFIC_ARM = "exact-float-decimal-control-group"


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
    report = compare_reports(
        _load(args.origin),
        _load(args.replay),
        experiment_id=EXPERIMENT_ID,
        scientific_arm=SCIENTIFIC_ARM,
    )
    _write(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["scientific_payload_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
