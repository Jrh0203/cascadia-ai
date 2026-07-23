#!/usr/bin/env python3
"""Durable runner for the frozen AAAAA split-Salmon bitset screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

from ortools import __version__ as ORTOOLS_VERSION

from tools.aaaaa_wildlife_split_salmon_dp import split_branch_packing

CASES = (
    ((4, 5, 2, 3, 6), 67),
    ((5, 5, 2, 2, 6), 64),
    ((3, 5, 2, 4, 6), 63),
    ((3, 6, 2, 3, 6), 63),
)
DEPENDENCIES = (
    "tools/aaaaa_wildlife_split_salmon_dp.py",
    "tools/aaaaa_wildlife_gap_two_salmon_pair_bound.py",
    "tools/aaaaa_wildlife_zero_hawk_bound.py",
    "tools/aaaaa_wildlife_exact.py",
    "tools/aaaaa_wildlife_motif_certificate.py",
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def run(case_index: int) -> dict[str, Any]:
    if not 0 <= case_index < len(CASES):
        raise ValueError("case index out of range")
    counts, target = CASES[case_index]
    result = split_branch_packing(counts, target)
    return {
        "schema": "aaaaa-split-salmon-bitset-shard-v1",
        "identity": {
            "runner_source_sha256": _sha256(__file__),
            "dependency_sha256": {
                path: _sha256(path) for path in DEPENDENCIES
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "ortools": ORTOOLS_VERSION,
        },
        "case_index": case_index,
        "counts": list(counts),
        "target": target,
        "result": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.case_index)
    _write_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "case_index": payload["case_index"],
                "status": payload["result"]["status"],
                "wall_seconds": payload["result"]["wall_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
