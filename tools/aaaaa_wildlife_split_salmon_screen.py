#!/usr/bin/env python3
"""Run fixed threshold-feasibility screens for split Salmon-A branches."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from tools import aaaaa_wildlife_split_salmon_feasibility as feasibility
from tools.aaaaa_wildlife_gap_two_salmon_pair_bound import SCREEN_CASES


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*, workers: int, per_shape_time_limit: float) -> dict[str, Any]:
    started = time.monotonic()
    results = []
    for index, (counts, target) in enumerate(SCREEN_CASES, 1):
        result = feasibility.split_branch_bound(
            counts,
            target,
            workers=workers,
            per_shape_time_limit=per_shape_time_limit,
        )
        results.append({"counts": list(counts), "target": target, "bound": result})
        print(
            f"case={index}/{len(SCREEN_CASES)} counts={counts} target={target} "
            f"status={result['status']} cases={result['cases']}",
            flush=True,
        )
    source = Path(__file__).resolve()
    feasibility_source = Path(feasibility.__file__).resolve()
    return {
        "schema": "aaaaa-split-salmon-feasibility-screen-v1",
        "proof_complete": all(row["bound"]["status"] == "INFEASIBLE" for row in results),
        "scope": "split two-singleton Salmon-A branch only",
        "configuration": {
            "workers": workers,
            "per_submodel_time_limit_seconds": per_shape_time_limit,
            "cases_sequential": True,
        },
        "source_sha256": file_sha256(source),
        "feasibility_source_sha256": file_sha256(feasibility_source),
        "results": results,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--per-shape-time-limit", type=float, default=30.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = run(workers=args.workers, per_shape_time_limit=args.per_shape_time_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "proof_complete": payload["proof_complete"],
                "elapsed_seconds": payload["elapsed_seconds"],
            }
        )
    )
    return 0 if payload["proof_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
