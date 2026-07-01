#!/usr/bin/env python3
"""Run ADR 0106 through the validated dynamic four-host queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import frontier_arbitrary_precision_cluster as queue

EXPERIMENT_ID = "complete-action-frontier-exact-float-decimal-control-v1"
ARTIFACT_ROOT = (
    "artifacts/experiments/"
    "complete-action-frontier-exact-float-decimal-control-v1"
)
MODULE_NAME = "cascadia_mlx.graded_oracle_frontier_exact_float_decimal"


def main() -> int:
    queue.EXPERIMENT_ID = EXPERIMENT_ID
    queue.ARTIFACT_ROOT = ARTIFACT_ROOT
    queue.MODULE_NAME = MODULE_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        type=Path,
        default=Path(ARTIFACT_ROOT) / "scheduler" / "state.json",
    )
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    args = parser.parse_args()
    state = queue.run_queue(args.state, args.poll_seconds)
    print(
        json.dumps(
            {
                "experiment_id": state["experiment_id"],
                "campaign_wall_seconds": state["campaign_wall_seconds"],
                "completed_tasks": sum(
                    int(task["status"] == "done")
                    for task in state["tasks"].values()
                ),
                "host_capacities": {
                    host: state["hosts"][host]["capacity"]
                    for host in queue.HOSTS
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
