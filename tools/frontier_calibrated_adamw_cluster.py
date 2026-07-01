#!/usr/bin/env python3
"""Run one ADR 0107 stage through a one-MLX-process-per-host queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import frontier_arbitrary_precision_cluster as queue

EXPERIMENT_ID = "complete-action-frontier-calibrated-monotone-adamw-v1"
BASE_ARTIFACT_ROOT = (
    "artifacts/experiments/"
    "complete-action-frontier-calibrated-monotone-adamw-v1"
)
MODULE_NAME = "cascadia_mlx.graded_oracle_frontier_calibrated_adamw"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("free", "neural"), required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    args = parser.parse_args()
    queue.EXPERIMENT_ID = EXPERIMENT_ID
    queue.MODULE_NAME = MODULE_NAME
    queue.INITIAL_CAPACITY = 1
    queue.MAXIMUM_CAPACITY = 1
    queue.CAPACITY_STEP = 1
    if args.stage == "free":
        queue.GROUPS = 24
        queue.COMMAND_NAME = "free-group"
    else:
        queue.GROUPS = 4
        queue.COMMAND_NAME = "neural-group"
    queue.ARTIFACT_ROOT = f"{BASE_ARTIFACT_ROOT}/{args.stage}"
    state_path = (
        Path(queue.ARTIFACT_ROOT) / "scheduler" / "state.json"
    )
    state = queue.run_queue(state_path, args.poll_seconds)
    print(
        json.dumps(
            {
                "experiment_id": state["experiment_id"],
                "stage": args.stage,
                "campaign_wall_seconds": state["campaign_wall_seconds"],
                "completed_tasks": sum(
                    int(task["status"] == "done")
                    for task in state["tasks"].values()
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
