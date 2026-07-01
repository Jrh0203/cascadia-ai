#!/usr/bin/env python3
"""Run ADR 0112 through the dynamic four-host static-audit queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import frontier_arbitrary_precision_cluster as queue

EXPERIMENT_ID = (
    "complete-action-frontier-local-geometry-feasibility-forensic-v1"
)
ARTIFACT_ROOT = (
    "artifacts/experiments/"
    "complete-action-frontier-local-geometry-feasibility-forensic-v1"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    args = parser.parse_args()
    queue.EXPERIMENT_ID = EXPERIMENT_ID
    queue.MODULE_NAME = (
        "cascadia_mlx.graded_oracle_frontier_local_geometry_feasibility"
    )
    queue.COMMAND_NAME = "group"
    queue.GROUPS = 4
    queue.GROUP_INDICES = None
    queue.INITIAL_CAPACITY = 1
    queue.MAXIMUM_CAPACITY = 1
    queue.CAPACITY_STEP = 1
    queue.ARTIFACT_ROOT = ARTIFACT_ROOT
    queue.ANALYTIC = (
        "artifacts/experiments/"
        "complete-action-frontier-calibrated-local-geometry-adapter-v1/"
        "reports/combined.json"
    )
    state_path = Path(ARTIFACT_ROOT) / "scheduler" / "state.json"
    state = queue.run_queue(state_path, args.poll_seconds)
    print(
        json.dumps(
            {
                "experiment_id": state["experiment_id"],
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
