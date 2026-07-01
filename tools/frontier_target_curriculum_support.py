#!/usr/bin/env python3
"""Independent reachability and trajectory audits for ADR 0091."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    GradedOracleDataset,
)
from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
    build_frontier_anchored_target_mask,
)

RESIDUAL_RANGES = (0.0, 3.0, 6.0, 12.0)


def audit_reachability(dataset: GradedOracleDataset) -> dict[str, Any]:
    """Measure the target ceiling under optimistic bounded residual scores."""
    hits = {value: 0 for value in RESIDUAL_RANGES}
    exact = {value: 0 for value in RESIDUAL_RANGES}
    required: list[float] = []
    groups = 0
    targets = 0
    for batch in dataset.batches(
        1,
        maximum_actions_per_batch=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
        maximum_group_actions=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    ):
        mask = np.asarray(batch.candidate_mask)[0]
        count = int(np.sum(mask))
        screen = np.asarray(batch.screen_value)[0, :count]
        flags = np.asarray(batch.source_flags)[0, :count]
        hashes = np.asarray(batch.action_hash)[0, :count]
        target = build_frontier_anchored_target_mask(
            r1200_mean=np.asarray(batch.r1200_mean),
            r1200_mask=np.asarray(batch.r1200_mask),
            source_flags=np.asarray(batch.source_flags),
            candidate_mask=np.asarray(batch.candidate_mask),
            action_hashes=np.asarray(batch.action_hash),
        )[0, :count]
        eligible = (flags & GRADED_SOURCE_CHAMPION_FRONTIER) == 0
        non_target = eligible & ~target
        quota = int(np.sum(target))
        required.append(
            max(
                0.0,
                float(np.max(screen[non_target]) - np.min(screen[target])) / 2.0,
            )
        )
        eligible_indices = np.flatnonzero(eligible)
        for residual_range in RESIDUAL_RANGES:
            optimistic = screen.copy()
            optimistic[target] += residual_range
            optimistic[non_target] -= residual_range
            ranked = np.asarray(
                sorted(
                    (int(index) for index in eligible_indices),
                    key=lambda index: (
                        -float(optimistic[index]),
                        bytes(hashes[index]),
                    ),
                ),
                dtype=np.int32,
            )[:quota]
            recalled = int(np.sum(target[ranked]))
            hits[residual_range] += recalled
            exact[residual_range] += int(recalled == quota)
        groups += 1
        targets += quota
    return {
        "split": dataset.split,
        "groups": groups,
        "target_positives": targets,
        "required_symmetric_residual_range": {
            "mean": float(np.mean(required)),
            "p50": float(np.quantile(required, 0.50)),
            "p90": float(np.quantile(required, 0.90)),
            "p95": float(np.quantile(required, 0.95)),
            "max": float(np.max(required)),
        },
        "ceilings": {
            str(value): {
                "target_positive_recall": hits[value] / targets,
                "target_set_exact_fraction": exact[value] / groups,
            }
            for value in RESIDUAL_RANGES
        },
    }


def audit_trajectory(metrics_path: Path) -> dict[str, Any]:
    """Summarize whether the rejected run ever learned its target set."""
    events = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    if not events:
        raise ValueError("training trajectory is empty")
    epochs = [
        {
            "epoch": int(event["epoch"]),
            "target_positive_recall": float(
                event["validation"]["target_positive_recall"]
            ),
            "target_set_exact_fraction": float(
                event["validation"]["target_set_exact_fraction"]
            ),
            "exact_winner_recall": float(
                event["validation"]["top64_r4800_winner_recall"]
            ),
            "training_objective": float(
                event["validation"]["training_objective"]
            ),
        }
        for event in events
    ]
    best = max(
        epochs,
        key=lambda event: (
            event["target_positive_recall"],
            event["target_set_exact_fraction"],
            event["exact_winner_recall"],
            -event["epoch"],
        ),
    )
    return {
        "epochs": epochs,
        "best_target_epoch": best,
        "target_recall_range": {
            "min": min(event["target_positive_recall"] for event in epochs),
            "max": max(event["target_positive_recall"] for event in epochs),
        },
        "exact_target_sets_ever_recovered": any(
            event["target_set_exact_fraction"] > 0.0 for event in epochs
        ),
    }


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    reachability = subparsers.add_parser("reachability")
    reachability.add_argument("--train-dataset", type=Path, required=True)
    reachability.add_argument("--validation-dataset", type=Path, required=True)
    reachability.add_argument("--output", type=Path, required=True)
    trajectory = subparsers.add_parser("trajectory")
    trajectory.add_argument("--metrics", type=Path, required=True)
    trajectory.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "reachability":
        report = {
            "schema_version": 1,
            "experiment_id": "complete-action-frontier-target-curriculum-v1",
            "audit": "residual-reachability",
            "train": audit_reachability(GradedOracleDataset(args.train_dataset)),
            "validation": audit_reachability(
                GradedOracleDataset(args.validation_dataset)
            ),
            "test_split_opened": False,
        }
    else:
        report = {
            "schema_version": 1,
            "experiment_id": "complete-action-frontier-target-curriculum-v1",
            "audit": "source-training-trajectory",
            "trajectory": audit_trajectory(args.metrics),
            "test_split_opened": False,
        }
    _write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
