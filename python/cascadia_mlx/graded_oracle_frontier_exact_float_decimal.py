"""Exact-float arbitrary-precision frontier control for ADR 0106."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cascadia_mlx.graded_oracle_frontier_arbitrary_precision import (
    MEMORY_GATE_BYTES,
    combine_group_reports,
    run_decimal_group,
)

EXPERIMENT_ID = "complete-action-frontier-exact-float-decimal-control-v1"
SCIENTIFIC_ARM = "exact-float-decimal-control-group"
INVALID_CLASSIFICATION = "exact_float_decimal_control_invalid"


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    group = subparsers.add_parser("group")
    group.add_argument("--dataset", type=Path, required=True)
    group.add_argument("--cache", type=Path, required=True)
    group.add_argument("--selected-run", type=Path, required=True)
    group.add_argument("--analytic", type=Path, required=True)
    group.add_argument("--group-index", type=int, required=True)
    group.add_argument("--output", type=Path, required=True)
    combine = subparsers.add_parser("combine")
    combine.add_argument("--group", type=Path, action="append", required=True)
    combine.add_argument(
        "--replay-comparison",
        type=Path,
        action="append",
        required=True,
    )
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "group":
        report = run_decimal_group(
            args.dataset,
            args.cache,
            args.selected_run,
            args.analytic,
            args.group_index,
            experiment_id=EXPERIMENT_ID,
            scientific_arm=SCIENTIFIC_ARM,
            exact_float_ranks=True,
        )
    else:
        report = combine_group_reports(
            args.group,
            args.replay_comparison,
            experiment_id=EXPERIMENT_ID,
            scientific_arm=SCIENTIFIC_ARM,
            invalid_classification=INVALID_CLASSIFICATION,
        )
    _write_json(args.output, report)
    if args.command == "group":
        telemetry = report["telemetry"]
        print(
            json.dumps(
                {
                    "group_index": report["scientific"]["group_index"],
                    "group_passed": report["scientific"]["gates"][
                        "group_passed"
                    ],
                    "resource_qualification_passed": bool(
                        int(telemetry["peak_process_rss_bytes"])
                        <= MEMORY_GATE_BYTES
                        and int(telemetry["process_swaps"]) == 0
                        and telemetry["system_swap_delta_bytes"] is not None
                        and int(telemetry["system_swap_delta_bytes"]) <= 0
                    ),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
