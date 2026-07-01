"""ADR 0109 calibrated neural local-fit stage."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import blake3

from cascadia_mlx.graded_oracle_frontier_calibrated_adamw import (
    _report,
    _resource_passed,
    run_neural_group,
)
from cascadia_mlx.graded_oracle_frontier_fit_interference import (
    _aggregate_metrics,
)
from cascadia_mlx.graded_oracle_frontier_free_residual import (
    _closed_domains,
)

EXPERIMENT_ID = "complete-action-frontier-calibrated-neural-stage-v1"
ARM = "calibrated-neural-local-fit-group"
GROUPS = 4
FROZEN_FREE_BLAKE3 = (
    "84d59e71f117250546f21118688ec93d40060e39547d464936c7fd4223b8630a"
)
EXPECTED_HOSTS = {"john1", "john2", "john3", "john4"}


def _canonical_host(value: str) -> str:
    return "john1" if value.lower().startswith("johns-mac-mini") else value


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON input is not an object: {path}")
    return value


def _validate_source_identities(
    paths: list[Path],
) -> dict[str, Any]:
    reports = [_load(path) for path in paths]
    hosts = {
        _canonical_host(str(report["host"]))
        for report in reports
    }
    identities = {
        (int(report["files"]), str(report["bundle_sha256"]))
        for report in reports
    }
    if hosts != EXPECTED_HOSTS or len(identities) != 1:
        raise ValueError("ADR 0109 cluster source identity differs")
    files, bundle_sha256 = next(iter(identities))
    return {
        "hosts": sorted(hosts),
        "files": files,
        "bundle_sha256": bundle_sha256,
    }


def run_group(
    dataset_root: Path,
    cache_root: Path,
    selected_run: Path,
    group_index: int,
) -> dict[str, Any]:
    """Run one preregistered full-model local-fit group."""
    return run_neural_group(
        dataset_root,
        cache_root,
        selected_run,
        group_index,
        experiment_id=EXPERIMENT_ID,
        arm=ARM,
        allow_numerical_convergence=True,
    )


def combine(
    *,
    group_paths: list[Path],
    comparison_paths: list[Path],
    free_path: Path,
    source_identity_paths: list[Path],
) -> dict[str, Any]:
    """Combine and mechanically classify the four neural groups."""
    free_identity = blake3.blake3(free_path.read_bytes()).hexdigest()
    free = _load(free_path)
    if (
        free_identity != FROZEN_FREE_BLAKE3
        or free.get("experiment_id")
        != "complete-action-frontier-monotone-adamw-stop-repair-v1"
        or free["scientific"].get("classification")
        != "free_stage_passed"
    ):
        raise ValueError("ADR 0109 frozen free-stage evidence differs")

    reports: dict[int, dict[str, Any]] = {}
    telemetry: dict[int, dict[str, Any]] = {}
    for path in group_paths:
        report = _load(path)
        scientific = report["scientific"]
        index = int(scientific["group_index"])
        if (
            report.get("experiment_id") != EXPERIMENT_ID
            or scientific.get("arm") != ARM
            or index not in range(GROUPS)
            or index in reports
        ):
            raise ValueError(f"invalid ADR 0109 group report: {path}")
        reports[index] = scientific
        telemetry[index] = report["telemetry"]
    if set(reports) != set(range(GROUPS)):
        raise ValueError("ADR 0109 group set is incomplete")

    comparisons = {
        int(comparison["group_index"]): comparison
        for comparison in map(_load, comparison_paths)
    }
    if set(comparisons) != set(range(GROUPS)):
        raise ValueError("ADR 0109 replay set is incomplete")
    replay_pipeline = all(
        comparison.get("experiment_id") == EXPERIMENT_ID
        and comparison.get("arm") == ARM
        and comparison.get("scientific_payload_identical") is True
        and _canonical_host(str(comparison["origin_host"]))
        != _canonical_host(str(comparison["replay_host"]))
        and _resource_passed(comparison["origin_telemetry"])
        and _resource_passed(comparison["replay_telemetry"])
        for comparison in comparisons.values()
    )
    group_pipeline = all(
        all(bool(value) for value in report["gates"].values())
        and report["failure"] is None
        and _resource_passed(telemetry[index])
        and report["test_split_opened"] is False
        and report["gameplay_opened"] is False
        and report["new_teacher_compute_used"] is False
        and report["external_compute_used"] is False
        for index, report in reports.items()
    )
    source_identity = _validate_source_identities(source_identity_paths)
    ordered = [reports[index] for index in range(GROUPS)]
    aggregate = _aggregate_metrics(
        [report["final"] for report in ordered]
    )
    aggregate_at_120 = _aggregate_metrics(
        [
            next(
                event["metrics"]
                for event in report["trajectory"]
                if event["exposures_per_group"] == 120
            )
            for report in ordered
        ]
    )
    terminal_strength = bool(
        aggregate["target_positive_recall"] >= 0.90
        and aggregate["target_set_exact_fraction"] >= 0.75
    )
    strength_at_120 = bool(
        aggregate_at_120["target_positive_recall"] >= 0.90
        and aggregate_at_120["target_set_exact_fraction"] >= 0.75
    )
    pipeline = bool(group_pipeline and replay_pipeline)
    if not pipeline:
        classification = "calibrated_optimizer_pipeline_invalid"
    elif not terminal_strength:
        classification = "public_observable_representation_insufficient"
    elif not strength_at_120:
        classification = "full_model_local_budget_insufficient"
    else:
        classification = "local_failure_not_reproduced"
    scientific = {
        "arm": "calibrated-neural-local-fit-combined",
        "classification": classification,
        "free_stage_blake3": free_identity,
        "groups": ordered,
        "aggregate_at_120": aggregate_at_120,
        "aggregate": aggregate,
        "source_identity": source_identity,
        "gates": {
            "neural_pipeline_passed": pipeline,
            "group_pipeline_passed": group_pipeline,
            "all_four_replays_identical": replay_pipeline,
            "strength_gate_at_120_passed": strength_at_120,
            "terminal_strength_gate_passed": terminal_strength,
        },
        **_closed_domains(),
    }
    return _report(
        scientific,
        time.perf_counter(),
        0,
        experiment_id=EXPERIMENT_ID,
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
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
    group.add_argument("--analytic", type=Path)
    group.add_argument("--group-index", type=int, required=True)
    group.add_argument("--output", type=Path, required=True)
    combined = subparsers.add_parser("combine")
    combined.add_argument(
        "--group",
        type=Path,
        action="append",
        required=True,
    )
    combined.add_argument(
        "--replay-comparison",
        type=Path,
        action="append",
        required=True,
    )
    combined.add_argument("--free", type=Path, required=True)
    combined.add_argument(
        "--source-identity",
        type=Path,
        action="append",
        required=True,
    )
    combined.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "group":
        report = run_group(
            args.dataset,
            args.cache,
            args.selected_run,
            args.group_index,
        )
    else:
        report = combine(
            group_paths=args.group,
            comparison_paths=args.replay_comparison,
            free_path=args.free,
            source_identity_paths=args.source_identity,
        )
    _write_json(args.output, report)
    if args.command == "group":
        print(
            json.dumps(
                {
                    "group_index": report["scientific"]["group_index"],
                    "resource_qualification_passed": _resource_passed(
                        report["telemetry"]
                    ),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
