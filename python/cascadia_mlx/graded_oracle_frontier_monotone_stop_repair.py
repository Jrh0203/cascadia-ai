"""ADR 0108 numerical-stop repair for saturated monotone AdamW groups."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import blake3

from cascadia_mlx.graded_oracle_frontier_calibrated_adamw import (
    MEMORY_GATE_BYTES,
    _report,
    _resource_passed,
    run_free_group,
)
from cascadia_mlx.graded_oracle_frontier_fit_interference import (
    _aggregate_metrics,
)
from cascadia_mlx.graded_oracle_frontier_free_residual import (
    _closed_domains,
)

EXPERIMENT_ID = "complete-action-frontier-monotone-adamw-stop-repair-v1"
ARM = "monotone-adamw-stop-repair-group"
FROZEN_EXPERIMENT_ID = (
    "complete-action-frontier-calibrated-monotone-adamw-v1"
)
REPAIR_GROUPS = (0, 2, 8, 14, 23)
FROZEN_GROUPS = tuple(
    group_index
    for group_index in range(24)
    if group_index not in REPAIR_GROUPS
)
FROZEN_FREE_BLAKE3 = (
    "f550f552a14400ffe6f33ec1b3cacea355c0b138651f815a5767336455b1b184"
)
FROZEN_SOURCE_BLAKE3 = (
    "1ded82bb44d6d43cd0e5ac097d68c6621f763bcf7409e44114f985b363206546"
)
EXPECTED_HOSTS = {"john1", "john2", "john3", "john4"}


def _canonical_host(value: str) -> str:
    return "john1" if value.lower().startswith("johns-mac-mini") else value


def _digest_path(path: Path) -> str:
    return blake3.blake3(path.read_bytes()).hexdigest()


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
    if hosts != EXPECTED_HOSTS:
        raise ValueError("ADR 0108 source identities do not cover four hosts")
    if len(identities) != 1:
        raise ValueError("ADR 0108 cluster source identities differ")
    files, bundle_sha256 = next(iter(identities))
    return {
        "hosts": sorted(hosts),
        "files": files,
        "bundle_sha256": bundle_sha256,
    }


def _validate_comparison(
    comparison: dict[str, Any],
    *,
    experiment_id: str,
    arm: str,
    group_index: int,
) -> bool:
    return bool(
        comparison.get("experiment_id") == experiment_id
        and comparison.get("arm") == arm
        and int(comparison.get("group_index", -1)) == group_index
        and comparison.get("scientific_payload_identical") is True
        and _canonical_host(str(comparison["origin_host"]))
        != _canonical_host(str(comparison["replay_host"]))
        and _resource_passed(comparison["origin_telemetry"])
        and _resource_passed(comparison["replay_telemetry"])
    )


def run_repair_group(
    dataset_root: Path,
    cache_root: Path,
    selected_run: Path,
    group_index: int,
) -> dict[str, Any]:
    """Run one of the five preregistered saturated groups."""
    if group_index not in REPAIR_GROUPS:
        raise ValueError("group is not in the ADR 0108 repair set")
    return run_free_group(
        dataset_root,
        cache_root,
        selected_run,
        group_index,
        experiment_id=EXPERIMENT_ID,
        arm=ARM,
        allow_numerical_convergence=True,
    )


def combine_repaired_free(
    *,
    repair_paths: list[Path],
    repair_comparison_paths: list[Path],
    frozen_free_path: Path,
    frozen_comparison_paths: list[Path],
    frozen_source_bundle: Path,
    source_identity_paths: list[Path],
) -> dict[str, Any]:
    """Merge five repaired groups with 19 byte-frozen ADR 0107 groups."""
    frozen_free_identity = _digest_path(frozen_free_path)
    frozen_source_identity = _digest_path(frozen_source_bundle)
    lineage_passed = bool(
        frozen_free_identity == FROZEN_FREE_BLAKE3
        and frozen_source_identity == FROZEN_SOURCE_BLAKE3
    )
    frozen_free = _load(frozen_free_path)
    if (
        frozen_free.get("experiment_id") != FROZEN_EXPERIMENT_ID
        or not lineage_passed
    ):
        raise ValueError("ADR 0107 frozen evidence identity differs")
    frozen_by_group = {
        int(group["group_index"]): group
        for group in frozen_free["scientific"]["groups"]
        if int(group["group_index"]) in FROZEN_GROUPS
    }
    if set(frozen_by_group) != set(FROZEN_GROUPS):
        raise ValueError("ADR 0107 frozen completed group set is incomplete")

    repairs: dict[int, dict[str, Any]] = {}
    repair_telemetry: dict[int, dict[str, Any]] = {}
    for path in repair_paths:
        report = _load(path)
        scientific = report["scientific"]
        group_index = int(scientific["group_index"])
        if (
            report.get("experiment_id") != EXPERIMENT_ID
            or scientific.get("arm") != ARM
            or group_index not in REPAIR_GROUPS
            or group_index in repairs
        ):
            raise ValueError(f"invalid ADR 0108 repair report: {path}")
        repairs[group_index] = scientific
        repair_telemetry[group_index] = report["telemetry"]
    if set(repairs) != set(REPAIR_GROUPS):
        raise ValueError("ADR 0108 repair group set is incomplete")

    repair_comparisons = {
        int(comparison["group_index"]): comparison
        for comparison in map(_load, repair_comparison_paths)
    }
    frozen_comparisons = {
        int(comparison["group_index"]): comparison
        for comparison in map(_load, frozen_comparison_paths)
    }
    if set(repair_comparisons) != set(REPAIR_GROUPS):
        raise ValueError("ADR 0108 repair replay set is incomplete")
    if set(frozen_comparisons) != set(FROZEN_GROUPS):
        raise ValueError("ADR 0107 frozen replay set is incomplete")

    repair_pipeline = all(
        all(bool(value) for value in repairs[index]["gates"].values())
        and repairs[index]["failure"] is None
        and repairs[index]["numerical_convergence"] is not None
        and _resource_passed(repair_telemetry[index])
        and _validate_comparison(
            repair_comparisons[index],
            experiment_id=EXPERIMENT_ID,
            arm=ARM,
            group_index=index,
        )
        and repairs[index]["test_split_opened"] is False
        and repairs[index]["gameplay_opened"] is False
        and repairs[index]["new_teacher_compute_used"] is False
        and repairs[index]["external_compute_used"] is False
        for index in REPAIR_GROUPS
    )
    frozen_pipeline = all(
        all(bool(value) for value in frozen_by_group[index]["gates"].values())
        and frozen_by_group[index]["failure"] is None
        and _validate_comparison(
            frozen_comparisons[index],
            experiment_id=FROZEN_EXPERIMENT_ID,
            arm="calibrated-free-residual-group",
            group_index=index,
        )
        and frozen_by_group[index]["test_split_opened"] is False
        and frozen_by_group[index]["gameplay_opened"] is False
        and frozen_by_group[index]["new_teacher_compute_used"] is False
        and frozen_by_group[index]["external_compute_used"] is False
        for index in FROZEN_GROUPS
    )
    source_identity = _validate_source_identities(source_identity_paths)
    groups = [
        repairs.get(index, frozen_by_group.get(index))
        for index in range(24)
    ]
    if any(group is None for group in groups):
        raise AssertionError("recombined ADR 0108 group list is incomplete")
    complete_groups = [group for group in groups if group is not None]
    aggregate = _aggregate_metrics(
        [group["final"] for group in complete_groups]
    )
    aggregate_at_120 = _aggregate_metrics(
        [
            next(
                event["metrics"]
                for event in group["trajectory"]
                if event["updates"] == 120
            )
            for group in complete_groups
        ]
    )
    strength_gate = bool(
        aggregate["target_positive_recall"] >= 0.95
        and aggregate["target_set_exact_fraction"] >= 0.75
    )
    pipeline = bool(
        lineage_passed and repair_pipeline and frozen_pipeline
    )
    if not pipeline:
        classification = "monotone_adamw_stop_repair_invalid"
    elif not strength_gate:
        classification = "calibrated_optimizer_mechanism_insufficient"
    else:
        classification = "free_stage_passed"
    scientific = {
        "arm": "monotone-adamw-stop-repair-combined",
        "classification": classification,
        "groups": complete_groups,
        "aggregate_at_120": aggregate_at_120,
        "aggregate": aggregate,
        "source_identity": source_identity,
        "frozen_lineage": {
            "free_combined_blake3": frozen_free_identity,
            "source_bundle_blake3": frozen_source_identity,
            "reused_groups": list(FROZEN_GROUPS),
            "repair_groups": list(REPAIR_GROUPS),
        },
        "gates": {
            "frozen_lineage_passed": lineage_passed,
            "repair_pipeline_passed": repair_pipeline,
            "frozen_19_pipeline_passed": frozen_pipeline,
            "recombined_pipeline_passed": pipeline,
            "free_strength_gate_passed": strength_gate,
            "all_five_repair_replays_identical": all(
                repair_comparisons[index][
                    "scientific_payload_identical"
                ]
                is True
                for index in REPAIR_GROUPS
            ),
            "all_19_frozen_replays_identical": all(
                frozen_comparisons[index][
                    "scientific_payload_identical"
                ]
                is True
                for index in FROZEN_GROUPS
            ),
        },
        "memory_gate_bytes": MEMORY_GATE_BYTES,
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
    group = subparsers.add_parser("repair-group")
    group.add_argument("--dataset", type=Path, required=True)
    group.add_argument("--cache", type=Path, required=True)
    group.add_argument("--selected-run", type=Path, required=True)
    group.add_argument("--analytic", type=Path)
    group.add_argument("--group-index", type=int, required=True)
    group.add_argument("--output", type=Path, required=True)
    combine = subparsers.add_parser("combine-free")
    combine.add_argument(
        "--repair-group",
        type=Path,
        action="append",
        required=True,
    )
    combine.add_argument(
        "--repair-comparison",
        type=Path,
        action="append",
        required=True,
    )
    combine.add_argument("--frozen-free", type=Path, required=True)
    combine.add_argument(
        "--frozen-comparison",
        type=Path,
        action="append",
        required=True,
    )
    combine.add_argument("--frozen-source-bundle", type=Path, required=True)
    combine.add_argument(
        "--source-identity",
        type=Path,
        action="append",
        required=True,
    )
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "repair-group":
        report = run_repair_group(
            args.dataset,
            args.cache,
            args.selected_run,
            args.group_index,
        )
    else:
        report = combine_repaired_free(
            repair_paths=args.repair_group,
            repair_comparison_paths=args.repair_comparison,
            frozen_free_path=args.frozen_free,
            frozen_comparison_paths=args.frozen_comparison,
            frozen_source_bundle=args.frozen_source_bundle,
            source_identity_paths=args.source_identity,
        )
    _write_json(args.output, report)
    if args.command == "repair-group":
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
