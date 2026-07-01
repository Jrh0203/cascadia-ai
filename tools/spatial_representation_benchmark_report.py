#!/usr/bin/env python3
"""Merge and classify distributed R0 spatial extraction benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import blake3

AGGREGATE_SCHEMA_VERSION = 2
AGGREGATE_ID = "r0-spatial-representation-extraction-aggregate-v2"
BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_ID = "r0-spatial-representation-extraction-v1"
DEFAULT_REQUIRED_REPLICATES = 3
MEDIAN_SELECTION_CRITERION = "exact-entity-control.extraction_ns_per_record"

EXACT_ENTITY_CONTROL = "exact-entity-control"
HISTORICAL_SQUARE_21 = "historical-square-21x21-441"
REQUIRED_ARMS = (
    EXACT_ENTITY_CONTROL,
    "hex-radius-6-127",
    "hex-radius-5-91",
    "hex-radius-4-61",
    HISTORICAL_SQUARE_21,
)
REQUIRED_ARM_SET = frozenset(REQUIRED_ARMS)

COMPLETE = "r0_extraction_benchmark_complete"
INCOMPLETE = "r0_extraction_benchmark_incomplete"
SEMANTIC_FAILURE = "r0_extraction_benchmark_semantic_failure"
INSUFFICIENT_PERFORMANCE = "r0_extraction_benchmark_insufficient_performance_evidence"

EXIT_CODES = {
    COMPLETE: 0,
    INCOMPLETE: 2,
    SEMANTIC_FAILURE: 3,
    INSUFFICIENT_PERFORMANCE: 4,
}

ORDINAL_RULE = (
    "concatenated CLI dataset-root order, manifest shard order, in-shard row order; "
    "global_ordinal % shard_count == shard_index"
)
TIMING_RELATIVE_TOLERANCE = 1e-9
TIMING_ABSOLUTE_TOLERANCE = 1e-6

WEIGHTED_MEAN_FIELDS = (
    "mean_packed_bytes",
    "mean_packed_bytes_vs_position_record",
    "mean_local_capacity_rows",
    "mean_active_local_rows",
    "mean_exact_entity_rows",
    "mean_overflow_entity_rows",
    "mean_dense_raw_scalar_slots",
)
TIMING_PHASES = ("extraction", "serialization", "deserialization")
ARM_TIMING_FIELDS = frozenset(
    {
        "extraction_seconds",
        "extraction_ns_per_record",
        "extraction_records_per_second",
        "serialization_seconds",
        "serialization_ns_per_record",
        "deserialization_seconds",
        "deserialization_ns_per_record",
    }
)


class ReportError(RuntimeError):
    """Raised when an input cannot be interpreted without guessing."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def canonical_json(value: object) -> bytes:
    """Return the stable JSON encoding used for scientific identities."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def scientific_blake3(value: object) -> str:
    return blake3.blake3(canonical_json(value)).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ReportError(f"cannot read benchmark JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReportError(f"benchmark JSON root must be an object: {path}")
    return value


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReportError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReportError(f"{label} must be an array")
    return value


def _string(value: object, label: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ReportError(f"{label} must be a{' nonempty' if nonempty else ''} string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReportError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: object, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{label} must be a finite number >= {minimum}")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ReportError(f"{label} must be a finite number >= {minimum}")
    return result


def _optional_fraction(value: object, label: str) -> float | None:
    if value is None:
        return None
    result = _number(value, label)
    if result > 1.0:
        raise ReportError(f"{label} must be <= 1")
    return result


def _digest(value: object, label: str) -> str:
    result = _string(value, label)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ReportError(f"{label} must be a lowercase 64-character hexadecimal digest")
    return result


def _dataset_identity(dataset: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": dataset["dataset_id"],
        "feature_schema": dataset["feature_schema"],
        "split": dataset["split"],
        "completed_games": dataset["completed_games"],
        "total_records": dataset["total_records"],
        "global_ordinal_start": dataset["global_ordinal_start"],
        "global_ordinal_end_exclusive": dataset["global_ordinal_end_exclusive"],
    }


def _normalize_dataset(value: object, label: str) -> dict[str, Any]:
    dataset = _object(value, label)
    _string(dataset.get("root"), f"{label}.root")
    dataset_id = _string(dataset.get("dataset_id"), f"{label}.dataset_id")
    feature_schema = _string(dataset.get("feature_schema"), f"{label}.feature_schema")
    split = _string(dataset.get("split"), f"{label}.split")
    completed_games = _integer(
        dataset.get("completed_games"),
        f"{label}.completed_games",
    )
    total_records = _integer(
        dataset.get("total_records"),
        f"{label}.total_records",
        minimum=1,
    )
    global_start = _integer(
        dataset.get("global_ordinal_start"),
        f"{label}.global_ordinal_start",
    )
    global_end = _integer(
        dataset.get("global_ordinal_end_exclusive"),
        f"{label}.global_ordinal_end_exclusive",
        minimum=1,
    )
    if global_end - global_start != total_records:
        raise ReportError(f"{label} ordinal range does not equal total_records")
    eligible_records = _integer(
        dataset.get("eligible_records"),
        f"{label}.eligible_records",
    )
    loaded_records = _integer(
        dataset.get("loaded_records"),
        f"{label}.loaded_records",
    )
    if loaded_records > eligible_records:
        raise ReportError(f"{label}.loaded_records exceeds eligible_records")
    return {
        "dataset_id": dataset_id,
        "feature_schema": feature_schema,
        "split": split,
        "completed_games": completed_games,
        "total_records": total_records,
        "global_ordinal_start": global_start,
        "global_ordinal_end_exclusive": global_end,
        "eligible_records": eligible_records,
        "loaded_records": loaded_records,
        "manifest_blake3": _digest(
            dataset.get("manifest_blake3"),
            f"{label}.manifest_blake3",
        ),
    }


def _normalize_arm(value: object, label: str) -> dict[str, Any]:
    arm = _object(value, label)
    arm_id = _string(arm.get("arm"), f"{label}.arm")
    records = _integer(arm.get("records"), f"{label}.records", minimum=1)
    iterations = _integer(arm.get("iterations"), f"{label}.iterations", minimum=1)
    round_trip_verified = arm.get("round_trip_verified")
    if not isinstance(round_trip_verified, bool):
        raise ReportError(f"{label}.round_trip_verified must be boolean")

    normalized = {
        "arm": arm_id,
        "records": records,
        "iterations": iterations,
        "round_trip_verified": round_trip_verified,
        "semantic_blake3": _digest(
            arm.get("semantic_blake3"),
            f"{label}.semantic_blake3",
        ),
        "extraction_seconds": _number(
            arm.get("extraction_seconds"),
            f"{label}.extraction_seconds",
        ),
        "extraction_ns_per_record": _number(
            arm.get("extraction_ns_per_record"),
            f"{label}.extraction_ns_per_record",
        ),
        "extraction_records_per_second": _number(
            arm.get("extraction_records_per_second"),
            f"{label}.extraction_records_per_second",
        ),
        "serialization_seconds": _number(
            arm.get("serialization_seconds"),
            f"{label}.serialization_seconds",
        ),
        "serialization_ns_per_record": _number(
            arm.get("serialization_ns_per_record"),
            f"{label}.serialization_ns_per_record",
        ),
        "deserialization_seconds": _number(
            arm.get("deserialization_seconds"),
            f"{label}.deserialization_seconds",
        ),
        "deserialization_ns_per_record": _number(
            arm.get("deserialization_ns_per_record"),
            f"{label}.deserialization_ns_per_record",
        ),
        "mean_packed_bytes": _number(
            arm.get("mean_packed_bytes"),
            f"{label}.mean_packed_bytes",
        ),
        "mean_packed_bytes_vs_position_record": _number(
            arm.get("mean_packed_bytes_vs_position_record"),
            f"{label}.mean_packed_bytes_vs_position_record",
        ),
        "mean_local_capacity_rows": _number(
            arm.get("mean_local_capacity_rows"),
            f"{label}.mean_local_capacity_rows",
        ),
        "mean_active_local_rows": _number(
            arm.get("mean_active_local_rows"),
            f"{label}.mean_active_local_rows",
        ),
        "local_occupancy_fraction": _optional_fraction(
            arm.get("local_occupancy_fraction"),
            f"{label}.local_occupancy_fraction",
        ),
        "mean_exact_entity_rows": _number(
            arm.get("mean_exact_entity_rows"),
            f"{label}.mean_exact_entity_rows",
        ),
        "mean_overflow_entity_rows": _number(
            arm.get("mean_overflow_entity_rows"),
            f"{label}.mean_overflow_entity_rows",
        ),
        "positions_with_overflow": _integer(
            arm.get("positions_with_overflow"),
            f"{label}.positions_with_overflow",
        ),
        "overflow_position_fraction": _optional_fraction(
            arm.get("overflow_position_fraction"),
            f"{label}.overflow_position_fraction",
        ),
        "mean_dense_raw_scalar_slots": _number(
            arm.get("mean_dense_raw_scalar_slots"),
            f"{label}.mean_dense_raw_scalar_slots",
        ),
    }
    if normalized["positions_with_overflow"] > records:
        raise ReportError(f"{label}.positions_with_overflow exceeds records")
    if normalized["overflow_position_fraction"] is None:
        raise ReportError(f"{label}.overflow_position_fraction cannot be null")
    return normalized


def _optional_positive_integer(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _integer(value, label, minimum=1)


def _normalize_execution_provenance(value: object, label: str) -> dict[str, Any]:
    provenance = _object(value, label)
    return {
        "hostname": _string(provenance.get("hostname"), f"{label}.hostname"),
        "os": _string(provenance.get("os"), f"{label}.os"),
        "arch": _string(provenance.get("arch"), f"{label}.arch"),
        "logical_parallelism": _optional_positive_integer(
            provenance.get("logical_parallelism"),
            f"{label}.logical_parallelism",
        ),
        "cpu_brand": _string(provenance.get("cpu_brand"), f"{label}.cpu_brand"),
        "memory_bytes": _optional_positive_integer(
            provenance.get("memory_bytes"),
            f"{label}.memory_bytes",
        ),
        "hardware_description": _string(
            provenance.get("hardware_description"),
            f"{label}.hardware_description",
        ),
    }


def normalize_report(value: dict[str, Any], label: str = "report") -> dict[str, Any]:
    """Validate one Rust benchmark report without applying cross-report gates."""

    schema_version = _integer(
        value.get("schema_version"),
        f"{label}.schema_version",
        minimum=1,
    )
    benchmark_id = _string(value.get("benchmark_id"), f"{label}.benchmark_id")
    record_count = _integer(
        value.get("record_count"),
        f"{label}.record_count",
        minimum=1,
    )
    iterations = _integer(
        value.get("iterations"),
        f"{label}.iterations",
        minimum=1,
    )
    replicate_index = _integer(
        value.get("replicate_index"),
        f"{label}.replicate_index",
    )
    selected_arms = sorted(
        _string(arm, f"{label}.selected_arms[{index}]")
        for index, arm in enumerate(_list(value.get("selected_arms"), f"{label}.selected_arms"))
    )

    shard_value = _object(value.get("shard"), f"{label}.shard")
    shard = {
        "shard_index": _integer(
            shard_value.get("shard_index"),
            f"{label}.shard.shard_index",
        ),
        "shard_count": _integer(
            shard_value.get("shard_count"),
            f"{label}.shard.shard_count",
            minimum=1,
        ),
        "ordinal_rule": _string(
            shard_value.get("ordinal_rule"),
            f"{label}.shard.ordinal_rule",
        ),
        "record_limit_after_partition": _integer(
            shard_value.get("record_limit_after_partition"),
            f"{label}.shard.record_limit_after_partition",
        ),
        "total_manifest_records": _integer(
            shard_value.get("total_manifest_records"),
            f"{label}.shard.total_manifest_records",
            minimum=1,
        ),
        "total_eligible_records": _integer(
            shard_value.get("total_eligible_records"),
            f"{label}.shard.total_eligible_records",
            minimum=1,
        ),
        "loaded_records": _integer(
            shard_value.get("loaded_records"),
            f"{label}.shard.loaded_records",
            minimum=1,
        ),
    }
    datasets = [
        _normalize_dataset(dataset, f"{label}.datasets[{index}]")
        for index, dataset in enumerate(_list(value.get("datasets"), f"{label}.datasets"))
    ]
    if not datasets:
        raise ReportError(f"{label}.datasets cannot be empty")
    datasets.sort(key=lambda dataset: dataset["global_ordinal_start"])
    expected_start = 0
    for dataset in datasets:
        if dataset["global_ordinal_start"] != expected_start:
            raise ReportError(f"{label}.datasets ordinal ranges are not contiguous")
        expected_start = dataset["global_ordinal_end_exclusive"]

    arms = [
        _normalize_arm(arm, f"{label}.arms[{index}]")
        for index, arm in enumerate(_list(value.get("arms"), f"{label}.arms"))
    ]
    if not arms:
        raise ReportError(f"{label}.arms cannot be empty")
    arms.sort(key=lambda arm: arm["arm"])

    normalized = {
        "schema_version": schema_version,
        "benchmark_id": benchmark_id,
        "record_count": record_count,
        "iterations": iterations,
        "replicate_index": replicate_index,
        "selected_arms": selected_arms,
        "shard": shard,
        "execution_provenance": _normalize_execution_provenance(
            value.get("execution_provenance"),
            f"{label}.execution_provenance",
        ),
        "validation_and_read_seconds": _number(
            value.get("validation_and_read_seconds"),
            f"{label}.validation_and_read_seconds",
        ),
        "source_semantic_blake3": _digest(
            value.get("source_semantic_blake3"),
            f"{label}.source_semantic_blake3",
        ),
        "datasets": datasets,
        "arms": arms,
    }
    normalized["report_scientific_blake3"] = scientific_blake3(normalized)
    return normalized


def _eligible_count(
    start: int,
    end_exclusive: int,
    shard_index: int,
    shard_count: int,
) -> int:
    start_remainder = start % shard_count
    offset = (shard_index + shard_count - start_remainder) % shard_count
    first = start + offset
    if first >= end_exclusive:
        return 0
    return 1 + (end_exclusive - 1 - first) // shard_count


def _expected_loaded_by_dataset(report: dict[str, Any]) -> list[int]:
    limit = report["shard"]["record_limit_after_partition"]
    remaining = math.inf if limit == 0 else limit
    expected: list[int] = []
    for dataset in report["datasets"]:
        loaded = min(dataset["eligible_records"], remaining)
        expected.append(int(loaded))
        remaining -= loaded
    return expected


def _gate(passed: bool, observed: object) -> dict[str, Any]:
    return {"passed": bool(passed), "observed": observed}


def _same(values: list[object]) -> bool:
    return len({canonical_json(value) for value in values}) == 1


def _arm_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {arm["arm"]: arm for arm in report["arms"]}


def _timing_consistent(arm: dict[str, Any]) -> bool:
    operations = arm["records"] * arm["iterations"]
    extraction_seconds = arm["extraction_seconds"]
    if (
        operations <= 0
        or extraction_seconds <= 0.0
        or arm["serialization_seconds"] <= 0.0
        or arm["deserialization_seconds"] <= 0.0
        or arm["mean_packed_bytes"] <= 0.0
        or arm["mean_packed_bytes_vs_position_record"] <= 0.0
    ):
        return False
    expected_extraction_ns = extraction_seconds * 1_000_000_000.0 / operations
    expected_extraction_rate = operations / extraction_seconds
    expected_serialization_ns = arm["serialization_seconds"] * 1_000_000_000.0 / operations
    expected_deserialization_ns = arm["deserialization_seconds"] * 1_000_000_000.0 / operations
    comparisons = (
        (arm["extraction_ns_per_record"], expected_extraction_ns),
        (arm["extraction_records_per_second"], expected_extraction_rate),
        (arm["serialization_ns_per_record"], expected_serialization_ns),
        (arm["deserialization_ns_per_record"], expected_deserialization_ns),
    )
    if any(
        observed <= 0.0
        or not math.isclose(
            observed,
            expected,
            rel_tol=TIMING_RELATIVE_TOLERANCE,
            abs_tol=TIMING_ABSOLUTE_TOLERANCE,
        )
        for observed, expected in comparisons
    ):
        return False
    expected_overflow_fraction = arm["positions_with_overflow"] / arm["records"]
    if not math.isclose(
        arm["overflow_position_fraction"],
        expected_overflow_fraction,
        rel_tol=TIMING_RELATIVE_TOLERANCE,
        abs_tol=TIMING_ABSOLUTE_TOLERANCE,
    ):
        return False
    capacity = arm["mean_local_capacity_rows"]
    active = arm["mean_active_local_rows"]
    occupancy = arm["local_occupancy_fraction"]
    if active > capacity:
        return False
    if capacity == 0.0:
        return occupancy is None and active == 0.0
    return occupancy is not None and math.isclose(
        occupancy,
        active / capacity,
        rel_tol=TIMING_RELATIVE_TOLERANCE,
        abs_tol=TIMING_ABSOLUTE_TOLERANCE,
    )


def _weighted_mean(
    reports: list[dict[str, Any]],
    arm_id: str,
    field: str,
) -> float:
    numerator = math.fsum(
        _arm_map(report)[arm_id][field] * report["record_count"] for report in reports
    )
    denominator = sum(report["record_count"] for report in reports)
    return numerator / denominator


def _aggregate_arm(
    reports: list[dict[str, Any]],
    arm_id: str,
    *,
    include_throughput: bool,
) -> dict[str, Any]:
    arms = [_arm_map(report)[arm_id] for report in reports]
    total_records = sum(arm["records"] for arm in arms)
    total_operations = sum(arm["records"] * arm["iterations"] for arm in arms)
    total_active_rows = math.fsum(arm["mean_active_local_rows"] * arm["records"] for arm in arms)
    total_capacity_rows = math.fsum(
        arm["mean_local_capacity_rows"] * arm["records"] for arm in arms
    )
    positions_with_overflow = sum(arm["positions_with_overflow"] for arm in arms)
    result: dict[str, Any] = {
        "arm": arm_id,
        "records": total_records,
        "operations": total_operations,
        "weighted_metrics": {
            field: _weighted_mean(reports, arm_id, field) for field in WEIGHTED_MEAN_FIELDS
        },
        "positions_with_overflow": positions_with_overflow,
        "overflow_position_fraction": positions_with_overflow / total_records,
        "local_occupancy_fraction": (
            total_active_rows / total_capacity_rows if total_capacity_rows > 0.0 else None
        ),
        "throughput": None,
    }
    if include_throughput:
        throughput: dict[str, Any] = {}
        for phase in TIMING_PHASES:
            seconds = math.fsum(arm[f"{phase}_seconds"] for arm in arms)
            throughput[phase] = {
                "seconds": seconds,
                "ns_per_record": seconds * 1_000_000_000.0 / total_operations,
                "records_per_second": total_operations / seconds,
            }
        result["throughput"] = throughput
    return result


def _comparison_ratios(
    arm: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, float]:
    return {
        "extraction_speedup": (
            reference["extraction_ns_per_record"] / arm["extraction_ns_per_record"]
        ),
        "serialization_speedup": (
            reference["serialization_ns_per_record"] / arm["serialization_ns_per_record"]
        ),
        "deserialization_speedup": (
            reference["deserialization_ns_per_record"] / arm["deserialization_ns_per_record"]
        ),
        "packed_bytes_fraction": (arm["mean_packed_bytes"] / reference["mean_packed_bytes"]),
    }


def _replicate_identity(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": report["schema_version"],
        "benchmark_id": report["benchmark_id"],
        "record_count": report["record_count"],
        "iterations": report["iterations"],
        "selected_arms": report["selected_arms"],
        "shard": report["shard"],
        "execution_provenance": report["execution_provenance"],
        "source_semantic_blake3": report["source_semantic_blake3"],
        "datasets": report["datasets"],
        "arms": [
            {key: value for key, value in arm.items() if key not in ARM_TIMING_FIELDS}
            for arm in report["arms"]
        ],
    }


def _report_sort_key(report: dict[str, Any]) -> tuple[int, int, str]:
    return (
        report["shard"]["shard_index"],
        report["replicate_index"],
        report["report_scientific_blake3"],
    )


def _within_process_ratios(report: dict[str, Any]) -> dict[str, Any]:
    arms = _arm_map(report)
    exact = arms[EXACT_ENTITY_CONTROL]
    historical = arms[HISTORICAL_SQUARE_21]
    return {
        "shard_index": report["shard"]["shard_index"],
        "replicate_index": report["replicate_index"],
        "hostname": report["execution_provenance"]["hostname"],
        "hardware_description": report["execution_provenance"]["hardware_description"],
        "weight_operations": report["record_count"] * report["iterations"],
        "arms": {
            arm_id: {
                "vs_exact_entity_control": _comparison_ratios(
                    arms[arm_id],
                    exact,
                ),
                "vs_historical_square_21x21_441": _comparison_ratios(
                    arms[arm_id],
                    historical,
                ),
            }
            for arm_id in REQUIRED_ARMS
        },
    }


def _sample_distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ReportError("cannot summarize an empty distribution")
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    mean = math.fsum(ordered) / count
    middle = count // 2
    median = (
        ordered[middle]
        if count % 2 == 1
        else math.fsum((ordered[middle - 1], ordered[middle])) / 2.0
    )
    sample_variance = (
        math.fsum((value - mean) ** 2 for value in ordered) / (count - 1) if count > 1 else None
    )
    sample_standard_deviation = math.sqrt(sample_variance) if sample_variance is not None else None
    return {
        "count": count,
        "mean": mean,
        "median": median,
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "sample_variance": sample_variance,
        "sample_standard_deviation": sample_standard_deviation,
        "coefficient_of_variation": (
            sample_standard_deviation / mean
            if sample_standard_deviation is not None and mean != 0.0
            else None
        ),
    }


def _replicate_variance(
    groups: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    result = []
    ratio_references = (
        "vs_exact_entity_control",
        "vs_historical_square_21x21_441",
    )
    ratio_metrics = (
        "extraction_speedup",
        "serialization_speedup",
        "deserialization_speedup",
        "packed_bytes_fraction",
    )
    timing_metrics = tuple(sorted(ARM_TIMING_FIELDS))
    for shard_index, reports in sorted(groups.items()):
        process_ratios = [_within_process_ratios(report) for report in reports]
        arms = {}
        for arm_id in REQUIRED_ARMS:
            arms[arm_id] = {
                "absolute_timing": {
                    metric: _sample_distribution(
                        [_arm_map(report)[arm_id][metric] for report in reports]
                    )
                    for metric in timing_metrics
                },
                "within_process_ratios": {
                    reference: {
                        metric: _sample_distribution(
                            [
                                process["arms"][arm_id][reference][metric]
                                for process in process_ratios
                            ]
                        )
                        for metric in ratio_metrics
                    }
                    for reference in ratio_references
                },
            }
        result.append(
            {
                "shard_index": shard_index,
                "replicate_indices": sorted(report["replicate_index"] for report in reports),
                "sample_variance_denominator": "n-1",
                "validation_and_read_seconds": _sample_distribution(
                    [report["validation_and_read_seconds"] for report in reports]
                ),
                "arms": arms,
            }
        )
    return result


def _median_reports(
    groups: dict[int, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_reports = []
    selections = []
    for shard_index, reports in sorted(groups.items()):
        ordered = sorted(
            reports,
            key=lambda report: (
                _arm_map(report)[EXACT_ENTITY_CONTROL]["extraction_ns_per_record"],
                report["replicate_index"],
                report["report_scientific_blake3"],
            ),
        )
        selected = ordered[len(ordered) // 2]
        selected_reports.append(selected)
        selections.append(
            {
                "shard_index": shard_index,
                "ordered_invocations": [
                    {
                        "rank": rank,
                        "replicate_index": report["replicate_index"],
                        "criterion_ns_per_record": _arm_map(report)[EXACT_ENTITY_CONTROL][
                            "extraction_ns_per_record"
                        ],
                        "report_scientific_blake3": report["report_scientific_blake3"],
                    }
                    for rank, report in enumerate(ordered)
                ],
                "selected_replicate_index": selected["replicate_index"],
                "selected_report_scientific_blake3": selected["report_scientific_blake3"],
            }
        )
    return selected_reports, {
        "criterion": MEDIAN_SELECTION_CRITERION,
        "tie_break": "replicate_index ascending, then report_scientific_blake3",
        "selected_rank": "floor(required_replicates / 2) after ascending sort",
        "shards": selections,
    }


def _process_receipt(
    report: dict[str, Any],
    *,
    selected_receipts: set[tuple[int, int, str]],
    local_arms_valid: bool,
    local_timing_valid: bool,
) -> dict[str, Any]:
    receipt = _report_sort_key(report)
    return {
        "shard_index": report["shard"]["shard_index"],
        "replicate_index": report["replicate_index"],
        "selected_for_absolute_aggregate": receipt in selected_receipts,
        "timing_valid": local_timing_valid,
        "hostname": report["execution_provenance"]["hostname"],
        "hardware_description": report["execution_provenance"]["hardware_description"],
        "validation_and_read_seconds": report["validation_and_read_seconds"],
        "report_scientific_blake3": report["report_scientific_blake3"],
        "timings": {
            arm["arm"]: {field: arm[field] for field in sorted(ARM_TIMING_FIELDS)}
            for arm in report["arms"]
        },
        "within_process_ratios": (
            _within_process_ratios(report)["arms"]
            if local_arms_valid and local_timing_valid
            else None
        ),
    }


def _ratio_summary(
    ratios: list[dict[str, Any]],
    arm_id: str,
    reference: str,
    metric: str,
) -> dict[str, Any]:
    values = [
        (
            float(shard["arms"][arm_id][reference][metric]),
            int(shard["weight_operations"]),
        )
        for shard in ratios
    ]
    total_weight = sum(weight for _, weight in values)
    geometric_mean = math.exp(
        math.fsum(weight * math.log(value) for value, weight in values) / total_weight
    )
    return {
        "weighted_geometric_mean": geometric_mean,
        "minimum": min(value for value, _ in values),
        "maximum": max(value for value, _ in values),
        "shards": len(values),
        "weight_operations": total_weight,
    }


def _combined_ratios(ratios: list[dict[str, Any]]) -> dict[str, Any]:
    references = (
        "vs_exact_entity_control",
        "vs_historical_square_21x21_441",
    )
    metrics = (
        "extraction_speedup",
        "serialization_speedup",
        "deserialization_speedup",
        "packed_bytes_fraction",
    )
    return {
        arm_id: {
            reference: {
                metric: _ratio_summary(
                    ratios,
                    arm_id,
                    reference,
                    metric,
                )
                for metric in metrics
            }
            for reference in references
        }
        for arm_id in REQUIRED_ARMS
    }


def aggregate_reports(
    values: list[dict[str, Any]],
    *,
    required_replicates: int = DEFAULT_REQUIRED_REPLICATES,
) -> dict[str, Any]:
    """Return one deterministic fail-closed aggregate for Rust shard reports."""

    if not values:
        raise ReportError("at least one benchmark report is required")
    if required_replicates != DEFAULT_REQUIRED_REPLICATES:
        raise ReportError(
            "the preregistered R0 classifier requires exactly "
            f"{DEFAULT_REQUIRED_REPLICATES} replicates"
        )
    reports = [normalize_report(value, f"report[{index}]") for index, value in enumerate(values)]
    reports.sort(key=_report_sort_key)

    schema_versions = [report["schema_version"] for report in reports]
    benchmark_ids = [report["benchmark_id"] for report in reports]
    iterations = [report["iterations"] for report in reports]
    shard_counts = [report["shard"]["shard_count"] for report in reports]
    ordinal_rules = [report["shard"]["ordinal_rule"] for report in reports]
    record_limits = [report["shard"]["record_limit_after_partition"] for report in reports]
    manifest_record_counts = [report["shard"]["total_manifest_records"] for report in reports]
    dataset_identities = [
        [_dataset_identity(dataset) for dataset in report["datasets"]] for report in reports
    ]
    source_manifest_identities = [
        [dataset["manifest_blake3"] for dataset in report["datasets"]] for report in reports
    ]

    benchmark_schema_identity = (
        _same(schema_versions) and schema_versions[0] == BENCHMARK_SCHEMA_VERSION
    )
    benchmark_identity = _same(benchmark_ids) and benchmark_ids[0] == BENCHMARK_ID
    source_identity = _same(source_manifest_identities)
    dataset_identity = _same(dataset_identities)
    iteration_identity = _same(iterations)
    partition_identity = (
        _same(shard_counts)
        and _same(ordinal_rules)
        and _same(record_limits)
        and _same(manifest_record_counts)
        and ordinal_rules[0] == ORDINAL_RULE
    )

    shard_indexes = [report["shard"]["shard_index"] for report in reports]
    shard_replicates = [
        {
            "shard_index": report["shard"]["shard_index"],
            "replicate_index": report["replicate_index"],
        }
        for report in reports
    ]
    common_shard_count = shard_counts[0] if _same(shard_counts) else None
    expected_shards = list(range(common_shard_count)) if common_shard_count is not None else []
    expected_replicate_indices = list(range(required_replicates))
    expected_shard_replicates = (
        [
            {
                "shard_index": shard_index,
                "replicate_index": replicate_index,
            }
            for shard_index in expected_shards
            for replicate_index in expected_replicate_indices
        ]
        if common_shard_count is not None
        else []
    )
    exact_process_coverage = (
        common_shard_count is not None and shard_replicates == expected_shard_replicates
    )
    groups: dict[int, list[dict[str, Any]]] = {}
    for report in reports:
        groups.setdefault(report["shard"]["shard_index"], []).append(report)

    accounting_valid = True
    accounting_observed: list[dict[str, Any]] = []
    required_arms_valid = True
    arm_observed: list[dict[str, Any]] = []
    semantics_valid = True
    semantic_observed: list[dict[str, Any]] = []
    timing_valid = True
    timing_observed: list[dict[str, Any]] = []
    local_validation: dict[tuple[int, int, str], dict[str, bool]] = {}
    for report in reports:
        shard = report["shard"]
        datasets = report["datasets"]
        shard_index = shard["shard_index"]
        replicate_index = report["replicate_index"]
        local_accounting_valid = (
            shard_index < shard["shard_count"]
            and shard["loaded_records"] == report["record_count"]
            and sum(dataset["loaded_records"] for dataset in datasets) == report["record_count"]
            and sum(dataset["eligible_records"] for dataset in datasets)
            == shard["total_eligible_records"]
            and datasets[-1]["global_ordinal_end_exclusive"] == shard["total_manifest_records"]
        )
        expected_loaded = _expected_loaded_by_dataset(report)
        local_accounting_valid = (
            local_accounting_valid
            and [dataset["loaded_records"] for dataset in datasets] == expected_loaded
            and report["record_count"] == sum(expected_loaded)
            and all(
                dataset["eligible_records"]
                == _eligible_count(
                    dataset["global_ordinal_start"],
                    dataset["global_ordinal_end_exclusive"],
                    shard_index,
                    shard["shard_count"],
                )
                for dataset in datasets
            )
        )
        accounting_valid &= local_accounting_valid
        accounting_observed.append(
            {
                "shard_index": shard_index,
                "replicate_index": replicate_index,
                "valid": local_accounting_valid,
                "record_count": report["record_count"],
                "loaded_records": shard["loaded_records"],
                "total_eligible_records": shard["total_eligible_records"],
            }
        )

        arm_ids = [arm["arm"] for arm in report["arms"]]
        selected_ids = report["selected_arms"]
        local_arms_valid = (
            len(arm_ids) == len(set(arm_ids))
            and len(selected_ids) == len(set(selected_ids))
            and set(arm_ids) == REQUIRED_ARM_SET
            and set(selected_ids) == REQUIRED_ARM_SET
            and set(selected_ids) == set(arm_ids)
            and all(
                arm["records"] == report["record_count"]
                and arm["iterations"] == report["iterations"]
                for arm in report["arms"]
            )
        )
        required_arms_valid &= local_arms_valid
        arm_observed.append(
            {
                "shard_index": shard_index,
                "replicate_index": replicate_index,
                "valid": local_arms_valid,
                "selected_arms": sorted(selected_ids),
                "reported_arms": sorted(arm_ids),
            }
        )

        local_semantics_valid = all(
            arm["round_trip_verified"] is True
            and arm["semantic_blake3"] == report["source_semantic_blake3"]
            for arm in report["arms"]
        )
        semantics_valid &= local_semantics_valid
        semantic_observed.append(
            {
                "shard_index": shard_index,
                "replicate_index": replicate_index,
                "valid": local_semantics_valid,
                "source_semantic_blake3": report["source_semantic_blake3"],
                "arm_semantic_blake3": {
                    arm["arm"]: arm["semantic_blake3"] for arm in report["arms"]
                },
                "round_trip_verified": {
                    arm["arm"]: arm["round_trip_verified"] for arm in report["arms"]
                },
            }
        )

        local_timing_valid = all(_timing_consistent(arm) for arm in report["arms"])
        timing_valid &= local_timing_valid
        timing_observed.append(
            {
                "shard_index": shard_index,
                "replicate_index": replicate_index,
                "hostname": report["execution_provenance"]["hostname"],
                "valid": local_timing_valid,
                "arms": {
                    arm["arm"]: {phase: arm[f"{phase}_seconds"] for phase in TIMING_PHASES}
                    for arm in report["arms"]
                },
            }
        )
        local_validation[_report_sort_key(report)] = {
            "arms": local_arms_valid,
            "timing": local_timing_valid,
        }

    replicate_identity_valid = True
    replicate_identity_observed = []
    for shard_index, group in sorted(groups.items()):
        identities = [
            {
                "replicate_index": report["replicate_index"],
                "replicate_identity_blake3": scientific_blake3(_replicate_identity(report)),
            }
            for report in group
        ]
        local_identity_valid = [
            report["replicate_index"] for report in group
        ] == expected_replicate_indices and _same([_replicate_identity(report) for report in group])
        replicate_identity_valid &= local_identity_valid
        replicate_identity_observed.append(
            {
                "shard_index": shard_index,
                "valid": local_identity_valid,
                "replicates": identities,
            }
        )

    exact_coverage = exact_process_coverage and accounting_valid
    gates = {
        "benchmark_schema_identity": _gate(
            benchmark_schema_identity,
            sorted(set(schema_versions)),
        ),
        "benchmark_identity": _gate(
            benchmark_identity,
            sorted(set(benchmark_ids)),
        ),
        "source_manifest_identity": _gate(
            source_identity,
            source_manifest_identities,
        ),
        "dataset_identity": _gate(
            dataset_identity,
            dataset_identities,
        ),
        "iteration_identity": _gate(
            iteration_identity,
            sorted(set(iterations)),
        ),
        "partition_identity": _gate(
            partition_identity,
            {
                "shard_counts": sorted(set(shard_counts)),
                "ordinal_rules": sorted(set(ordinal_rules)),
                "record_limits_after_partition": sorted(set(record_limits)),
                "total_manifest_records": sorted(set(manifest_record_counts)),
            },
        ),
        "exact_nonoverlapping_shard_coverage": _gate(
            exact_coverage,
            {
                "expected_shards": expected_shards,
                "reported_shards": sorted(set(shard_indexes)),
                "expected_replicate_indices": expected_replicate_indices,
                "expected_shard_replicates": expected_shard_replicates,
                "reported_shard_replicates": shard_replicates,
                "accounting": accounting_observed,
            },
        ),
        "replicate_group_identity": _gate(
            replicate_identity_valid,
            replicate_identity_observed,
        ),
        "all_required_arms": _gate(
            required_arms_valid,
            arm_observed,
        ),
        "round_trip_and_semantic_identity": _gate(
            semantics_valid,
            semantic_observed,
        ),
        "performance_evidence": _gate(
            timing_valid,
            timing_observed,
        ),
    }
    structural_complete = all(
        gates[name]["passed"]
        for name in (
            "benchmark_schema_identity",
            "benchmark_identity",
            "source_manifest_identity",
            "dataset_identity",
            "iteration_identity",
            "partition_identity",
            "exact_nonoverlapping_shard_coverage",
            "replicate_group_identity",
            "all_required_arms",
        )
    )
    if not semantics_valid:
        classification = SEMANTIC_FAILURE
    elif not structural_complete:
        classification = INCOMPLETE
    elif not timing_valid:
        classification = INSUFFICIENT_PERFORMANCE
    else:
        classification = COMPLETE

    scientifically_aggregateable = structural_complete and semantics_valid
    performance_aggregateable = scientifically_aggregateable and timing_valid
    representative_reports = (
        [group[0] for _, group in sorted(groups.items())] if scientifically_aggregateable else []
    )
    selected_reports: list[dict[str, Any]] = []
    median_selection: dict[str, Any] | None = None
    if performance_aggregateable:
        selected_reports, median_selection = _median_reports(groups)
    aggregate_basis = selected_reports if performance_aggregateable else representative_reports
    within_shard_ratios = (
        [_within_process_ratios(report) for report in selected_reports]
        if performance_aggregateable
        else []
    )
    arm_aggregates = (
        {
            arm_id: _aggregate_arm(
                aggregate_basis,
                arm_id,
                include_throughput=performance_aggregateable,
            )
            for arm_id in REQUIRED_ARMS
        }
        if scientifically_aggregateable
        else {}
    )
    selected_receipts = {_report_sort_key(report) for report in selected_reports}
    all_replicate_processes = [
        _process_receipt(
            report,
            selected_receipts=selected_receipts,
            local_arms_valid=local_validation[_report_sort_key(report)]["arms"],
            local_timing_valid=local_validation[_report_sort_key(report)]["timing"],
        )
        for report in reports
    ]
    replicate_variance = _replicate_variance(groups) if performance_aggregateable else []
    combined_ratios = _combined_ratios(within_shard_ratios) if performance_aggregateable else {}

    common_dataset_identity = dataset_identities[0] if dataset_identity else None
    common_source_manifest_identity = source_manifest_identities[0] if source_identity else None
    unique_records = (
        sum(report["record_count"] for report in representative_reports)
        if scientifically_aggregateable
        else None
    )
    selected_operations = (
        sum(report["record_count"] * report["iterations"] for report in selected_reports)
        if performance_aggregateable
        else None
    )
    all_replicate_operations = (
        sum(report["record_count"] * report["iterations"] for report in reports)
        if exact_process_coverage and required_arms_valid
        else None
    )
    scientific = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "aggregate_id": AGGREGATE_ID,
        "benchmark_id": BENCHMARK_ID,
        "classification": classification,
        "complete": classification == COMPLETE,
        "claim_scope": {
            "stage": "extraction-and-serialization-only",
            "learned_model_evaluated": False,
            "player_strength_evaluated": False,
            "authorizes_learned_representation_promotion": False,
            "claims_progress_toward_100_mean": False,
        },
        "required_arms": list(REQUIRED_ARMS),
        "replication_protocol": {
            "required_replicates": required_replicates,
            "required_replicate_indices": expected_replicate_indices,
            "independent_unit": "one process invocation",
            "iterations_are_independent_replicates": False,
            "absolute_latency_selection": MEDIAN_SELECTION_CRITERION,
            "variance_estimator": ("within-shard sample variance with denominator n-1"),
        },
        "gates": gates,
        "identity": {
            "benchmark_schema_version": (schema_versions[0] if benchmark_schema_identity else None),
            "iterations": iterations[0] if iteration_identity else None,
            "shard_count": common_shard_count,
            "ordinal_rule": (ordinal_rules[0] if partition_identity else None),
            "record_limit_after_partition": (record_limits[0] if partition_identity else None),
            "total_manifest_records": (manifest_record_counts[0] if partition_identity else None),
            "datasets": common_dataset_identity,
            "source_manifest_blake3": common_source_manifest_identity,
            "source_identity_blake3": (
                scientific_blake3(
                    {
                        "datasets": common_dataset_identity,
                        "source_manifest_blake3": (common_source_manifest_identity),
                    }
                )
                if dataset_identity and source_identity
                else None
            ),
            "required_replicates": required_replicates,
        },
        "coverage": {
            "reported_shards": sorted(set(shard_indexes)),
            "expected_shards": expected_shards,
            "reported_process_invocations": len(reports),
            "expected_process_invocations": (
                common_shard_count * required_replicates if common_shard_count is not None else None
            ),
            "semantic_unique_records": unique_records,
            "total_records": unique_records,
            "selected_timing_operations_per_arm": selected_operations,
            "total_operations_per_arm": selected_operations,
            "all_replicate_timed_operations_per_arm": (all_replicate_operations),
        },
        "median_invocation_selection": median_selection,
        "all_replicate_processes": all_replicate_processes,
        "replicate_variance_by_shard": replicate_variance,
        "arms": arm_aggregates,
        "within_shard_ratios": within_shard_ratios,
        "selected_within_process_ratios": within_shard_ratios,
        "combined_within_shard_ratios": combined_ratios,
        "combined_selected_within_process_ratios": combined_ratios,
    }
    return {
        "scientific": scientific,
        "scientific_blake3": scientific_blake3(scientific),
    }


def write_json(path: Path, value: dict[str, Any]) -> str:
    encoded = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)
    return encoded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        action="append",
        required=True,
        help="Rust process report; repeat for replicas 0, 1, and 2 of every shard.",
    )
    parser.add_argument(
        "--required-replicates",
        type=int,
        default=DEFAULT_REQUIRED_REPLICATES,
        help="Frozen independent process count per shard.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        aggregate = aggregate_reports(
            [load_json(path) for path in args.report],
            required_replicates=args.required_replicates,
        )
    except ReportError as error:
        parser.error(str(error))
    encoded = write_json(args.output, aggregate)
    print(encoded, end="")
    return EXIT_CODES[aggregate["scientific"]["classification"]]


if __name__ == "__main__":
    raise SystemExit(main())
