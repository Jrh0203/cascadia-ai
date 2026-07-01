from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pytest
import spatial_representation_benchmark_report as reporter


def _digest(value: int) -> str:
    return f"{value:064x}"


def _arm(
    arm_id: str,
    *,
    records: int,
    iterations: int,
    semantic_blake3: str,
    control_extraction_ns: float,
) -> dict:
    base_extraction_ns = {
        reporter.EXACT_ENTITY_CONTROL: 100.0,
        "hex-radius-6-127": 50.0,
        "hex-radius-5-91": 40.0,
        "hex-radius-4-61": 35.0,
        reporter.HISTORICAL_SQUARE_21: 80.0,
    }[arm_id]
    invocation_scale = control_extraction_ns / 100.0
    extraction_ns = base_extraction_ns * invocation_scale
    serialization_ns = extraction_ns / 2.0
    deserialization_ns = extraction_ns * 0.75
    packed_bytes = {
        reporter.EXACT_ENTITY_CONTROL: 1000.0,
        "hex-radius-6-127": 800.0,
        "hex-radius-5-91": 700.0,
        "hex-radius-4-61": 650.0,
        reporter.HISTORICAL_SQUARE_21: 1600.0,
    }[arm_id]
    local_capacity = {
        reporter.EXACT_ENTITY_CONTROL: 0.0,
        "hex-radius-6-127": 127.0,
        "hex-radius-5-91": 91.0,
        "hex-radius-4-61": 61.0,
        reporter.HISTORICAL_SQUARE_21: 441.0,
    }[arm_id]
    active_local = 0.0 if local_capacity == 0.0 else 20.0
    positions_with_overflow = 1 if arm_id == "hex-radius-4-61" else 0
    operations = records * iterations
    return {
        "arm": arm_id,
        "records": records,
        "iterations": iterations,
        "round_trip_verified": True,
        "semantic_blake3": semantic_blake3,
        "extraction_seconds": extraction_ns * operations / 1_000_000_000.0,
        "extraction_ns_per_record": extraction_ns,
        "extraction_records_per_second": 1_000_000_000.0 / extraction_ns,
        "serialization_seconds": serialization_ns * operations / 1_000_000_000.0,
        "serialization_ns_per_record": serialization_ns,
        "deserialization_seconds": deserialization_ns * operations / 1_000_000_000.0,
        "deserialization_ns_per_record": deserialization_ns,
        "mean_packed_bytes": packed_bytes,
        "mean_packed_bytes_vs_position_record": packed_bytes / 4096.0,
        "mean_local_capacity_rows": local_capacity,
        "mean_active_local_rows": active_local,
        "local_occupancy_fraction": (
            None if local_capacity == 0.0 else active_local / local_capacity
        ),
        "mean_exact_entity_rows": (20.0 if arm_id == reporter.EXACT_ENTITY_CONTROL else 0.0),
        "mean_overflow_entity_rows": (1.0 if arm_id == "hex-radius-4-61" else 0.0),
        "positions_with_overflow": positions_with_overflow,
        "overflow_position_fraction": positions_with_overflow / records,
        "mean_dense_raw_scalar_slots": local_capacity * 6.0,
    }


def _report(
    shard_index: int,
    replicate_index: int,
    *,
    shard_count: int = 4,
    total_records: int = 40,
    iterations: int = 20,
    control_extraction_ns: float | None = None,
) -> dict:
    eligible_records = reporter._eligible_count(
        0,
        total_records,
        shard_index,
        shard_count,
    )
    source_semantic_blake3 = _digest(shard_index + 1)
    if control_extraction_ns is None:
        control_extraction_ns = {
            0: 120.0,
            1: 80.0,
            2: 100.0,
        }[replicate_index]
    return {
        "schema_version": reporter.BENCHMARK_SCHEMA_VERSION,
        "benchmark_id": reporter.BENCHMARK_ID,
        "record_count": eligible_records,
        "iterations": iterations,
        "replicate_index": replicate_index,
        "selected_arms": list(reporter.REQUIRED_ARMS),
        "shard": {
            "shard_index": shard_index,
            "shard_count": shard_count,
            "ordinal_rule": reporter.ORDINAL_RULE,
            "record_limit_after_partition": 0,
            "total_manifest_records": total_records,
            "total_eligible_records": eligible_records,
            "loaded_records": eligible_records,
        },
        "execution_provenance": {
            "hostname": f"john{shard_index + 1}",
            "os": "macos",
            "arch": "aarch64",
            "logical_parallelism": 10,
            "cpu_brand": "Apple Test",
            "memory_bytes": 16 * 2**30,
            "hardware_description": ("cpu=Apple Test; memory_bytes=17179869184"),
        },
        "validation_and_read_seconds": 0.01 + replicate_index * 0.001,
        "source_semantic_blake3": source_semantic_blake3,
        "datasets": [
            {
                "root": f"/host-{shard_index}/dataset",
                "dataset_id": "r0-open-corpus-v1",
                "feature_schema": "position-record-v1",
                "split": "train",
                "completed_games": 10,
                "total_records": total_records,
                "global_ordinal_start": 0,
                "global_ordinal_end_exclusive": total_records,
                "eligible_records": eligible_records,
                "loaded_records": eligible_records,
                "manifest_blake3": _digest(99),
            }
        ],
        "arms": [
            _arm(
                arm_id,
                records=eligible_records,
                iterations=iterations,
                semantic_blake3=source_semantic_blake3,
                control_extraction_ns=control_extraction_ns,
            )
            for arm_id in reporter.REQUIRED_ARMS
        ],
    }


def _complete_reports(
    control_ns_by_replicate: dict[int, float] | None = None,
) -> list[dict]:
    return [
        _report(
            shard_index,
            replicate_index,
            control_extraction_ns=(
                control_ns_by_replicate[replicate_index]
                if control_ns_by_replicate is not None
                else None
            ),
        )
        for shard_index in range(4)
        for replicate_index in range(reporter.DEFAULT_REQUIRED_REPLICATES)
    ]


def _find_report(
    reports: list[dict],
    shard_index: int,
    replicate_index: int,
) -> dict:
    return next(
        report
        for report in reports
        if report["shard"]["shard_index"] == shard_index
        and report["replicate_index"] == replicate_index
    )


def _set_arm_timing(
    report: dict,
    arm_id: str,
    *,
    extraction_ns: float,
    serialization_ns: float,
    deserialization_ns: float,
) -> None:
    arm = next(arm for arm in report["arms"] if arm["arm"] == arm_id)
    operations = arm["records"] * arm["iterations"]
    arm["extraction_seconds"] = extraction_ns * operations / 1_000_000_000.0
    arm["extraction_ns_per_record"] = extraction_ns
    arm["extraction_records_per_second"] = 1_000_000_000.0 / extraction_ns
    arm["serialization_seconds"] = serialization_ns * operations / 1_000_000_000.0
    arm["serialization_ns_per_record"] = serialization_ns
    arm["deserialization_seconds"] = deserialization_ns * operations / 1_000_000_000.0
    arm["deserialization_ns_per_record"] = deserialization_ns


def test_complete_twelve_report_merge_does_not_triple_count_rows() -> None:
    aggregate = reporter.aggregate_reports(_complete_reports())
    scientific = aggregate["scientific"]

    assert scientific["classification"] == reporter.COMPLETE
    assert all(gate["passed"] for gate in scientific["gates"].values())
    assert scientific["coverage"] == {
        "reported_shards": [0, 1, 2, 3],
        "expected_shards": [0, 1, 2, 3],
        "reported_process_invocations": 12,
        "expected_process_invocations": 12,
        "semantic_unique_records": 40,
        "total_records": 40,
        "selected_timing_operations_per_arm": 800,
        "total_operations_per_arm": 800,
        "all_replicate_timed_operations_per_arm": 2400,
    }
    assert scientific["arms"]["hex-radius-6-127"]["records"] == 40
    assert scientific["arms"]["hex-radius-6-127"]["operations"] == 800
    assert scientific["arms"]["hex-radius-6-127"]["weighted_metrics"]["mean_packed_bytes"] == 800.0
    assert len(scientific["all_replicate_processes"]) == 12
    assert len(scientific["within_shard_ratios"]) == 4
    assert len(scientific["replicate_variance_by_shard"]) == 4
    assert (
        sum(
            process["selected_for_absolute_aggregate"]
            for process in scientific["all_replicate_processes"]
        )
        == 4
    )

    combined = scientific["combined_within_shard_ratios"]["hex-radius-6-127"]
    assert combined["vs_exact_entity_control"]["extraction_speedup"][
        "weighted_geometric_mean"
    ] == pytest.approx(2.0)
    assert combined["vs_historical_square_21x21_441"]["packed_bytes_fraction"][
        "weighted_geometric_mean"
    ] == pytest.approx(0.5)
    assert scientific["claim_scope"]["learned_model_evaluated"] is False
    assert scientific["claim_scope"]["player_strength_evaluated"] is False


def test_missing_replicate_fails_closed_as_incomplete() -> None:
    reports = _complete_reports()
    reports.remove(_find_report(reports, 2, 1))

    aggregate = reporter.aggregate_reports(reports)
    scientific = aggregate["scientific"]

    assert scientific["classification"] == reporter.INCOMPLETE
    assert scientific["gates"]["exact_nonoverlapping_shard_coverage"]["passed"] is False
    assert scientific["gates"]["replicate_group_identity"]["passed"] is False
    assert scientific["arms"] == {}


def test_duplicate_replicate_fails_closed_as_incomplete() -> None:
    reports = _complete_reports()
    reports.append(copy.deepcopy(_find_report(reports, 1, 2)))

    aggregate = reporter.aggregate_reports(reports)
    scientific = aggregate["scientific"]

    assert scientific["classification"] == reporter.INCOMPLETE
    assert scientific["gates"]["exact_nonoverlapping_shard_coverage"]["passed"] is False
    assert scientific["gates"]["replicate_group_identity"]["passed"] is False
    assert scientific["coverage"]["reported_process_invocations"] == 13


def test_replicate_source_identity_drift_fails_closed() -> None:
    reports = _complete_reports()
    drifted = _find_report(reports, 3, 2)
    drifted["source_semantic_blake3"] = _digest(500)
    for arm in drifted["arms"]:
        arm["semantic_blake3"] = _digest(500)

    aggregate = reporter.aggregate_reports(reports)
    scientific = aggregate["scientific"]

    assert scientific["classification"] == reporter.INCOMPLETE
    assert scientific["gates"]["replicate_group_identity"]["passed"] is False
    assert scientific["gates"]["round_trip_and_semantic_identity"]["passed"] is True


def test_replicate_arm_identity_drift_fails_closed() -> None:
    reports = _complete_reports()
    drifted = _find_report(reports, 0, 1)
    drifted["arms"][1]["mean_packed_bytes"] += 1.0

    aggregate = reporter.aggregate_reports(reports)

    assert aggregate["scientific"]["classification"] == reporter.INCOMPLETE
    assert aggregate["scientific"]["gates"]["replicate_group_identity"]["passed"] is False


def test_merge_order_is_byte_deterministic() -> None:
    reports = _complete_reports()
    shuffled = copy.deepcopy(reports)
    random.Random(136).shuffle(shuffled)

    ordered = reporter.aggregate_reports(copy.deepcopy(reports))
    reordered = reporter.aggregate_reports(shuffled)

    assert ordered == reordered
    assert reporter.canonical_json(ordered) == reporter.canonical_json(reordered)


def test_median_control_invocation_selects_the_whole_paired_process() -> None:
    reports = _complete_reports()
    treatment_timings = {
        0: (40.0, 20.0, 30.0),
        1: (50.0, 25.0, 37.5),
        2: (90.0, 45.0, 67.5),
    }
    for report in reports:
        extraction_ns, serialization_ns, deserialization_ns = treatment_timings[
            report["replicate_index"]
        ]
        _set_arm_timing(
            report,
            "hex-radius-6-127",
            extraction_ns=extraction_ns,
            serialization_ns=serialization_ns,
            deserialization_ns=deserialization_ns,
        )

    aggregate = reporter.aggregate_reports(reports)
    scientific = aggregate["scientific"]

    selections = scientific["median_invocation_selection"]["shards"]
    assert [selection["selected_replicate_index"] for selection in selections] == [
        2,
        2,
        2,
        2,
    ]
    assert scientific["arms"][reporter.EXACT_ENTITY_CONTROL]["throughput"]["extraction"][
        "ns_per_record"
    ] == pytest.approx(100.0)
    assert scientific["arms"]["hex-radius-6-127"]["throughput"]["extraction"][
        "ns_per_record"
    ] == pytest.approx(90.0)
    assert scientific["combined_within_shard_ratios"]["hex-radius-6-127"][
        "vs_exact_entity_control"
    ]["extraction_speedup"]["weighted_geometric_mean"] == pytest.approx(100.0 / 90.0)
    selected = [
        process
        for process in scientific["all_replicate_processes"]
        if process["selected_for_absolute_aggregate"]
    ]
    assert {process["replicate_index"] for process in selected} == {2}


def test_median_tie_break_is_replicate_index() -> None:
    reports = _complete_reports(control_ns_by_replicate={0: 100.0, 1: 100.0, 2: 100.0})
    aggregate = reporter.aggregate_reports(reports)

    selections = aggregate["scientific"]["median_invocation_selection"]["shards"]
    assert [selection["selected_replicate_index"] for selection in selections] == [
        1,
        1,
        1,
        1,
    ]


def test_all_invocation_sample_variance_is_reported() -> None:
    aggregate = reporter.aggregate_reports(_complete_reports())
    variance = aggregate["scientific"]["replicate_variance_by_shard"][0]
    extraction = variance["arms"][reporter.EXACT_ENTITY_CONTROL]["absolute_timing"][
        "extraction_ns_per_record"
    ]

    assert extraction["count"] == 3
    assert extraction["mean"] == pytest.approx(100.0)
    assert extraction["median"] == pytest.approx(100.0)
    assert extraction["minimum"] == pytest.approx(80.0)
    assert extraction["maximum"] == pytest.approx(120.0)
    assert extraction["sample_variance"] == pytest.approx(400.0)
    assert extraction["sample_standard_deviation"] == pytest.approx(20.0)
    assert extraction["coefficient_of_variation"] == pytest.approx(0.2)
    ratio_variance = variance["arms"]["hex-radius-6-127"]["within_process_ratios"][
        "vs_exact_entity_control"
    ]["extraction_speedup"]
    assert ratio_variance["sample_variance"] == pytest.approx(0.0)


def test_source_manifest_drift_fails_closed_as_incomplete() -> None:
    reports = _complete_reports()
    _find_report(reports, 1, 1)["datasets"][0]["manifest_blake3"] = _digest(100)

    aggregate = reporter.aggregate_reports(reports)

    assert aggregate["scientific"]["classification"] == reporter.INCOMPLETE
    assert aggregate["scientific"]["gates"]["source_manifest_identity"]["passed"] is False


def test_semantic_mismatch_has_priority_over_structural_claims() -> None:
    reports = _complete_reports()
    _find_report(reports, 1, 2)["arms"][2]["semantic_blake3"] = _digest(1234)

    aggregate = reporter.aggregate_reports(reports)

    assert aggregate["scientific"]["classification"] == reporter.SEMANTIC_FAILURE
    assert aggregate["scientific"]["gates"]["round_trip_and_semantic_identity"]["passed"] is False
    assert aggregate["scientific"]["arms"] == {}


def test_false_round_trip_is_a_semantic_failure() -> None:
    reports = _complete_reports()
    _find_report(reports, 0, 0)["arms"][0]["round_trip_verified"] = False

    aggregate = reporter.aggregate_reports(reports)

    assert aggregate["scientific"]["classification"] == reporter.SEMANTIC_FAILURE


def test_missing_required_arm_fails_closed_as_incomplete() -> None:
    reports = _complete_reports()
    report = _find_report(reports, 2, 0)
    missing = "hex-radius-4-61"
    report["selected_arms"].remove(missing)
    report["arms"] = [arm for arm in report["arms"] if arm["arm"] != missing]

    aggregate = reporter.aggregate_reports(reports)

    assert aggregate["scientific"]["classification"] == reporter.INCOMPLETE
    assert aggregate["scientific"]["gates"]["all_required_arms"]["passed"] is False
    assert aggregate["scientific"]["combined_within_shard_ratios"] == {}


def test_iteration_drift_fails_exact_identity() -> None:
    reports = _complete_reports()
    report = _find_report(reports, 1, 0)
    report["iterations"] += 1
    for arm in report["arms"]:
        arm["iterations"] += 1

    aggregate = reporter.aggregate_reports(reports)

    assert aggregate["scientific"]["classification"] == reporter.INCOMPLETE
    assert aggregate["scientific"]["gates"]["iteration_identity"]["passed"] is False


def test_invalid_replicate_timing_is_insufficient_performance_evidence() -> None:
    reports = _complete_reports()
    arm = _find_report(reports, 0, 0)["arms"][0]
    arm["extraction_seconds"] = 0.0
    arm["extraction_ns_per_record"] = 0.0
    arm["extraction_records_per_second"] = 0.0

    aggregate = reporter.aggregate_reports(reports)
    scientific = aggregate["scientific"]

    assert scientific["classification"] == reporter.INSUFFICIENT_PERFORMANCE
    assert scientific["gates"]["performance_evidence"]["passed"] is False
    assert scientific["arms"][reporter.EXACT_ENTITY_CONTROL]["throughput"] is None
    assert scientific["median_invocation_selection"] is None
    assert scientific["replicate_variance_by_shard"] == []
    invalid = next(
        process
        for process in scientific["all_replicate_processes"]
        if process["shard_index"] == 0 and process["replicate_index"] == 0
    )
    assert invalid["timing_valid"] is False
    assert invalid["within_process_ratios"] is None


def test_host_local_dataset_roots_are_not_scientific_identity() -> None:
    original = _complete_reports()
    remounted = copy.deepcopy(original)
    for index, report in enumerate(remounted):
        report["datasets"][0]["root"] = f"/different/mount/{index}"

    assert reporter.aggregate_reports(original) == reporter.aggregate_reports(remounted)


def test_cli_writes_deterministic_complete_output(tmp_path: Path) -> None:
    paths = []
    for report in _complete_reports():
        path = tmp_path / (
            f"shard-{report['shard']['shard_index']}-replicate-{report['replicate_index']}.json"
        )
        path.write_text(json.dumps(report))
        paths.append(path)
    random.Random(136).shuffle(paths)
    output = tmp_path / "aggregate.json"

    arguments = []
    for path in paths:
        arguments.extend(("--report", str(path)))
    arguments.extend(
        (
            "--required-replicates",
            "3",
            "--output",
            str(output),
        )
    )
    exit_code = reporter.main(arguments)

    assert exit_code == 0
    written = json.loads(output.read_text())
    assert written["scientific"]["classification"] == reporter.COMPLETE
    assert written["scientific"]["coverage"]["total_records"] == 40


def test_malformed_digest_is_rejected_without_guessing() -> None:
    reports = _complete_reports()
    reports[0]["source_semantic_blake3"] = "not-a-digest"

    with pytest.raises(reporter.ReportError, match="64-character"):
        reporter.aggregate_reports(reports)


def test_preregistered_replicate_count_is_frozen() -> None:
    with pytest.raises(reporter.ReportError, match="exactly 3 replicates"):
        reporter.aggregate_reports(_complete_reports(), required_replicates=5)
