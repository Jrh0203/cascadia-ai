#!/usr/bin/env python3
"""Build the untouched-C0 control and classify the ADR 0166 tournament."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import blake3
from cascadia_mlx.opportunity_cross_attention_mlx_model import ARMS
from cascadia_mlx.opportunity_cross_attention_mlx_pairwise import (
    BOOTSTRAP_REPLICATES,
    collect_decision_panel,
    compare_decision_panels,
    factorial_effects,
    panel_identity,
)
from cascadia_mlx.opportunity_cross_attention_mlx_protocol import (
    ADR_ID,
    ARM_HOSTS,
    EXPERIMENT_ID,
    PROTOCOL_ID,
    RELATIONAL_DATA_ARM,
    TRAINING_STEPS,
)
from cascadia_mlx.opportunity_cross_attention_mlx_train import (
    load_verified_warm_start,
)
from cascadia_mlx.r3_action_edit_mlx_cache import R3ActionEditMlxCache
from cascadia_mlx.relational_substrate_mlx_cache import (
    RelationalSubstrateMlxCache,
)
from cascadia_mlx.s1_exact_supply_mlx_cache import S1ExactSupplyCache

CONTROL_KIND = "untouched-c0-exact-r2"
PARENT_ARM = "c0-parent-conditioned"
TREATMENT_ARMS = tuple(arm for arm in ARMS if arm != PARENT_ARM)

CLASSIFICATION_ADVANCE = "opportunity_query_candidate_advance"
CLASSIFICATION_PARENT = "opportunity_parent_context_only"
CLASSIFICATION_NULL = "opportunity_query_factorial_null"
CLASSIFICATION_QUALITY = "opportunity_query_quality_regression"
CLASSIFICATION_PROTECTED = "opportunity_query_protected_slice_regression"
CLASSIFICATION_SERVING = "opportunity_query_serving_failure"
CLASSIFICATION_INVALID = "opportunity_query_structurally_invalid"
CLASSIFICATION_CROSS_HOST = "opportunity_query_cross_host_inconsistent"

ABSOLUTE_SERVING = {
    "groups": 240,
    "actions": 860_203,
    "p99_ms_max": 250.0,
    "memory_max_bytes": 4 * 1024**3,
}
ADVANCEMENT_LIMITS = {
    "global_probability_min": 0.95,
    "strategic_probability_min": 0.90,
    "protected_delta_min": -0.02,
    "rmse_delta_max": 0.03,
    "regret_delta_max": 0.02,
}
TIE_ORDER = {arm: index for index, arm in enumerate(ARMS)}


class OpportunityReportError(ValueError):
    """ADR 0166 evidence is incomplete or internally inconsistent."""


class OpportunityCrossHostError(OpportunityReportError):
    """A supposedly shared scientific identity differs across hosts."""


def build_untouched_c0_control(
    *,
    warm_start_run_dir: Path,
    warm_start_report: Path,
    validation_dataset: Path,
    r3_cache: Path,
    relational_cache: Path,
    s1_cache: Path,
) -> dict[str, Any]:
    """Score the exact C0 checkpoint once on the paired validation panel."""
    model, warm_start, checkpoint = load_verified_warm_start(
        warm_start_run_dir,
        warm_start_report,
    )
    report = _read_json(warm_start_report, "C0 warm-start report")
    run_report = _read_json(
        warm_start_run_dir / "final-report.json",
        "C0 run final report",
    )
    if report != run_report:
        raise OpportunityReportError(
            "C0 report differs from the run final report"
        )
    r3 = R3ActionEditMlxCache(
        r3_cache,
        verify_checksums=False,
        verify_semantics=False,
        require_complete=True,
    )
    relational = RelationalSubstrateMlxCache(
        relational_cache,
        r3_cache=r3,
        verify_checksums=False,
        verify_semantics=False,
        require_complete=True,
    )
    supply = S1ExactSupplyCache(
        s1_cache,
        verify_checksums=False,
        verify_semantics=False,
        require_complete=True,
    )
    validation = relational.bind_dataset(
        validation_dataset,
        s1_cache=supply,
        verify_dataset_checksums=False,
    )
    panel = collect_decision_panel(model, validation)
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "control_kind": CONTROL_KIND,
        "warm_start": warm_start,
        "source_report_id": report["report_id"],
        "checkpoint": {
            "checkpoint_id": checkpoint.name,
            "manifest_blake3": _file_blake3(
                checkpoint / "checkpoint.json"
            ),
            "model_blake3": _file_blake3(
                checkpoint / "model.safetensors"
            ),
        },
        "r3_cache_id": r3.cache_id,
        "relational_cache_id": relational.cache_id,
        "s1_cache_id": supply.cache_id,
        "validation_dataset_id": validation.base.manifest["dataset_id"],
        "metrics": report["metrics"],
        "performance": report["performance"],
        "paired_panel": panel,
        "paired_panel_id": panel_identity(panel),
        "information_boundary": {
            "open_validation_used": True,
            "sealed_test_opened": False,
            "gameplay_run": False,
        },
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "control_kind": CONTROL_KIND,
        "control_id": _canonical_blake3(identity),
        "scientific_identity": identity,
        **{
            key: identity[key]
            for key in (
                "warm_start",
                "source_report_id",
                "checkpoint",
                "r3_cache_id",
                "relational_cache_id",
                "s1_cache_id",
                "validation_dataset_id",
                "metrics",
                "performance",
                "paired_panel",
                "paired_panel_id",
                "information_boundary",
            )
        },
    }


def aggregate_with_order_proof(
    reports: list[dict[str, Any]],
    untouched_c0: dict[str, Any],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Classify in both input orders and require byte-identical science."""
    forward = classify_reports(
        reports,
        untouched_c0,
        bootstrap_replicates=bootstrap_replicates,
    )
    reverse = classify_reports(
        list(reversed(reports)),
        untouched_c0,
        bootstrap_replicates=bootstrap_replicates,
    )
    forward_bytes = _canonical_bytes(forward["scientific_identity"])
    reverse_bytes = _canonical_bytes(reverse["scientific_identity"])
    proof_identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "forward_aggregate_id": forward["aggregate_id"],
        "reverse_aggregate_id": reverse["aggregate_id"],
        "byte_identical": forward_bytes == reverse_bytes,
        "forward_blake3": blake3.blake3(forward_bytes).hexdigest(),
        "reverse_blake3": blake3.blake3(reverse_bytes).hexdigest(),
    }
    if not proof_identity["byte_identical"]:
        raise OpportunityReportError(
            "ADR 0166 classification depends on report order"
        )
    proof = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "proof_id": _canonical_blake3(proof_identity),
        "scientific_identity": proof_identity,
    }
    return forward, reverse, proof


def classify_reports(
    reports: list[dict[str, Any]],
    untouched_c0: dict[str, Any],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Apply every preregistered paired, protected, and serving gate."""
    by_arm = _validate_reports(reports)
    control = _validate_untouched_c0(untouched_c0, by_arm)
    parent = by_arm[PARENT_ARM]
    parent_vs_c0 = compare_decision_panels(
        parent["paired_panel"],
        control["paired_panel"],
        bootstrap_replicates=bootstrap_replicates,
    )
    parent_serving = _serving(parent)
    parent_checks = _comparison_checks(
        parent_vs_c0,
        global_strict=True,
        strategic_strict=False,
        require_global_probability=True,
        serving=parent_serving,
    )

    assessments: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    utility_positive = False
    protected_failure = False
    serving_failure = False
    for arm in TREATMENT_ARMS:
        report = by_arm[arm]
        versus_parent = compare_decision_panels(
            report["paired_panel"],
            parent["paired_panel"],
            bootstrap_replicates=bootstrap_replicates,
        )
        versus_c0 = compare_decision_panels(
            report["paired_panel"],
            control["paired_panel"],
            bootstrap_replicates=bootstrap_replicates,
        )
        serving = _serving(report)
        checks = _comparison_checks(
            versus_parent,
            global_strict=True,
            strategic_strict=True,
            require_global_probability=True,
            serving=serving,
        )
        checks["global_above_untouched_c0"] = (
            versus_c0["global_top64_recall"]["delta"] > 0.0
        )
        non_serving = {
            key: value
            for key, value in checks.items()
            if key != "absolute_serving"
        }
        arm_utility_positive = (
            checks["global_above_parent"]
            and checks["global_above_untouched_c0"]
            and checks["global_probability"]
            and checks["strategic_above_parent"]
            and checks["strategic_probability"]
        )
        utility_positive |= arm_utility_positive
        protected_failure |= (
            arm_utility_positive
            and (
                not checks["low_supply_noninferior"]
                or not checks["independent_draft_noninferior"]
            )
        )
        serving_failure |= all(non_serving.values()) and not serving["passed"]
        eligible_arm = all(checks.values())
        if eligible_arm:
            eligible.append(arm)
        assessments[arm] = {
            "host": report["host"],
            "report_id": report["report_id"],
            "paired_panel_id": report["paired_panel_id"],
            "versus_parent": versus_parent,
            "versus_untouched_c0": versus_c0,
            "serving": serving,
            "checks": checks,
            "eligible": eligible_arm,
        }

    if eligible:
        selected = _select_arm(eligible, assessments)
        classification = CLASSIFICATION_ADVANCE
    elif serving_failure:
        selected = None
        classification = CLASSIFICATION_SERVING
    elif protected_failure:
        selected = None
        classification = CLASSIFICATION_PROTECTED
    elif all(parent_checks.values()):
        selected = PARENT_ARM
        classification = CLASSIFICATION_PARENT
    elif utility_positive:
        selected = None
        classification = CLASSIFICATION_QUALITY
    else:
        selected = None
        classification = CLASSIFICATION_NULL

    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "classification": classification,
        "selected_arm": selected,
        "report_ids": {
            arm: by_arm[arm]["report_id"] for arm in ARMS
        },
        "paired_panel_ids": {
            arm: by_arm[arm]["paired_panel_id"] for arm in ARMS
        },
        "untouched_c0_control_id": control["control_id"],
        "common_identity": _common_identity(by_arm),
        "parent": {
            "versus_untouched_c0": parent_vs_c0,
            "serving": parent_serving,
            "checks": parent_checks,
        },
        "treatments": assessments,
        "factorial_effects": factorial_effects(
            {
                arm: by_arm[arm]["paired_panel"]
                for arm in ARMS
            }
        ),
        "limits": {
            "absolute_serving": ABSOLUTE_SERVING,
            "advancement": ADVANCEMENT_LIMITS,
            "bootstrap_replicates": bootstrap_replicates,
        },
        "claim_boundary": {
            "offline_evidence_complete": True,
            "paired_gameplay_qualification_authorized": (
                classification == CLASSIFICATION_ADVANCE
            ),
            "gameplay_strength_established": False,
            "champion_changed": False,
            "progress_to_100_established": False,
        },
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "classification": classification,
        "selected_arm": selected,
        "aggregate_id": _canonical_blake3(identity),
        "scientific_identity": identity,
    }


def invalid_outputs(
    error: Exception,
    report_paths: list[Path],
    c0_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    classification = (
        CLASSIFICATION_CROSS_HOST
        if isinstance(error, OpportunityCrossHostError)
        else CLASSIFICATION_INVALID
    )
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "classification": classification,
        "selected_arm": None,
        "error_type": type(error).__name__,
        "error": str(error),
        "inputs": {
            "reports": sorted(
                (_path_identity(path) for path in report_paths),
                key=lambda identity: identity["path"],
            ),
            "untouched_c0": _path_identity(c0_path),
        },
        "claim_boundary": {
            "offline_evidence_complete": False,
            "paired_gameplay_qualification_authorized": False,
            "gameplay_strength_established": False,
            "progress_to_100_established": False,
        },
    }
    aggregate = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "classification": classification,
        "selected_arm": None,
        "aggregate_id": _canonical_blake3(identity),
        "scientific_identity": identity,
    }
    proof_identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "forward_aggregate_id": aggregate["aggregate_id"],
        "reverse_aggregate_id": aggregate["aggregate_id"],
        "byte_identical": True,
        "invalid_evidence": True,
    }
    proof = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "proof_id": _canonical_blake3(proof_identity),
        "scientific_identity": proof_identity,
    }
    return aggregate, dict(aggregate), proof


def validate_collected_checkpoints(
    reports: list[dict[str, Any]],
    checkpoint_dirs: dict[str, Path],
) -> None:
    """Bind every collected production report to its exact checkpoint bytes."""
    if set(checkpoint_dirs) != set(ARMS):
        raise OpportunityReportError(
            "ADR 0166 requires one collected checkpoint directory per arm"
        )
    by_arm = {
        report.get("arm"): report
        for report in reports
        if report.get("arm") in ARMS
    }
    if set(by_arm) != set(ARMS):
        raise OpportunityReportError(
            "ADR 0166 checkpoint validation lacks complete arm reports"
        )
    for arm in ARMS:
        checkpoint = checkpoint_dirs[arm]
        report_identity = by_arm[arm].get("checkpoint")
        manifest = checkpoint / "checkpoint.json"
        model = checkpoint / "model.safetensors"
        try:
            manifest_identity = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            manifest_identity = None
        if (
            not isinstance(report_identity, dict)
            or not manifest.is_file()
            or not model.is_file()
            or not isinstance(manifest_identity, dict)
            or _file_blake3(manifest)
            != report_identity.get("manifest_blake3")
            or _file_blake3(model) != report_identity.get("model_blake3")
            or manifest_identity.get("checkpoint_id")
            != Path(str(report_identity.get("path", ""))).name
            or manifest_identity.get("model_config", {}).get("arm") != arm
        ):
            raise OpportunityReportError(
                f"ADR 0166 collected checkpoint differs for {arm}"
            )


def _validate_reports(
    reports: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if len(reports) != len(ARMS):
        raise OpportunityReportError(
            "ADR 0166 requires exactly four production reports"
        )
    by_arm: dict[str, dict[str, Any]] = {}
    traces: dict[str, list[tuple[int, str, int]]] = {}
    for report in reports:
        arm = report.get("arm")
        identity = report.get("scientific_identity")
        panel = report.get("paired_panel")
        if (
            report.get("schema_version") != 1
            or report.get("experiment_id") != EXPERIMENT_ID
            or report.get("protocol_id") != PROTOCOL_ID
            or report.get("adr") != ADR_ID
            or report.get("mode") != "production"
            or arm not in ARMS
            or report.get("host") != ARM_HOSTS[arm]
            or report.get("data_arm") != RELATIONAL_DATA_ARM
            or not isinstance(identity, dict)
            or _canonical_blake3(identity) != report.get("report_id")
            or not isinstance(panel, list)
            or len(panel) != ABSOLUTE_SERVING["groups"]
            or panel_identity(panel) != report.get("paired_panel_id")
            or report.get("claims", {}).get(
                "offline_comparison_complete"
            )
            is not True
            or report.get("claims", {}).get("base_parameters_frozen")
            is not True
            or report.get("information_boundary", {}).get(
                "sealed_test_opened"
            )
            is not False
            or arm in by_arm
        ):
            raise OpportunityReportError(
                "ADR 0166 production report is malformed or duplicated"
            )
        optimization = report.get("optimization", {})
        trace = optimization.get("loss_trace")
        if (
            optimization.get("global_step") != TRAINING_STEPS
            or not isinstance(trace, list)
            or len(trace) != TRAINING_STEPS
        ):
            raise OpportunityReportError(
                f"{arm} production training trace is incomplete"
            )
        normalized = []
        for step, event in enumerate(trace, start=1):
            if (
                not isinstance(event, dict)
                or event.get("step") != step
                or not isinstance(event.get("batch_blake3"), str)
                or len(event["batch_blake3"]) != 64
                or not isinstance(event.get("candidates"), int)
                or not _finite(event.get("loss"))
                or not _finite(event.get("elapsed_seconds"))
            ):
                raise OpportunityReportError(
                    f"{arm} production trace is malformed"
                )
            normalized.append(
                (step, event["batch_blake3"], event["candidates"])
            )
        traces[arm] = normalized
        by_arm[arm] = report
    if set(by_arm) != set(ARMS):
        raise OpportunityReportError("ADR 0166 arm coverage is incomplete")
    if len({tuple(trace) for trace in traces.values()}) != 1:
        raise OpportunityCrossHostError(
            "ADR 0166 arms consumed different scientific batches"
        )
    _common_identity(by_arm)
    return by_arm


def _common_identity(
    by_arm: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    parent = by_arm[PARENT_ARM]
    common = _shared_identity(parent)
    for arm, report in by_arm.items():
        if _shared_identity(report) != common:
            raise OpportunityCrossHostError(
                f"ADR 0166 common identity drifted on {arm}"
            )
    return common


def _shared_identity(report: dict[str, Any]) -> dict[str, Any]:
    model = report.get("model", {})
    source = report.get("source", {})
    controls = report.get("controls", {})
    warm_start = report.get("warm_start", {})
    r6 = report.get("r6_binary", {})
    return {
        "r3_cache_id": report.get("r3_cache_id"),
        "relational_cache_id": report.get("relational_cache_id"),
        "s1_cache_id": report.get("s1_cache_id"),
        "r6_binary_blake3": r6.get("blake3"),
        "protocol": report.get("protocol"),
        "warm_start_id": warm_start.get("warm_start_id"),
        "source_blake3": source.get("v2_source_blake3"),
        "authorization_id": controls.get("authorization_id"),
        "open_data_verification_id": controls.get(
            "open_data_verification_id"
        ),
        "total_parameter_count": model.get("total_parameter_count"),
        "total_parameter_layout_blake3": model.get(
            "total_parameter_layout_blake3"
        ),
        "adapter_parameter_count": model.get("adapter_parameter_count"),
        "adapter_parameter_layout_blake3": model.get(
            "adapter_parameter_layout_blake3"
        ),
        "initial_all_parameter_tensor_blake3": model.get(
            "initial_all_parameter_tensor_blake3"
        ),
        "initial_adapter_parameter_tensor_blake3": model.get(
            "initial_adapter_parameter_tensor_blake3"
        ),
        "base_parameter_tensor_blake3": model.get(
            "base_parameter_tensor_blake3"
        ),
    }


def _validate_untouched_c0(
    control: dict[str, Any],
    by_arm: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    identity = control.get("scientific_identity")
    panel = control.get("paired_panel")
    parent = by_arm[PARENT_ARM]
    if (
        control.get("schema_version") != 1
        or control.get("experiment_id") != EXPERIMENT_ID
        or control.get("protocol_id") != PROTOCOL_ID
        or control.get("adr") != ADR_ID
        or control.get("control_kind") != CONTROL_KIND
        or not isinstance(identity, dict)
        or _canonical_blake3(identity) != control.get("control_id")
        or not isinstance(panel, list)
        or len(panel) != ABSOLUTE_SERVING["groups"]
        or panel_identity(panel) != control.get("paired_panel_id")
        or control.get("warm_start")
        != parent.get("warm_start")
        or control.get("r3_cache_id") != parent.get("r3_cache_id")
        or control.get("relational_cache_id")
        != parent.get("relational_cache_id")
        or control.get("s1_cache_id") != parent.get("s1_cache_id")
        or control.get("information_boundary", {}).get(
            "sealed_test_opened"
        )
        is not False
    ):
        raise OpportunityReportError(
            "ADR 0166 untouched C0 control is malformed or mismatched"
        )
    return control


def _comparison_checks(
    comparison: dict[str, Any],
    *,
    global_strict: bool,
    strategic_strict: bool,
    require_global_probability: bool,
    serving: dict[str, Any],
) -> dict[str, bool]:
    global_delta = comparison["global_top64_recall"]["delta"]
    strategic_delta = comparison["strategic_top64_recall"]["delta"]
    return {
        "global_above_parent": (
            global_delta > 0.0 if global_strict else global_delta >= 0.0
        ),
        "global_probability": (
            not require_global_probability
            or comparison["global_top64_recall"]["probability_favorable"]
            >= ADVANCEMENT_LIMITS["global_probability_min"]
        ),
        "strategic_above_parent": (
            strategic_delta > 0.0
            if strategic_strict
            else strategic_delta >= 0.0
        ),
        "strategic_probability": (
            comparison["strategic_top64_recall"]["probability_favorable"]
            >= ADVANCEMENT_LIMITS["strategic_probability_min"]
        ),
        "low_supply_noninferior": (
            comparison["protected"]["low_supply"]["delta"]
            >= ADVANCEMENT_LIMITS["protected_delta_min"]
        ),
        "independent_draft_noninferior": (
            comparison["protected"]["independent_draft_winner"]["delta"]
            >= ADVANCEMENT_LIMITS["protected_delta_min"]
        ),
        "rmse_noninferior": (
            comparison["r4800_rmse"]["delta"]
            <= ADVANCEMENT_LIMITS["rmse_delta_max"]
        ),
        "regret_noninferior": (
            comparison["top64_regret"]["delta"]
            <= ADVANCEMENT_LIMITS["regret_delta_max"]
        ),
        "absolute_serving": serving["passed"],
    }


def _serving(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    performance = report.get("performance", {})
    combined = performance.get("combined_with_r6", {})
    memory = performance.get("memory", {})
    r6 = performance.get("r6_apply_undo", {})
    latency = combined.get("latency_milliseconds", {})
    checks = {
        "metrics_groups": metrics.get("groups")
        == ABSOLUTE_SERVING["groups"],
        "metrics_actions": metrics.get("candidates")
        == ABSOLUTE_SERVING["actions"],
        "all_groups_scored_once": metrics.get("all_groups_scored_once")
        is True,
        "all_candidates_scored_once": metrics.get(
            "all_candidates_scored_once"
        )
        is True,
        "finite": metrics.get("all_scores_and_uncertainties_finite")
        is True,
        "combined_groups": combined.get("groups")
        == ABSOLUTE_SERVING["groups"],
        "combined_actions": combined.get("actions")
        == ABSOLUTE_SERVING["actions"],
        "r6_exact": (
            combined.get("r6_exact_parity_pass") is True
            and r6.get("exact_parity_pass") is True
            and r6.get("apply_failures") == 0
            and r6.get("undo_failures") == 0
        ),
        "p99_latency": (
            _finite(latency.get("p99"))
            and latency["p99"] <= ABSOLUTE_SERVING["p99_ms_max"]
        ),
        "peak_rss": (
            isinstance(memory.get("peak_process_rss_bytes"), int)
            and memory["peak_process_rss_bytes"]
            <= ABSOLUTE_SERVING["memory_max_bytes"]
        ),
        "peak_active": (
            isinstance(memory.get("peak_active_bytes"), int)
            and memory["peak_active_bytes"]
            <= ABSOLUTE_SERVING["memory_max_bytes"]
        ),
        "swap": (
            memory.get("process_swaps") == 0
            and memory.get("system_swap_delta_bytes") in (None, 0)
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "measurements": {
            "p99_milliseconds": latency.get("p99"),
            "peak_process_rss_bytes": memory.get(
                "peak_process_rss_bytes"
            ),
            "peak_active_bytes": memory.get("peak_active_bytes"),
            "combined_actions_per_second": combined.get(
                "action_scores_per_second"
            ),
        },
    }


def _select_arm(
    eligible: list[str],
    assessments: dict[str, dict[str, Any]],
) -> str:
    def key(
        arm: str,
    ) -> tuple[float, float, float, float, float, int, int]:
        candidate = assessments[arm]
        parent = candidate["versus_parent"]
        serving = candidate["serving"]["measurements"]
        return (
            -parent["global_top64_recall"]["delta"],
            -parent["strategic_top64_recall"]["delta"],
            parent["top64_regret"]["delta"],
            parent["r4800_rmse"]["delta"],
            float(serving["p99_milliseconds"]),
            int(serving["peak_process_rss_bytes"]),
            TIE_ORDER[arm],
        )

    return min(eligible, key=key)


def _path_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "blake3": _file_blake3(path),
    }


def _file_blake3(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _canonical_blake3(value: object) -> str:
    return blake3.blake3(_canonical_bytes(value)).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise OpportunityReportError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise OpportunityReportError(f"{label} must be a JSON object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    c0 = subparsers.add_parser("build-c0-control")
    c0.add_argument("--warm-start-run-dir", type=Path, required=True)
    c0.add_argument("--warm-start-report", type=Path, required=True)
    c0.add_argument("--validation-dataset", type=Path, required=True)
    c0.add_argument("--r3-cache", type=Path, required=True)
    c0.add_argument("--relational-cache", type=Path, required=True)
    c0.add_argument("--s1-cache", type=Path, required=True)
    c0.add_argument("--output", type=Path, required=True)

    classify = subparsers.add_parser("classify")
    classify.add_argument("--report", type=Path, action="append", required=True)
    classify.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="ARM=DIR",
    )
    classify.add_argument("--untouched-c0", type=Path, required=True)
    classify.add_argument("--forward-output", type=Path, required=True)
    classify.add_argument("--reverse-output", type=Path, required=True)
    classify.add_argument("--order-proof-output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "build-c0-control":
        result = build_untouched_c0_control(
            warm_start_run_dir=args.warm_start_run_dir,
            warm_start_report=args.warm_start_report,
            validation_dataset=args.validation_dataset,
            r3_cache=args.r3_cache,
            relational_cache=args.relational_cache,
            s1_cache=args.s1_cache,
        )
        _write_json_atomic(args.output, result)
        print(
            json.dumps(
                {
                    "control_id": result["control_id"],
                    "paired_panel_id": result["paired_panel_id"],
                },
                sort_keys=True,
            )
        )
        return 0

    report_paths = list(args.report)
    try:
        reports = [
            _read_json(path, f"ADR 0166 report {path}")
            for path in report_paths
        ]
        validate_collected_checkpoints(
            reports,
            _parse_arm_paths(args.checkpoint),
        )
        untouched = _read_json(
            args.untouched_c0,
            "ADR 0166 untouched C0 control",
        )
        forward, reverse, proof = aggregate_with_order_proof(
            reports,
            untouched,
        )
        exit_code = 0
    except (
        KeyError,
        TypeError,
        OpportunityReportError,
        ValueError,
    ) as error:
        forward, reverse, proof = invalid_outputs(
            error,
            report_paths,
            args.untouched_c0,
        )
        exit_code = 2
    _write_json_atomic(args.forward_output, forward)
    _write_json_atomic(args.reverse_output, reverse)
    _write_json_atomic(args.order_proof_output, proof)
    print(
        json.dumps(
            {
                "classification": forward["classification"],
                "selected_arm": forward["selected_arm"],
                "aggregate_id": forward["aggregate_id"],
                "order_proof_id": proof["proof_id"],
            },
            sort_keys=True,
        )
    )
    return exit_code


def _parse_arm_paths(values: list[str]) -> dict[str, Path]:
    parsed = {}
    for value in values:
        arm, separator, path = value.partition("=")
        if not separator or arm not in ARMS or not path or arm in parsed:
            raise OpportunityReportError(
                "checkpoint arguments must be unique ARM=DIR values"
            )
        parsed[arm] = Path(path)
    if set(parsed) != set(ARMS):
        raise OpportunityReportError(
            "checkpoint arguments omit a registered ADR 0166 arm"
        )
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
