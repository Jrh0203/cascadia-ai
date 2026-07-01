#!/usr/bin/env python3
"""Validate the four-host ADR 0166 common-arm numerical smoke."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np
from cascadia_mlx.opportunity_cross_attention_mlx_model import ARMS
from cascadia_mlx.opportunity_cross_attention_mlx_protocol import (
    ADR_ID,
    EXPERIMENT_ID,
    PROTOCOL_ID,
)

SMOKE_ARM = ARMS[0]
SMOKE_STEPS = 3
EXPECTED_HOSTS = ("john1", "john2", "john3", "john4")

LOSS_MAX_ABS = 1e-4
LOSS_MAX_REL = 1e-5
PARAMETER_MAX_ABS = 1e-4
PARAMETER_MEAN_ABS = 1e-6
SCORE_MAX_ABS = 1e-4
UNCERTAINTY_MAX_ABS = 1e-5

PASS = "opportunity_query_cross_host_smoke_pass"
INVALID = "opportunity_query_cross_host_smoke_invalid"


class SmokeParityError(ValueError):
    """The bounded four-host smoke is malformed or outside tolerance."""


def compare_smoke(
    report_paths: list[Path],
    checkpoint_paths: dict[str, Path],
) -> dict[str, Any]:
    """Compare four reports independent of CLI ordering."""
    if len(report_paths) != len(EXPECTED_HOSTS) or set(
        checkpoint_paths
    ) != set(EXPECTED_HOSTS):
        raise SmokeParityError("ADR 0166 smoke requires four reports and checkpoints")
    reports = [_read_smoke_report(path) for path in report_paths]
    by_host = {report["host"]: report for report in reports}
    if set(by_host) != set(EXPECTED_HOSTS) or len(by_host) != len(reports):
        raise SmokeParityError("ADR 0166 smoke host set is incomplete or duplicated")
    checkpoints = {
        host: _load_checkpoint(checkpoint_paths[host], by_host[host])
        for host in EXPECTED_HOSTS
    }

    reference = by_host["john1"]
    shared_fields = (
        "arm",
        "data_arm",
        "r3_cache_id",
        "relational_cache_id",
        "s1_cache_id",
        "protocol",
        "warm_start",
        "zero_init_prediction_parity",
    )
    for host in EXPECTED_HOSTS[1:]:
        candidate = by_host[host]
        for field in shared_fields:
            if candidate.get(field) != reference.get(field):
                raise SmokeParityError(
                    f"ADR 0166 smoke shared field differs on {host}: {field}"
                )
        if (
            candidate["r6_binary"]["blake3"]
            != reference["r6_binary"]["blake3"]
            or candidate["source"]["v2_source_blake3"]
            != reference["source"]["v2_source_blake3"]
        ):
            raise SmokeParityError(
                f"ADR 0166 smoke binary or source identity differs on {host}"
            )

    reference_trace = reference["optimization"]["loss_trace"]
    reference_batches = [event["batch_blake3"] for event in reference_trace]
    reference_candidates = [event["candidates"] for event in reference_trace]
    reference_model = reference["model"]
    reference_panel = reference["metrics"]["prediction_panel"]
    reference_losses = np.asarray(
        [event["loss"] for event in reference_trace],
        dtype=np.float64,
    )
    reference_scores = np.asarray(reference_panel["scores"], dtype=np.float64)
    reference_uncertainties = np.asarray(
        reference_panel["standard_errors"],
        dtype=np.float64,
    )
    reference_ranking = _stable_ranking(
        reference_scores,
        reference_panel["action_hashes"],
    )

    per_host = {}
    all_checks = {}
    for host in EXPECTED_HOSTS:
        report = by_host[host]
        trace = report["optimization"]["loss_trace"]
        panel = report["metrics"]["prediction_panel"]
        model = report["model"]
        losses = np.asarray(
            [event["loss"] for event in trace],
            dtype=np.float64,
        )
        scores = np.asarray(panel["scores"], dtype=np.float64)
        uncertainties = np.asarray(
            panel["standard_errors"],
            dtype=np.float64,
        )
        for name, values in (
            ("losses", losses),
            ("scores", scores),
            ("uncertainties", uncertainties),
        ):
            if not np.all(np.isfinite(values)):
                raise SmokeParityError(
                    f"ADR 0166 smoke contains nonfinite {name} on {host}"
                )
        parameter = _parameter_drift(
            checkpoints["john1"],
            checkpoints[host],
        )
        if parameter["parameters"] != model["total_parameter_count"]:
            raise SmokeParityError(
                f"ADR 0166 checkpoint parameter count differs on {host}"
            )
        loss_abs = np.abs(reference_losses - losses)
        loss_rel = loss_abs / np.maximum(np.abs(reference_losses), 1e-12)
        score_abs = np.abs(reference_scores - scores)
        uncertainty_abs = np.abs(reference_uncertainties - uncertainties)
        measurements = {
            "loss_max_abs": float(loss_abs.max(initial=0.0)),
            "loss_max_rel": float(loss_rel.max(initial=0.0)),
            "parameter_max_abs": parameter["max_abs"],
            "parameter_mean_abs": parameter["mean_abs"],
            "parameter_changed_scalars": parameter["changed_scalars"],
            "prediction_score_max_abs": float(score_abs.max(initial=0.0)),
            "prediction_uncertainty_max_abs": float(
                uncertainty_abs.max(initial=0.0)
            ),
        }
        checks = {
            "batch_identity_exact": (
                [event["batch_blake3"] for event in trace]
                == reference_batches
            ),
            "candidate_counts_exact": (
                [event["candidates"] for event in trace]
                == reference_candidates
            ),
            "initialization_exact": (
                model["initial_all_parameter_tensor_blake3"]
                == reference_model["initial_all_parameter_tensor_blake3"]
                and model["initial_adapter_parameter_tensor_blake3"]
                == reference_model["initial_adapter_parameter_tensor_blake3"]
            ),
            "base_frozen_exact": (
                model["final_base_parameter_tensor_blake3"]
                == report["warm_start"]["base_parameter_tensor_blake3"]
            ),
            "panel_action_identity_exact": (
                panel["action_hashes"] == reference_panel["action_hashes"]
            ),
            "panel_stable_ranking_exact": np.array_equal(
                _stable_ranking(scores, panel["action_hashes"]),
                reference_ranking,
            ),
            "loss_max_abs_within_tolerance": (
                measurements["loss_max_abs"] <= LOSS_MAX_ABS
            ),
            "loss_max_rel_within_tolerance": (
                measurements["loss_max_rel"] <= LOSS_MAX_REL
            ),
            "parameter_max_abs_within_tolerance": (
                measurements["parameter_max_abs"] <= PARAMETER_MAX_ABS
            ),
            "parameter_mean_abs_within_tolerance": (
                measurements["parameter_mean_abs"] <= PARAMETER_MEAN_ABS
            ),
            "prediction_score_max_abs_within_tolerance": (
                measurements["prediction_score_max_abs"] <= SCORE_MAX_ABS
            ),
            "prediction_uncertainty_max_abs_within_tolerance": (
                measurements["prediction_uncertainty_max_abs"]
                <= UNCERTAINTY_MAX_ABS
            ),
            "r6_exact_parity": (
                report["performance"]["combined_with_r6"][
                    "r6_exact_parity_pass"
                ]
                is True
            ),
        }
        per_host[host] = {
            "report_id": report["report_id"],
            "checkpoint_model_blake3": report["checkpoint"]["model_blake3"],
            "measurements": measurements,
            "checks": checks,
        }
        all_checks[host] = all(checks.values())
    if not all(all_checks.values()):
        raise SmokeParityError(
            f"ADR 0166 smoke numerical parity failed: {all_checks}"
        )

    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "arm": SMOKE_ARM,
        "steps": SMOKE_STEPS,
        "hosts": list(EXPECTED_HOSTS),
        "r3_cache_id": reference["r3_cache_id"],
        "relational_cache_id": reference["relational_cache_id"],
        "s1_cache_id": reference["s1_cache_id"],
        "r6_binary_blake3": reference["r6_binary"]["blake3"],
        "source_blake3": reference["source"]["v2_source_blake3"],
        "warm_start_id": reference["warm_start"]["warm_start_id"],
        "tolerances": {
            "loss_max_abs": LOSS_MAX_ABS,
            "loss_max_rel": LOSS_MAX_REL,
            "parameter_max_abs": PARAMETER_MAX_ABS,
            "parameter_mean_abs": PARAMETER_MEAN_ABS,
            "prediction_score_max_abs": SCORE_MAX_ABS,
            "prediction_uncertainty_max_abs": UNCERTAINTY_MAX_ABS,
        },
        "per_host": per_host,
        "checks": all_checks,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "classification": PASS,
        "proof_id": _canonical_blake3(identity),
        "scientific_identity": identity,
        "claims": {
            "cross_host_smoke_complete": True,
            "production_training_started": False,
            "gameplay_strength_measured": False,
            "promotion_authorized": False,
            "progress_to_100_claimed": False,
        },
    }


def _read_smoke_report(path: Path) -> dict[str, Any]:
    report = _read_json(path, f"ADR 0166 smoke report {path}")
    identity = report.get("scientific_identity")
    if (
        report.get("schema_version") != 1
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("adr") != ADR_ID
        or report.get("mode") != "bounded-smoke"
        or report.get("arm") != SMOKE_ARM
        or report.get("host") not in EXPECTED_HOSTS
        or report.get("optimization", {}).get("global_step") != SMOKE_STEPS
        or report.get("claims", {}).get("bounded_smoke_complete") is not True
        or report.get("claims", {}).get("base_parameters_frozen") is not True
        or report.get("claims", {}).get("offline_comparison_complete")
        is not False
        or not isinstance(identity, dict)
        or identity != _report_scientific_identity(report)
        or _canonical_blake3(identity) != report.get("report_id")
    ):
        raise SmokeParityError(
            f"ADR 0166 smoke report is malformed: {path}"
        )
    trace = report["optimization"].get("loss_trace")
    if (
        not isinstance(trace, list)
        or len(trace) != SMOKE_STEPS
        or [event.get("step") for event in trace]
        != list(range(1, SMOKE_STEPS + 1))
    ):
        raise SmokeParityError(
            f"ADR 0166 smoke loss trace is malformed: {path}"
        )
    return report


def _load_checkpoint(
    path: Path,
    report: dict[str, Any],
) -> dict[str, mx.array]:
    if (
        not path.is_file()
        or _checksum(path) != report["checkpoint"].get("model_blake3")
    ):
        raise SmokeParityError(
            f"ADR 0166 smoke checkpoint checksum differs: {path}"
        )
    tensors = mx.load(path)
    if not isinstance(tensors, dict) or not tensors:
        raise SmokeParityError(
            f"ADR 0166 smoke checkpoint is empty: {path}"
        )
    return tensors


def _parameter_drift(
    left: dict[str, mx.array],
    right: dict[str, mx.array],
) -> dict[str, int | float]:
    if left.keys() != right.keys():
        raise SmokeParityError(
            "ADR 0166 smoke checkpoint tensor names differ"
        )
    total = 0
    changed = 0
    absolute_sum = 0.0
    maximum = 0.0
    for name in left:
        left_value = np.asarray(left[name])
        right_value = np.asarray(right[name])
        if (
            left_value.shape != right_value.shape
            or left_value.dtype != right_value.dtype
        ):
            raise SmokeParityError(
                f"ADR 0166 smoke checkpoint tensor layout differs: {name}"
            )
        difference = np.abs(
            left_value.astype(np.float64)
            - right_value.astype(np.float64)
        )
        if not np.all(np.isfinite(difference)):
            raise SmokeParityError(
                f"ADR 0166 smoke checkpoint is nonfinite: {name}"
            )
        total += difference.size
        changed += int(np.count_nonzero(difference))
        absolute_sum += float(difference.sum())
        maximum = max(maximum, float(difference.max(initial=0.0)))
    return {
        "parameters": total,
        "changed_scalars": changed,
        "max_abs": maximum,
        "mean_abs": absolute_sum / max(total, 1),
    }


def _stable_ranking(scores: np.ndarray, hashes: list[str]) -> np.ndarray:
    return np.lexsort((np.asarray(hashes), -scores))


def _report_scientific_identity(
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in (
            "experiment_id",
            "protocol_id",
            "adr",
            "mode",
            "arm",
            "host",
            "data_arm",
            "r3_cache_id",
            "relational_cache_id",
            "s1_cache_id",
            "r6_binary",
            "protocol",
            "warm_start",
            "zero_init_prediction_parity",
            "model",
            "optimization",
            "checkpoint",
            "metrics",
            "paired_panel",
            "paired_panel_id",
            "performance",
            "runtime",
            "source",
            "controls",
            "information_boundary",
            "claims",
        )
    }


def _canonical_blake3(value: object) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SmokeParityError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise SmokeParityError(f"{label} must be a JSON object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare four ADR 0166 common-arm smoke runs"
    )
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="HOST=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        checkpoint_paths = _parse_host_paths(args.checkpoint)
        result = compare_smoke(args.report, checkpoint_paths)
        exit_code = 0
    except (SmokeParityError, KeyError, TypeError, ValueError) as error:
        identity = {
            "experiment_id": EXPERIMENT_ID,
            "protocol_id": PROTOCOL_ID,
            "adr": ADR_ID,
            "classification": INVALID,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        result = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "protocol_id": PROTOCOL_ID,
            "adr": ADR_ID,
            "classification": INVALID,
            "proof_id": _canonical_blake3(identity),
            "scientific_identity": identity,
            "claims": {
                "cross_host_smoke_complete": False,
                "production_training_started": False,
            },
        }
        exit_code = 2
    _write_json_atomic(args.output, result)
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "proof_id": result["proof_id"],
            },
            sort_keys=True,
        )
    )
    raise SystemExit(exit_code)


def _parse_host_paths(values: list[str]) -> dict[str, Path]:
    parsed = {}
    for value in values:
        host, separator, path = value.partition("=")
        if (
            not separator
            or host not in EXPECTED_HOSTS
            or not path
            or host in parsed
        ):
            raise SmokeParityError(
                "checkpoint arguments must be unique HOST=PATH values"
            )
        parsed[host] = Path(path)
    if set(parsed) != set(EXPECTED_HOSTS):
        raise SmokeParityError("checkpoint arguments omit a registered host")
    return parsed


if __name__ == "__main__":
    main()
