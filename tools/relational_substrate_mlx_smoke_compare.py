#!/usr/bin/env python3
"""Validate the frozen john1/john4 ADR 0161 numerical-parity smoke."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np
from cascadia_mlx.relational_substrate_mlx_cache import (
    ADR_ID,
    EXPERIMENT_ID,
    PROTOCOL_ID,
)

SMOKE_ARM = "d3-r5-s3-s5"
SMOKE_STEPS = 10
EXPECTED_HOSTS = ("john1", "john4")

LOSS_MAX_ABS = 1e-4
LOSS_MAX_REL = 1e-5
PARAMETER_MAX_ABS = 1e-4
PARAMETER_MEAN_ABS = 1e-6
SCORE_MAX_ABS = 1e-4
UNCERTAINTY_MAX_ABS = 1e-5

PASS = "relational_substrate_mlx_cross_host_smoke_pass"
INVALID = "relational_substrate_mlx_cross_host_smoke_invalid"


class SmokeParityError(ValueError):
    """The bounded cross-host smoke is malformed or outside tolerance."""


def compare_smoke(
    left_report_path: Path,
    right_report_path: Path,
    left_checkpoint_path: Path,
    right_checkpoint_path: Path,
) -> dict[str, Any]:
    left = _read_smoke_report(left_report_path)
    right = _read_smoke_report(right_report_path)
    if (left["host"], right["host"]) != EXPECTED_HOSTS:
        raise SmokeParityError(
            "ADR 0161 smoke reports must be ordered john1 then john4"
        )
    for field in (
        "arm",
        "r3_cache_id",
        "relational_cache_id",
        "s1_cache_id",
        "protocol",
    ):
        if left.get(field) != right.get(field):
            raise SmokeParityError(
                f"ADR 0161 smoke shared field differs: {field}"
            )
    if left["r6_binary"]["blake3"] != right["r6_binary"]["blake3"]:
        raise SmokeParityError(
            "ADR 0161 smoke R6 binary identity differs"
        )

    left_trace = left["optimization"]["loss_trace"]
    right_trace = right["optimization"]["loss_trace"]
    left_batches = [event["batch_blake3"] for event in left_trace]
    right_batches = [event["batch_blake3"] for event in right_trace]
    left_candidates = [event["candidates"] for event in left_trace]
    right_candidates = [event["candidates"] for event in right_trace]
    if left_batches != right_batches or left_candidates != right_candidates:
        raise SmokeParityError(
            "ADR 0161 smoke scientific batch identity differs"
        )

    left_model = left["model"]
    right_model = right["model"]
    for field in (
        "parameter_count",
        "parameter_layout_blake3",
        "initial_parameter_tensor_blake3",
    ):
        if left_model.get(field) != right_model.get(field):
            raise SmokeParityError(
                f"ADR 0161 smoke initialization differs: {field}"
            )

    left_panel = left["metrics"]["prediction_panel"]
    right_panel = right["metrics"]["prediction_panel"]
    if left_panel["action_hashes"] != right_panel["action_hashes"]:
        raise SmokeParityError(
            "ADR 0161 smoke prediction-panel action identities differ"
        )

    left_losses = np.asarray(
        [event["loss"] for event in left_trace],
        dtype=np.float64,
    )
    right_losses = np.asarray(
        [event["loss"] for event in right_trace],
        dtype=np.float64,
    )
    left_scores = np.asarray(left_panel["scores"], dtype=np.float64)
    right_scores = np.asarray(right_panel["scores"], dtype=np.float64)
    left_uncertainties = np.asarray(
        left_panel["standard_errors"],
        dtype=np.float64,
    )
    right_uncertainties = np.asarray(
        right_panel["standard_errors"],
        dtype=np.float64,
    )
    for name, values in (
        ("left losses", left_losses),
        ("right losses", right_losses),
        ("left scores", left_scores),
        ("right scores", right_scores),
        ("left uncertainties", left_uncertainties),
        ("right uncertainties", right_uncertainties),
    ):
        if not np.all(np.isfinite(values)):
            raise SmokeParityError(
                f"ADR 0161 smoke contains nonfinite {name}"
            )

    loss_abs = np.abs(left_losses - right_losses)
    loss_rel = loss_abs / np.maximum(np.abs(left_losses), 1e-12)
    score_abs = np.abs(left_scores - right_scores)
    uncertainty_abs = np.abs(
        left_uncertainties - right_uncertainties
    )
    left_ranking = _stable_ranking(
        left_scores,
        left_panel["action_hashes"],
    )
    right_ranking = _stable_ranking(
        right_scores,
        right_panel["action_hashes"],
    )

    left_checkpoint = _load_checkpoint(left_checkpoint_path, left)
    right_checkpoint = _load_checkpoint(right_checkpoint_path, right)
    parameter = _parameter_drift(left_checkpoint, right_checkpoint)
    if parameter["parameters"] != left_model["parameter_count"]:
        raise SmokeParityError(
            "ADR 0161 checkpoint parameter count differs from report"
        )

    measurements = {
        "loss_max_abs": float(loss_abs.max(initial=0.0)),
        "loss_max_rel": float(loss_rel.max(initial=0.0)),
        "parameter_max_abs": parameter["max_abs"],
        "parameter_mean_abs": parameter["mean_abs"],
        "parameter_changed_scalars": parameter["changed_scalars"],
        "prediction_score_max_abs": float(
            score_abs.max(initial=0.0)
        ),
        "prediction_uncertainty_max_abs": float(
            uncertainty_abs.max(initial=0.0)
        ),
    }
    checks = {
        "batch_identity_exact": left_batches == right_batches,
        "candidate_counts_exact": left_candidates == right_candidates,
        "initialization_exact": (
            left_model["initial_parameter_tensor_blake3"]
            == right_model["initial_parameter_tensor_blake3"]
        ),
        "panel_action_identity_exact": (
            left_panel["action_hashes"]
            == right_panel["action_hashes"]
        ),
        "panel_stable_ranking_exact": np.array_equal(
            left_ranking,
            right_ranking,
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
    }
    if not all(checks.values()):
        raise SmokeParityError(
            f"ADR 0161 smoke numerical parity failed: {checks}"
        )

    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "arm": SMOKE_ARM,
        "steps": SMOKE_STEPS,
        "hosts": list(EXPECTED_HOSTS),
        "r3_cache_id": left["r3_cache_id"],
        "relational_cache_id": left["relational_cache_id"],
        "s1_cache_id": left["s1_cache_id"],
        "r6_binary_blake3": left["r6_binary"]["blake3"],
        "report_ids": {
            "john1": left["report_id"],
            "john4": right["report_id"],
        },
        "checkpoint_blake3": {
            "john1": _checksum(left_checkpoint_path),
            "john4": _checksum(right_checkpoint_path),
        },
        "tolerances": {
            "loss_max_abs": LOSS_MAX_ABS,
            "loss_max_rel": LOSS_MAX_REL,
            "parameter_max_abs": PARAMETER_MAX_ABS,
            "parameter_mean_abs": PARAMETER_MEAN_ABS,
            "prediction_score_max_abs": SCORE_MAX_ABS,
            "prediction_uncertainty_max_abs": UNCERTAINTY_MAX_ABS,
        },
        "measurements": measurements,
        "checks": checks,
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
    report = _read_json(path, f"ADR 0161 smoke report {path}")
    identity = report.get("scientific_identity")
    if (
        report.get("schema_version") != 1
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("adr") != ADR_ID
        or report.get("mode") != "bounded-smoke"
        or report.get("arm") != SMOKE_ARM
        or report.get("host") not in EXPECTED_HOSTS
        or report.get("optimization", {}).get("global_step")
        != SMOKE_STEPS
        or report.get("claims", {}).get("bounded_smoke_complete")
        is not True
        or report.get("claims", {}).get("offline_comparison_complete")
        is not False
        or not isinstance(identity, dict)
        or identity != _report_scientific_identity(report)
        or _canonical_blake3(identity) != report.get("report_id")
    ):
        raise SmokeParityError(
            f"ADR 0161 smoke report is malformed: {path}"
        )
    trace = report["optimization"].get("loss_trace")
    if (
        not isinstance(trace, list)
        or len(trace) != SMOKE_STEPS
        or [event.get("step") for event in trace]
        != list(range(1, SMOKE_STEPS + 1))
    ):
        raise SmokeParityError(
            f"ADR 0161 smoke loss trace is malformed: {path}"
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
            f"ADR 0161 smoke checkpoint checksum differs: {path}"
        )
    tensors = mx.load(path)
    if not isinstance(tensors, dict) or not tensors:
        raise SmokeParityError(
            f"ADR 0161 smoke checkpoint is empty: {path}"
        )
    return tensors


def _parameter_drift(
    left: dict[str, mx.array],
    right: dict[str, mx.array],
) -> dict[str, int | float]:
    if left.keys() != right.keys():
        raise SmokeParityError(
            "ADR 0161 smoke checkpoint tensor names differ"
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
                "ADR 0161 smoke checkpoint tensor layout differs: "
                f"{name}"
            )
        left_float = left_value.astype(np.float64)
        right_float = right_value.astype(np.float64)
        if (
            not np.all(np.isfinite(left_float))
            or not np.all(np.isfinite(right_float))
        ):
            raise SmokeParityError(
                f"ADR 0161 smoke checkpoint is nonfinite: {name}"
            )
        difference = np.abs(left_float - right_float)
        total += difference.size
        changed += int(np.count_nonzero(difference))
        absolute_sum += float(difference.sum())
        maximum = max(
            maximum,
            float(difference.max(initial=0.0)),
        )
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
            "r3_cache_id",
            "relational_cache_id",
            "s1_cache_id",
            "r6_binary",
            "protocol",
            "model",
            "optimization",
            "checkpoint",
            "metrics",
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
        raise SmokeParityError(
            f"cannot read {label}: {error}"
        ) from error
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
        description="Compare john1 and john4 ADR 0161 smoke runs"
    )
    parser.add_argument("--left-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--left-checkpoint", type=Path, required=True)
    parser.add_argument("--right-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        result = compare_smoke(
            args.left_report,
            args.right_report,
            args.left_checkpoint,
            args.right_checkpoint,
        )
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


if __name__ == "__main__":
    main()
