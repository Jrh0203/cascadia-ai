#!/usr/bin/env python3
"""Mechanically authorize or cancel the contingent ADR 0124 branch."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import blake3

EXPERIMENT_ID = "conditional-tile-local-geometry-dropout-v1"
SOURCE_EXPERIMENT_ID = "conditional-tile-optimizer-schedule-v1"
PREFLIGHT_EXPERIMENT_ID = "conditional-tile-local-geometry-dropout-preflight-repair-v1"
PREFLIGHT_CLASSIFICATION = "local_geometry_dropout_preflight_passed"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _blake3(value: object) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def authorize_branch(
    *,
    source_combined: dict[str, Any],
    preflight_combined: dict[str, Any],
    manifest: dict[str, Any],
    now_ms: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the mechanically updated manifest and authorization report."""
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("ADR 0124 manifest identity drifted")
    if manifest.get("status") != "contingently_authorized":
        raise ValueError("ADR 0124 manifest is not awaiting branch authorization")
    source_scientific = source_combined.get("scientific", {})
    preflight_scientific = preflight_combined.get("scientific", {})
    if source_combined.get("experiment_id") != SOURCE_EXPERIMENT_ID:
        raise ValueError("ADR 0120 combined report identity drifted")
    if (
        preflight_combined.get("experiment_id") != PREFLIGHT_EXPERIMENT_ID
        or preflight_scientific.get("classification") != PREFLIGHT_CLASSIFICATION
    ):
        raise ValueError("ADR 0124 repaired preflight is not valid")
    source_gates = source_scientific.get("gates", {})
    if not bool(source_gates.get("pipeline_passed")):
        raise ValueError("ADR 0120 pipeline is invalid")
    classification = source_scientific.get("classification")
    if classification == "optimizer_schedule_tile_insufficient":
        status = "authorized"
        decision = "launch_local_geometry_dropout"
    elif classification == "optimizer_schedule_tile_sufficient":
        status = "cancelled"
        decision = "cancel_optimizer_schedule_sufficient"
    else:
        raise ValueError("ADR 0120 classification cannot resolve ADR 0124")
    source_hash = str(source_combined["scientific_blake3"])
    preflight_hash = str(preflight_combined["scientific_blake3"])
    updated = json.loads(json.dumps(manifest))
    updated["status"] = status
    updated["branch_authorization"].update(
        {
            "required_preflight_experiment": PREFLIGHT_EXPERIMENT_ID,
            "source_combined_scientific_blake3": source_hash,
            "preflight_combined_scientific_blake3": preflight_hash,
            "authorized_unix_ms": now_ms if status == "authorized" else None,
            "cancelled_unix_ms": now_ms if status == "cancelled" else None,
            "decision": decision,
        }
    )
    scientific = {
        "decision": decision,
        "manifest_status": status,
        "source_classification": classification,
        "source_pipeline_passed": True,
        "source_combined_scientific_blake3": source_hash,
        "preflight_classification": PREFLIGHT_CLASSIFICATION,
        "preflight_combined_scientific_blake3": preflight_hash,
        "training_authorized": status == "authorized",
        "training_cancelled": status == "cancelled",
        "sealed_test_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "preflight_experiment_id": PREFLIGHT_EXPERIMENT_ID,
        "scientific": scientific,
        "scientific_blake3": _blake3(scientific),
    }
    return updated, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    updated, report = authorize_branch(
        source_combined=_load(args.source),
        preflight_combined=_load(args.preflight),
        manifest=_load(args.manifest),
        now_ms=time.time_ns() // 1_000_000,
    )
    _write_json(args.manifest, updated)
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "decision": report["scientific"]["decision"],
                "scientific_blake3": report["scientific_blake3"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
