#!/usr/bin/env python3
"""Classify the crossed-host O1 policy-held-out corpus audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import blake3

SCHEMA_VERSION = 1
EXPERIMENT_ID = "o1-opponent-intent-policy-heldout-corpus-v1"
EXPECTED_AUDIT_CLASSIFICATION = "policy_held_out_corpus_passed"
FINAL_CLASSIFICATION = "policy_held_out_draft_survival_corpus_passed"
EXPECTED_ROLES = {
    "train-part-0",
    "train-part-1",
    "validation",
    "test",
    "final-stress",
}


class ClassificationError(RuntimeError):
    """Raised when a corpus audit is incomplete, inconsistent, or non-exact."""


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def load_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ClassificationError(f"cannot read report {path}: {error}") from error
    if not isinstance(value, dict):
        raise ClassificationError(f"report root is not an object: {path}")
    return value


def _valid_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_report(report: dict[str, Any], *, role: str) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ClassificationError(f"{role} has an unsupported schema version")
    if report.get("experiment_id") != EXPERIMENT_ID:
        raise ClassificationError(f"{role} belongs to another experiment")
    if report.get("status") != "complete":
        raise ClassificationError(f"{role} did not complete")
    if report.get("classification") != EXPECTED_AUDIT_CLASSIFICATION:
        raise ClassificationError(f"{role} did not pass the corpus audit")
    if not _valid_digest(report.get("scientific_blake3")):
        raise ClassificationError(f"{role} has an invalid scientific digest")

    scientific = report.get("scientific")
    if not isinstance(scientific, dict):
        raise ClassificationError(f"{role} lacks scientific results")
    totals = scientific.get("totals")
    if not isinstance(totals, dict) or totals != {
        "games": 1664,
        "records": 126464,
        "shards": 104,
        "unique_model_inputs": 126464,
        "duplicate_model_inputs_within_datasets": 0,
        "identity_exclusion_checks": 126464,
    }:
        raise ClassificationError(f"{role} has inconsistent corpus totals")

    datasets = scientific.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 5:
        raise ClassificationError(f"{role} must contain five dataset results")
    dataset_roles = {dataset.get("role") for dataset in datasets if isinstance(dataset, dict)}
    if dataset_roles != EXPECTED_ROLES:
        raise ClassificationError(f"{role} has incorrect dataset roles")
    if any(
        not isinstance(dataset, dict)
        or dataset.get("unique_model_inputs") != dataset.get("records")
        or dataset.get("duplicate_model_inputs") != 0
        or dataset.get("identity_exclusion_checks") != dataset.get("records")
        or dataset.get("model_input_bytes") != 1189
        for dataset in datasets
    ):
        raise ClassificationError(f"{role} failed dataset-level input checks")

    overlaps = scientific.get("overlaps")
    if (
        not isinstance(overlaps, list)
        or len(overlaps) != 10
        or any(
            not isinstance(overlap, dict)
            or overlap.get("exact_hash_overlap") != 0
            or overlap.get("sample_hashes") != []
            for overlap in overlaps
        )
    ):
        raise ClassificationError(f"{role} has cross-corpus model-input overlap")

    factor_coverage = scientific.get("action_factor_coverage")
    if (
        not isinstance(factor_coverage, list)
        or len(factor_coverage) != 9
        or any(
            not isinstance(factor, dict)
            or factor.get("passed") is not True
            or factor.get("missing_from_training") != []
            for factor in factor_coverage
        )
    ):
        raise ClassificationError(f"{role} lacks held-out action-factor coverage")
    survival = scientific.get("survival_coverage")
    if not isinstance(survival, dict) or survival.get("passed") is not True:
        raise ClassificationError(f"{role} lacks survival-class coverage")
    gates = scientific.get("gates")
    if (
        not isinstance(gates, list)
        or len(gates) != 7
        or any(not isinstance(gate, dict) or gate.get("passed") is not True for gate in gates)
    ):
        raise ClassificationError(f"{role} failed a preregistered corpus gate")

    limitations = scientific.get("limitations")
    if not isinstance(limitations, list) or len(limitations) != 3:
        raise ClassificationError(f"{role} lacks the frozen scope limitations")
    limitations_by_label = {
        limitation.get("label"): limitation
        for limitation in limitations
        if isinstance(limitation, dict)
    }
    paid_wipe = limitations_by_label.get("Paid wildlife-wipe intent is unsupported")
    if (
        not isinstance(paid_wipe, dict)
        or paid_wipe.get("observed") != "0 of 379392 target actions contain a paid wipe"
    ):
        raise ClassificationError(f"{role} obscures the paid-wipe support gap")

    provenance = report.get("provenance")
    if (
        not isinstance(provenance, dict)
        or not isinstance(provenance.get("hostname"), str)
        or not _valid_digest(provenance.get("executable_blake3"))
        or not isinstance(provenance.get("dataset_roots"), dict)
        or set(provenance["dataset_roots"]) != EXPECTED_ROLES
    ):
        raise ClassificationError(f"{role} lacks complete execution provenance")


def classify(
    primary: dict[str, Any],
    replay: dict[str, Any],
    *,
    primary_path: Path,
    replay_path: Path,
) -> dict[str, Any]:
    validate_report(primary, role="primary")
    validate_report(replay, role="replay")
    if primary["scientific"] != replay["scientific"]:
        raise ClassificationError("primary and replay scientific results differ")
    if primary["scientific_blake3"] != replay["scientific_blake3"]:
        raise ClassificationError("primary and replay scientific digests differ")
    primary_provenance = primary["provenance"]
    replay_provenance = replay["provenance"]
    if primary_provenance["hostname"] == replay_provenance["hostname"]:
        raise ClassificationError("primary and replay must run on distinct hosts")
    if primary_provenance["executable_blake3"] != replay_provenance["executable_blake3"]:
        raise ClassificationError("primary and replay executable digests differ")

    scientific = primary["scientific"]
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "classification": FINAL_CLASSIFICATION,
        "matched_scientific_blake3": primary["scientific_blake3"],
        "primary": {
            "path": str(primary_path),
            "hostname": primary_provenance["hostname"],
            "executable_blake3": primary_provenance["executable_blake3"],
        },
        "replay": {
            "path": str(replay_path),
            "hostname": replay_provenance["hostname"],
            "executable_blake3": replay_provenance["executable_blake3"],
        },
        "totals": scientific["totals"],
        "gates": scientific["gates"],
        "action_factor_coverage": scientific["action_factor_coverage"],
        "survival_coverage": scientific["survival_coverage"],
        "scope_limitations": scientific["limitations"],
        "authorization": {
            "public_state_control_training": True,
            "recent_history_intent_training": True,
            "next_draft_auxiliary_training": True,
            "market_survival_training": True,
            "policy_held_out_calibration": True,
            "paid_wipe_intent_training": False,
            "strategy_switch_training": False,
            "champion_generalization_claim": False,
            "gameplay_promotion": False,
        },
        "required_successor": (
            "Run the matched MLX public-state/history/intent/survival learnability "
            "factorial. Separately collect nature-token-active and champion-like "
            "policy cohorts before any paid-wipe or champion-generalization claim."
        ),
    }
    result["classification_blake3"] = blake3.blake3(canonical_json(result)).hexdigest()
    return result


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canonical-output", type=Path)
    args = parser.parse_args()
    result = classify(
        load_report(args.primary),
        load_report(args.replay),
        primary_path=args.primary,
        replay_path=args.replay,
    )
    write_json(args.output, result)
    if args.canonical_output is not None and args.canonical_output != args.output:
        write_json(args.canonical_output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
