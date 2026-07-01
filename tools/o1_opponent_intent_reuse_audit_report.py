#!/usr/bin/env python3
"""Classify the crossed-host O1 imitation-corpus reuse audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import blake3

SCHEMA_VERSION = 2
EXPERIMENT_ID = "o1-opponent-intent-corpus-reuse-audit-v1"
EXPECTED_CLASSIFICATION = "exact_replay_foundation_reusable_policy_holdout_required"


class ClassificationError(RuntimeError):
    """Raised when an audit report is incomplete, inconsistent, or non-exact."""


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


def validate_report(report: dict[str, Any], *, role: str) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ClassificationError(f"{role} has an unsupported schema version")
    if report.get("experiment_id") != EXPERIMENT_ID:
        raise ClassificationError(f"{role} belongs to another experiment")
    if report.get("status") != "complete":
        raise ClassificationError(f"{role} did not complete")
    if report.get("classification") != EXPECTED_CLASSIFICATION:
        raise ClassificationError(f"{role} has an unexpected classification")
    scientific_blake3 = report.get("scientific_blake3")
    if (
        not isinstance(scientific_blake3, str)
        or len(scientific_blake3) != 64
        or any(character not in "0123456789abcdef" for character in scientific_blake3)
    ):
        raise ClassificationError(f"{role} has an invalid scientific digest")
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ClassificationError(f"{role} lacks execution provenance")
    dataset_roots = provenance.get("dataset_roots")
    if (
        not isinstance(dataset_roots, list)
        or len(dataset_roots) != 2
        or any(not isinstance(root, str) or not root for root in dataset_roots)
    ):
        raise ClassificationError(f"{role} lacks two host-local dataset roots")

    datasets = report.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 2:
        raise ClassificationError(f"{role} must contain exactly two datasets")
    if {dataset.get("split") for dataset in datasets if isinstance(dataset, dict)} != {
        "train",
        "validation",
    }:
        raise ClassificationError(f"{role} does not contain train and validation")
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ClassificationError(f"{role} contains a malformed dataset result")
        positions = dataset.get("positions")
        candidates = dataset.get("candidates")
        games = dataset.get("games")
        exact = dataset.get("exact_checks")
        identity = dataset.get("identity_recovery")
        if (
            not isinstance(positions, int)
            or not isinstance(candidates, int)
            or not isinstance(games, int)
            or positions != games * 80
            or not isinstance(exact, dict)
            or not isinstance(identity, dict)
        ):
            raise ClassificationError(f"{role} dataset totals are inconsistent")
        expected_exact = {
            "exact_turn_order": positions,
            "exact_active_seat": positions,
            "exact_position_bytes": positions,
            "exact_candidate_action_hashes": candidates,
            "exactly_one_selected_action": positions,
            "exact_state_transitions": positions,
            "terminal_games": games,
        }
        for field, expected in expected_exact.items():
            if exact.get(field) != expected:
                raise ClassificationError(
                    f"{role} dataset failed {field}: {exact.get(field)} != {expected}"
                )
        if identity.get("positions_with_four_unique_tile_ids") != positions:
            raise ClassificationError(f"{role} did not recover four unique tiles everywhere")
        survival = dataset.get("survival_windows")
        expected_windows = games * 76
        if (
            not isinstance(survival, dict)
            or survival.get("focal_post_action_windows") != expected_windows
            or survival.get("market_tile_labels") != expected_windows * 4
        ):
            raise ClassificationError(f"{role} survival-window coverage is incomplete")

    overlaps = report.get("cross_dataset_overlaps")
    if not isinstance(overlaps, list) or len(overlaps) != 1:
        raise ClassificationError(f"{role} must contain one cross-split overlap result")
    for overlap in overlaps:
        if not isinstance(overlap, dict):
            raise ClassificationError(f"{role} contains a malformed overlap result")
        for field in (
            "group_id_overlap",
            "position_record_overlap",
            "public_state_overlap",
            "initial_hidden_state_overlap",
        ):
            if overlap.get(field) != 0:
                raise ClassificationError(f"{role} has nonzero {field}")

    recoverability = report.get("recoverability")
    required_true = (
        "exact_sequential_replay",
        "exact_candidate_action_reconstruction",
        "exact_selected_action_labels",
        "exact_unique_tile_identity",
        "exact_post_action_tile_survival",
        "exact_next_pick_slots_and_species",
        "exact_nature_token_action",
        "public_recent_draft_history",
    )
    if not isinstance(recoverability, dict) or any(
        recoverability.get(field) is not True for field in required_true
    ):
        raise ClassificationError(f"{role} lacks a required recoverability result")
    if recoverability.get("wildlife_token_physical_identity") is not False:
        raise ClassificationError(f"{role} invented physical wildlife identity")

    boundary = report.get("claim_boundary")
    if (
        not isinstance(boundary, dict)
        or boundary.get("foundation_reuse_authorized") is not True
        or boundary.get("final_o1_training_corpus_authorized") is not False
        or boundary.get("policy_held_out_evaluation_available") is not False
        or boundary.get("checkpoint_identity_shortcut_testable") is not False
        or boundary.get("strategy_switch_target_available") is not False
    ):
        raise ClassificationError(f"{role} violates the preregistered claim boundary")


def classify(
    primary: dict[str, Any],
    replay: dict[str, Any],
    *,
    primary_path: Path,
    replay_path: Path,
) -> dict[str, Any]:
    validate_report(primary, role="primary")
    validate_report(replay, role="replay")
    for field in (
        "datasets",
        "cross_dataset_overlaps",
        "recoverability",
        "claim_boundary",
        "scientific_blake3",
    ):
        if primary[field] != replay[field]:
            raise ClassificationError(f"primary and replay differ in {field}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "classification": EXPECTED_CLASSIFICATION,
        "foundation_reuse_authorized": True,
        "final_o1_training_corpus_authorized": False,
        "policy_holdout_required": True,
        "matched_scientific_blake3": primary["scientific_blake3"],
        "primary": {
            "path": str(primary_path),
            "hostname": primary.get("provenance", {}).get("hostname"),
            "executable_blake3": primary.get("provenance", {}).get("executable_blake3"),
        },
        "replay": {
            "path": str(replay_path),
            "hostname": replay.get("provenance", {}).get("hostname"),
            "executable_blake3": replay.get("provenance", {}).get("executable_blake3"),
        },
        "datasets": primary["datasets"],
        "cross_dataset_overlaps": primary["cross_dataset_overlaps"],
        "claim_boundary": primary["claim_boundary"],
    }
    payload["classification_blake3"] = blake3.blake3(canonical_json(payload)).hexdigest()
    return payload


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
