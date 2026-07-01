#!/usr/bin/env python3
"""Classify a complete F1 feature-schema activation census fail-closed."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import blake3

EXPERIMENT_ID = "feature-schema-activation-census-v1"
COMPLETE = "feature_schema_activation_census_complete"
INCOMPLETE = "feature_schema_activation_census_incomplete"
PHASES = ("opening", "early", "middle", "late")
SEATS = ("0", "1", "2", "3")
EXPECTED_IMPLEMENTED_BLOCKS = 109
EXPECTED_MODERN_BLOCKS = 78
EXPECTED_LEGACY_BLOCKS = 31
EXPECTED_EVIDENCE = 628
EXPECTED_CANDIDATE_ROWS = 2_995_314
EXPECTED_LEGACY_ROWS = 200_000
EXPECTED_MANIFEST_SCIENTIFIC_BLAKE3 = (
    "1ebc86d586453548cb6109780f3c86d05867936f61eb1e34690b7bfd086fc9de"
)


class ClassificationError(RuntimeError):
    """Raised when a required input is malformed or scientifically invalid."""


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def scientific_blake3(value: object) -> str:
    return blake3.blake3(canonical_json(value)).hexdigest()


def checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ClassificationError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ClassificationError(f"JSON root must be an object: {path}")
    return value


def validate_report(value: dict[str, Any], label: str) -> None:
    scientific = value.get("scientific")
    if not isinstance(scientific, dict):
        raise ClassificationError(f"{label} report lacks a scientific object")
    if scientific.get("experiment_id") != EXPERIMENT_ID:
        raise ClassificationError(f"{label} report experiment identity drifted")
    if scientific_blake3(scientific) != value.get("scientific_blake3"):
        raise ClassificationError(f"{label} report scientific BLAKE3 mismatch")


def phase_seat_complete(block: dict[str, Any]) -> bool:
    phase_seat = block.get("census", {}).get("phase_seat", {})
    return all(
        int(phase_seat.get(phase, {}).get(seat, {}).get("rows", 0)) > 0
        for phase in PHASES
        for seat in SEATS
    )


def _gate(passed: bool, observed: str) -> dict[str, Any]:
    return {"passed": bool(passed), "observed": observed}


def classify(
    forward: dict[str, Any],
    reverse: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    validate_report(forward, "forward")
    validate_report(reverse, "reverse")
    manifest_payload = {
        key: value for key, value in manifest.items() if key != "scientific_blake3"
    }
    manifest_hash_valid = (
        scientific_blake3(manifest_payload) == manifest.get("scientific_blake3")
    )
    scientific = forward["scientific"]
    reverse_scientific = reverse["scientific"]
    blocks = scientific.get("blocks", [])
    evidence = scientific.get("evidence", [])
    if not isinstance(blocks, list) or not isinstance(evidence, list):
        raise ClassificationError("scientific blocks and evidence must be lists")

    implemented = [
        block for block in blocks if block.get("implementation_status") == "implemented"
    ]
    measured = [
        block
        for block in implemented
        if int(block.get("census", {}).get("rows", 0)) > 0
    ]
    legacy = [
        block for block in implemented if str(block.get("schema", "")).startswith("legacy-")
    ]
    modern = [block for block in implemented if block not in legacy]
    future_measured = [
        block
        for block in blocks
        if block.get("implementation_status") != "implemented"
        and int(block.get("census", {}).get("rows", 0)) > 0
    ]
    block_by_id = {str(block.get("block_id")): block for block in blocks}

    evidence_ids = [str(item.get("evidence_id")) for item in evidence]
    kind_rows = Counter()
    splits_by_kind: dict[str, set[str]] = {}
    for item in evidence:
        kind = str(item.get("kind"))
        kind_rows[kind] += int(item.get("rows_scanned", 0))
        splits_by_kind.setdefault(kind, set()).add(str(item.get("split")))

    dead_blocks = sorted(
        str(block["block_id"])
        for block in measured
        if int(block["census"].get("dead_channel_count", 0))
        == int(block.get("width", 0))
    )
    constant_blocks = sorted(
        str(block["block_id"])
        for block in measured
        if int(block["census"].get("constant_channel_count", 0))
        == int(block.get("width", 0))
    )
    rare_blocks = sorted(
        str(block["block_id"])
        for block in measured
        if int(block["census"].get("rare_channel_count", 0)) > 0
    )
    collision_statuses = sorted(
        {
            str(block.get("census", {}).get("collision_status"))
            for block in measured
        }
    )
    closed_domains = scientific.get("closed_domains", {})
    merged_shards = scientific.get("config", {}).get("merged_shards")
    expected_shards = [[0, 4], [1, 4], [2, 4], [3, 4]]

    gates = {
        "manifest_identity": _gate(
            manifest_hash_valid
            and manifest.get("scientific_blake3")
            == EXPECTED_MANIFEST_SCIENTIFIC_BLAKE3
            and scientific.get("manifest_scientific_blake3")
            == EXPECTED_MANIFEST_SCIENTIFIC_BLAKE3,
            str(manifest.get("scientific_blake3")),
        ),
        "merge_order_determinism": _gate(
            forward.get("scientific_blake3") == reverse.get("scientific_blake3")
            and scientific == reverse_scientific,
            (
                f"forward={forward.get('scientific_blake3')} "
                f"reverse={reverse.get('scientific_blake3')}"
            ),
        ),
        "exact_shard_set": _gate(
            merged_shards == expected_shards,
            json.dumps(merged_shards, sort_keys=True),
        ),
        "unique_evidence": _gate(
            len(evidence_ids) == EXPECTED_EVIDENCE
            and len(evidence_ids) == len(set(evidence_ids)),
            f"{len(evidence_ids)} evidence IDs; {len(set(evidence_ids))} unique",
        ),
        "candidate_rows": _gate(
            kind_rows["graded_dataset_shard"] == EXPECTED_CANDIDATE_ROWS
            and kind_rows["candidate_factor_cache_batch"] == EXPECTED_CANDIDATE_ROWS,
            (
                f"graded={kind_rows['graded_dataset_shard']} "
                f"factor={kind_rows['candidate_factor_cache_batch']}"
            ),
        ),
        "legacy_rows": _gate(
            kind_rows["legacy_sparse_feature_shard"] == EXPECTED_LEGACY_ROWS,
            str(kind_rows["legacy_sparse_feature_shard"]),
        ),
        "train_validation_present": _gate(
            splits_by_kind.get("graded_dataset_shard") == {"train", "validation"}
            and splits_by_kind.get("candidate_factor_cache_batch")
            == {"train", "validation"},
            json.dumps(
                {
                    key: sorted(value)
                    for key, value in sorted(splits_by_kind.items())
                },
                sort_keys=True,
            ),
        ),
        "all_implemented_blocks_measured": _gate(
            len(implemented) == EXPECTED_IMPLEMENTED_BLOCKS
            and len(measured) == EXPECTED_IMPLEMENTED_BLOCKS
            and len(modern) == EXPECTED_MODERN_BLOCKS
            and len(legacy) == EXPECTED_LEGACY_BLOCKS,
            (
                f"implemented={len(implemented)} measured={len(measured)} "
                f"modern={len(modern)} legacy={len(legacy)}"
            ),
        ),
        "future_schemas_unmeasured": _gate(
            not future_measured,
            f"{len(future_measured)} proposed-only blocks measured",
        ),
        "phase_seat_coverage": _gate(
            phase_seat_complete(block_by_id.get("v2.board.coordinates", {}))
            and phase_seat_complete(block_by_id.get("legacy.cell_core", {})),
            "modern and legacy focal blocks cover seats 0-3 in all four phases",
        ),
        "historical_mid_tail_measured": _gate(
            int(
                block_by_id.get(
                    "legacy.mid_tail_historical_adjacency_prefix", {}
                )
                .get("census", {})
                .get("rows", 0)
            )
            == EXPECTED_LEGACY_ROWS,
            str(
                block_by_id.get(
                    "legacy.mid_tail_historical_adjacency_prefix", {}
                )
                .get("census", {})
                .get("rows", 0)
            ),
        ),
        "status_categories_distinct": _gate(
            {"unknown", "no_channel_alias_detected"}.issubset(
                set(collision_statuses)
            )
            or "structural_or_empirical_alias" in collision_statuses,
            json.dumps(collision_statuses),
        ),
        "closed_domains": _gate(
            isinstance(closed_domains, dict)
            and bool(closed_domains)
            and all(value is False for value in closed_domains.values()),
            json.dumps(closed_domains, sort_keys=True),
        ),
    }
    complete = all(gate["passed"] for gate in gates.values())
    result_scientific = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "classification": COMPLETE if complete else INCOMPLETE,
        "complete": complete,
        "input_scientific_blake3": forward["scientific_blake3"],
        "manifest_scientific_blake3": manifest.get("scientific_blake3"),
        "gates": gates,
        "metrics": {
            "evidence_payloads": len(evidence),
            "graded_candidate_rows": kind_rows["graded_dataset_shard"],
            "factor_candidate_rows": kind_rows["candidate_factor_cache_batch"],
            "hierarchical_rows": kind_rows["hierarchical_factor_cache_shard"],
            "legacy_rows": kind_rows["legacy_sparse_feature_shard"],
            "implemented_blocks": len(implemented),
            "measured_blocks": len(measured),
            "dead_blocks": dead_blocks,
            "constant_blocks": constant_blocks,
            "rare_blocks": rare_blocks,
            "collision_statuses": collision_statuses,
        },
    }
    return {
        "schema_version": 1,
        "scientific": result_scientific,
        "scientific_blake3": scientific_blake3(result_scientific),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--reverse", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = classify(
            load_json(args.forward),
            load_json(args.reverse),
            load_json(args.manifest),
        )
    except ClassificationError as error:
        parser.error(str(error))
    report["provenance"] = {
        "forward": {
            "path": str(args.forward),
            "blake3": checksum(args.forward),
        },
        "reverse": {
            "path": str(args.reverse),
            "blake3": checksum(args.reverse),
        },
        "manifest": {
            "path": str(args.manifest),
            "blake3": checksum(args.manifest),
        },
        "classifier_blake3": checksum(Path(__file__)),
    }
    report["created_unix_seconds"] = int(time.time())
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["scientific"]["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
