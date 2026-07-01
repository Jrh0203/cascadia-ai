#!/usr/bin/env python3
"""Merge four ADR 0150 failure atlases into one mechanism report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import blake3
import numpy as np
from cascadia_mlx.r3_action_edit_mlx_cache import (
    ADR_ID,
    ARMS,
    CONTROL_ARM,
    EXPERIMENT_ID,
    PROTOCOL_ID,
)
from cascadia_mlx.r3_action_edit_mlx_forensics import (
    ATLAS_KIND,
    ATLAS_SCHEMA_VERSION,
)

REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "r3-action-edit-four-arm-failure-mechanism-v1"
TREATMENTS = ARMS[1:]


def _canonical_blake3(value: Any) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text().splitlines()
        values = [json.loads(line) for line in lines if line]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"atlas details are unreadable: {path}") from error
    if not values or any(not isinstance(value, dict) for value in values):
        raise ValueError(f"atlas details are empty or malformed: {path}")
    return values


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _validate_atlas(
    report_path: Path,
    details_path: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    report = _read_json(report_path, "failure-atlas report")
    identity = report.get("scientific_identity")
    report_without_id = dict(report)
    report_id = report_without_id.pop("report_id", None)
    if (
        report.get("schema_version") != ATLAS_SCHEMA_VERSION
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("adr") != ADR_ID
        or report.get("atlas_kind") != ATLAS_KIND
        or report.get("arm") not in ARMS
        or report.get("classifier_eligible") is not False
        or not isinstance(identity, dict)
        or identity.get("arm") != report.get("arm")
        or identity.get("details_blake3") != _checksum(details_path)
        or _canonical_blake3(report_without_id) != report_id
    ):
        raise ValueError(f"failure-atlas identity is invalid: {report_path}")
    records = _read_jsonl(details_path)
    by_group: dict[int, dict[str, Any]] = {}
    for record in records:
        raw_group_id = record.get("group_id")
        if isinstance(raw_group_id, bool) or not isinstance(raw_group_id, int):
            raise ValueError(
                f"failure-atlas group identity is invalid: {details_path}"
            )
        group_id = raw_group_id
        if (
            record.get("schema_version") != ATLAS_SCHEMA_VERSION
            or group_id in by_group
        ):
            raise ValueError(f"failure-atlas group identity is invalid: {details_path}")
        by_group[group_id] = record
    if (
        len(by_group) != report.get("validation_groups")
        or sum(record["candidate_count"] for record in records)
        != report.get("validation_candidates")
    ):
        raise ValueError(f"failure-atlas coverage differs: {details_path}")
    return report, by_group


def _rate(records: list[dict[str, Any]], key: str) -> float:
    return sum(bool(record[key]) for record in records) / max(len(records), 1)


def _mean(records: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(record[key]) for record in records]))


def _pairwise_rank(left: list[int], right: list[int]) -> dict[str, int | float]:
    if len(left) != len(right) or not left:
        raise ValueError("pairwise rank vectors are inconsistent")
    differences = np.asarray(right, dtype=np.int64) - np.asarray(left, dtype=np.int64)
    return {
        "left_better": int(np.sum(differences > 0)),
        "equal": int(np.sum(differences == 0)),
        "right_better": int(np.sum(differences < 0)),
        "mean_right_minus_left": float(np.mean(differences)),
    }


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def compare_atlas_records(
    records_by_arm: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    """Compare group-level outcomes under the fixed four-arm identity."""
    if set(records_by_arm) != set(ARMS):
        raise ValueError("mechanism report requires all four ADR 0150 arms")
    group_ids = sorted(records_by_arm[CONTROL_ARM])
    if not group_ids or any(sorted(records_by_arm[arm]) != group_ids for arm in ARMS):
        raise ValueError("failure atlases do not cover identical groups")
    stable_fields = (
        "game_index",
        "turn",
        "personal_turn",
        "phase",
        "low_supply",
        "independent_draft_winner",
        "candidate_count",
        "winner_index",
        "winner_r4800",
    )
    for group_id in group_ids:
        control = records_by_arm[CONTROL_ARM][group_id]
        for arm in ARMS[1:]:
            treatment = records_by_arm[arm][group_id]
            if any(treatment[field] != control[field] for field in stable_fields):
                raise ValueError(f"atlas public group facts differ for group {group_id}")

    pattern_counts = {
        "confidence_coverage_t1_t2_t3": {},
        "winner_recall_t1_t2_t3": {},
    }
    group_mechanisms: list[dict[str, Any]] = []
    for group_id in group_ids:
        rows = {arm: records_by_arm[arm][group_id] for arm in ARMS}
        confidence_pattern = "".join(
            "1" if rows[arm]["confidence_set_covered_top64"] else "0"
            for arm in TREATMENTS
        )
        recall_pattern = "".join(
            "1" if rows[arm]["winner_recalled_top64"] else "0" for arm in TREATMENTS
        )
        for key, pattern in (
            ("confidence_coverage_t1_t2_t3", confidence_pattern),
            ("winner_recall_t1_t2_t3", recall_pattern),
        ):
            pattern_counts[key][pattern] = pattern_counts[key].get(pattern, 0) + 1
        group_mechanisms.append(
            {
                "group_id": group_id,
                "control_confidence_covered": rows[CONTROL_ARM][
                    "confidence_set_covered_top64"
                ],
                "control_winner_recalled": rows[CONTROL_ARM]["winner_recalled_top64"],
                "treatment_confidence_pattern": confidence_pattern,
                "treatment_recall_pattern": recall_pattern,
                "winner_ranks": {arm: rows[arm]["winner_rank"] for arm in ARMS},
                "confidence_ranks": {
                    arm: rows[arm]["best_confidence_set_rank"] for arm in ARMS
                },
                "candidate_count": rows[CONTROL_ARM]["candidate_count"],
                "phase": rows[CONTROL_ARM]["phase"],
                "low_supply": rows[CONTROL_ARM]["low_supply"],
                "independent_draft_winner": rows[CONTROL_ARM][
                    "independent_draft_winner"
                ],
            }
        )

    arm_summary: dict[str, Any] = {}
    for arm in ARMS:
        records = [records_by_arm[arm][group_id] for group_id in group_ids]
        global_tokens = [float(record["winner_global_token_count"]) for record in records]
        winner_ranks = [float(record["winner_rank"]) for record in records]
        pass_global = [
            float(record["winner_global_token_count"])
            for record in records
            if record["confidence_set_covered_top64"]
        ]
        fail_global = [
            float(record["winner_global_token_count"])
            for record in records
            if not record["confidence_set_covered_top64"]
        ]
        arm_summary[arm] = {
            "top64_winner_recall": _rate(records, "winner_recalled_top64"),
            "top64_confidence_set_coverage_95": _rate(
                records,
                "confidence_set_covered_top64",
            ),
            "mean_top64_retained_r4800_regret": _mean(
                records,
                "top64_retained_r4800_regret",
            ),
            "mean_winner_rank": _mean(records, "winner_rank"),
            "mean_winner_token_count": _mean(records, "winner_token_count"),
            "mean_winner_local_token_count": _mean(
                records,
                "winner_local_token_count",
            ),
            "mean_winner_global_token_count": _mean(
                records,
                "winner_global_token_count",
            ),
            "global_token_count_to_winner_rank_correlation": _correlation(
                global_tokens,
                winner_ranks,
            ),
            "mean_global_tokens_when_confidence_passes": (
                float(np.mean(pass_global)) if pass_global else None
            ),
            "mean_global_tokens_when_confidence_fails": (
                float(np.mean(fail_global)) if fail_global else None
            ),
        }

    ranks = {
        arm: [records_by_arm[arm][group_id]["winner_rank"] for group_id in group_ids]
        for arm in TREATMENTS
    }
    mechanism_counts = {
        "control_pass_all_treatments_fail_confidence": sum(
            row["control_confidence_covered"]
            and row["treatment_confidence_pattern"] == "000"
            for row in group_mechanisms
        ),
        "control_fail_all_treatments_pass_confidence": sum(
            not row["control_confidence_covered"]
            and row["treatment_confidence_pattern"] == "111"
            for row in group_mechanisms
        ),
        "all_four_fail_confidence": sum(
            not row["control_confidence_covered"]
            and row["treatment_confidence_pattern"] == "000"
            for row in group_mechanisms
        ),
        "radius1_only_passes_confidence": pattern_counts[
            "confidence_coverage_t1_t2_t3"
        ].get("001", 0),
        "radius3_only_passes_confidence": pattern_counts[
            "confidence_coverage_t1_t2_t3"
        ].get("100", 0),
        "smaller_radius1_passes_while_radius3_fails": sum(
            row["treatment_confidence_pattern"][0] == "0"
            and row["treatment_confidence_pattern"][2] == "1"
            for row in group_mechanisms
        ),
        "larger_radius3_passes_while_radius1_fails": sum(
            row["treatment_confidence_pattern"][0] == "1"
            and row["treatment_confidence_pattern"][2] == "0"
            for row in group_mechanisms
        ),
    }
    return {
        "groups": len(group_ids),
        "arm_summary": arm_summary,
        "treatment_patterns": {
            key: dict(sorted(values.items())) for key, values in pattern_counts.items()
        },
        "pairwise_winner_rank": {
            "radius3_vs_radius2": _pairwise_rank(
                ranks[TREATMENTS[0]],
                ranks[TREATMENTS[1]],
            ),
            "radius2_vs_radius1": _pairwise_rank(
                ranks[TREATMENTS[1]],
                ranks[TREATMENTS[2]],
            ),
            "radius3_vs_radius1": _pairwise_rank(
                ranks[TREATMENTS[0]],
                ranks[TREATMENTS[2]],
            ),
        },
        "mechanism_counts": mechanism_counts,
        "highest_disagreement_groups": sorted(
            group_mechanisms,
            key=lambda row: (
                max(row["winner_ranks"].values()) - min(row["winner_ranks"].values()),
                row["candidate_count"],
            ),
            reverse=True,
        )[:30],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atlas",
        action="append",
        nargs=2,
        metavar=("REPORT", "DETAILS"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.atlas) != len(ARMS):
        raise ValueError("exactly four report/details atlas pairs are required")

    reports: dict[str, dict[str, Any]] = {}
    records_by_arm: dict[str, dict[int, dict[str, Any]]] = {}
    for report_name, details_name in args.atlas:
        report, records = _validate_atlas(Path(report_name), Path(details_name))
        arm = str(report["arm"])
        if arm in reports:
            raise ValueError(f"duplicate failure atlas for {arm}")
        reports[arm] = report
        records_by_arm[arm] = records
    comparison = compare_atlas_records(records_by_arm)
    scientific_identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "report_kind": REPORT_KIND,
        "atlas_report_ids": {arm: reports[arm]["report_id"] for arm in ARMS},
        "atlas_details_blake3": {
            arm: reports[arm]["details_blake3"] for arm in ARMS
        },
        "classifier_eligible": False,
        "scientific_use": "post-classification-diagnostic-only",
    }
    output = {
        "schema_version": REPORT_SCHEMA_VERSION,
        **scientific_identity,
        "scientific_identity": scientific_identity,
        "comparison": comparison,
    }
    output["report_id"] = _canonical_blake3(output)
    _write_json_atomic(args.output, output)
    print(
        json.dumps(
            {
                "report_id": output["report_id"],
                "mechanism_counts": comparison["mechanism_counts"],
                "pairwise_winner_rank": comparison["pairwise_winner_rank"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
