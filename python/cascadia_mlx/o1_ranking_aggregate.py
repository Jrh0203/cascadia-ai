"""Replicated paired classification for ADR 0188."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import blake3
import numpy as np

from cascadia_mlx.o1_ranking_intent_cache import (
    ARMS,
    HISTORY_ARM,
    PUBLIC_STATE_ARM,
    SHUFFLE_ARM,
    ZERO_ARM,
)
from cascadia_mlx.o1_ranking_metrics import game_clustered_bootstrap
from cascadia_mlx.o1_ranking_protocol import (
    ADR_ID,
    EXPERIMENT_ID,
    PROTOCOL_ID,
    WAVE_HOSTS,
)

PRIMARY_IMPROVEMENT = 0.05
HIGH_REGRET_THRESHOLD = 0.50
HIGH_REGRET_IMPROVEMENT = 0.10
PAIRWISE_REGRESSION_TOLERANCE = 0.005


def aggregate_o1_ranking(
    *,
    experiment_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Verify both waves, apply every frozen gate, and seal classification."""
    reports = _load_reports(experiment_root)
    replication = {}
    primary = {}
    for arm in ARMS:
        first = reports["primary"][arm]
        second = reports["rotated"][arm]
        if (
            first["replication_id"] != second["replication_id"]
            or first["replication_identity"] != second["replication_identity"]
        ):
            raise ValueError(f"{arm} primary and rotated scientific artifacts differ")
        replication[arm] = {
            "replication_id": first["replication_id"],
            "primary_report_id": first["report_id"],
            "rotated_report_id": second["report_id"],
            "primary_host": first["host"],
            "rotated_host": second["host"],
            "exact_match": True,
        }
        primary[arm] = first

    aligned = _aligned_group_records(primary)
    control_records = aligned[ZERO_ARM]
    scorable = np.asarray(
        [record["r4800_scorable"] for record in control_records],
        dtype=np.bool_,
    )
    if int(scorable.sum()) != 235:
        raise ValueError("frozen validation R4800-scorable count drifted")
    control_regret = _regret_array(control_records, scorable)
    games = np.asarray(
        [record["game_index"] for record in control_records],
        dtype=np.uint64,
    )[scorable]
    high_regret = control_regret >= HIGH_REGRET_THRESHOLD
    comparisons = {}
    for arm in (PUBLIC_STATE_ARM, HISTORY_ARM, SHUFFLE_ARM):
        treatment_records = aligned[arm]
        treatment_regret = _regret_array(treatment_records, scorable)
        inference = game_clustered_bootstrap(
            treatment_regret,
            control_regret,
            games,
        )
        high_delta = float(
            np.mean(treatment_regret[high_regret] - control_regret[high_regret])
        )
        treatment_metrics = primary[arm]["metrics"]
        control_metrics = primary[ZERO_ARM]["metrics"]
        gates = {
            "primary_improvement_at_least_0_05": (
                inference["mean_difference"] <= -PRIMARY_IMPROVEMENT
            ),
            "paired_ci_wholly_below_zero": inference["ci95_upper"] < 0.0,
            "high_regret_improvement_at_least_0_10": (
                high_delta <= -HIGH_REGRET_IMPROVEMENT
            ),
            "top1_recall_nonregression": (
                treatment_metrics[
                    "top1_retained_r4800_winner_recall"
                ]
                >= control_metrics[
                    "top1_retained_r4800_winner_recall"
                ]
            ),
            "r1200_pairwise_regression_within_0_5pp": (
                treatment_metrics["r1200_pairwise_ordering_accuracy"]
                >= control_metrics["r1200_pairwise_ordering_accuracy"]
                - PAIRWISE_REGRESSION_TOLERANCE
            ),
            "all_actions_scored_exactly_once": (
                treatment_metrics["all_candidates_scored_once"] is True
                and treatment_metrics["all_groups_scored_once"] is True
            ),
            "primary_rotated_artifacts_match": True,
        }
        comparisons[arm] = {
            "control_mean_regret": float(np.mean(control_regret)),
            "treatment_mean_regret": float(np.mean(treatment_regret)),
            "improvement": float(np.mean(control_regret - treatment_regret)),
            "paired_treatment_minus_control": inference,
            "high_regret": {
                "threshold": HIGH_REGRET_THRESHOLD,
                "groups": int(high_regret.sum()),
                "control_mean_regret": float(
                    np.mean(control_regret[high_regret])
                ),
                "treatment_mean_regret": float(
                    np.mean(treatment_regret[high_regret])
                ),
                "mean_difference": high_delta,
                "improvement": -high_delta,
            },
            "top1_recall": {
                "control": control_metrics[
                    "top1_retained_r4800_winner_recall"
                ],
                "treatment": treatment_metrics[
                    "top1_retained_r4800_winner_recall"
                ],
                "difference": (
                    treatment_metrics[
                        "top1_retained_r4800_winner_recall"
                    ]
                    - control_metrics[
                        "top1_retained_r4800_winner_recall"
                    ]
                ),
            },
            "r1200_pairwise_accuracy": {
                "control": control_metrics[
                    "r1200_pairwise_ordering_accuracy"
                ],
                "treatment": treatment_metrics[
                    "r1200_pairwise_ordering_accuracy"
                ],
                "difference": (
                    treatment_metrics[
                        "r1200_pairwise_ordering_accuracy"
                    ]
                    - control_metrics[
                        "r1200_pairwise_ordering_accuracy"
                    ]
                ),
            },
            "gates": gates,
            "eligible": (
                arm in (PUBLIC_STATE_ARM, HISTORY_ARM)
                and all(gates.values())
            ),
        }

    eligible = [
        arm
        for arm in (PUBLIC_STATE_ARM, HISTORY_ARM)
        if comparisons[arm]["eligible"]
    ]
    selected = (
        min(
            eligible,
            key=lambda arm: (
                primary[arm]["metrics"][
                    "mean_top1_retained_r4800_regret"
                ],
                -primary[arm]["metrics"][
                    "top1_retained_r4800_winner_recall"
                ],
                -primary[arm]["metrics"][
                    "r1200_pairwise_ordering_accuracy"
                ],
                arm,
            ),
        )
        if eligible
        else None
    )
    p2_vs_b1 = _paired_comparison(
        aligned[HISTORY_ARM],
        aligned[PUBLIC_STATE_ARM],
        scorable,
        games,
    )
    p2_vs_shuffle = _paired_comparison(
        aligned[HISTORY_ARM],
        aligned[SHUFFLE_ARM],
        scorable,
        games,
    )
    history_aligned_intent_supported = (
        selected == HISTORY_ARM
        and p2_vs_b1["ci95_upper"] < 0.0
        and p2_vs_shuffle["ci95_upper"] < 0.0
    )
    validation_classification = (
        "o1_ranking_validation_arm_selected"
        if selected is not None
        else "o1_ranking_validation_factorial_null"
    )
    test_classification = (
        "o1_ranking_test_pending"
        if selected is not None
        else "o1_ranking_test_not_opened"
    )
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "replication": replication,
        "validation_groups": 240,
        "r4800_scorable_groups": int(scorable.sum()),
        "r1200_groups": 240,
        "comparisons": comparisons,
        "p2_vs_b1": p2_vs_b1,
        "p2_vs_shuffle": p2_vs_shuffle,
        "selected_arm": selected,
        "history_aligned_intent_supported": (
            history_aligned_intent_supported
        ),
        "validation_classification": validation_classification,
        "test_classification": test_classification,
        "sealed_test_opened": False,
        "gameplay_run": False,
        "claim_boundary": (
            "offline fixed-top64 reranking only; no gameplay or score claim"
        ),
    }
    aggregate = {
        "schema_version": 1,
        **identity,
        "aggregate_id": _canonical_blake3(identity),
        "scientific_identity": identity,
    }
    _write_json_atomic(output, aggregate)
    return aggregate


def _load_reports(
    experiment_root: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    collected: dict[str, dict[str, dict[str, Any]]] = {
        "primary": {},
        "rotated": {},
    }
    reports_root = experiment_root / "reports"
    for path in sorted(reports_root.glob("*.json")):
        report = _read_json(path, "O1 ranking report")
        wave = report.get("wave")
        arm = report.get("arm")
        if wave not in collected or arm not in ARMS:
            continue
        if arm in collected[wave]:
            raise ValueError(f"duplicate {wave} report for {arm}")
        _verify_report(experiment_root, report)
        collected[wave][arm] = report
    for wave in collected:
        if set(collected[wave]) != set(ARMS):
            raise ValueError(f"{wave} report set is incomplete")
    return collected


def _verify_report(
    experiment_root: Path,
    report: dict[str, Any],
) -> None:
    scientific_identity = report.get("scientific_identity")
    replication_identity = report.get("replication_identity")
    if (
        report.get("experiment_id") != EXPERIMENT_ID
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("adr") != ADR_ID
        or report.get("mode") != "production"
        or report.get("host")
        != WAVE_HOSTS[report["wave"]][report["arm"]]
        or not isinstance(scientific_identity, dict)
        or _canonical_blake3(scientific_identity)
        != report.get("report_id")
        or not isinstance(replication_identity, dict)
        or _canonical_blake3(replication_identity)
        != report.get("replication_id")
        or report.get("claims", {}).get("offline_validation_complete")
        is not True
        or report.get("claims", {}).get("base_parameters_frozen")
        is not True
        or report.get("information_boundary", {}).get("sealed_test_opened")
        is not False
    ):
        raise ValueError("O1 report envelope or claim boundary is invalid")
    run_root = (
        experiment_root
        / "runs"
        / report["wave"]
        / f"{report['arm']}-{report['host']}"
    )
    checkpoint_name = Path(report["checkpoint"]["path"]).name
    checkpoint = run_root / "checkpoints" / checkpoint_name
    if (
        _checksum(checkpoint / "checkpoint.json")
        != report["checkpoint"]["manifest_blake3"]
        or _checksum(checkpoint / "model.safetensors")
        != report["checkpoint"]["model_blake3"]
    ):
        raise ValueError("O1 collected checkpoint differs from report")
    for name, specification in report["prediction_files"].items():
        path = run_root / name
        if (
            path.stat().st_size != specification["bytes"]
            or _checksum(path) != specification["blake3"]
        ):
            raise ValueError("O1 collected prediction tensor differs")


def _aligned_group_records(
    reports: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    aligned = {
        arm: sorted(
            report["metrics"]["group_records"],
            key=lambda record: record["group_id"],
        )
        for arm, report in reports.items()
    }
    control_ids = [
        record["group_id"]
        for record in aligned[ZERO_ARM]
    ]
    for arm, records in aligned.items():
        if (
            len(records) != 240
            or [record["group_id"] for record in records] != control_ids
            or [
                record["r4800_scorable"]
                for record in records
            ]
            != [
                record["r4800_scorable"]
                for record in aligned[ZERO_ARM]
            ]
        ):
            raise ValueError(f"{arm} paired group records do not align")
    return aligned


def _regret_array(
    records: list[dict[str, Any]],
    scorable: np.ndarray,
) -> np.ndarray:
    values = [
        record["top1_retained_r4800_regret"]
        for record, include in zip(records, scorable, strict=True)
        if include
    ]
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("paired R4800 regrets are invalid")
    return result


def _paired_comparison(
    treatment_records: list[dict[str, Any]],
    control_records: list[dict[str, Any]],
    scorable: np.ndarray,
    games: np.ndarray,
) -> dict[str, float | int]:
    return game_clustered_bootstrap(
        _regret_array(treatment_records, scorable),
        _regret_array(control_records, scorable),
        games,
    )


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
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate replicated ADR 0188 validation reports"
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = aggregate_o1_ranking(
        experiment_root=args.experiment_root,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "aggregate_id": report["aggregate_id"],
                "validation_classification": report[
                    "validation_classification"
                ],
                "selected_arm": report["selected_arm"],
                "test_classification": report["test_classification"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
