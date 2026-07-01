"""Preregistration, authorization, and classification for O1 public-belief search."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import blake3
import numpy as np

EXPERIMENT_ID = "o1-public-belief-one-rotation-search-v1"
PROTOCOL_ID = "o1-public-belief-one-rotation-high-regret-v2"
SOURCE_EXPERIMENT_ID = "o1-high-regret-draft-ranking-integration-v1"
SOURCE_CONTROL_ARM = "z0-zero-intent"
HIGH_REGRET_THRESHOLD = 0.50
EXPECTED_PANEL_GROUPS = 99
COHORT_WIDTH = 64
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2_026_061_721
MIN_REGRET_IMPROVEMENT = 0.05
MAX_PAIRWISE_REGRESSION = 0.005

ARMS = (
    "c0-pattern-prior",
    "a0-public-state-intent",
    "a2-history-intent",
    "s3-shuffled-history-intent",
)
PRIMARY_ROLES = {
    "c0-pattern-prior": "c0-primary",
    "a0-public-state-intent": "a0-primary",
    "a2-history-intent": "a2-primary",
    "s3-shuffled-history-intent": "s3-primary",
}
REPLAY_ROLES = {
    "c0-pattern-prior": "c0-replay",
    "a0-public-state-intent": "a0-replay",
    "a2-history-intent": "a2-replay",
    "s3-shuffled-history-intent": "s3-replay",
}
ROLE_ARMS = {
    role: arm
    for arm in ARMS
    for role in (PRIMARY_ROLES[arm], REPLAY_ROLES[arm])
}


class PublicBeliefSearchError(ValueError):
    """The frozen public-belief search contract was violated."""


def frozen_protocol() -> dict[str, Any]:
    return {
        "root_candidates": 64,
        "stage_additional_samples": [4, 4, 8, 16],
        "stage_retain": [32, 16, 8, 1],
        "trajectories_per_group": 640,
        "opponent_turns": 3,
        "control_temperature": 1.0,
        "pattern_config": {
            "immediate_candidate_limit": 8,
            "habitat_candidate_limit": 6,
            "bear_candidate_limit": 8,
            "future_market_draws": 4,
        },
        "leaf_model": "qualified-legacy-v4opp-exact-mlx-v1",
        "leaf_value": "v2-current-base-score-plus-legacy-nnue-remaining-value",
        "root_chance_policy": (
            "condition-on-frozen-complete-turn-staged-prelude-context"
        ),
        "hidden_order_policy": (
            "sort-and-redeterminize-after-frozen-root-before-opponent-rotation"
        ),
    }


def freeze_high_regret_panel(
    *,
    control_report: Path,
    output: Path,
) -> dict[str, Any]:
    report = _read_json(control_report, "source O1 control report")
    if (
        report.get("experiment_id") != SOURCE_EXPERIMENT_ID
        or report.get("arm") != SOURCE_CONTROL_ARM
        or report.get("claims", {}).get("offline_validation_complete") is not True
        or report.get("information_boundary", {}).get("sealed_test_opened") is not False
    ):
        raise PublicBeliefSearchError("source O1 control report is not qualified")
    records = report.get("metrics", {}).get("group_records")
    if not isinstance(records, list):
        raise PublicBeliefSearchError("source O1 report has no validation group records")
    groups = []
    for record in records:
        regret = record.get("top1_retained_r4800_regret")
        if (
            record.get("r4800_scorable") is True
            and isinstance(regret, (int, float))
            and math.isfinite(float(regret))
            and float(regret) >= HIGH_REGRET_THRESHOLD
        ):
            groups.append(
                {
                    "row": _nonnegative_int(record.get("row"), "panel row"),
                    "group_id": _nonnegative_int(
                        record.get("group_id"),
                        "panel group ID",
                    ),
                    "game_index": _nonnegative_int(
                        record.get("game_index"),
                        "panel game index",
                    ),
                    "turn": _nonnegative_int(record.get("turn"), "panel turn"),
                    "control_regret": float(regret),
                }
            )
    groups.sort(key=lambda value: value["row"])
    if (
        len(groups) != EXPECTED_PANEL_GROUPS
        or len({value["row"] for value in groups}) != len(groups)
        or len({value["group_id"] for value in groups}) != len(groups)
    ):
        raise PublicBeliefSearchError(
            f"high-regret panel has {len(groups)} groups, expected {EXPECTED_PANEL_GROUPS}"
        )
    identity = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_report_blake3": _checksum(control_report),
        "threshold": HIGH_REGRET_THRESHOLD,
        "split": "validation",
        "groups": groups,
    }
    panel = {**identity, "panel_id": canonical_blake3(identity)}
    _write_json(output, panel)
    return panel


def build_authorization(
    *,
    bundle_id: str,
    dataset_root: Path,
    cohort_root: Path,
    intent_root: Path,
    panel_path: Path,
    model_dir: Path,
) -> dict[str, Any]:
    _require_digest(bundle_id, "bundle ID")
    dataset = _read_json(dataset_root / "dataset.json", "validation dataset manifest")
    cohort = _read_json(cohort_root / "cache.json", "O1 cohort manifest")
    intent = _read_json(intent_root / "cache.json", "O1 intent manifest")
    panel = _read_json(panel_path, "high-regret panel")
    model = _read_json(model_dir / "model.json", "legacy MLX model manifest")
    validation = cohort.get("splits", {}).get("validation")
    intent_validation = intent.get("splits", {}).get("validation")
    if (
        dataset.get("split") != "validation"
        or dataset.get("dataset_id") != validation.get("dataset_id")
        or validation.get("groups") != 240
        or intent_validation.get("groups") != 240
        or intent.get("cohort_id") != cohort.get("cache_id")
        or panel.get("experiment_id") != EXPERIMENT_ID
        or panel.get("panel_id") != canonical_blake3(
            {key: value for key, value in panel.items() if key != "panel_id"}
        )
        or len(panel.get("groups", [])) != EXPECTED_PANEL_GROUPS
        or model.get("architecture") != "legacy-sparse-nnue-v4opp-mlx-v1"
        or model.get("dimensions", {}).get("features") != 11_231
    ):
        raise PublicBeliefSearchError("authorization inputs do not match the frozen campaign")
    inputs = {
        "dataset_id": dataset["dataset_id"],
        "dataset_manifest_blake3": _checksum(dataset_root / "dataset.json"),
        "cohort_id": cohort["cache_id"],
        "cohort_manifest_blake3": _checksum(cohort_root / "cache.json"),
        "intent_id": intent["cache_id"],
        "intent_manifest_blake3": _checksum(intent_root / "cache.json"),
        "panel_id": panel["panel_id"],
        "panel_blake3": _checksum(panel_path),
        "model_manifest_blake3": _checksum(model_dir / "model.json"),
        "model_safetensors_blake3": _checksum(model_dir / "model.safetensors"),
    }
    identity = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "bundle_id": bundle_id,
        "roles": ROLE_ARMS,
        "protocol": frozen_protocol(),
        "inputs": inputs,
        "claim_boundary": {
            "open_high_regret_validation_only": True,
            "sealed_test_opened": False,
            "gameplay_run": False,
            "score_claim_authorized": False,
            "progress_to_100_claim_authorized": False,
            "legacy_441_role": "frozen-leaf-evaluator-only",
            "v2_search_state": "canonical-public-game-state",
        },
    }
    return {**identity, "authorization_id": canonical_blake3(identity)}


def authorize(**kwargs: Any) -> dict[str, Any]:
    output = Path(kwargs.pop("output"))
    result = build_authorization(**kwargs)
    _write_json(output, result)
    return result


def verify_authorization(
    *,
    authorization: Path,
    role: str,
    output: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    if role not in ROLE_ARMS:
        raise PublicBeliefSearchError(f"unknown public-belief role {role}")
    observed = _read_json(authorization, "public-belief authorization")
    expected = build_authorization(**kwargs)
    if observed != expected:
        raise PublicBeliefSearchError("authorization bytes do not rebuild exactly")
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "role": role,
        "arm": ROLE_ARMS[role],
        "authorization_id": expected["authorization_id"],
        "bundle_id": expected["bundle_id"],
        "passed": True,
    }
    _write_json(output, report)
    return report


def aggregate_reports(
    *,
    role_reports: dict[str, Path],
    output: Path,
) -> dict[str, Any]:
    if set(role_reports) != set(ROLE_ARMS):
        raise PublicBeliefSearchError("aggregate requires every primary and replay role")
    reports = {
        role: _read_json(path, f"{role} search report")
        for role, path in role_reports.items()
    }
    authorization_ids = {report.get("authorization_id") for report in reports.values()}
    bundle_ids = {report.get("bundle_id") for report in reports.values()}
    if len(authorization_ids) != 1 or len(bundle_ids) != 1:
        raise PublicBeliefSearchError("search reports do not share one authorization and bundle")
    replication = {}
    primary_metrics = {}
    for arm in ARMS:
        primary = reports[PRIMARY_ROLES[arm]]
        replay = reports[REPLAY_ROLES[arm]]
        _validate_report(primary, PRIMARY_ROLES[arm], arm)
        _validate_report(replay, REPLAY_ROLES[arm], arm)
        exact = primary["scientific_result_id"] == replay["scientific_result_id"]
        replication[arm] = {
            "primary_role": PRIMARY_ROLES[arm],
            "replay_role": REPLAY_ROLES[arm],
            "primary_host": primary["host"],
            "replay_host": replay["host"],
            "scientific_result_id": primary["scientific_result_id"],
            "exact_match": exact,
        }
        primary_metrics[arm] = _report_metrics(primary)

    comparisons = {
        "a2_vs_c0": _compare(
            primary_metrics["a2-history-intent"],
            primary_metrics["c0-pattern-prior"],
        ),
        "a2_vs_a0": _compare(
            primary_metrics["a2-history-intent"],
            primary_metrics["a0-public-state-intent"],
        ),
        "a2_vs_shuffle": _compare(
            primary_metrics["a2-history-intent"],
            primary_metrics["s3-shuffled-history-intent"],
        ),
    }
    a2 = primary_metrics["a2-history-intent"]
    c0 = primary_metrics["c0-pattern-prior"]
    a0 = primary_metrics["a0-public-state-intent"]
    gates = {
        "all_primary_replays_exact": all(
            value["exact_match"] for value in replication.values()
        ),
        "all_reports_complete_and_accounted": all(
            value["complete_accounting"] for value in primary_metrics.values()
        ),
        "a2_improves_c0_regret_by_0_05": (
            comparisons["a2_vs_c0"]["regret_improvement"] >= MIN_REGRET_IMPROVEMENT
        ),
        "a2_improves_a0_regret_by_0_05": (
            comparisons["a2_vs_a0"]["regret_improvement"] >= MIN_REGRET_IMPROVEMENT
        ),
        "a2_improves_shuffle_regret_by_0_05": (
            comparisons["a2_vs_shuffle"]["regret_improvement"]
            >= MIN_REGRET_IMPROVEMENT
        ),
        "a2_vs_c0_ci_wholly_below_zero": (
            comparisons["a2_vs_c0"]["paired_a2_minus_reference"]["ci95_upper"] < 0.0
        ),
        "a2_vs_a0_ci_wholly_below_zero": (
            comparisons["a2_vs_a0"]["paired_a2_minus_reference"]["ci95_upper"] < 0.0
        ),
        "a2_vs_shuffle_ci_wholly_below_zero": (
            comparisons["a2_vs_shuffle"]["paired_a2_minus_reference"]["ci95_upper"]
            < 0.0
        ),
        "a2_recall_nonregression": (
            a2["top1_recall"] >= max(c0["top1_recall"], a0["top1_recall"])
        ),
        "a2_pairwise_within_guardrail": (
            a2["r1200_pairwise_accuracy"]
            >= max(c0["r1200_pairwise_accuracy"], a0["r1200_pairwise_accuracy"])
            - MAX_PAIRWISE_REGRESSION
        ),
    }
    eligible = all(gates.values())
    identity = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "authorization_id": next(iter(authorization_ids)),
        "bundle_id": next(iter(bundle_ids)),
        "classification": (
            "o1_public_belief_search_validation_passed"
            if eligible
            else "o1_public_belief_search_validation_null"
        ),
        "eligible": eligible,
        "selected_arm": "a2-history-intent" if eligible else None,
        "gates": gates,
        "replication": replication,
        "arms": primary_metrics,
        "comparisons": comparisons,
        "claim_boundary": {
            "offline_high_regret_validation_only": True,
            "sealed_test_opened": False,
            "gameplay_run": False,
            "score_claim": False,
            "progress_to_100_claim": False,
        },
    }
    result = {**identity, "aggregate_id": canonical_blake3(identity)}
    _write_json(output, result)
    return result


def _report_metrics(report: dict[str, Any]) -> dict[str, Any]:
    groups = report["groups"]
    regrets = []
    recalls = []
    game_indices = []
    pairwise_correct = 0.0
    pairwise_total = 0
    selected_hashes = []
    for group in groups:
        candidates = group["candidates"]
        if len(candidates) != COHORT_WIDTH:
            raise PublicBeliefSearchError("search report group lost root candidates")
        scores = np.asarray(
            [candidate["search_mean"] for candidate in candidates],
            dtype=np.float64,
        )
        hashes = [bytes.fromhex(candidate["action_hash"]) for candidate in candidates]
        ranking = sorted(range(COHORT_WIDTH), key=lambda index: (-scores[index], hashes[index]))
        selected = ranking[0]
        if (
            selected != group["selected_cohort_index"]
            or hashes[selected].hex() != group["selected_action_hash"]
        ):
            raise PublicBeliefSearchError("search report selected root is inconsistent")
        r4800 = np.asarray(
            [
                np.nan if candidate["r4800_mean"] is None else candidate["r4800_mean"]
                for candidate in candidates
            ],
            dtype=np.float64,
        )
        labeled = np.flatnonzero(np.isfinite(r4800))
        if not len(labeled):
            raise PublicBeliefSearchError("high-regret group has no R4800 labels")
        winner = sorted(labeled, key=lambda index: (-r4800[index], hashes[index]))[0]
        best = float(r4800[winner])
        regret = (
            best - float(r4800[selected])
            if np.isfinite(r4800[selected])
            else best - float(np.min(r4800[labeled]))
        )
        regrets.append(regret)
        recalls.append(float(selected == winner))
        game_indices.append(int(group["game_index"]))
        selected_hashes.append(hashes[selected].hex())
        r1200 = np.asarray(
            [
                np.nan if candidate["r1200_mean"] is None else candidate["r1200_mean"]
                for candidate in candidates
            ],
            dtype=np.float64,
        )
        valid = np.flatnonzero(np.isfinite(r1200))
        for left_position, left in enumerate(valid):
            for right in valid[left_position + 1 :]:
                target_delta = r1200[left] - r1200[right]
                if target_delta == 0:
                    continue
                prediction_delta = scores[left] - scores[right]
                pairwise_total += 1
                if prediction_delta == 0:
                    pairwise_correct += 0.5
                elif math.copysign(1.0, prediction_delta) == math.copysign(
                    1.0,
                    target_delta,
                ):
                    pairwise_correct += 1.0
    regret_values = np.asarray(regrets, dtype=np.float64)
    recall_values = np.asarray(recalls, dtype=np.float64)
    return {
        "groups": len(groups),
        "mean_regret": float(np.mean(regret_values)),
        "median_regret": float(np.median(regret_values)),
        "top1_recall": float(np.mean(recall_values)),
        "r1200_pairwise_accuracy": (
            pairwise_correct / pairwise_total if pairwise_total else 0.0
        ),
        "r1200_pairwise_correct": pairwise_correct,
        "r1200_pairwise_total": pairwise_total,
        "regret_values": regrets,
        "game_indices": game_indices,
        "selected_action_hashes": selected_hashes,
        "wall_seconds": float(report["wall_seconds"]),
        "trajectories": int(report["trajectories"]),
        "leaf_model_rows": int(report["leaf_model_rows"]),
        "complete_accounting": (
            report["groups_completed"] == EXPECTED_PANEL_GROUPS
            and report["groups_expected"] == EXPECTED_PANEL_GROUPS
            and report["root_candidates"] == EXPECTED_PANEL_GROUPS * COHORT_WIDTH
            and report["trajectories"] == EXPECTED_PANEL_GROUPS * 640
            and report["hidden_order_invariance_checks"] == EXPECTED_PANEL_GROUPS
            and report["candidate_hash_checks"] == EXPECTED_PANEL_GROUPS * COHORT_WIDTH
        ),
    }


def _compare(a2: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    a2_regret = np.asarray(a2["regret_values"], dtype=np.float64)
    reference_regret = np.asarray(reference["regret_values"], dtype=np.float64)
    games = np.asarray(a2["game_indices"], dtype=np.int64)
    if (
        a2_regret.shape != reference_regret.shape
        or not np.array_equal(games, np.asarray(reference["game_indices"], dtype=np.int64))
    ):
        raise PublicBeliefSearchError("paired comparison reports do not align")
    bootstrap = _game_clustered_bootstrap(
        a2_regret,
        reference_regret,
        games,
    )
    return {
        "a2_mean_regret": float(np.mean(a2_regret)),
        "reference_mean_regret": float(np.mean(reference_regret)),
        "regret_improvement": float(np.mean(reference_regret - a2_regret)),
        "paired_a2_minus_reference": bootstrap,
        "selected_action_agreement": float(
            np.mean(
                np.asarray(a2["selected_action_hashes"], dtype=object)
                == np.asarray(reference["selected_action_hashes"], dtype=object)
            )
        ),
    }


def _game_clustered_bootstrap(
    treatment: np.ndarray,
    control: np.ndarray,
    game_indices: np.ndarray,
) -> dict[str, Any]:
    unique_games = np.unique(game_indices)
    if not len(unique_games):
        raise PublicBeliefSearchError("bootstrap has no games")
    by_game = {
        game: np.flatnonzero(game_indices == game)
        for game in unique_games
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_games = rng.choice(unique_games, size=len(unique_games), replace=True)
        rows = np.concatenate([by_game[int(game)] for game in sampled_games])
        differences[replicate] = float(np.mean(treatment[rows] - control[rows]))
    return {
        "mean_difference": float(np.mean(treatment - control)),
        "ci95_lower": float(np.quantile(differences, 0.025)),
        "ci95_upper": float(np.quantile(differences, 0.975)),
        "games": len(unique_games),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
    }


def _validate_report(report: dict[str, Any], role: str, arm: str) -> None:
    if (
        report.get("schema_version") != 1
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("role") != role
        or report.get("arm") != arm
        or report.get("production") is not True
        or report.get("protocol") != frozen_protocol()
        or report.get("groups_completed") != EXPECTED_PANEL_GROUPS
        or not isinstance(report.get("scientific_result_id"), str)
        or not isinstance(report.get("groups"), list)
    ):
        raise PublicBeliefSearchError(f"{role} report violates the frozen contract")


def canonical_blake3(value: object) -> str:
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
        raise PublicBeliefSearchError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise PublicBeliefSearchError(f"{label} must be an object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def _require_digest(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PublicBeliefSearchError(f"{label} is not a lowercase BLAKE3 digest")


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PublicBeliefSearchError(f"{label} must be a nonnegative integer")
    return value


def _add_common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--intent-root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)


def _parse_role_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        role, separator, path = value.partition("=")
        if not separator or role in result or role not in ROLE_ARMS or not path:
            raise PublicBeliefSearchError(f"invalid role report mapping {value}")
        result[role] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    panel_parser = subparsers.add_parser("freeze-panel")
    panel_parser.add_argument("--control-report", type=Path, required=True)
    panel_parser.add_argument("--output", type=Path, required=True)

    authorize_parser = subparsers.add_parser("authorize")
    _add_common_inputs(authorize_parser)
    authorize_parser.add_argument("--output", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify-authorization")
    _add_common_inputs(verify_parser)
    verify_parser.add_argument("--authorization", type=Path, required=True)
    verify_parser.add_argument("--role", required=True)
    verify_parser.add_argument("--output", type=Path, required=True)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--report", action="append", default=[], required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "freeze-panel":
        freeze_high_regret_panel(
            control_report=args.control_report,
            output=args.output,
        )
    elif args.command == "authorize":
        authorize(
            bundle_id=args.bundle_id,
            dataset_root=args.dataset_root,
            cohort_root=args.cohort_root,
            intent_root=args.intent_root,
            panel_path=args.panel,
            model_dir=args.model_dir,
            output=args.output,
        )
    elif args.command == "verify-authorization":
        verify_authorization(
            authorization=args.authorization,
            role=args.role,
            bundle_id=args.bundle_id,
            dataset_root=args.dataset_root,
            cohort_root=args.cohort_root,
            intent_root=args.intent_root,
            panel_path=args.panel,
            model_dir=args.model_dir,
            output=args.output,
        )
    else:
        aggregate_reports(
            role_reports=_parse_role_paths(args.report),
            output=args.output,
        )


if __name__ == "__main__":
    main()
