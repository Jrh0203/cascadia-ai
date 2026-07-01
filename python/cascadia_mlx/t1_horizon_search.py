"""Authorization and classification for the T1 search-horizon decomposition."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import blake3
import numpy as np

EXPERIMENT_ID = "t1-search-horizon-decomposition-v1"
PROTOCOL_ID = "t1-strict-train-horizon-decomposition-v1"
EXPECTED_GROUPS = 560
COHORT_WIDTH = 64
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2_026_061_722
FAMILYWISE_ALPHA = 0.05
MIN_DIRECT_IMPROVEMENT = 0.05
MIN_H0_IMPROVEMENT = 0.03
MAX_PAIRWISE_REGRESSION = 0.005

ARMS = (
    "h0-root-leaf",
    "h1-one-opponent",
    "h2-two-opponents",
    "h3-full-rotation",
)
SEARCHED_ARMS = ARMS[1:]
PRIMARY_ROLES = {
    "h0-root-leaf": "h0-primary",
    "h1-one-opponent": "h1-primary",
    "h2-two-opponents": "h2-primary",
    "h3-full-rotation": "h3-primary",
}
REPLAY_ROLES = {
    "h0-root-leaf": "h0-replay",
    "h1-one-opponent": "h1-replay",
    "h2-two-opponents": "h2-replay",
    "h3-full-rotation": "h3-replay",
}
ROLE_ARMS = {
    role: arm
    for arm in ARMS
    for role in (PRIMARY_ROLES[arm], REPLAY_ROLES[arm])
}
HORIZON_TURNS = {
    "h0-root-leaf": 0,
    "h1-one-opponent": 1,
    "h2-two-opponents": 2,
    "h3-full-rotation": 3,
}


class HorizonSearchError(ValueError):
    """The frozen T1 horizon-search contract was violated."""


def frozen_protocol() -> dict[str, Any]:
    return {
        "root_candidates": COHORT_WIDTH,
        "h0_evaluations_per_group": COHORT_WIDTH,
        "stage_additional_samples": [4, 4, 8, 16],
        "stage_retain": [32, 16, 8, 1],
        "trajectories_per_search_group": 640,
        "horizon_opponent_turns": HORIZON_TURNS,
        "control_temperature_milli": 1_000,
        "pattern_config": {
            "immediate_candidate_limit": 8,
            "habitat_candidate_limit": 6,
            "bear_candidate_limit": 8,
            "future_market_draws": 4,
        },
        "leaf_model": "qualified-legacy-v4opp-exact-mlx-v1",
        "leaf_value": "v2-current-base-score-plus-legacy-nnue-remaining-value",
        "root_chance_policy": (
            "apply-frozen-complete-root-before-future-redeterminization"
        ),
        "hidden_order_policy": (
            "arm-independent-sort-and-redeterminize-post-root-by-group-root-sample"
        ),
        "prefix_coupling": (
            "h1-prefix-of-h2-prefix-of-h3-by-shared-opponent-uniforms"
        ),
    }


def build_authorization(
    *,
    bundle_id: str,
    dataset_root: Path,
    cohort_root: Path,
    model_dir: Path,
) -> dict[str, Any]:
    _require_digest(bundle_id, "bundle ID")
    dataset = _read_json(dataset_root / "dataset.json", "T1 train dataset")
    cohort = _read_json(cohort_root / "cohort.json", "T1 strict cohort")
    model = _read_json(model_dir / "model.json", "qualified MLX model")
    if (
        dataset.get("split") != "train"
        or dataset.get("total_groups") != EXPECTED_GROUPS
        or cohort.get("experiment_id") != EXPERIMENT_ID
        or cohort.get("protocol_id") != "t1-strict-train-top64-cohort-v1"
        or cohort.get("cohort_schema") != "t1-strict-exact-r2-top64-cohort-v1"
        or cohort.get("complete_train_corpus") is not True
        or cohort.get("groups") != EXPECTED_GROUPS
        or cohort.get("dataset_id") != dataset.get("dataset_id")
        or model.get("architecture") != "legacy-sparse-nnue-v4opp-mlx-v1"
        or model.get("dimensions", {}).get("features") != 11_231
    ):
        raise HorizonSearchError("authorization inputs violate the frozen T1 campaign")
    inputs = {
        "dataset_id": dataset["dataset_id"],
        "dataset_manifest_blake3": _checksum(dataset_root / "dataset.json"),
        "cohort_id": cohort["cohort_id"],
        "cohort_manifest_blake3": _checksum(cohort_root / "cohort.json"),
        "model_manifest_blake3": _checksum(model_dir / "model.json"),
        "model_safetensors_blake3": _checksum(
            model_dir / "model.safetensors"
        ),
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
            "open_train_only": True,
            "validation_opened": False,
            "sealed_test_opened": False,
            "gameplay_run": False,
            "score_claim_authorized": False,
            "progress_to_100_claim_authorized": False,
            "legacy_441_role": "frozen-leaf-evaluator-only",
            "v2_search_state": "canonical-sparse-game-state",
        },
    }
    return {**identity, "authorization_id": canonical_blake3(identity)}


def authorize(**kwargs: Any) -> dict[str, Any]:
    output = Path(kwargs.pop("output"))
    authorization = build_authorization(**kwargs)
    _write_json(output, authorization)
    return authorization


def verify_authorization(
    *,
    authorization: Path,
    role: str,
    output: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    if role not in ROLE_ARMS:
        raise HorizonSearchError(f"unknown T1 role {role}")
    observed = _read_json(authorization, "T1 authorization")
    expected = build_authorization(**kwargs)
    if observed != expected:
        raise HorizonSearchError("T1 authorization bytes do not rebuild exactly")
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
        raise HorizonSearchError("T1 aggregate requires every primary and replay role")
    reports = {
        role: _read_json(path, f"{role} T1 report")
        for role, path in role_reports.items()
    }
    authorization_ids = {report.get("authorization_id") for report in reports.values()}
    bundle_ids = {report.get("bundle_id") for report in reports.values()}
    if len(authorization_ids) != 1 or len(bundle_ids) != 1:
        raise HorizonSearchError("T1 reports do not share one authorization and bundle")

    replication = {}
    primary_metrics = {}
    cohort_signatures = set()
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
        cohort_signatures.add(_cohort_signature(primary))
    if len(cohort_signatures) != 1:
        raise HorizonSearchError("T1 arms did not evaluate the same frozen roots")

    direct = _direct_metrics(reports[PRIMARY_ROLES["h0-root-leaf"]])
    h0 = primary_metrics["h0-root-leaf"]
    comparisons = {
        "h0_vs_direct": _compare(h0, direct),
    }
    superiority_keys = []
    for arm in SEARCHED_ARMS:
        short = arm.split("-", 1)[0]
        direct_key = f"{short}_vs_direct"
        h0_key = f"{short}_vs_h0"
        comparisons[direct_key] = _compare(primary_metrics[arm], direct)
        comparisons[h0_key] = _compare(primary_metrics[arm], h0)
        superiority_keys.extend((direct_key, h0_key))
    comparisons["h2_vs_h1"] = _compare(
        primary_metrics["h2-two-opponents"],
        primary_metrics["h1-one-opponent"],
    )
    comparisons["h3_vs_h2"] = _compare(
        primary_metrics["h3-full-rotation"],
        primary_metrics["h2-two-opponents"],
    )

    holm = _holm_bonferroni(
        {
            key: comparisons[key]["paired_treatment_minus_reference"][
                "one_sided_superiority_p"
            ]
            for key in superiority_keys
        },
        alpha=FAMILYWISE_ALPHA,
    )
    global_integrity = {
        "all_primary_replays_exact": all(
            value["exact_match"] for value in replication.values()
        ),
        "all_reports_complete_and_accounted": all(
            value["complete_accounting"] for value in primary_metrics.values()
        ),
        "all_arms_share_frozen_roots": len(cohort_signatures) == 1,
    }
    horizon_gates = {}
    eligible_arms = []
    for arm in SEARCHED_ARMS:
        short = arm.split("-", 1)[0]
        versus_direct = comparisons[f"{short}_vs_direct"]
        versus_h0 = comparisons[f"{short}_vs_h0"]
        gates = {
            "improves_direct_by_0_05": (
                versus_direct["regret_improvement"] >= MIN_DIRECT_IMPROVEMENT
            ),
            "improves_h0_by_0_03": (
                versus_h0["regret_improvement"] >= MIN_H0_IMPROVEMENT
            ),
            "direct_superiority_survives_holm": holm[f"{short}_vs_direct"][
                "rejected"
            ],
            "h0_superiority_survives_holm": holm[f"{short}_vs_h0"]["rejected"],
            "recall_nonregression": (
                primary_metrics[arm]["top1_recall"]
                >= max(direct["top1_recall"], h0["top1_recall"])
            ),
            "pairwise_within_guardrail": (
                primary_metrics[arm]["r1200_pairwise_accuracy"]
                >= max(
                    direct["r1200_pairwise_accuracy"],
                    h0["r1200_pairwise_accuracy"],
                )
                - MAX_PAIRWISE_REGRESSION
            ),
        }
        eligible = all(gates.values()) and all(global_integrity.values())
        horizon_gates[arm] = {**gates, "eligible": eligible}
        if eligible:
            eligible_arms.append(arm)

    selected_arm = min(
        eligible_arms,
        key=lambda arm: (
            primary_metrics[arm]["mean_regret"],
            HORIZON_TURNS[arm],
        ),
        default=None,
    )
    h0_vs_direct = comparisons["h0_vs_direct"]
    leaf_only = (
        selected_arm is None
        and h0_vs_direct["regret_improvement"] >= MIN_DIRECT_IMPROVEMENT
        and h0_vs_direct["paired_treatment_minus_reference"]["ci95_upper"] < 0.0
        and all(global_integrity.values())
    )
    if selected_arm is not None:
        classification = "t1_search_horizon_decomposition_development_passed"
    elif leaf_only:
        classification = "t1_search_horizon_leaf_only"
    else:
        classification = "t1_search_horizon_decomposition_development_null"
    identity = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "authorization_id": next(iter(authorization_ids)),
        "bundle_id": next(iter(bundle_ids)),
        "classification": classification,
        "eligible": selected_arm is not None,
        "selected_arm": selected_arm,
        "selected_opponent_turns": (
            None if selected_arm is None else HORIZON_TURNS[selected_arm]
        ),
        "global_integrity": global_integrity,
        "horizon_gates": horizon_gates,
        "holm_bonferroni": holm,
        "replication": replication,
        "direct": direct,
        "arms": primary_metrics,
        "comparisons": comparisons,
        "claim_boundary": {
            "offline_open_train_only": True,
            "validation_opened": False,
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
    arm = report["arm"]
    expected_per_group = 64 if arm == "h0-root-leaf" else 640
    selection = []
    score_rows = []
    for group in report["groups"]:
        candidates = group["candidates"]
        scores = np.asarray(
            [candidate["search_mean"] for candidate in candidates],
            dtype=np.float64,
        )
        hashes = [bytes.fromhex(candidate["action_hash"]) for candidate in candidates]
        ranking = sorted(
            range(COHORT_WIDTH),
            key=lambda index: (-scores[index], hashes[index]),
        )
        if (
            ranking[0] != group["selected_cohort_index"]
            or hashes[ranking[0]].hex() != group["selected_action_hash"]
        ):
            raise HorizonSearchError("T1 selected root is inconsistent")
        selection.append(ranking[0])
        score_rows.append(scores)
    metrics = _selection_metrics(report, selection, score_rows)
    metrics.update(
        {
            "wall_seconds": float(report["wall_seconds"]),
            "trajectories": int(report["trajectories"]),
            "leaf_model_rows": int(report["leaf_model_rows"]),
            "terminal_leaves": int(report["terminal_leaves"]),
            "opponent_decisions": int(report["opponent_decisions"]),
            "opponent_options": int(report["opponent_options"]),
            "complete_accounting": (
                report["groups_completed"] == EXPECTED_GROUPS
                and report["groups_expected"] == EXPECTED_GROUPS
                and report["root_candidates"] == EXPECTED_GROUPS * COHORT_WIDTH
                and report["trajectories"] == EXPECTED_GROUPS * expected_per_group
                and report["hidden_order_invariance_checks"] == EXPECTED_GROUPS
                and report["prefix_coupling_checks"] == EXPECTED_GROUPS
                and report["candidate_hash_checks"]
                == EXPECTED_GROUPS * COHORT_WIDTH
            ),
        }
    )
    return metrics


def _direct_metrics(report: dict[str, Any]) -> dict[str, Any]:
    selection = []
    score_rows = []
    for group in report["groups"]:
        candidates = group["candidates"]
        direct = [
            index
            for index, candidate in enumerate(candidates)
            if candidate["direct_rank"] == 0
        ]
        if (
            len(direct) != 1
            or direct[0] != group["direct_cohort_index"]
        ):
            raise HorizonSearchError("T1 direct comparator is not unique")
        selection.append(direct[0])
        score_rows.append(
            np.asarray(
                [candidate["direct_score"] for candidate in candidates],
                dtype=np.float64,
            )
        )
    return _selection_metrics(report, selection, score_rows)


def _selection_metrics(
    report: dict[str, Any],
    selection: list[int],
    score_rows: list[np.ndarray],
) -> dict[str, Any]:
    regrets = []
    recalls = []
    game_indices = []
    pairwise_correct = 0.0
    pairwise_total = 0
    selected_hashes = []
    phase_regrets: dict[str, list[float]] = {"early": [], "middle": [], "late": []}
    for group, selected, scores in zip(
        report["groups"],
        selection,
        score_rows,
        strict=True,
    ):
        candidates = group["candidates"]
        hashes = [bytes.fromhex(candidate["action_hash"]) for candidate in candidates]
        r4800 = np.asarray(
            [
                np.nan
                if candidate["r4800_mean"] is None
                else candidate["r4800_mean"]
                for candidate in candidates
            ],
            dtype=np.float64,
        )
        labeled = np.flatnonzero(np.isfinite(r4800))
        if not len(labeled):
            raise HorizonSearchError("T1 group has no R4800 labels")
        winner = sorted(
            labeled,
            key=lambda index: (-r4800[index], hashes[index]),
        )[0]
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
        phase_regrets[_phase_name(int(group["completed_turns"]))].append(regret)
        r1200 = np.asarray(
            [
                np.nan
                if candidate["r1200_mean"] is None
                else candidate["r1200_mean"]
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
    return {
        "groups": len(regrets),
        "mean_regret": float(np.mean(regret_values)),
        "median_regret": float(np.median(regret_values)),
        "top1_recall": float(np.mean(recalls)),
        "r1200_pairwise_accuracy": (
            pairwise_correct / pairwise_total if pairwise_total else 0.0
        ),
        "r1200_pairwise_correct": pairwise_correct,
        "r1200_pairwise_total": pairwise_total,
        "phase_mean_regret": {
            phase: (
                None if not values else float(np.mean(np.asarray(values)))
            )
            for phase, values in phase_regrets.items()
        },
        "regret_values": regrets,
        "game_indices": game_indices,
        "selected_action_hashes": selected_hashes,
    }


def _compare(
    treatment: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    treatment_regret = np.asarray(treatment["regret_values"], dtype=np.float64)
    reference_regret = np.asarray(reference["regret_values"], dtype=np.float64)
    games = np.asarray(treatment["game_indices"], dtype=np.int64)
    if (
        treatment_regret.shape != reference_regret.shape
        or not np.array_equal(
            games,
            np.asarray(reference["game_indices"], dtype=np.int64),
        )
    ):
        raise HorizonSearchError("T1 paired reports do not align")
    bootstrap = _game_clustered_bootstrap(
        treatment_regret,
        reference_regret,
        games,
    )
    return {
        "treatment_mean_regret": float(np.mean(treatment_regret)),
        "reference_mean_regret": float(np.mean(reference_regret)),
        "regret_improvement": float(
            np.mean(reference_regret - treatment_regret)
        ),
        "paired_treatment_minus_reference": bootstrap,
        "selected_action_agreement": float(
            np.mean(
                np.asarray(treatment["selected_action_hashes"], dtype=object)
                == np.asarray(
                    reference["selected_action_hashes"],
                    dtype=object,
                )
            )
        ),
    }


def _game_clustered_bootstrap(
    treatment: np.ndarray,
    reference: np.ndarray,
    game_indices: np.ndarray,
) -> dict[str, Any]:
    unique_games = np.unique(game_indices)
    if not len(unique_games):
        raise HorizonSearchError("T1 bootstrap has no source games")
    by_game = {
        int(game): np.flatnonzero(game_indices == game)
        for game in unique_games
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_games = rng.choice(
            unique_games,
            size=len(unique_games),
            replace=True,
        )
        rows = np.concatenate([by_game[int(game)] for game in sampled_games])
        differences[replicate] = float(
            np.mean(treatment[rows] - reference[rows])
        )
    return {
        "mean_difference": float(np.mean(treatment - reference)),
        "ci95_lower": float(np.quantile(differences, 0.025)),
        "ci95_upper": float(np.quantile(differences, 0.975)),
        "one_sided_superiority_p": float(
            (1 + np.count_nonzero(differences >= 0.0))
            / (BOOTSTRAP_REPLICATES + 1)
        ),
        "games": len(unique_games),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
    }


def _holm_bonferroni(
    p_values: dict[str, float],
    *,
    alpha: float,
) -> dict[str, dict[str, Any]]:
    if (
        not p_values
        or not 0.0 < alpha < 1.0
        or any(not 0.0 <= value <= 1.0 for value in p_values.values())
    ):
        raise HorizonSearchError("invalid Holm-Bonferroni inputs")
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    count = len(ordered)
    reject_prefix = True
    running_adjusted = 0.0
    result: dict[str, dict[str, Any]] = {}
    for rank, key in enumerate(ordered, start=1):
        multiplier = count - rank + 1
        threshold = alpha / multiplier
        raw = p_values[key]
        rejected = reject_prefix and raw <= threshold
        reject_prefix = reject_prefix and rejected
        running_adjusted = max(running_adjusted, min(1.0, raw * multiplier))
        result[key] = {
            "raw_p": raw,
            "adjusted_p": running_adjusted,
            "rank": rank,
            "threshold": threshold,
            "rejected": rejected,
        }
    return {key: result[key] for key in sorted(result)}


def _cohort_signature(report: dict[str, Any]) -> str:
    identity = []
    for group in report["groups"]:
        identity.append(
            {
                "cohort_row": group["cohort_row"],
                "group_id": group["group_id"],
                "game_index": group["game_index"],
                "completed_turns": group["completed_turns"],
                "direct_cohort_index": group["direct_cohort_index"],
                "candidates": [
                    {
                        "source_index": candidate["source_index"],
                        "action_hash": candidate["action_hash"],
                        "direct_rank": candidate["direct_rank"],
                        "direct_score": candidate["direct_score"],
                        "r1200_mean": candidate["r1200_mean"],
                        "r4800_mean": candidate["r4800_mean"],
                    }
                    for candidate in group["candidates"]
                ],
            }
        )
    return canonical_blake3(identity)


def _validate_report(report: dict[str, Any], role: str, arm: str) -> None:
    if (
        report.get("schema_version") != 1
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("role") != role
        or report.get("arm") != arm
        or report.get("opponent_turns") != HORIZON_TURNS[arm]
        or report.get("production") is not True
        or report.get("protocol") != frozen_protocol()
        or report.get("groups_completed") != EXPECTED_GROUPS
        or not isinstance(report.get("scientific_result_id"), str)
        or not isinstance(report.get("groups"), list)
        or len(report["groups"]) != EXPECTED_GROUPS
    ):
        raise HorizonSearchError(f"{role} report violates the frozen T1 contract")


def _phase_name(completed_turns: int) -> str:
    personal_turn = completed_turns // 4
    if personal_turn < 7:
        return "early"
    if personal_turn < 14:
        return "middle"
    return "late"


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
        raise HorizonSearchError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise HorizonSearchError(f"{label} must be an object")
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
        raise HorizonSearchError(f"{label} is not a lowercase BLAKE3 digest")


def _add_common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)


def _parse_role_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        role, separator, path = value.partition("=")
        if not separator or role in result or role not in ROLE_ARMS or not path:
            raise HorizonSearchError(f"invalid T1 role report mapping {value}")
        result[role] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    authorize_parser = subparsers.add_parser("authorize")
    _add_common_inputs(authorize_parser)
    authorize_parser.add_argument("--output", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify-authorization")
    _add_common_inputs(verify_parser)
    verify_parser.add_argument("--authorization", type=Path, required=True)
    verify_parser.add_argument("--role", required=True)
    verify_parser.add_argument("--output", type=Path, required=True)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument(
        "--report",
        action="append",
        default=[],
        required=True,
    )
    aggregate_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "authorize":
        authorize(
            bundle_id=args.bundle_id,
            dataset_root=args.dataset_root,
            cohort_root=args.cohort_root,
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

