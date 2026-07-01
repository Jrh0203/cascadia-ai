#!/usr/bin/env python3
"""Frozen O2 exact-opportunity identifiability and B1 analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

EXPERIMENT_ID = "o2-exact-opportunity-matching-v1"
PROTOCOL_ID = "o2-strict-train-top64-foundation-identifiability-v1"
EXPECTED_GROUPS = 560
COHORT_WIDTH = 64
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2_026_061_702
RIDGE_LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0)

FEATURE_NAMES = (
    "demand_count",
    "supply_count",
    "edge_count",
    "matched_demands",
    "unmatched_demands",
    "wildlife_matches",
    "habitat_matches",
    "market_matches",
    "unseen_matches",
    "exact_completion_value",
    "teacher_value_points",
    "matched_demand_fraction",
    "market_match_fraction",
    "wildlife_match_fraction",
    "mean_matched_exposure",
)


class AnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class Group:
    game: int
    group_id: int
    personal_turn: int
    nature_tokens: int
    rows: tuple[dict[str, Any], ...]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisError(f"{path} is not a JSON object")
    return value


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    try:
        import blake3  # type: ignore

        return blake3.blake3(payload).hexdigest()
    except ImportError:
        return "sha256:" + hashlib.sha256(payload).hexdigest()


def load_export(root: Path) -> tuple[dict[str, Any], list[Group]]:
    report = _read_json(root / "export-report.json")
    required = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "production": True,
        "groups_expected": EXPECTED_GROUPS,
        "groups_completed": EXPECTED_GROUPS,
        "candidate_rows": EXPECTED_GROUPS * COHORT_WIDTH,
        "unique_action_hashes": EXPECTED_GROUPS * COHORT_WIDTH,
        "hidden_order_invariance_checks": EXPECTED_GROUPS,
        "replay_construction_checks": EXPECTED_GROUPS * COHORT_WIDTH,
        "codec_round_trip_checks": EXPECTED_GROUPS * COHORT_WIDTH * 2,
        "d6_covariance_checks": EXPECTED_GROUPS * 12,
        "candidate_hash_checks": EXPECTED_GROUPS * COHORT_WIDTH,
        "zero_swap_growth": True,
        "validation_opened": False,
        "sealed_test_opened": False,
        "gameplay_run": False,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise AnalysisError(
                f"foundation report field {key} is {report.get(key)!r}, expected {expected!r}"
            )
    groups: list[Group] = []
    for index in range(EXPECTED_GROUPS):
        value = _read_json(root / "groups" / f"row-{index:03}.json")
        rows = value.get("rows")
        if (
            value.get("cohort_row") != index
            or not isinstance(rows, list)
            or len(rows) != COHORT_WIDTH
            or [row.get("cohort_index") for row in rows] != list(range(COHORT_WIDTH))
        ):
            raise AnalysisError(f"group row {index} violates cohort ordering")
        hashes = [row.get("action_hash") for row in rows]
        if len(set(hashes)) != COHORT_WIDTH:
            raise AnalysisError(f"group row {index} repeats action hashes")
        groups.append(
            Group(
                game=int(value["game_index"]),
                group_id=int(value["group_id"]),
                personal_turn=int(value["personal_turn"]),
                nature_tokens=int(value["nature_tokens"]),
                rows=tuple(rows),
            )
        )
    if len({group.group_id for group in groups}) != EXPECTED_GROUPS:
        raise AnalysisError("group IDs are not unique")
    games = sorted({group.game for group in groups})
    if games != [61000, 61001, 61002, 61005, 61006, 61009, 61010]:
        raise AnalysisError(f"unexpected game blocks: {games}")
    return report, groups


def row_features(row: dict[str, Any]) -> np.ndarray:
    raw = np.asarray(
        [
            row["demand_count"],
            row["supply_count"],
            row["edge_count"],
            row["matched_demands"],
            row["unmatched_demands"],
            row["wildlife_matches"],
            row["habitat_matches"],
            row["market_matches"],
            row["unseen_matches"],
            row["exact_completion_value"],
            row["teacher_value_micros"] / 1_000_000.0,
            row["matched_demand_fraction"],
            row["market_match_fraction"],
            row["wildlife_match_fraction"],
            row["mean_matched_exposure"],
        ],
        dtype=np.float64,
    )
    if raw.shape != (len(FEATURE_NAMES),) or not np.isfinite(raw).all() or np.any(raw < 0):
        raise AnalysisError("matching feature row is invalid")
    return np.log1p(raw)


def flatten(groups: list[Group]) -> dict[str, np.ndarray]:
    rows = [(group, row) for group in groups for row in group.rows]
    features = np.stack([row_features(row) for _, row in rows])
    direct = np.asarray([row["direct_score"] for _, row in rows], dtype=np.float64)
    target = np.asarray(
        [
            np.nan if row["r4800_mean"] is None else float(row["r4800_mean"])
            for _, row in rows
        ],
        dtype=np.float64,
    )
    arrays = {
        "features": features,
        "direct": direct,
        "target": target,
        "residual": target - direct,
        "game": np.asarray([group.game for group, _ in rows], dtype=np.int64),
        "group": np.repeat(np.arange(len(groups), dtype=np.int64), COHORT_WIDTH),
    }
    if not np.isfinite(features).all() or not np.isfinite(direct).all():
        raise AnalysisError("flattened features contain non-finite values")
    return arrays


def _fit_ridge(
    features: np.ndarray,
    target: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (features - mean) / scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return mean, scale, coefficients


def _predict_ridge(
    features: np.ndarray,
    fit: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    mean, scale, coefficients = fit
    design = np.column_stack([np.ones(len(features)), (features - mean) / scale])
    return design @ coefficients


def nested_game_crossfit(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[int, float]]:
    features = arrays["features"]
    residual = arrays["residual"]
    game = arrays["game"]
    labeled = np.isfinite(residual)
    predictions = np.full(len(residual), np.nan, dtype=np.float64)
    selected_lambdas: dict[int, float] = {}
    games = sorted(set(int(value) for value in game))
    for outer_game in games:
        outer_train = (game != outer_game) & labeled
        if np.count_nonzero(outer_train) <= len(FEATURE_NAMES):
            raise AnalysisError("outer O2 fold has insufficient labeled rows")
        inner_games = [value for value in games if value != outer_game]
        losses: dict[float, float] = {}
        for ridge in RIDGE_LAMBDAS:
            squared_errors: list[np.ndarray] = []
            for inner_game in inner_games:
                inner_train = outer_train & (game != inner_game)
                inner_test = outer_train & (game == inner_game)
                if not np.any(inner_test):
                    raise AnalysisError("inner O2 fold has no labeled rows")
                fit = _fit_ridge(features[inner_train], residual[inner_train], ridge)
                predicted = _predict_ridge(features[inner_test], fit)
                squared_errors.append((predicted - residual[inner_test]) ** 2)
            losses[ridge] = float(np.mean(np.concatenate(squared_errors)))
        selected = min(RIDGE_LAMBDAS, key=lambda ridge: (losses[ridge], ridge))
        selected_lambdas[outer_game] = selected
        fit = _fit_ridge(features[outer_train], residual[outer_train], selected)
        predictions[game == outer_game] = _predict_ridge(features[game == outer_game], fit)
    if not np.isfinite(predictions).all():
        raise AnalysisError("cross-fitted O2 predictions are incomplete")
    return predictions, selected_lambdas


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) <= 0 or np.std(right) <= 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def bootstrap_correlation(
    residual: np.ndarray,
    prediction: np.ndarray,
    games: np.ndarray,
) -> list[float]:
    unique = np.asarray(sorted(set(int(value) for value in games)), dtype=np.int64)
    rows = {game: np.flatnonzero(games == game) for game in unique}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([rows[int(game)] for game in sampled])
        values[replicate] = _pearson(prediction[indices], residual[indices])
    return _percentile_interval(values)


def select_index(scores: np.ndarray, hashes: list[str]) -> int:
    if len(scores) != len(hashes) or not np.isfinite(scores).all():
        raise AnalysisError("selection inputs are invalid")
    return min(range(len(scores)), key=lambda index: (-float(scores[index]), hashes[index]))


def group_metrics(
    groups: list[Group],
    arrays: dict[str, np.ndarray],
    residual_prediction: np.ndarray,
) -> tuple[list[dict[str, Any]], float, float]:
    results: list[dict[str, Any]] = []
    baseline_pairwise_correct = baseline_pairwise_total = 0
    treatment_pairwise_correct = treatment_pairwise_total = 0
    for group_index, group in enumerate(groups):
        start = group_index * COHORT_WIDTH
        stop = start + COHORT_WIDTH
        target = arrays["target"][start:stop]
        direct = arrays["direct"][start:stop]
        treatment = direct + residual_prediction[start:stop]
        hashes = [str(row["action_hash"]) for row in group.rows]
        labeled = np.flatnonzero(np.isfinite(target))
        if len(labeled) == 0:
            raise AnalysisError(f"group {group.group_id} has no R4800 labels")
        winner = min(labeled, key=lambda index: (-float(target[index]), hashes[index]))
        best = float(target[winner])
        worst = float(np.min(target[labeled]))
        baseline = select_index(direct, hashes)
        selected = select_index(treatment, hashes)

        def regret(index: int) -> float:
            return best - float(target[index]) if np.isfinite(target[index]) else best - worst

        for left_pos, left in enumerate(labeled):
            for right in labeled[left_pos + 1 :]:
                truth = np.sign(target[left] - target[right])
                if truth == 0:
                    continue
                baseline_pairwise_correct += int(np.sign(direct[left] - direct[right]) == truth)
                treatment_pairwise_correct += int(
                    np.sign(treatment[left] - treatment[right]) == truth
                )
                baseline_pairwise_total += 1
                treatment_pairwise_total += 1
        results.append(
            {
                "group_index": group_index,
                "group_id": group.group_id,
                "game": group.game,
                "personal_turn": group.personal_turn,
                "nature_tokens": group.nature_tokens,
                "baseline_index": baseline,
                "treatment_index": selected,
                "winner_index": int(winner),
                "baseline_regret": regret(baseline),
                "treatment_regret": regret(selected),
                "difference": regret(selected) - regret(baseline),
                "baseline_top1": int(baseline == winner),
                "treatment_top1": int(selected == winner),
                "baseline_draft_kind": int(group.rows[baseline]["draft_kind"]),
                "winner_wildlife": int(group.rows[winner]["drafted_wildlife"]),
                "median_unseen_fraction": float(
                    np.median([row["unseen_matches"] / max(row["matched_demands"], 1) for row in group.rows])
                ),
                "median_exposure": float(
                    np.median([row["mean_matched_exposure"] for row in group.rows])
                ),
            }
        )
    if baseline_pairwise_total == 0 or treatment_pairwise_total != baseline_pairwise_total:
        raise AnalysisError("pairwise metric has no comparable labeled pairs")
    return (
        results,
        baseline_pairwise_correct / baseline_pairwise_total,
        treatment_pairwise_correct / treatment_pairwise_total,
    )


def bootstrap_regret_difference(results: list[dict[str, Any]]) -> list[float]:
    games = np.asarray(sorted({int(row["game"]) for row in results}), dtype=np.int64)
    differences = {
        game: np.asarray(
            [row["difference"] for row in results if row["game"] == int(game)],
            dtype=np.float64,
        )
        for game in games
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(games, size=len(games), replace=True)
        values[replicate] = float(
            np.mean(np.concatenate([differences[int(game)] for game in sampled]))
        )
    return _percentile_interval(values)


def protected_slices(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    game_values = sorted({int(row["game"]) for row in results})
    scarcity_threshold: dict[int, float] = {}
    exposure_threshold: dict[int, float] = {}
    for held_out in game_values:
        training = [row for row in results if row["game"] != held_out]
        scarcity_threshold[held_out] = float(
            np.median([row["median_unseen_fraction"] for row in training])
        )
        exposure_threshold[held_out] = float(
            np.median([row["median_exposure"] for row in training])
        )
    predicates: dict[str, Any] = {
        "phase-early": lambda row: row["personal_turn"] <= 6,
        "phase-middle": lambda row: 7 <= row["personal_turn"] <= 13,
        "phase-late": lambda row: row["personal_turn"] >= 14,
        "nature-zero": lambda row: row["nature_tokens"] == 0,
        "nature-positive": lambda row: row["nature_tokens"] > 0,
        "baseline-paired": lambda row: row["baseline_draft_kind"] == 0,
        "baseline-independent": lambda row: row["baseline_draft_kind"] == 1,
        "scarce-supply": lambda row: row["median_unseen_fraction"]
        < scarcity_threshold[int(row["game"])],
        "high-competition": lambda row: row["median_exposure"]
        > exposure_threshold[int(row["game"])],
    }
    for wildlife, name in enumerate(("bear", "elk", "salmon", "hawk", "fox")):
        predicates[f"winner-{name}"] = (
            lambda row, wildlife=wildlife: row["winner_wildlife"] == wildlife
        )
    slices: dict[str, dict[str, Any]] = {}
    for name, predicate in predicates.items():
        selected = [row for row in results if predicate(row)]
        mean_difference = (
            float(np.mean([row["difference"] for row in selected])) if selected else None
        )
        gated = len(selected) >= 20
        slices[name] = {
            "groups": len(selected),
            "gated": gated,
            "mean_treatment_minus_baseline_regret": mean_difference,
            "passed": not gated or (mean_difference is not None and mean_difference <= 0.05),
        }
    return slices


def analyze(root: Path) -> dict[str, Any]:
    export, groups = load_export(root)
    arrays = flatten(groups)
    prediction, lambdas = nested_game_crossfit(arrays)
    labeled = np.isfinite(arrays["residual"])
    residual = arrays["residual"][labeled]
    labeled_prediction = prediction[labeled]
    baseline_mse = float(np.mean(residual**2))
    treatment_mse = float(np.mean((labeled_prediction - residual) ** 2))
    mse_improvement = (baseline_mse - treatment_mse) / baseline_mse
    correlation = _pearson(labeled_prediction, residual)
    correlation_ci = bootstrap_correlation(
        residual,
        labeled_prediction,
        arrays["game"][labeled],
    )
    variable_groups = 0
    for group_index in range(len(groups)):
        block = arrays["features"][
            group_index * COHORT_WIDTH : (group_index + 1) * COHORT_WIDTH
        ]
        variable_groups += int(np.any(np.ptp(block, axis=0) > 1e-12))
    variable_fraction = variable_groups / len(groups)
    f2_passed = (
        mse_improvement >= 0.01
        and correlation >= 0.05
        and correlation_ci[0] > 0.0
        and variable_fraction >= 0.5
    )

    decisions, baseline_pairwise, treatment_pairwise = group_metrics(
        groups, arrays, prediction
    )
    baseline_regret = float(np.mean([row["baseline_regret"] for row in decisions]))
    treatment_regret = float(np.mean([row["treatment_regret"] for row in decisions]))
    mean_difference = treatment_regret - baseline_regret
    regret_ci = bootstrap_regret_difference(decisions)
    baseline_top1 = float(np.mean([row["baseline_top1"] for row in decisions]))
    treatment_top1 = float(np.mean([row["treatment_top1"] for row in decisions]))
    slices = protected_slices(decisions)
    b1_passed = (
        f2_passed
        and baseline_regret - treatment_regret >= 0.05
        and regret_ci[1] < 0.0
        and treatment_top1 >= baseline_top1 - 0.01
        and treatment_pairwise >= baseline_pairwise - 0.005
        and all(value["passed"] for value in slices.values())
    )
    if not f2_passed:
        classification = "o2_exact_teacher_unidentifiable"
    elif not b1_passed:
        classification = "o2_exact_teacher_signal_b1_null"
    else:
        classification = "o2_exact_teacher_b1_confirmation_eligible"
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "classification": classification,
        "foundation": {
            "passed": True,
            "scientific_result_id": export["scientific_result_id"],
            "groups": len(groups),
            "candidate_rows": len(arrays["target"]),
            "labeled_candidates": int(np.count_nonzero(labeled)),
            "unlabeled_candidates": int(np.count_nonzero(~labeled)),
            "wall_seconds": export["wall_seconds"],
            "rows_per_second": export["rows_per_second"],
            "peak_rss_bytes": export["peak_rss_bytes"],
            "swap_delta_bytes": export["swap_delta_bytes"],
            "zero_swap_growth": export["zero_swap_growth"],
        },
        "identifiability": {
            "passed": f2_passed,
            "baseline_residual_mse": baseline_mse,
            "treatment_residual_mse": treatment_mse,
            "relative_mse_improvement": mse_improvement,
            "residual_pearson": correlation,
            "residual_pearson_game_block_ci95": correlation_ci,
            "variable_feature_group_fraction": variable_fraction,
            "selected_ridge_lambda_by_held_out_game": {
                str(key): value for key, value in lambdas.items()
            },
        },
        "decision_treatment": {
            "passed": b1_passed,
            "baseline_mean_regret": baseline_regret,
            "treatment_mean_regret": treatment_regret,
            "regret_improvement": baseline_regret - treatment_regret,
            "treatment_minus_baseline_regret": mean_difference,
            "game_block_ci95": regret_ci,
            "baseline_top1_recall": baseline_top1,
            "treatment_top1_recall": treatment_top1,
            "baseline_pairwise_accuracy": baseline_pairwise,
            "treatment_pairwise_accuracy": treatment_pairwise,
            "protected_slices": slices,
        },
        "claim_boundary": {
            "open_train_only": True,
            "validation_opened": False,
            "sealed_test_opened": False,
            "gameplay_run": False,
            "score_claim": False,
            "progress_to_100_claim": False,
        },
    }
    result["result_id"] = _canonical_hash(result)
    return result


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json_atomic(args.output, analyze(args.export_root))


if __name__ == "__main__":
    main()
