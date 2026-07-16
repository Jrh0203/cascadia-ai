"""Validate policy parity and timing for matched Gumbel execution ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

RULESET_ID = "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_16"
DEFAULT_MIN_WALL_SPEEDUP = 1.05
DEFAULT_MAX_ROOT_VALUE_DRIFT = 2.0e-5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise ValueError(f"report is not passing: {path}")
    if report.get("ruleset_id") != RULESET_ID:
        raise ValueError(f"ruleset mismatch in {path}")
    if report.get("control", {}).get("kind") != "none":
        raise ValueError(f"execution comparison requires control=none in {path}")
    if report.get("scientific_eligibility") != "candidate_only_search_arm":
        raise ValueError(f"unexpected scientific eligibility in {path}")
    if not report.get("source_revision"):
        raise ValueError(f"missing source revision in {path}")
    return report


def _search_without_parallel_rollouts(report: dict[str, Any]) -> dict[str, Any]:
    search = dict(report.get("search", {}))
    search.pop("parallel_leaf_rollouts", None)
    return search


def _artifact_identity(report: dict[str, Any]) -> dict[str, Any]:
    artifacts = report.get("artifacts", {})
    keys = (
        "binary_sha256",
        "manifest_sha256",
        "weights_sha256",
        "checkpoint_tag",
        "checkpoint_step",
        "q_quantiles",
    )
    identity = {key: artifacts.get(key) for key in keys}
    identity["q_decomposition"] = bool(artifacts.get("q_decomposition", False))
    if any(value is None for value in identity.values()):
        raise ValueError("report artifact identity is incomplete")
    return identity


def _load_decisions(path: Path, seeds: list[int]) -> dict[tuple[int, int], dict[str, Any]]:
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw:
            continue
        row = json.loads(raw)
        if row.get("type") != "gumbel_decision":
            raise ValueError(f"non-decision row in {path}")
        if row.get("ruleset_id") != RULESET_ID:
            raise ValueError(f"decision ruleset mismatch in {path}")
        key = (int(row["seed"]), int(row["ply"]))
        if key in rows:
            raise ValueError(f"duplicate decision {key} in {path}")
        rows[key] = row
    expected = {(seed, ply) for seed in seeds for ply in range(80)}
    if rows.keys() != expected:
        raise ValueError(f"decision coverage mismatch in {path}")
    return rows


def _load_games(path: Path, seeds: list[int]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw:
            continue
        row = json.loads(raw)
        if row.get("type") != "gumbel_game_done":
            raise ValueError(f"non-game row in {path}")
        if row.get("ruleset_id") != RULESET_ID:
            raise ValueError(f"game ruleset mismatch in {path}")
        seed = int(row["seed"])
        if seed in rows:
            raise ValueError(f"duplicate game seed {seed} in {path}")
        scores = row.get("scores", [])
        if len(scores) != 4:
            raise ValueError(f"game seed {seed} does not contain four scores")
        for score in scores:
            category_total = (
                sum(int(value) for value in score.get("wildlife", []))
                + sum(int(value) for value in score.get("habitat", []))
                + int(score.get("nature_tokens", 0))
            )
            if category_total != int(score.get("total", -1)):
                raise ValueError(f"score category sum mismatch for seed {seed}")
        rows[seed] = row
    if rows.keys() != set(seeds):
        raise ValueError(f"game seed coverage mismatch in {path}")
    return rows


def build_comparison(
    baseline_report_path: Path,
    candidate_report_path: Path,
    baseline_decisions_path: Path,
    candidate_decisions_path: Path,
    baseline_games_path: Path,
    candidate_games_path: Path,
    source_revision: str | None = None,
    min_wall_speedup: float = DEFAULT_MIN_WALL_SPEEDUP,
    max_root_value_drift: float = DEFAULT_MAX_ROOT_VALUE_DRIFT,
) -> dict[str, Any]:
    if min_wall_speedup <= 0.0:
        raise ValueError("minimum wall speedup must be positive")
    if max_root_value_drift < 0.0:
        raise ValueError("maximum root-value drift must be non-negative")
    baseline = _load_report(baseline_report_path)
    candidate = _load_report(candidate_report_path)
    revision = str(baseline["source_revision"])
    if candidate.get("source_revision") != revision:
        raise ValueError("source revision mismatch between execution reports")
    if source_revision is not None and revision != source_revision:
        raise ValueError("reports do not match the required source revision")
    if baseline.get("seeds") != candidate.get("seeds"):
        raise ValueError("seed mismatch between execution reports")
    seeds = [int(seed) for seed in baseline["seeds"]]
    if not seeds:
        raise ValueError("execution comparison requires at least one seed")
    if baseline.get("execution") != candidate.get("execution"):
        raise ValueError("execution topology mismatch between reports")
    if _search_without_parallel_rollouts(baseline) != _search_without_parallel_rollouts(candidate):
        raise ValueError("search settings differ beyond parallel_leaf_rollouts")
    if baseline.get("search", {}).get("parallel_leaf_rollouts") is not False:
        raise ValueError("baseline must disable parallel leaf rollouts")
    if candidate.get("search", {}).get("parallel_leaf_rollouts") is not True:
        raise ValueError("candidate must enable parallel leaf rollouts")
    artifact_identity = _artifact_identity(baseline)
    if _artifact_identity(candidate) != artifact_identity:
        raise ValueError("artifact identity mismatch between execution reports")

    baseline_decisions = _load_decisions(baseline_decisions_path, seeds)
    candidate_decisions = _load_decisions(candidate_decisions_path, seeds)
    action_differences: list[dict[str, Any]] = []
    root_value_differences: list[float] = []
    timing_pairs: list[tuple[float, float]] = []
    invariant_fields = (
        "action_count",
        "simulations_run",
        "market_branches_searched",
        "market_chance_samples",
        "total_simulations_run",
        "exact_endgame",
    )
    for key in sorted(baseline_decisions):
        left = baseline_decisions[key]
        right = candidate_decisions[key]
        if any(left.get(field) != right.get(field) for field in invariant_fields):
            raise ValueError(f"decision invariant mismatch at seed {key[0]} ply {key[1]}")
        left_action = (left.get("chosen_action_id"), left.get("free_three_of_a_kind_choice"))
        right_action = (right.get("chosen_action_id"), right.get("free_three_of_a_kind_choice"))
        if left_action != right_action:
            action_differences.append({"seed": key[0], "ply": key[1]})
        root_value_differences.append(abs(float(left["root_value"]) - float(right["root_value"])))
        timing_pairs.append((float(left["decision_seconds"]), float(right["decision_seconds"])))

    baseline_games = _load_games(baseline_games_path, seeds)
    candidate_games = _load_games(candidate_games_path, seeds)
    score_differences = [
        seed for seed in seeds if baseline_games[seed]["scores"] != candidate_games[seed]["scores"]
    ]
    decision_count_differences = [
        seed
        for seed in seeds
        if baseline_games[seed].get("decision_count")
        != candidate_games[seed].get("decision_count")
    ]

    baseline_summary = baseline["strategies"]["gumbel-search"]
    candidate_summary = candidate["strategies"]["gumbel-search"]
    baseline_wall = float(baseline["candidate_wall_seconds"])
    candidate_wall = float(candidate["candidate_wall_seconds"])
    baseline_mean_decision = float(baseline_summary["mean_total_decision_seconds"])
    candidate_mean_decision = float(candidate_summary["mean_total_decision_seconds"])
    wall_speedup = baseline_wall / candidate_wall if candidate_wall > 0.0 else None
    mean_decision_speedup = (
        baseline_mean_decision / candidate_mean_decision
        if candidate_mean_decision > 0.0
        else None
    )
    policy_parity = not action_differences and not score_differences
    observed_max_root_value_drift = max(root_value_differences, default=0.0)
    exact_numeric_parity = observed_max_root_value_drift == 0.0
    numeric_parity_within_tolerance = observed_max_root_value_drift <= max_root_value_drift
    performance_gate_pass = bool(
        policy_parity
        and numeric_parity_within_tolerance
        and wall_speedup is not None
        and wall_speedup >= min_wall_speedup
    )
    inputs = {
        "baseline_report_sha256": _sha256(baseline_report_path),
        "candidate_report_sha256": _sha256(candidate_report_path),
        "baseline_decisions_sha256": _sha256(baseline_decisions_path),
        "candidate_decisions_sha256": _sha256(candidate_decisions_path),
        "baseline_games_sha256": _sha256(baseline_games_path),
        "candidate_games_sha256": _sha256(candidate_games_path),
    }
    return {
        "status": "pass",
        "scientific_eligibility": "engineering_execution_parity_only",
        "ruleset_id": RULESET_ID,
        "source_revision": revision,
        "seeds": seeds,
        "execution": baseline["execution"],
        "search": _search_without_parallel_rollouts(baseline)
        | {"baseline_parallel_leaf_rollouts": False, "candidate_parallel_leaf_rollouts": True},
        "artifact_identity": artifact_identity,
        "comparison": {
            "decision_count": len(baseline_decisions),
            "action_difference_count": len(action_differences),
            "first_action_difference": action_differences[0] if action_differences else None,
            "score_difference_seeds": score_differences,
            "decision_count_difference_seeds": decision_count_differences,
            "root_value_max_abs_difference": observed_max_root_value_drift,
            "root_value_mean_abs_difference": (
                sum(root_value_differences) / len(root_value_differences)
                if root_value_differences
                else 0.0
            ),
            "policy_parity": policy_parity,
            "exact_numeric_parity": exact_numeric_parity,
            "numeric_parity_within_tolerance": numeric_parity_within_tolerance,
        },
        "timing": {
            "baseline_wall_seconds": baseline_wall,
            "candidate_wall_seconds": candidate_wall,
            "wall_speedup": wall_speedup,
            "baseline_mean_decision_seconds": baseline_mean_decision,
            "candidate_mean_decision_seconds": candidate_mean_decision,
            "mean_decision_speedup": mean_decision_speedup,
            "baseline_p50_decision_seconds": baseline["candidate_decision_seconds_p50"],
            "candidate_p50_decision_seconds": candidate["candidate_decision_seconds_p50"],
            "baseline_p95_decision_seconds": baseline["candidate_decision_seconds_p95"],
            "candidate_p95_decision_seconds": candidate["candidate_decision_seconds_p95"],
            "baseline_decision_seconds_sum": sum(left for left, _ in timing_pairs),
            "candidate_decision_seconds_sum": sum(right for _, right in timing_pairs),
        },
        "minimum_wall_speedup": min_wall_speedup,
        "maximum_root_value_drift": max_root_value_drift,
        "performance_gate_pass": performance_gate_pass,
        "inputs": inputs,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    comparison = report["comparison"]
    timing = report["timing"]
    lines = [
        "# Gumbel Execution Parity and Performance",
        "",
        f"- Source revision: `{report['source_revision']}`",
        f"- Seeds: `{len(report['seeds'])}`",
        f"- Decisions compared: `{comparison['decision_count']}`",
        f"- Action differences: `{comparison['action_difference_count']}`",
        f"- Score differences: `{len(comparison['score_difference_seeds'])}`",
        f"- Maximum root-value drift: `{comparison['root_value_max_abs_difference']:.9g}`",
        f"- Root-value drift tolerance: `{report['maximum_root_value_drift']:.9g}`",
        f"- Wall: `{timing['baseline_wall_seconds']:.3f}s` -> "
        f"`{timing['candidate_wall_seconds']:.3f}s` (`{timing['wall_speedup']:.3f}x`)",
        f"- Mean decision: `{timing['baseline_mean_decision_seconds']:.6f}s` -> "
        f"`{timing['candidate_mean_decision_seconds']:.6f}s` "
        f"(`{timing['mean_decision_speedup']:.3f}x`)",
        f"- Performance gate pass: `{report['performance_gate_pass']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--baseline-decisions", required=True)
    parser.add_argument("--candidate-decisions", required=True)
    parser.add_argument("--baseline-games", required=True)
    parser.add_argument("--candidate-games", required=True)
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--min-wall-speedup", type=float, default=DEFAULT_MIN_WALL_SPEEDUP)
    parser.add_argument(
        "--max-root-value-drift", type=float, default=DEFAULT_MAX_ROOT_VALUE_DRIFT
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    report = build_comparison(
        Path(args.baseline_report),
        Path(args.candidate_report),
        Path(args.baseline_decisions),
        Path(args.candidate_decisions),
        Path(args.baseline_games),
        Path(args.candidate_games),
        args.source_revision or None,
        args.min_wall_speedup,
        args.max_root_value_drift,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, Path(args.summary_out))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
