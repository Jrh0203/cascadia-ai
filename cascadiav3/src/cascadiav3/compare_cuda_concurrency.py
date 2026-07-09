"""Validate and select the CUDA jobs12/16/24 Gumbel batch-concurrency knee."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from .compare_gumbel_execution import (
    DEFAULT_MAX_ROOT_VALUE_DRIFT,
    RULESET_ID,
    _artifact_identity,
    _load_decisions,
    _load_games,
    _load_report,
    _sha256,
)

JOBS = (12, 16, 24)
REFERENCE_JOBS = 12
DEFAULT_MIN_THROUGHPUT_SPEEDUP = 1.05
DEFAULT_KNEE_TOLERANCE = 0.02


def _load_gpu_profile(path: Path) -> dict[str, Any]:
    rows: list[tuple[float, float, float, float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        fields = [field.strip() for field in raw.split(",")]
        if len(fields) != 4:
            raise ValueError(f"GPU profile row does not have four fields: {path}")
        try:
            row = tuple(float(field) for field in fields)
        except ValueError as error:
            raise ValueError(f"GPU profile contains a non-numeric field: {path}") from error
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f"GPU profile contains a non-finite field: {path}")
        utilization, power, memory, _temperature = row
        if not 0.0 <= utilization <= 100.0 or power < 0.0 or memory < 0.0:
            raise ValueError(f"GPU profile contains an invalid measurement: {path}")
        rows.append(row)
    if len(rows) < 30:
        raise ValueError(f"GPU profile needs at least 30 samples: {path}")

    def aggregate(index: int) -> dict[str, float]:
        values = [row[index] for row in rows]
        return {"mean": mean(values), "min": min(values), "max": max(values)}

    return {
        "samples": len(rows),
        "gpu_utilization_percent": aggregate(0),
        "power_draw_watts": aggregate(1),
        "memory_used_mib": aggregate(2),
        "temperature_celsius": aggregate(3),
        "sha256": _sha256(path),
    }


def _execution_without_concurrency(report: dict[str, Any]) -> dict[str, Any]:
    execution = dict(report.get("execution", {}))
    execution.pop("requested_jobs", None)
    execution.pop("parallel_game_cap", None)
    return execution


def _validate_execution(report: dict[str, Any], jobs: int, seed_count: int) -> None:
    execution = report.get("execution", {})
    if execution.get("requested_jobs") != jobs:
        raise ValueError(f"jobs{jobs} report requested_jobs mismatch")
    if execution.get("parallel_game_cap") != min(jobs, seed_count):
        raise ValueError(f"jobs{jobs} report parallel_game_cap mismatch")
    expected = {
        "runner": "gumbel-benchmark-batch",
        "batch_runner": True,
        "seed_scheduler": "dynamic_seed_queue",
        "shared_model_session": True,
        "bridge_process_topology": "one_shared_bridge",
        "maximum_concurrent_bridge_processes": 1,
        "device": "cuda",
        "seed_count": seed_count,
    }
    for key, value in expected.items():
        if execution.get(key) != value:
            raise ValueError(f"jobs{jobs} execution field {key} mismatch")


def _compare_arm(
    baseline_decisions: dict[tuple[int, int], dict[str, Any]],
    candidate_decisions: dict[tuple[int, int], dict[str, Any]],
    baseline_games: dict[int, dict[str, Any]],
    candidate_games: dict[int, dict[str, Any]],
    max_root_value_drift: float,
) -> dict[str, Any]:
    action_differences: list[dict[str, int]] = []
    root_value_differences: list[float] = []
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

    score_difference_seeds = [
        seed
        for seed in sorted(baseline_games)
        if baseline_games[seed]["scores"] != candidate_games[seed]["scores"]
    ]
    decision_count_difference_seeds = [
        seed
        for seed in sorted(baseline_games)
        if baseline_games[seed].get("decision_count")
        != candidate_games[seed].get("decision_count")
    ]
    observed_drift = max(root_value_differences, default=0.0)
    policy_parity = (
        not action_differences
        and not score_difference_seeds
        and not decision_count_difference_seeds
    )
    numeric_parity = observed_drift <= max_root_value_drift
    return {
        "decision_count": len(baseline_decisions),
        "action_difference_count": len(action_differences),
        "first_action_difference": action_differences[0] if action_differences else None,
        "score_difference_seeds": score_difference_seeds,
        "decision_count_difference_seeds": decision_count_difference_seeds,
        "root_value_max_abs_difference": observed_drift,
        "root_value_mean_abs_difference": (
            sum(root_value_differences) / len(root_value_differences)
            if root_value_differences
            else 0.0
        ),
        "policy_parity": policy_parity,
        "numeric_parity_within_tolerance": numeric_parity,
        "eligible_for_knee_selection": policy_parity and numeric_parity,
    }


def build_comparison(
    arms: dict[int, tuple[Path, Path, Path]],
    source_revision: str | None = None,
    min_throughput_speedup: float = DEFAULT_MIN_THROUGHPUT_SPEEDUP,
    knee_tolerance: float = DEFAULT_KNEE_TOLERANCE,
    max_root_value_drift: float = DEFAULT_MAX_ROOT_VALUE_DRIFT,
    gpu_profiles: dict[int, Path] | None = None,
) -> dict[str, Any]:
    if set(arms) != set(JOBS):
        raise ValueError(f"concurrency comparison requires exactly jobs {JOBS}")
    if not math.isfinite(min_throughput_speedup) or min_throughput_speedup <= 0.0:
        raise ValueError("minimum throughput speedup must be positive")
    if not math.isfinite(knee_tolerance) or not 0.0 <= knee_tolerance < 1.0:
        raise ValueError("knee tolerance must be in [0, 1)")
    if not math.isfinite(max_root_value_drift) or max_root_value_drift < 0.0:
        raise ValueError("maximum root-value drift must be non-negative")
    if gpu_profiles is not None and set(gpu_profiles) != set(JOBS):
        raise ValueError(f"GPU profiles must cover exactly jobs {JOBS}")

    reports = {jobs: _load_report(paths[0]) for jobs, paths in arms.items()}
    reference = reports[REFERENCE_JOBS]
    revision = str(reference["source_revision"])
    if source_revision is not None and revision != source_revision:
        raise ValueError("reports do not match the required source revision")
    seeds = [int(seed) for seed in reference.get("seeds", [])]
    if len(seeds) < max(JOBS):
        raise ValueError("concurrency comparison needs at least 24 seeds to fill jobs24")
    reference_search = reference.get("search")
    reference_execution = _execution_without_concurrency(reference)
    reference_artifacts = _artifact_identity(reference)
    for jobs, report in reports.items():
        if report.get("source_revision") != revision:
            raise ValueError("source revision mismatch between concurrency arms")
        if report.get("seeds") != seeds:
            raise ValueError("seed mismatch between concurrency arms")
        if report.get("search") != reference_search:
            raise ValueError("search settings differ between concurrency arms")
        if report.get("search", {}).get("parallel_leaf_rollouts") is not False:
            raise ValueError("CUDA concurrency calibration requires serial leaf rollouts")
        if _execution_without_concurrency(report) != reference_execution:
            raise ValueError("execution topology differs beyond requested jobs")
        _validate_execution(report, jobs, len(seeds))
        if _artifact_identity(report) != reference_artifacts:
            raise ValueError("artifact identity mismatch between concurrency arms")

    loaded_decisions = {
        jobs: _load_decisions(paths[1], seeds) for jobs, paths in arms.items()
    }
    loaded_games = {jobs: _load_games(paths[2], seeds) for jobs, paths in arms.items()}
    reference_decisions = loaded_decisions[REFERENCE_JOBS]
    reference_games = loaded_games[REFERENCE_JOBS]
    arm_results: dict[str, Any] = {}
    reference_wall = float(reference["candidate_wall_seconds"])
    for jobs in JOBS:
        report = reports[jobs]
        wall = float(report["candidate_wall_seconds"])
        mean_decision = float(
            report["strategies"]["gumbel-search"]["mean_total_decision_seconds"]
        )
        p50_decision = float(report["candidate_decision_seconds_p50"])
        p95_decision = float(report["candidate_decision_seconds_p95"])
        timings = (wall, mean_decision, p50_decision, p95_decision)
        if (
            not all(math.isfinite(value) for value in timings)
            or wall <= 0.0
            or mean_decision <= 0.0
            or p50_decision < 0.0
            or p95_decision < 0.0
        ):
            raise ValueError(f"jobs{jobs} report contains invalid timing data")
        comparison = _compare_arm(
            reference_decisions,
            loaded_decisions[jobs],
            reference_games,
            loaded_games[jobs],
            max_root_value_drift,
        )
        arm_results[str(jobs)] = {
            "jobs": jobs,
            "wall_seconds": wall,
            "games_per_hour": len(seeds) * 3600.0 / wall,
            "throughput_speedup_vs_jobs12": reference_wall / wall,
            "mean_decision_seconds": mean_decision,
            "mean_decision_ratio_vs_jobs12": (
                mean_decision
                / float(reference["strategies"]["gumbel-search"]["mean_total_decision_seconds"])
            ),
            "p50_decision_seconds": p50_decision,
            "p95_decision_seconds": p95_decision,
            "comparison_vs_jobs12": comparison,
            "inputs": {
                "report_sha256": _sha256(arms[jobs][0]),
                "decisions_sha256": _sha256(arms[jobs][1]),
                "games_sha256": _sha256(arms[jobs][2]),
            },
            "gpu_profile": (
                _load_gpu_profile(gpu_profiles[jobs]) if gpu_profiles is not None else None
            ),
        }

    eligible = [
        arm for arm in arm_results.values() if arm["comparison_vs_jobs12"]["eligible_for_knee_selection"]
    ]
    if not eligible:
        raise ValueError("no concurrency arm passed policy and numeric parity")
    fastest = min(eligible, key=lambda arm: arm["wall_seconds"])
    best_speedup = float(fastest["throughput_speedup_vs_jobs12"])
    if best_speedup < min_throughput_speedup:
        recommended_jobs = REFERENCE_JOBS
        recommendation = "retain_jobs12_no_material_throughput_gain"
    else:
        near_fastest = [
            arm
            for arm in eligible
            if arm["wall_seconds"] <= fastest["wall_seconds"] * (1.0 + knee_tolerance)
        ]
        recommended_jobs = min(int(arm["jobs"]) for arm in near_fastest)
        recommendation = "adopt_measured_cuda_concurrency_knee"

    return {
        "status": "pass",
        "scientific_eligibility": "engineering_cuda_concurrency_only",
        "ruleset_id": RULESET_ID,
        "source_revision": revision,
        "seeds": seeds,
        "search": reference_search,
        "execution_common": reference_execution,
        "artifact_identity": reference_artifacts,
        "maximum_root_value_drift": max_root_value_drift,
        "minimum_throughput_speedup": min_throughput_speedup,
        "knee_tolerance": knee_tolerance,
        "arms": arm_results,
        "selection": {
            "fastest_jobs": int(fastest["jobs"]),
            "best_speedup_vs_jobs12": best_speedup,
            "recommended_jobs": recommended_jobs,
            "recommendation": recommendation,
            "change_from_jobs12": recommended_jobs != REFERENCE_JOBS,
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# CUDA Gumbel Concurrency Calibration",
        "",
        f"Source revision: `{report['source_revision']}`",
        f"Seeds: `{len(report['seeds'])}`",
        "",
        "| Jobs | Wall | Games/hour | Speedup vs 12 | Mean decision | P95 | GPU mean | Power mean | Action diffs | Max root drift |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for jobs in JOBS:
        arm = report["arms"][str(jobs)]
        comparison = arm["comparison_vs_jobs12"]
        gpu_profile = arm.get("gpu_profile")
        gpu_mean = (
            f"{gpu_profile['gpu_utilization_percent']['mean']:.1f}%"
            if gpu_profile is not None
            else "n/a"
        )
        power_mean = (
            f"{gpu_profile['power_draw_watts']['mean']:.1f}W"
            if gpu_profile is not None
            else "n/a"
        )
        lines.append(
            f"| {jobs} | {arm['wall_seconds']:.1f}s | {arm['games_per_hour']:.2f} | "
            f"{arm['throughput_speedup_vs_jobs12']:.3f}x | "
            f"{arm['mean_decision_seconds']:.3f}s | {arm['p95_decision_seconds']:.3f}s | "
            f"{gpu_mean} | {power_mean} | "
            f"{comparison['action_difference_count']} | "
            f"{comparison['root_value_max_abs_difference']:.3g} |"
        )
    selection = report["selection"]
    lines.extend(
        [
            "",
            f"Recommendation: `{selection['recommendation']}`; jobs=`{selection['recommended_jobs']}`.",
            "Engineering throughput/parity evidence only; no gameplay-strength claim.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for jobs in JOBS:
        parser.add_argument(f"--jobs{jobs}-report", required=True)
        parser.add_argument(f"--jobs{jobs}-decisions", required=True)
        parser.add_argument(f"--jobs{jobs}-games", required=True)
        parser.add_argument(f"--jobs{jobs}-gpu-profile", required=True)
    parser.add_argument("--source-revision", default="")
    parser.add_argument(
        "--min-throughput-speedup", type=float, default=DEFAULT_MIN_THROUGHPUT_SPEEDUP
    )
    parser.add_argument("--knee-tolerance", type=float, default=DEFAULT_KNEE_TOLERANCE)
    parser.add_argument(
        "--max-root-value-drift", type=float, default=DEFAULT_MAX_ROOT_VALUE_DRIFT
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    arms = {
        jobs: (
            Path(getattr(args, f"jobs{jobs}_report")),
            Path(getattr(args, f"jobs{jobs}_decisions")),
            Path(getattr(args, f"jobs{jobs}_games")),
        )
        for jobs in JOBS
    }
    gpu_profiles = {
        jobs: Path(getattr(args, f"jobs{jobs}_gpu_profile")) for jobs in JOBS
    }
    report = build_comparison(
        arms,
        args.source_revision or None,
        args.min_throughput_speedup,
        args.knee_tolerance,
        args.max_root_value_drift,
        gpu_profiles,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, Path(args.summary_out))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
